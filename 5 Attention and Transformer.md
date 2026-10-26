# Attention and Transformer

> [!tip] 互动入口
> [[互动实验入口]] · [[assets/4 Attention andTransformer/interactive/causal-mask-explorer.html|打开 causal mask 互动]]
>
> 交互的主视觉是 query–key mask 矩阵：按步骤只增加一个 query 行，或显示完整矩阵。每次变化都应同时读当前行、允许读取的 token 和被遮住的未来 token；静态课件截图仍保留在相邻小节。

> Lecture 08｜面向 AI chip design 博士生的零基础讲义
>
> 本章承接 [[4 Sequence Models and Attention]]：上一章已经介绍了 RNN encoder-decoder、固定 context 的 information bottleneck 和基础 attention。本章回答更大的问题：**如何把 attention 抽象成通用计算层，并用它构建可以高度并行的 Transformer？**

> **学习导航**：[[1 Basic neural networks]] → [[2 CNN]] → [[3 RNN]] → [[4 Sequence Models and Attention]] → 当前。读完本篇后，你已具备用 tensor shape 和数据流分析 Transformer 的基础；继续深入芯片设计还需要补齐数字逻辑、体系结构、片上/片外存储、并行调度、编译器和实测性能分析。

## 老师开场：Transformer 到底改变了什么

Transformer 不是一个神秘的“更大的 RNN”。它把“当前 token 应该读取哪些 token”变成一组可以批量计算的矩阵操作。给定输入矩阵 $X$，模型通过三个投影得到：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

可以把它们先记成三个问题：`Query` 是“我在找什么”，`Key` 是“我有什么标签”，`Value` 是“真正要交付的内容”。$QK^T$ 计算匹配分数，softmax 把每一行变成权重，再用权重混合 $V$。

### 为什么它更容易并行，却不代表一定更省

训练时整段 token 已经存在，所以多个位置可以同时计算 Q/K/V、score 和 weighted sum；这比一条 RNN 时间链更适合大规模矩阵乘法。但全局 self-attention 要考虑很多 query-key pairs，序列长度为 $N$ 时，score matrix 的形状是 $N\times N$。当 $N$ 变大，问题可能从“算力不够”变成“中间结果和 K/V 搬不动”。

自回归生成又是另一种工作负载：下一个 token 依赖上一个刚生成的 token，仍然逐步 decode。`KV cache` 保存过去 token 的 K/V，避免每一步重新计算，但 cache 的容量和带宽会随层数、head 数、序列长度和 batch 增长。

所以本章我会反复让你区分三件事：**训练能否并行、生成能否并行、数据是否能被近端存储复用**。只说“Transformer 并行”是不完整的工程结论。

## 0. 学习目标与先修知识

你不需要先会 Transformer 或 NLP。读完本章，你应该能够：

1. 用输入/输出 shape 写出 General Attention Layer 的完整数据流；
2. 解释 `Query`、`Key`、`Value` 为什么要分开，以及 scaled dot-product attention 为什么除以 $\sqrt{d_k}$；
3. 说明 self-attention 与 cross-attention 的区别；
4. 解释为什么 self-attention 需要 positional encoding，以及 causal mask 如何防止看到未来 token；
5. 拆解 multi-head attention、residual connection、LayerNorm、MLP 和 Transformer encoder/decoder block；
6. 从 AI chip 角度分析 attention 的矩阵乘法、$O(N^2)$ 的 score matrix、memory bandwidth、训练并行性和 autoregressive decoding latency；
7. 说明 Transformer 如何支持 pretraining、transfer learning、GPT 和 Vision Transformer（ViT）。

### 0.0 如果你完全没有 AI 基础：先建立这 8 个概念

下面这些词不是本章的“附加知识”，而是读懂后面公式的最低地基。第一次阅读时，先能用自己的话解释它们，不必马上背公式。

| 词              | 小白解释                                   | 在芯片里对应什么                                       |
| -------------- | -------------------------------------- | ---------------------------------------------- |
| feature / 特征   | 用数字描述一个对象的一组量。例如一个 token 先被表示成 512 个数字 | 一段定长数据，通常存放在 SRAM/DRAM                         |
| vector / 向量    | 一排数字，例如 `[0.2, -1.1, 0.7]`             | 一个 token 的 feature row                         |
| matrix / 矩阵    | 很多向量排成表格                               | 适合映射到 GEMM、systolic array、Tensor Core          |
| tensor / 张量    | 带有多个维度的数字容器；矩阵只是二维 tensor              | batch、sequence、head、channel 等维度的统一表示           |
| parameter / 参数 | 模型通过训练学出来的数字，例如 $W_Q$                  | weight memory、片上 buffer 和数据搬运                  |
| forward / 前向计算 | 把输入送进模型，得到预测                           | inference datapath 或 training forward pipeline |
| loss / 损失      | 用一个数字表示预测有多差，越小通常越好                    | 训练时的 reduction 和统计量                            |
| gradient / 梯度  | 告诉每个参数应该往哪个方向调整                        | backward 中的乘加、存储和同步                            |

一个很小的例子：若 batch 中有 2 个句子，每句 4 个 token，每个 token 用 8 个数字表示，输入 shape 就是 `X: [2, 4, 8]`。这里 `2` 是 batch，`4` 是 sequence length，`8` 是 feature dimension。**shape 不是装饰信息，它决定矩阵乘法是否合法，也决定硬件需要多少存储。**

### 0.1a 训练、验证、推理分别在做什么？

- **训练 training**：模型先做 forward 得到预测，再计算 loss，接着用 gradient 更新 parameter。重复很多次，模型才逐渐学会规律。
- **验证 validation**：用模型没有参与更新的数据检查是否真的学会，而不是只记住训练样本。
- **推理 inference**：参数固定，只做 forward，例如给大语言模型一个 prompt 后生成下一个 token。

本章重点是 inference 中的 attention 数据流，但训练和推理的硬件行为不同：训练可以同时处理整段已知序列；自回归推理必须先生成前一个 token，才能计算下一个 token。

### 0.1b 读公式的固定方法

以后看到任何公式，按四步读：

1. 先找每个符号代表什么数据，不要先做代数；
2. 给每个数据写 shape，例如 `Q: [Nq, dk]`；
3. 检查乘法的中间维度是否相等；
4. 最后问这个结果要被谁读取、存在哪里、搬运几次。

例如 $Y=AV$ 不是一句“矩阵相乘”就结束：$A$ 是“每个 query 对所有 input 的权重”，$V$ 是被读取的内容，所以 $Y$ 的行数必须等于 query 数。

### 0.2 分层术语 roadmap

```text
Level 0  复习：一次“读取信息”的操作
  query -> input features -> score -> weighted context
      |
Level 1  通用 attention（第 1–2 节）
  key/value -> Q,K,V projections -> score matrix E
  -> softmax weights A -> output Y
      |
Level 2  self-attention 的约束（第 3–5 节）
  self-attention -> permutation equivariance -> positional encoding
  -> mask -> causal self-attention
      |
Level 3  Transformer block（第 6–8 节）
  multi-head -> residual -> LayerNorm -> MLP
  -> encoder block -> decoder block -> cross-attention
      |
Level 4  规模化与视觉迁移（第 9–11 节）
  pretraining -> transfer learning -> GPT
  -> image patch token -> ViT -> DETR / Swin
```

| 读到哪一层 | 你应该能做什么 |
|---|---|
| Level 0 | 解释“当前查询从一组输入中读取加权信息” |
| Level 1 | 给定 $X,Q,W_Q,W_K,W_V$，写出 $E,A,Y$ 的 shape |
| Level 2 | 说明没有位置编码和没有 causal mask 时会发生什么 |
| Level 3 | 画出一个 Transformer block 的数据流 |
| Level 4 | 从训练吞吐、推理延迟和 memory traffic 角度比较 Transformer 与 RNN/CNN |

### 0.3 总体动机：为什么 attention 要从 RNN 中“独立出来”？

上一章的 RNN attention 已经能让 decoder 在不同时间步读取不同输入位置，但仍有一个结构性限制：RNN 的 hidden state 按时间串行更新。长序列需要等待前一个时间步，训练时还要保存很多 intermediate states。

Transformer 的设计选择是：把“从一组向量中读取相关信息”做成独立的 attention layer，让同一序列中的 token 可以通过矩阵乘法同时交互；再用 positional encoding 补回顺序信息，用 MLP 逐 token 做非线性变换。

**工程视角**：这不是简单地把 RNN 换成更大的网络。

- **性能/并行性**：训练时所有位置的 Q/K/V 和 score 可以批量计算；
- **内存系统**：attention score matrix 的元素数量随序列长度平方增长，可能由 bandwidth 和 SRAM 容量主导；
- **控制与部署**：causal generation 仍然是逐 token 的，训练并行不等于推理并行；
- **验证**：mask、padding、position index 和 residual shape 错一个，结果可能仍能运行但语义已经错误。

<svg viewBox="0 0 920 190" role="img" aria-label="From query and input features to attention output, then to a Transformer block" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="at-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#475569"/></marker></defs>
<rect width="920" height="190" rx="12" fill="#f8fafc"/>
<g font-family="Arial, sans-serif" font-size="14" text-anchor="middle">
<g fill="#dbeafe" stroke="#2563eb" stroke-width="2"><rect x="25" y="58" width="130" height="58" rx="10"/><text x="90" y="83" fill="#1e3a8a">Input X</text><text x="90" y="102" fill="#1e3a8a">N x D</text></g>
<g fill="#ffffff" stroke="#64748b" stroke-width="1.5"><rect x="195" y="58" width="135" height="58" rx="10"/><text x="262" y="83" fill="#1f2937">Q, K, V</text><text x="262" y="102" fill="#1f2937">linear projections</text><rect x="370" y="58" width="145" height="58" rx="10"/><text x="442" y="83" fill="#1f2937">QK^T / sqrt(d_k)</text><text x="442" y="102" fill="#1f2937">score matrix E</text><rect x="555" y="58" width="115" height="58" rx="10"/><text x="612" y="83" fill="#1f2937">softmax</text><text x="612" y="102" fill="#1f2937">A weights</text></g>
<g fill="#dcfce7" stroke="#16a34a" stroke-width="2"><rect x="710" y="58" width="85" height="58" rx="10"/><text x="752" y="83" fill="#14532d">AV</text><text x="752" y="102" fill="#14532d">Y</text><rect x="820" y="58" width="75" height="58" rx="10"/><text x="857" y="83" fill="#14532d">Block</text><text x="857" y="102" fill="#14532d">output</text></g>
<g stroke="#475569" stroke-width="2" marker-end="url(#at-arrow)"><line x1="155" y1="87" x2="195" y2="87"/><line x1="330" y1="87" x2="370" y2="87"/><line x1="515" y1="87" x2="555" y2="87"/><line x1="670" y1="87" x2="710" y2="87"/><line x1="795" y1="87" x2="820" y2="87"/></g>
<path d="M90 123 C90 165 752 165 752 123" fill="none" stroke="#16a34a" stroke-width="1.7" stroke-dasharray="6 4" marker-end="url(#at-arrow)"/><text x="420" y="181" fill="#166534">residual path: add the original representation back</text>
</g></svg>

读图时把 attention 看成一个可学习的 data movement：先决定“读谁”（score/softmax），再决定“读多少”（weighted sum），最后把结果送入 residual、normalization 和 MLP。

## 1. 从 seq2seq attention 到通用信息读取层

### 1.1 复习：上一章的 context vector

在 RNN encoder-decoder 中，encoder 产生一组 hidden outputs，decoder 当前状态产生一个 query。attention score 衡量 query 与每个输入位置的匹配程度，再把输入加权相加得到 context vector：

$$
\mathbf{c}_t=\sum_{i=1}^{T}\alpha_{t,i}\mathbf{h}^{\mathrm{enc}}_i
$$

这里的 `query` 可以理解成“我现在想找什么”，输入 hidden output 是“可被读取的信息”。

![[assets/4 Attention andTransformer/AT-12.png]]

> **截图导航（PDF p.12）**：课件把 image captioning 抽象为 input features 加 query，再通过 attention 读取信息。重点是 attention 不只属于翻译，它是一个通用 information-reading layer。

### 1.2 为什么要拆成 Query、Key、Value？

如果只有一组输入向量，同时用它们做“匹配”和“被读取的内容”，表达能力会受限。通用 attention 把每个输入拆成两种角色：

- **Key**：用于和 Query 比较，回答“这个位置是否相关？”
- **Value**：真正被加权读取的内容，回答“如果相关，要取什么信息？”

Query 来自当前需求；Key 像索引；Value 像索引指向的内容。一个 query 可以匹配多个 key，再把对应 values 混合起来。

这是一种直觉类比，不是计算机内存的字面实现。硬件上 Q/K/V 仍然是 tensor，需要通过矩阵乘法产生、存储和搬运。

**先记住一个不会混淆的版本：** Query 是“我正在寻找的条件”，Key 是“每个位置用来被比较的标签”，Value 是“真正要取出的内容”。Key 相似只能说明“应该读这个位置”，并不等于输出就是 Key；输出来自 Value。

![[assets/4 Attention andTransformer/AT-16.png]]

> **截图导航（PDF p.16）**：从一般输入向量到分开的 Q、K、V。Q/K 决定相关性，V 决定输出内容。

## 2. General Attention Layer：公式、shape 与最小例子

设输入向量集合为：

$$
X\in\mathbb{R}^{N_x\times D_x}
$$

其中 `N_x` 是输入向量个数，`D_x` 是每个向量的 feature dimension。查询集合为：

$$
Q\in\mathbb{R}^{N_q\times D_q}
$$

通过 learned projections 得到 Key 和 Value：

$$
K=XW_K,\qquad V=XW_V
$$

若 $W_K\in\mathbb{R}^{D_x\times d_k}$、$W_V\in\mathbb{R}^{D_x\times d_v}$，则：

$$
K\in\mathbb{R}^{N_x\times d_k},qquad V\in\mathbb{R}^{N_x\times d_v}
$$

### 2.1 Score、Normalize、Read

这一节可以拆成三个动作：

1. **Score**：用点积比较 query 和 key，得到“相关程度”；
2. **Normalize**：用 softmax 把一行分数变成非负且总和为 1 的权重；
3. **Read**：用权重对 value 加权求和，得到当前 query 的结果。

因此 attention 不是凭空生成信息，而是在已有 Value 中做一次 input-dependent read。softmax 前的 `E` 允许是负数，softmax 后的 `A` 才像概率分布；`A` 的每一行对应一个 query。

第一步是计算每个 query 与每个 key 的匹配分数：

$$
E=\frac{QK^T}{\sqrt{d_k}}
\in\mathbb{R}^{N_q\times N_x}
$$

`E[i,j]` 表示第 `i` 个 query 对第 `j` 个输入位置的 raw score。除以 $\sqrt{d_k}$ 是为了避免 $d_k$ 较大时 dot product 的方差变大，使 softmax 过早饱和，梯度变得不稳定。

然后沿 key/input axis 做 softmax：

$$
A=\operatorname{softmax}(E,\text{key axis})
\in\mathbb{R}^{N_q\times N_x}
$$

最后对 Value 做加权和：

$$
Y=AV\in\mathbb{R}^{N_q\times d_v}
$$

![[assets/4 Attention andTransformer/AT-17.png]]

> **截图导航（PDF p.17）**：这是本章最重要的公式总览。读 shape 时注意：score/weight matrix 是 `query 数量 × input 数量`；输出的行数跟 query 数量相同。

### 2.2 最小数值例子

假设只有一个 query、两个 key/value，raw scores 为：

$$
E=[2,1]
$$

softmax 得到：

$$
A\approx[0.731,0.269]
$$

如果 values 是二维向量：

$$
V=\begin{bmatrix}10&0\\0&4\end{bmatrix}
$$

则输出为：

$$
Y=AV\approx[7.31,1.08]
$$

这说明 attention 不是简单选择一个 value，而是根据相关性进行连续加权。若分数差距变大，权重会更接近 hard selection；若分数接近，输出会更像平均融合。

### 2.3 用一个生活化例子把 General Attention 想清楚

先暂时忘掉公式。想象你在一本很长的书里回答问题：**“主角为什么离开家？”**

- 书里每个句子是一个可读取的位置（input）。
- 问题本身是 query：它表达这次阅读的需求。
- 每个句子会被映射成 key（便于和问题比较的“索引特征”）以及 value（句子真正携带、供答案使用的信息）。
- 模型先比较问题和每个 key，判断哪些句子值得看；再把相应 value 按权重混合，形成回答当前问题的向量。

```text
问题 query ──与每个 key 比较──> 相关性分数 ──softmax──> 阅读比例
                                                              │
书中句子的 values ────────────────────────────────────────────┴──加权求和──> 结果
```

这个比喻的重点不是“模型真的像人一样查目录”，而是**匹配用的信息（K）和取出的内容（V）分工不同**。高分表示某个位置更值得读取；输出是 value 的加权组合，不是分数本身，也通常不是直接挑出唯一一句。

#### 把比喻对应回张量

假设输入有 3 个位置，每个位置用 4 个数字表示；现在有 2 个 query。于是 $X$ 是 $3\times4$，$Q$ 是 $2\times d_k$。投影矩阵把输入特征变换到 key/value 空间：$K=XW_K$、$V=XW_V$。然后：

1. $QK^T$ 得到 $2\times3$ 分数矩阵：两行分别对应两个问题，三列分别对应三个输入位置。
2. 每行各自做 softmax，得到两行读取比例；每一行非负且和为 1。
3. 用这张 $2\times3$ 权重表乘 $V$。结果有两行——**一个 query 产出一个结果**。

可用小表格记形状（暂不考虑 batch 和 head）：

| 张量 | 形状 | 直观含义 |
|---|---:|---|
| $Q$ | $N_q\times d_k$ | 有多少个读取需求，每个需求的特征长度 |
| $K$ | $N_x\times d_k$ | 有多少个可匹配位置 |
| $V$ | $N_x\times d_v$ | 每个位置真正可读取的内容 |
| $A=softmax(QK^T/\sqrt{d_k})$ | $N_q\times N_x$ | 每个 query 对每个位置分配多少权重 |
| $Y=AV$ | $N_q\times d_v$ | 每个 query 读出的结果 |

**最容易混淆的地方：** $Q$ 的行数决定输出有几行；$K$ 决定“看哪里”；$V$ 决定“读到什么”。Key/value 的行数相同，是因为它们描述同一批输入位置，但特征维度可以不同。$Q$ 可以来自另一组数据，因此 general attention 的 query 数与输入位置数不必相等；当 Q、K、V 都由同一个序列产生时，才是 self-attention 的常见情形。

> [!example] 一句话心算
> “先拿 query 去和所有 key 打分，再把分数变成比例，最后按比例混合 value。”背住这句话，就能顺着读完整个 general attention 公式。

### 2.4 General attention 的工程代价

对 batch size $B$、query 数 $N_q$、input 数 $N_x$，核心 score 矩阵约有 $B N_q N_x$ 个元素。即便 QKᵀ 的乘法可以映射到 GEMM，softmax 和保存/读取 $A$ 仍可能成为 memory bottleneck。

**AI chip perspective**：

- Q/K/V projection 是规则 GEMM，适合 systolic array 或 tensor core；
- QKᵀ 是另一个 GEMM，但输出 score matrix 可能很大；
- Softmax 需要 row-wise max、exponentiation、sum 和 normalization；
- AV 是按 attention weights 对 V 做矩阵乘法；
- fused attention 可以减少把完整 $A$ 写回 SRAM/DRAM 的次数，但增加 control 和 verification 复杂度。

### 2.5 Batch 和 multi-head 下的完整 shape

前面的公式省略了 batch 和 head，便于第一次理解；真实 Transformer 通常同时处理多个样本和多个 head。设 batch size 为 $B$、query 数为 $N_q$、input 数为 $N_x$、head 数为 $H$、每 head 维度为 $d_k$，则可以把一个 head 的数据写成：

$$
Q,K,V\in\mathbb{R}^{B\times H\times N\times d_k}
$$

对于 self-attention，$N_q=N_x=N$，score 和 weight 的 shape 是：

$$
E,A\in\mathbb{R}^{B\times H\times N\times N}
$$

最后两个维度分别表示 query position 和 key position；每个 batch、每个 head、每一行都要独立做 softmax。输出仍然是 $[B,H,N,d_v]$，之后把 head 维和 $d_v$ 维拼回 model dimension。

这个扩展解释了为什么“一个公式看起来只是矩阵乘法”，实现却需要大量 reshape、transpose 和 memory layout 管理。任何一次维度排列错误都可能让程序仍然运行，却把 batch/head/query/key 的含义混在一起；验证时应同时检查 shape、mask 广播方向和每行权重和。

### 2.6 Softmax 的数值稳定性

直接计算 $e^{e_i}$ 可能 overflow；score 很小时也可能 underflow 到 0。实现通常使用 **log-sum-exp trick**：先减去当前行最大值 $m$，再计算：

$$
\operatorname{softmax}(e_i)=\frac{\exp(e_i-m)}{\sum_j\exp(e_j-m)},
\qquad m=\max_j e_j
$$

减去同一个常数不会改变 softmax 的结果，却能把最大指数限制在 $\exp(0)=1$ 附近。causal mask 使用 $-\infty$ 或足够小的有限值，也是为了让禁止位置的权重变为 0，同时避免产生 NaN。对 AI chip 设计，这意味着 softmax 不是普通逐元素 activation：它包含 row-wise max、指数近似、sum reduction 和 reciprocal，常与 score/weighted sum 融合以减少中间 tensor 的写回。

## 3. Self-attention：Query 也来自输入本身

### 3.1 定义与直觉

在 self-attention 中，输入只有一个集合 $X$。同一个 $X$ 经过三组不同的线性变换，产生 $Q=XW_Q$、$K=XW_K$、$V=XW_V$。所以“self”指的是信息来源相同，不是说每个 token 只能看自己。

对第 $i$ 个 token 来说，它会：

1. 生成自己的 query，表达“我需要哪些上下文”；
2. 与所有 token 的 key 比较；
3. 按得到的权重读取所有 token 的 value；
4. 用读取结果更新自己的表示。

这就是 token 之间建立联系的具体过程。一次 layer 后，每个 token 都可能混入其他位置的信息。

在 general attention 中，Q 可以来自另一个来源；在 **self-attention** 中，同一个输入序列 `X` 同时产生 Q、K、V：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

因此每个 input vector 都会问：“我应该从序列中的哪些位置读取信息？”输出仍然有一个向量对应每个输入位置。

![[assets/4 Attention andTransformer/AT-21.png]]

> **截图导航（PDF p.21）**：self-attention 的关键变化是 query 不再来自 decoder，而是由同一组输入向量投影得到。

### 3.2 Self-attention 做的不是“平均”，而是 input-dependent mixing

CNN 使用固定 kernel 在局部邻域混合；self-attention 使用由输入内容决定的权重，在所有位置之间混合。若两个 token 的内容关系强，它们之间的 score 就可能高；不同输入会产生不同的 mixing matrix `A`。

![[assets/4 Attention andTransformer/AT-34.png]]

| 机制 | 混合邻域 | 权重 | 主要优势 | 主要代价 |
|---|---|---|---|---|
| CNN | 固定 local neighborhood | 共享 kernel，输入无关 | local locality、规则访存、成本随图像大小近似线性 | 远距离关系要靠堆层或扩大 receptive field |
| RNN | 前一步 hidden state | recurrent update | 顺序建模直观、状态可流式 | 时间方向串行、长依赖难训练 |
| Self-attention | 全部或被 mask 的位置 | input-dependent score | 任意位置直接交互、训练位置维度并行 | score matrix 和带宽随长度平方增长 |

这三者不是简单的“旧方法/新方法”关系，而是不同的 inductive bias：CNN 偏好局部规则，RNN 偏好时间递推，self-attention 偏好内容相关的全局读取。

## 4. 为什么 self-attention 需要 positional encoding？

### 4.1 没有位置时，顺序会丢失

attention 本身只看到一组向量和它们之间的相似度。把输入 token 的顺序打乱，若不提供位置线索，模型无法知道“猫追狗”和“狗追猫”的顺序差异。更准确地说，self-attention 对输入排列具有 permutation equivariance：输入怎么换序，输出也会跟着换序，但它不会自动知道哪个位置是第 1 个、第 2 个。

如果只给 self-attention 一组向量，而不给任何位置或顺序相关的 mask，那么它只看到“有哪些向量”和它们之间的内容关系。把输入行重新排列，输出也会按同样方式重新排列：

$$
F(PX)=PF(X)
$$

这叫 **permutation equivariance**。它对无序 set 是好性质，但语言中 `dog bites man` 和 `man bites dog` 的 token 集合相同、顺序不同，意义却不同。

### 4.2 设计选择：把位置向量加到 token embedding

原始 Transformer 使用：

$$
Z=X+P
$$

`P` 是 position representation，与 token embedding 具有相同 shape。它可以是 learned lookup table，也可以是预先设计的 fixed function。加法保持模型宽度不变；concatenation 是另一种可能，但会增加输入 dimension。

![[assets/4 Attention andTransformer/AT-25.png]]

> **截图导航（PDF p.25）**：课件指出 self-attention 本身没有 order signal，必须额外编码位置。

**注意边界**：learned positional table 通常对训练长度内的位置最可靠；直接外推到更长 sequence 不自动保证有效。位置编码不是“让模型知道真实世界的绝对坐标”，而是给网络一个可利用的顺序信号。

## 5. Masked self-attention：防止偷看答案

### 5.1 为什么训练语言模型需要 mask？

#### 交互：逐步打开 causal mask

打开 [Causal mask explorer](assets/4 Attention andTransformer/interactive/causal-mask-explorer.html)，点击“下一步”或“显示完整 mask”。观察每个 query 位置只能看到自己与左侧的 key，右侧未来位置保持被屏蔽。

Obsidian 原生打开：[[assets/4 Attention andTransformer/interactive/causal-mask-explorer.html|打开 Causal mask 互动页面]]

> [!tip]- Obsidian 内原生练习（不需要 HTML）
> 1. 对第 1 个 token，允许读取哪些位置？
> 2. 对第 3 个 token，允许读取哪些位置？
> 3. 在一个 5×5 矩阵中，把主对角线及其左下方标为“允许”，右上方标为“禁止”。
> 4. 上面的 HTML 链接是可选的独立互动页；如果按钮不动，使用本节的公式和课件截图。

这是一个教学插图；静态结论不依赖 HTML：允许的位置满足 j<=i，未来位置的 score 加上极小值后，softmax 权重接近 0。mask 限制可见性，不删除 token，也不减少参数。

> **老师讲解重点**：把矩阵的行读成 query、列读成 key。先固定一个 query 行，问“它能看到哪些列”；再把这个可见范围和 next-token prediction 对照。mask 的作用是保证训练时遵守生成时的信息边界，不是减少参数。

**Obsidian 内互动（JS Engine）**

点击“下一步”，逐行查看 causal mask。允许读取的位置用 ✓ 表示，未来位置用 × 表示。

```js-engine
const box = container.createDiv();
box.createEl("b", { text: "老师讲解：每行 query 只能读取自己和过去的 key" });
const controls = box.createDiv();
const previous = controls.createEl("button", { text: "上一个 query" });
const next = controls.createEl("button", { text: "下一个 query" });
const all = controls.createEl("button", { text: "完整 mask" });
const result = box.createEl("pre");
let position = 0;
function show() {
  const lines = [];
  for (let i = 0; i < 5; i++) {
    let line = "";
    for (let j = 0; j < 5; j++) line += j <= i ? "✓ " : "× ";
    lines.push((i === position ? "> " : "  ") + "query " + i + ": " + line);
  }
  result.textContent = "列 = key 位置；✓ = 可读取，× = 被 mask\n当前 query " + position + " 可读 key 0–" + position +
    "；未来位置不可见。mask 限制信息，不删除 token，也不减少参数。\n" + lines.join("\n");
}
previous.addEventListener("click", () => { position = Math.max(0, position - 1); show(); });
next.addEventListener("click", () => { position = Math.min(4, position + 1); show(); });
all.addEventListener("click", () => { position = 4; show(); });
show();
return;
```

语言模型训练的目标是：给定前面的 token，预测下一个 token。假设句子是 `我 喜欢 芯片`，预测“芯片”时只能使用“我 喜欢”，不能直接读取答案“芯片”。causal mask 就是把未来位置的 score 设为 $-\infty$，这样 softmax 后这些位置权重为 0。

注意 mask 改变的是“允许读取哪些位置”，不是删除 token，也不是减少模型参数。

训练 next-token prediction 时，位置 `t` 的目标是预测下一个 token。如果 query 在训练时能读取未来 token，模型就会直接看到答案，loss 虽低，推理时却无法复现。

因此在 score matrix `E` 上加入 mask `M`：

$$
A=\operatorname{softmax}(E+M,\text{key axis})
$$

允许的位置 $M_{ij}=0$，禁止未来读取的位置设为一个足够小的值（概念上是 $-\infty$）。softmax 后，被 mask 的位置权重接近 0。

![[assets/4 Attention andTransformer/AT-40.png]]

> **截图导航（PDF p.40）**：causal mask 让当前位置只能看自己和过去，不能看未来。它是训练自回归语言模型的正确性约束。

### 5.2 训练并行 ≠ 自回归生成并行

这是 Transformer 最容易被误解的地方：

- **训练**：完整 target sequence 已知，causal mask 保证每个位置不能看未来，但所有位置的矩阵计算可以一次性 batch 完成；
- **生成**：第 `t+1` 个 token 依赖模型刚生成的第 `t` 个 token，所以仍然必须逐步生成。

![[assets/4 Attention andTransformer/AT-36.png]]

对 AI chip 设计，这意味着 training accelerator 可以追求大矩阵吞吐，而 decode accelerator 更关心单 token latency、KV cache、batching 和 memory bandwidth。

## 6. Multi-head self-attention

### 6.1 为什么需要多个 head？

一个 head 只有一组 Q/K/V 投影，往往会把所有关系压在同一个表示空间里。Multi-head 把 $d_{\\text{model}}$ 切成多个较小的子空间：每个 head 独立算一次 attention，最后把各 head 的结果拼接，再做一次线性变换。

可以把它理解成多个观察者：一个 head 关注相邻词，一个 head 关注主谓关系，另一个 head 关注长距离依赖。这个类比帮助理解功能，但具体关注模式必须通过训练得到，不能靠 head 编号预先指定。

每个 head 独立产生一组 Q/K/V 和输出：

$$
\operatorname{head}_r=\operatorname{Attention}(XW_Q^{(r)},XW_K^{(r)},XW_V^{(r)})
$$

然后拼接所有 head，再通过输出投影：

$$
\operatorname{MHA}(X)=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_H)W_O
$$

![[assets/4 Attention andTransformer/AT-41.png]]

> **截图导航（PDF p.41）**：multi-head 的数据流是 project & split → 每个 head 做 attention → concatenate → output projection。

**硬件视角**：head 之间天然可并行，但每个 head 的小矩阵可能降低阵列利用率；实现通常会把 head dimension、batch 和 sequence dimension 重新排布，以获得连续内存访问和更高 GEMM occupancy。head 数不是越多越好，更多 head 也意味着更多 metadata、reshape 和 kernel 调度边界。

## 7. Transformer block：Attention + MLP + Residual + LayerNorm

### 7.1 Block 的总体结构

Transformer block 可以先看成一条流水线：

`输入表示 -> attention 混合 token 间信息 -> residual 加回原输入 -> LayerNorm 稳定数值 -> MLP 逐 token 做非线性变换 -> 再次 residual/LayerNorm`。

Attention 负责“token 之间通信”，MLP 负责“每个 token 内部加工”。两者分工不同：去掉 attention，token 之间几乎不能交换信息；去掉 MLP，模型的逐 token 非线性表达能力会明显减弱。

Transformer 不是只有 attention。一个 block 通常包含：

1. self-attention 或 masked self-attention；
2. residual connection，把 block 输入加回 attention 输出；
3. LayerNorm，稳定每个 token 的 feature scale；
4. position-wise MLP，对每个 token 独立做两层线性变换和 activation；
5. 第二条 residual connection 和 LayerNorm。

课件强调：block 中 token 之间真正发生 interaction 的地方是 self-attention；LayerNorm 和 MLP 可以对每个 token 独立执行。

![[assets/4 Attention andTransformer/AT-45.png]]

> **截图导航（PDF p.45）**：Transformer block 的关键组成是 attention、MLP、residual 和 normalization。先理解数据流，再记具体的 pre-norm/post-norm 变体。

### 7.2 LayerNorm 的公式

对一个 token 的 feature vector $h\in\mathbb{R}^{D}$，LayerNorm 沿 feature 维度计算均值和方差：

$$
\mu=\frac{1}{D}\sum_{j=1}^{D}h_j,
\qquad
\sigma^2=\frac{1}{D}\sum_{j=1}^{D}(h_j-\mu)^2
$$

归一化并恢复可学习的尺度和偏移：

$$
z=\frac{h-\mu}{\sqrt{\sigma^2+\varepsilon}},
\qquad
y=\gamma\odot z+\beta
$$

其中 $\gamma,\beta\in\mathbb{R}^{D}$ 是 learned parameters。LayerNorm 与 BatchNorm 的关键区别是：它对每个 token 的 feature 维度做归一化，不依赖 batch 中其他样本的统计量。

![[assets/4 Attention andTransformer/AT-48.png]]

### 7.3 Position-wise MLP

attention 负责 token 之间的信息交换；MLP 负责每个 token 内部的 feature transformation：

$$
\operatorname{MLP}(x)=W_2\,\phi(W_1x+b_1)+b_2
$$

同一组 $W_1,W_2$ 复用于所有 sequence positions。因而 MLP 可以视为对最后一维做 batched GEMM，不会在 token 之间直接通信。

### 7.4 为什么 residual connection 有用？

如果一个新 block 还没有学会有用变换，至少可以通过 shortcut 近似 identity：

$$
y=x+F(x)
$$

这样信息和 gradient 都有较直接的路径。代价是两个分支的 shape 和 dtype 必须匹配；在硬件上还要付出 activation read/add/write，以及可能的 synchronization。

## 8. Transformer Encoder–Decoder Architecture

### 8.1 Encoder

Encoder 接收完整 source sequence，通常允许每个 token 读取 source 中的所有位置，因此适合理解输入，例如分类、检索和编码。

encoder 接收带 positional information 的 source representations。每层 encoder block 通常使用 unmasked self-attention，因此 source 中的每个位置可以读取其他 source positions；然后经过 MLP、residual 和 normalization，得到 encoder outputs。

### 8.2 Decoder

Decoder 通常有两种 attention：第一种是 masked self-attention，保证生成时不偷看未来；第二种是 cross-attention，让 decoder 当前状态读取 encoder 输出。前者回答“我已经生成了什么”，后者回答“source 中哪些内容与当前输出有关”。

decoder 有两种 attention：

1. **Masked self-attention**：decoder target 位置只能读取已经生成的 target prefix；
2. **Cross-attention**：decoder query 读取 encoder outputs 的 key/value，把 source 信息注入当前 decoder state。

![[assets/4 Attention andTransformer/AT-61.png]]

> **截图导航（PDF p.61）**：decoder block 的核心是 masked self-attention + cross-attention。前者维持自回归因果性，后者读取 encoder 的 source representation。

### 8.3 Transformer 与 RNN 的区别

| 方面 | RNN | Transformer |
|---|---|---|
| 训练时间方向 | recurrent dependency，难完全并行 | sequence positions 可批量做 attention |
| 长距离关系 | 必须通过 hidden state 传播 | 一个 attention layer 可直接连接远距离位置 |
| 单层交互 | 前后相邻时间步 | 全局或 mask 允许的位置 |
| 推理生成 | 自回归串行 | decoder 仍然自回归串行 |
| 主要硬件压力 | state dependency、低时间并行度 | score matrix、bandwidth、KV cache |
| 主要 inductive bias | 顺序递推 | 内容相关的全局 mixing |

Transformer 解决的是训练并行和长距离交互问题，但没有让 autoregressive generation 变成完全并行。这个区分对性能评估非常重要。

## 9. Pretraining、Transfer Learning 与 GPT

### 9.1 为什么要先 pretrain？

从零训练一个大模型需要大量数据和计算。Pretraining 先用大规模数据学习通用的 token 表示和语言规律；下游任务再用较少数据做 fine-tuning 或 prompting。对芯片设计来说，这意味着同一套模型权重会被许多任务复用，部署时必须关注权重存储、量化和不同 batch/sequence 形状下的利用率。

从零训练一个适用于具体任务的 Transformer，通常需要大量数据才能学到词、句法和长距离模式。**Pretraining** 先在大规模、经过筛选的数据上训练 task-appropriate language objective，例如 next-token prediction；然后把学到的 parameters 迁移到下游任务。

课件的基本流程是：

```text
large corpus -> data curation/filtering -> pretraining objective
             -> pretrained Transformer -> fine-tuning or prompting
```

**Transfer learning** 的关键不是“免费获得知识”，而是复用参数和表示，减少目标任务需要的标注数据和训练时间。代价是预训练数据质量、目标任务分布差异、参数规模和部署成本都需要管理。

![[assets/4 Attention andTransformer/AT-64.png]]

> **截图导航（PDF p.64）**：课件将 pretraining 与 task adaptation 分开。读图时区分“学通用表示”和“适配具体任务”两个阶段。

### 9.2 GPT 应该怎样放入本章？

在本章的抽象中，GPT 属于 decoder-only、causal self-attention 的自回归 Transformer。它使用 masked self-attention 预测后续 token；本章只需要掌握这一结构位置，不把 GPT 的规模、训练配方或产品能力等同于某一个 Transformer block。

## 10. Vision Transformer 与跨模态统一

### 10.1 Image patch 作为 token

ViT 的第一步不是把整张图片直接塞进 Transformer，而是把图片切成许多小 patch。每个 patch 展平后经过线性投影，变成一个 token embedding；再加上二维位置编码，送入 Transformer。这样图像问题被改写成“处理一串 token”的问题。

ViT 的核心想法是把图像切成固定大小的 patches，再把每个 patch flatten 并投影成一个 token embedding：

$$
H\times W\times C
\longrightarrow
N\text{ patches}\times(P^2C)
\longrightarrow
N\text{ tokens}\times D
$$

如果图像大小是 $H\times W$，patch 大小是 $P\times P$，则 patch 数约为：

$$
N=\frac{H}{P}\times\frac{W}{P}
$$

之后可以像处理文字 token 一样加 positional embedding，并送入 Transformer encoder。

![[assets/4 Attention andTransformer/AT-71.png]]

> **截图导航（PDF p.71）**：图像 patch 被当成 Transformer tokens。关键不是把图像变成文字，而是把二维数据转成一串可由 attention 处理的向量。

### 10.2 ViT 的收益与代价

- 收益：统一 token/attention/block 结构，patch 之间可以建模全局关系，适合大规模 pretraining 和 transfer；
- 代价：patch 数量增加会使 attention score matrix 增长，细粒度 patch 带来更高 memory 和 computation；
- 工程边界：较大的 patch 降低 $N$ 和 attention 成本，但可能丢失局部细节；较小的 patch 保留细节，但增加序列长度。

课件还提到 DETR、Swin Transformer、DALL·E 等扩展：它们说明同一 attention/block 基础可以服务 object detection、分层视觉建模和 text-to-image，但不要把这些应用混为一种训练目标。

![[assets/4 Attention andTransformer/AT-74.png]]

> **截图导航（PDF p.74）**：DETR 使用 learned queries 预测无序 object set，体现了“query 从输入特征中读取结果”的 General Attention 视角。

![[assets/4 Attention andTransformer/AT-76.png]]

> **截图导航（PDF p.76）**：课件提到 hierarchy、windows 和 set prediction 等视觉 Transformer 方向；它们通常是在全局 attention 的成本或视觉 inductive bias 上做调整。

## 11. AI chip design：从公式到数据通路

### 11.1 训练阶段：规则 GEMM 与大规模并行

训练时 source/target 的整个 sequence 已知。虽然 causal mask 限制信息可见性，但 Q/K/V projection、QKᵀ、softmax 和 AV 都可以在 batch 和 sequence 维度上批量执行。适合：

- large GEMM / tensor core / systolic array；
- sequence 和 batch tiling；
- fused kernels，减少 score/attention intermediate 的读写；
- activation checkpointing，在计算量和 memory capacity 之间折中。

### 11.2 推理阶段：prefill 与 decode

部署时可区分两个阶段：

- **Prefill**：一次处理用户已有的 prompt，sequence 维度较多，矩阵运算并行度高；
- **Decode**：每次只生成一个新 token，必须使用之前的历史，单 token GEMM 可能变成 memory-bound。

在 decoder-only 模型中，历史 token 的 Key/Value 通常缓存为 **KV cache**。下一步只需为新 token 计算 query，并与历史 K/V 交互；收益是避免重复计算历史 projection，代价是 KV cache 的容量和带宽随 batch、层数、head 数和 sequence length 增长。

这也解释了为什么一次 prompt 长度为 $N$ 的处理和之后生成 $N$ 个 token 不是同一件事：prefill 对 prompt 中许多位置并行计算，单次 attention 的 pair 数随长度平方增长；decode 每次只增加一个 query，但每个新 query 通常仍需读取越来越长的历史 K/V。KV cache 节省重复计算，却没有让长上下文的读取变成常数成本。服务系统会在并发请求的 batch、cache 容量、首 token latency 和每 token latency 之间做取舍。

对单层、batch $B$、head 数 $H$、历史长度 $T$、每 head 维度 $d_k$，KV cache 元素数量近似为：

$$
2BHTd_k
$$

乘以 layer 数和每元素 bytes 后，就是主要的 cache storage 估算。这个公式是芯片设计中把模型配置转成 SRAM/DRAM 压力的第一步。

### 11.3 Attention 的主要成本

| 模块 | 主要计算 | 主要瓶颈 |
|---|---|---|
| Q/K/V projection | GEMM | weight bandwidth、阵列利用率 |
| QKᵀ | 矩阵乘法 | score matrix 大小、片上存储 |
| softmax | max/reduce/exp/sum/divide | reduction、数值稳定性、融合 |
| AV | weighted GEMM | 读取 weights 和 V 的带宽 |
| MLP | 两个大 GEMM + activation | 参数和 activation movement |
| residual/LayerNorm | add、reduce、scale/shift | 小算子 launch、读写和同步 |
| KV cache | append/read K,V | decode 时容量和 memory bandwidth |

### 11.4 为什么“attention 更并行”不等于“总能耗更低”？

#### 当章互动：KV cache 容量估算

改变一个参数，观察 decode 阶段的 KV cache 如何增长。这个计算是理想化容量估算，不是某个模型的实测显存占用；它用于训练你把模型配置翻译成 memory budget。

```js-engine
const box = container.createDiv();
box.createEl("b", { text: "老师讲解：KV cache = 层数 × batch × token 数 × KV 宽度 × K/V × 每元素字节" });
const fields = {};
for (const [name, value] of [["layers", 32], ["batch", 1], ["tokens", 4096], ["kvHeads", 8], ["headDim", 128], ["bytes", 2]]) {
  const row = box.createDiv();
  row.createEl("label", { text: name + " = " });
  const input = row.createEl("input");
  input.type = "number"; input.min = "1"; input.max = "100000"; input.step = "1"; input.value = String(value);
  fields[name] = input;
}
const result = box.createEl("pre");
function show() {
  const elements = 2 * Number(fields.layers.value) * Number(fields.batch.value) * Number(fields.tokens.value) * Number(fields.kvHeads.value) * Number(fields.headDim.value);
  const bytes = elements * Number(fields.bytes.value);
  const gib = bytes / 1024 / 1024 / 1024;
  result.textContent = "K + V 元素数：" + elements.toLocaleString() +
    "\n每元素字节：" + Number(fields.bytes.value) +
    "\n估算容量：" + bytes.toLocaleString() + " bytes ≈ " + gib.toFixed(3) + " GiB" +
    "\n公式：2 × layers × batch × tokens × KV_heads × head_dim × bytes/element" +
    "\n边界：未计入 allocator 对齐、分页/块管理、量化 scale 等额外开销。";
}
for (const input of Object.values(fields)) input.addEventListener("input", show);
show();
return;
```

Transformer 减少了 RNN 的时间串行依赖，但它可能访问更多 token pairs。若 sequence length 增长，理论上 attention score 的 pair 数增长为 $O(N^2)$；如果硬件无法把数据留在近端 memory，搬运 score、K/V 和 activation 的能量可能抵消 arithmetic parallelism 的收益。

因此实际 accelerator 设计会考虑 tiling、window/local attention、sparsity、quantization、operator fusion、KV cache compression 等方向。它们不是本课件全部展开的算法，而是由上述数据流和成本自然引出的系统问题。

## 12. 最小手算与 shape 检查

### 12.1 一次 self-attention 的 shape

设 $N=4$、$D=8$、$d_k=d_v=4$，输入：

$$
X\in\mathbb{R}^{4\times8},\quad W_Q,W_K,W_V\in\mathbb{R}^{8\times4}
$$

则：

$$
Q,K,V\in\mathbb{R}^{4\times4}
$$

$$
E=QK^T\in\mathbb{R}^{4\times4},\quad A\in\mathbb{R}^{4\times4},\quad Y=AV\in\mathbb{R}^{4\times4}
$$

每一行 `A[i,:]` 是第 `i` 个 token 对四个输入位置的 attention distribution，行和应为 1（允许少量浮点误差）。

### 12.2 读一个 Transformer 配置

给定 `d_model=512`、`num_heads=8`，每个 head 的 key dimension 常为：

$$
d_k=\frac{d_{\text{model}}}{H}=64
$$

给定 sequence length `N=2048`，单个 head 的 score matrix 有 $2048^2$ 个位置；8 个 heads 就有 8 倍的 attention weights（实现可能不显式保存全部矩阵）。这就是为什么 sequence length、head 数和 dtype 是硬件容量估算的基本参数。

## 13. 常见混淆

### 13.1 General attention vs self-attention

General attention 允许 Q 与 K/V 来自不同来源；self-attention 让 Q/K/V 都由同一个 X 投影得到。Transformer decoder 的 cross-attention 是 general attention 的一个重要实例：Q 来自 decoder，K/V 来自 encoder outputs。

### 13.2 Self-attention vs cross-attention

self-attention 在同一序列内部建立关系；cross-attention 在两个表示集合之间读取信息。encoder 通常用 self-attention，decoder 同时需要 masked self-attention 和 cross-attention。

### 13.3 Causal mask vs padding mask

causal mask 防止读取未来 token；padding mask 忽略补齐位置。二者解决不同错误，可以同时存在。把 padding mask 当 causal mask 会造成不必要的信息阻断；漏掉 causal mask 则会产生 label leakage。

### 13.4 LayerNorm vs BatchNorm

LayerNorm 通常对每个 token 的 feature dimension 归一化；BatchNorm 使用 batch 统计量。Transformer 常用 LayerNorm，因为 sequence length、batch size 和自回归推理时的 batch 形状可能变化。

### 13.5 Multi-head 不等于多层 Transformer

Multi-head 是同一 attention layer 内的多个投影子空间；多层 Transformer 是多个 block 纵向堆叠。head 数和 layer 数是两个不同的 architecture hyperparameters。

## 14. 从一句话到 Transformer 输出：完整走一遍

下面把前面的概念串成一条完整链路。假设输入是中文句子“我喜欢芯片”，目标是根据已有内容预测下一个 token。真实系统会使用更大的词表和更复杂的 tokenizer，这里只保留结构。

### 14.1 文本先变成数字

神经网络不能直接接收“我”“喜欢”这样的字符。Tokenizer 先把文本切成 token，再为每个 token 分配整数 ID，例如：

```text
文本:       我      喜欢      芯片
token ID:   17      204       891
```

token ID 只是索引，不表示大小关系。ID=891 并不意味着“芯片”比“我”更重要。Embedding table 根据 ID 查出一个向量，把每个 token 变成 `D` 个浮点数：

$$
\text{token IDs }[N]\longrightarrow X\in\mathbb{R}^{N\times D}
$$

这一步是 lookup，不是 attention。对芯片而言，它通常是 embedding memory 的随机或半随机读取。

### 14.2 加入位置后，产生 Q/K/V

把位置向量 $P$ 加到 token embedding $X$ 上得到 $Z=X+P$。随后用三组不同的权重矩阵做线性投影：

$$
Q=ZW_Q,\qquad K=ZW_K,\qquad V=ZW_V
$$

如果 $Z$ 是 $[N,D]$，而每个 head 的维度是 $d_k$，那么一个 head 的 Q/K/V 是 $[N,d_k]$。三个矩阵的输入相同，但参数不同，所以它们承担不同角色：Q/K 用于计算相关性，V 用于携带内容。

### 14.3 先算“看谁”，再算“取什么”

对每个 query 和所有 key 做点积，得到 $E:[N,N]$。第 `i` 行表示第 `i` 个 token 对整个句子的关注分数。加上 causal mask 后，未来位置被禁止。经过按行 softmax，得到 $A:[N,N]$，每一行加起来为 1。

最后 $Y=AV$。这一步把每个 token 的上下文读回来：如果第 `i` 行在第 `j` 列权重很大，输出 `Y[i]` 就会大量包含 `V[j]` 的信息。

### 14.4 经过 block 后得到更丰富的表示

attention 输出先通过 output projection，再与输入做 residual add；LayerNorm 调整 feature 的数值尺度；MLP 对每个 token 独立做非线性变换；第二个 residual 和 normalization 后，得到下一层输入。多层堆叠后，早期层通常先形成局部或词法关系，后续层可以组合出更高层的语义模式，但具体功能不是由结构硬编码的，而是由训练数据和参数共同决定。

### 14.5 从表示到下一个 token

decoder-only 模型通常取最后一个位置的 hidden vector，经过一个 vocabulary projection 得到每个候选 token 的 logit：

$$
\text{logits}=h_{\text{last}}W_{\text{vocab}}+b
$$

softmax 后得到词表上的概率。推理时可以选择概率最大的 token（greedy），也可以按概率采样。选出的 token 被追加到输入序列，下一轮再次执行 decode。这就是为什么自回归生成有时间上的循环，即使每一轮内部的矩阵运算可以并行。

## 14.5 骨灰级工程桥接：从 Transformer 公式到大模型系统

### 14.5.1 Attention 的真实瓶颈不只是 FLOPs

标准 self-attention 形成 $N\times N$ 的 score/weight 关系，因此全量 materialized attention 的空间和计算随序列长度平方增长。FlashAttention 的核心不是改变 attention 数学结果，而是使用 tiling 和 kernel fusion，避免完整 attention matrix 在高层 memory 中反复读写。这个案例说明：相同算法复杂度的实现，memory traffic 可以完全不同。

### 14.5.2 RoPE 与长度外推边界

RoPE（rotary positional embedding）把位置信息作用到 query/key 的旋转变换中，使点积关系带有相对位置信息。它常用于 decoder-only Transformer，但不能因此推断任意长度外推都可靠；训练长度、频率设置、插值策略和任务分布都会影响长 context 行为。

### 14.5.3 MHA、MQA、GQA 与 KV cache

MHA 为每个 head 保留独立 K/V；MQA 让多个 query heads 共享 K/V；GQA 在两者之间分组共享。共享 K/V 可以减少 KV cache 容量和 decode memory bandwidth，但会改变表示容量，必须在目标任务上验证。

KV cache 的近似元素数为：

$$
2\times L\times B\times T\times H_{kv}\times d_k
$$

其中 $L$ 是层数，$B$ 是 batch，$T$ 是历史长度，$H_{kv}$ 是 key/value heads，$d_k$ 是每个 head 的维度，前面的 2 对应 K 和 V；再乘每个元素的 bytes，才得到近似存储量。

### 14.5.4 Prefill、decode 与 continuous batching

Prefill 一次处理已有 prompt，sequence 维度长、矩阵运算密集；decode 每次只新增一个 token，却要读取历史 KV cache，常由 memory bandwidth 和单 token latency 主导。continuous batching 可以动态合并不同请求的 decode 步骤，提高吞吐，但增加调度、cache fragmentation 和尾延迟管理难度。

### 14.5.5 并行化不是一个维度

大模型系统可能组合 data parallel、tensor parallel、pipeline parallel、sequence/context parallel。每种并行都会改变通信量、同步点和显存占用。只说“用了更多 GPU”不能说明系统更快，必须同时报告 batch、sequence length、precision、通信拓扑、有效吞吐和 latency 定义。

### 14.5.6 Quantization 与 speculative decoding

低精度 quantization 可以减少权重和 KV cache 的容量与带宽，但 outlier、累加精度、激活范围和 kernel 支持会影响结果。speculative decoding 用较小 draft model 提议 token，再由 target model 验证；收益取决于接受率、验证 batch、额外模型成本和调度开销，不是无条件加速。

> **External context**：FlashAttention、RoPE、GQA、prefill/decode、continuous batching 和 speculative decoding 是在线论文或官方系统文档支持的进阶背景；本章课件主线仍是 Q/K/V、mask、multi-head、Transformer block 和 KV cache。

## 14.6 教授式完整算例：从 Q/K/V 到 attention 输出

设一个 query 为 $q=[1,0]$，两个 keys 为 $k_1=[1,0]$、$k_2=[0,1]$，两个 values 为 $v_1=[10,0]$、$v_2=[0,4]$。

### 1. Score：我应该读谁？

因为 $d_k=2$，scaled dot-product score 为：

$$
e_1=\frac{qk_1^T}{\sqrt2}=\frac{1}{\sqrt2}\approx0.707,
\qquad e_2=\frac{qk_2^T}{\sqrt2}=0
$$

第一个 key 与 query 更相似，所以第一个 raw score 更大。除以 $\sqrt{d_k}$ 是为了避免维度变大时点积尺度过大，使 softmax 过早饱和。

### 2. Normalize：把 score 变成权重

$$
\alpha_1=\frac{e^{0.707}}{e^{0.707}+e^0}\approx0.67,
\qquad \alpha_2\approx0.33
$$

两个位置仍都能贡献，并且权重和为 1。

### 3. Read：真正取出的是 Value

$$
y=0.67v_1+0.33v_2=[6.7,1.32]
$$

输出不是 key 的加权和，而是 value 的加权和。Key 负责被比较，Value 携带要读出的内容。

### 4. 扩展到矩阵 shape

若有 $N_q$ 个 queries、$N_k$ 个 keys、每个 value 维度为 $d_v$：

$$
Q:[N_q,d_k],\quad K:[N_k,d_k],\quad V:[N_k,d_v]
$$

$$
QK^T:[N_q,N_k],\quad A:[N_q,N_k],\quad AV:[N_q,d_v]
$$

因此 attention 输出的行数跟 query 数量相同，而不是跟 key 数量相同。

### 5. Causal mask 只改 score

如果当前位置不能读取未来 key，就把对应 score 加上概念上的 $-\infty$：

$$
\operatorname{softmax}([0,-\infty])=[1,0]
$$

mask 没有删除 value，也没有减少参数；它只改变哪些 pair 可以参与归一化。训练时位置维度仍可用规则矩阵并行，生成时新 token 仍要等待上一步输出。

> **讲授结论**：Q 提出读取条件，K 决定匹配程度，softmax 产生读取权重，V 提供内容。Transformer 的复杂之处主要来自规模、位置、mask、残差、归一化和 memory system。

### 6. 把这个算例放回整个 Transformer

本算例只展示了一个 query、两个 key/value；真实 multi-head attention 只是把它批量化并分成多个子空间。随后还要经过 output projection、residual connection、LayerNorm 和 MLP。训练时这些步骤一起进入 computation graph，loss 的梯度会回到 Q/K/V projection；推理时则要额外考虑 prefill、decode 和 KV cache。也就是说，**attention 的数学骨架很小，但系统代价来自 shape、重复次数和 memory movement**。

## 15. 一页学习链

```text
seq2seq attention
  -> general information-reading layer
  -> Q/K/V roles
  -> scaled dot-product: E -> softmax A -> Y
  -> self-attention
  -> positional encoding
  -> causal masking
  -> multi-head attention
  -> residual + LayerNorm + MLP
  -> Transformer encoder/decoder
  -> pretraining and transfer learning
  -> GPT / ViT / multimodal systems
```

这条链路可以作为本章的总复习：

```text
文本
  -> tokenizer / token IDs
  -> embedding + positional encoding
  -> Q/K/V projections
  -> score matrix -> mask -> softmax weights
  -> weighted values
  -> residual + LayerNorm + MLP
  -> stacked Transformer blocks
  -> vocabulary logits -> next-token probability
```

从 AI chip 角度，任何一个 Transformer 配置都可以沿三条线理解：

- **形状线**：每个 tensor 的 batch、sequence、head 和 feature 维度是什么；
- **计算线**：哪些部分是规则 GEMM，哪些部分是 reduction、elementwise 或控制依赖；
- **数据线**：权重、activation、score 和 KV cache 在 SRAM、片外 DRAM、寄存器之间如何移动。

## 16. 可选的现象展示

实验目录：[[experiments/README]]。这些脚本是对正文现象的可视化，不是作业，也不要求先完成才能理解本章。正文已经给出结论；运行脚本的作用是让矩阵、权重图和内存曲线变得可见。

- `05_attention_reading.py` 展示 Q/K 决定读取权重、V 决定输出内容；
- `06_position_and_causal_mask.py` 展示位置编码和未来位置的屏蔽；
- `07_multihead_attention.py` 展示一个 layer 内多个 head 的权重张量；
- `08_attention_system_cost.py` 展示 score matrix 的平方增长和 KV cache 的线性增长。

脚本和图片位于两个同步目录：桌面目录 `deep learning/experiments/` 与 Obsidian 目录 `deep learning/experiments/`。它们用于观察本文已经解释过的机制，不替代概念讲解或真实模型评测。

## 来源与截图

- 主要来源：`Attention and Transformer.pdf`，Lan Xu，CS 280 Deep Learning，Lecture 08，2026 Fall，82 pages。
- 截图保存在 `assets/4 Attention andTransformer/`，文件名中的数字对应 PDF 页码。
- Q/K/V 的直觉、shape 例子、AI chip 成本估算和配置分析是教学补充；课件中的 attention、self-attention、positional encoding、mask、multi-head、Transformer、pretraining、GPT 和 ViT 主线均已整合。

## 外部资料与进阶阅读

以下资料用于补充 Transformer 的现代实现边界。访问日期：2026-10-17。

- **External context：标准 attention API**：PyTorch 官方 [scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)，核对 mask、scale、dropout 和实现选择。
- **External context：原始 Transformer**：Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，作为 Q/K/V、multi-head、positional encoding 和 encoder-decoder 的原始论文来源。
- **External context：降低 attention memory traffic**：Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135)，理解 tiling、IO-aware attention 和为什么同样的数学式可以有不同系统代价。
- **External context：位置与 KV heads**：Su et al., [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)；Ainslie et al., [GQA](https://arxiv.org/abs/2305.13245)，分别补充 rotary position 和 grouped-query attention。
- **External context：部署视角**：NVIDIA [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/)，用于继续学习 precision、kernel、fusion 和推理优化；不要把文档中的硬件结果直接外推到其他设备。
- **External course-material extension — attention pooling 与高效实现**：D2L, [Attention Mechanisms and Transformers](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)，作为由简单 attention pooling 进入 Transformer 的教程；PyTorch [`scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 文档核对 API 对 mask、dropout 与 backend 的约定。接口行为可能随 PyTorch 版本变化，部署前应按本地版本核验。访问日期：2026-10-22。
