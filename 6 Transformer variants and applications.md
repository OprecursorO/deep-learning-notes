# Transformer variants and applications

> [!tip] 互动入口
> [[互动实验入口]] · [[assets/5 Transformer variants and applications/interactive/attention-pattern-explorer.html|打开 Attention pattern explorer]]
>
> 页面默认先显示同一坐标尺度下的 full / local / global-token / low-rank pattern；切换窗口或模式时，矩阵保留区域、连接比例和文字结论同步改变。静态 SVG 是脚本不可用时的 overview 回退。

> Lecture 09｜Transformer：Variants and Applications I
>
> 本章承接 [[5 Attention and Transformer]]。前一章已经回答了“attention 和 Transformer block 怎样工作”；本章继续追问：**当序列变长、任务变多、输入从文本扩展到图像和代码时，应该保留哪些连接、使用哪种 Transformer 变体？**
>
> 课件中的命令式句子、论文标题和链接是本章的学习材料，不是需要在笔记外执行的指令。

## 老师开场：没有“最好的 Transformer”，只有适合约束的选择

前一章讲清楚了 full self-attention 如何工作，本章要把它放进真实任务中判断。序列短时，全连接 attention 可以让每个 token 直接交流；序列变长后，$N^2$ 个 query-key pair 会带来计算、存储和带宽压力。此时我们不能只问“精度最高吗”，还要问“哪些关系真的必须保留”。

我会用一个简单的选择问题贯穿本章：如果一个 query 只看附近窗口，远距离信息会丢在哪里？如果保留几个 global token，谁负责汇总全局？如果用 low-rank 或 linear attention，省下的到底是连接数、矩阵形状，还是 IO？每一种 variant 都是在表达能力、近似误差、实现难度和硬件成本之间交换。

### 三类模型先分清楚

- **Encoder-only**（例如 BERT）：看到受规定的输入上下文，输出适合分类、检索或 token-level prediction 的表示；
- **Decoder-only**（例如 GPT）：使用 causal mask，只看过去，逐步预测下一个 token；
- **Encoder-decoder**：先理解输入，再根据输入条件生成输出，适合翻译和摘要等任务。

图像也可以进入这条路线：把图像切成 patch，每个 patch 展平并投影成一个 token，ViT 就能使用 Transformer；Swin 再把局部窗口和 shifted window 放回视觉结构，降低连接量并逐层扩大感受野。

因此本章不是模型名称背诵，而是训练一种工程判断：**输入的结构是什么、任务需要哪些可见关系、保留这些关系要付出多少计算和存储代价**。

## 0. 先确定本章要解决什么

读完后，你应该能：

1. 从 $N^2$ 个 query-key 连接解释为什么长序列 self-attention 会变贵；
2. 用“保留哪些连接”来区分 local、stride、global、clustering、learnable pattern、low-rank 和 linear attention；
3. 区分 encoder-only、decoder-only、encoder-decoder 三类 Transformer 的可见范围和训练目标；
4. 解释 BERT 的 masked language modeling（MLM）与 GPT 的 next-token prediction 为什么适合不同任务；
5. 用一个完整例子理解 Codex 的 `pass@k`，并说明代码为什么适合作为可测试的生成模态；
6. 解释 ViT 如何把图像 patch 变成 token，以及 Swin 为什么采用局部窗口和 shifted window；
7. 从“模型表达能力”和“系统计算成本”两个角度判断一个变体的收益与代价。

### 0.1 术语路线图

```text
问题：N² 连接太贵
  full attention -> attention pattern -> sparsity/locality -> low rank/linear
      |
模型家族：不同任务需要不同可见范围
  encoder-only -> BERT/MLM -> 表示学习与下游任务
  decoder-only -> GPT/causal mask -> next-token generation
  encoder-decoder -> 输入理解 + 输出生成
      |
跨模态：同一个 token-mixing 思路迁移
  text -> image patch -> code -> multimodal sequence
      |
视觉先验：图像有局部性、层次和位移规律
  ViT -> window attention -> shifted windows -> Swin
```

| 读到哪一层 | 先认识的词 | 学到的动作 |
|---|---|---|
| 1 | `attention pattern`、`sparsity`、`locality` | 看懂一个 query 到哪些 key 的连接被保留 |
| 2 | `low-rank`、`linear attention`、`clustering` | 说出每种近似省的是哪一部分计算 |
| 3 | `encoder-only`、`decoder-only`、`causal` | 根据任务选择可见范围 |
| 4 | `MLM`、`next-token`、`fine-tuning` | 解释预训练目标如何塑造模型用途 |
| 5 | `patch token`、`hierarchy`、`shifted window` | 把图像结构映射到 Transformer 设计 |

## 1. 从“每个人都看所有人”开始

### 1.1 学习者能看到的具体问题

在 full self-attention 中，长度为 $N$ 的序列有 $N$ 个 query，每个 query 都和 $N$ 个 key 比较，因此 score matrix 的大小是：

$$
N\times N=N^2.
$$

例如 $N=4$ 时只有 16 个 query-key 对；$N=16$ 时是 256 个；$N=1024$ 时已经是 1,048,576 个。这里的平方增长只描述连接数，不等于整个模型的全部 FLOPs，但它解释了长上下文为什么会给计算、片上存储和内存搬运带来压力。

**注意**：self-attention 只是更大网络中的一个模块。短序列时，投影、MLP 或其他层可能同样重要；当 $N$ 足够大时，attention 的连接矩阵会成为主要负担。课件第 5–6 页强调的正是这个条件，而不是“attention 永远占全部计算”。

### 1.2 `attention pattern`（注意力连接模式）是什么？

把 score matrix 想成一张网格：行是 query，列是 key。某个格子保留，表示这个 query 会计算并可能读取该 key；格子置零或跳过，表示这条连接不参与本层计算。**pattern** 只决定“看谁”，softmax 权重仍然决定“看多少”。

可打开上方互动页面，把序列长度从 4 调到 32，切换模式，再观察蓝色格子的数量。静态地说：full 模式保留全部 $N^2$ 个格子；稀疏模式只保留其中一部分，因此连接更少，但可能丢失信息。

![[assets/5 Transformer variants and applications/source/transformer-p08-08.png|课件中的局部注意力：把远处连接截断为零]]

> **看图方法（课件 PDF p.8）**：蓝色带状区域是局部窗口，灰色区域被跳过。它像 CNN 的局部感受野，但 self-attention 原本允许任意远距离连接，因此局部化会带来表达能力的取舍。

![[assets/5 Transformer variants and applications/diagrams/attention-patterns.svg|1200]]

> **看图方法**：横轴是 key，纵轴是 query。先看“蓝色格子占多少”，再看“蓝色格子连成什么形状”。连接更少通常更省，但也意味着模型不能在这一层直接读取被删掉的位置。

### 1.3 Local / truncated attention：先把视线限制在附近

`local attention`（局部注意力）让 query 只看附近的 key，例如 $|i-j|\leq w$。它的动机是：很多信号在局部范围内具有相关性，没必要为每个 query 扫遍整个序列。若窗口宽度 $w$ 固定，连接数量近似为 $O(Nw)$，比 $O(N^2)$ 更容易扩展。

**直觉**：读一句话时，当前词通常需要看邻近词；读图像时，一个小区域通常和相邻区域有强关系。**边界**：如果关键证据在很远位置，单层局部 attention 看不到它；多层堆叠可以逐步传播，但传播距离、层数和延迟都会增加。课件明确提醒“可能牺牲 performance”，所以局部化是约束，不是免费优化。

#### 一个具体的连接数例子

设序列长度 $N=8$，每个 query 只看自己左右各一个位置，即窗口半径 $w=1$。第 0 个 query 只能看 key 0 和 1；中间 query 通常看 3 个 key；最后一个 query 只能看 6 和 7。连接数为：

$$
2+3+3+3+3+3+3+2=22,
$$

而 full attention 需要 $8^2=64$ 个连接。这里的 22 不是严格的 $N(2w+1)=24$，因为序列边界会截断窗口。随着 $N$ 变大，固定 $w$ 时连接数近似 $N(2w+1)$，因此是线性增长；但模型要获取很远位置的信息，往往需要堆叠多个局部层。

**实现视角**：如果窗口大小固定，地址模式规则，硬件容易批量读取；如果每个 query 的窗口位置不同，虽然理论连接数相同，实际 kernel 可能因为不规则访存而变慢。

### 1.4 Stride attention：少量规则的远距离连接

`stride attention`（步进注意力）在局部连接外，按照固定步长采样远处位置。它试图兼顾两件事：保留邻近细节，同时让 query 能够看到非局部区域。

它的代价是采样规律可能与任务真正需要的依赖不对齐：规则地看每隔几个位置，不保证恰好看到了关键 token。工程上，规则模式有利于地址生成、缓存和并行调度；方法上，它用更少连接换取可能的信息缺失。

可以把 stride 看成两套连接的并集：

```text
局部连接：j ∈ [i-w, i+w]
步进连接：j = i mod s, i mod s + s, i mod s + 2s, ...
```

其中 $s$ 是 stride。第一套连接负责局部细节，第二套连接提供稀疏的远距离观察。stride 不是“随机抽样”：它通常是预先确定、规则可编程的模式，因此比动态稀疏更适合硬件，但也更依赖任务结构。

### 1.5 Global attention：用特殊 token 传递全局信息

`global token`（全局 token）是加入序列的特殊位置。它可以 attend to every token，收集全局信息；其他 token 也可以读取它，从而获得一条全局通道。相较于让所有 query 两两连接，这种设计把“全局汇总”集中到少量位置。

**什么时候有用？** 文档分类或图像分类常需要一个汇总表示；全局 token 可以承担这个角色。**边界是什么？** 如果所有细粒度关系都必须直接交互，只有汇总 token 可能成为信息瓶颈；因此它通常与局部连接或其他稀疏连接一起使用。

把一个特殊 token 记为 $g$，一种简单的 global pattern 可以写成：

$$
\mathcal{E}=\{(i,j): |i-j|\leq w\}\cup\{(i,g),(g,j)\;|\;0\leq i,j<N\}.
$$

第一部分是局部边，第二部分让所有位置都能写入或读取 $g$。这并不保证一层之内所有 token 都直接互相看见；它提供的是一条经过 $g$ 的信息中继路径。

## 2. 其他高效 attention 路线：它们分别省在哪里

前一节主要改变的是连接图。课件第 12–26 页继续展示了其他路线。它们不是一个“谁绝对最好”的排行榜，而是对不同瓶颈做近似。

### 2.1 Clustering：先把可能相关的位置分组

`clustering`（聚类）路线尝试把 query/key 按相似性分组，只在可能相关的组之间计算 attention。课件列出 Reformer 和 Routing Transformer 作为例子。它们的核心问题是：如果能快速估计哪些区域可能有较大 attention，就可以跳过大量小权重区域。

**困难**：聚类本身也需要计算，并且分组错误会把真正重要的连接一起删掉。实现者要在“寻找候选连接的开销”和“少做 attention 的收益”之间平衡。

### 2.2 Learnable patterns：让另一个模块学习连接图

`learnable pattern`（可学习模式）不预先固定网格，而是让另一个 learned module 决定哪些格子应该跳过。课件以 Sinkhorn Sorting Network 为例。

这把“连接谁”也变成学习问题，适应性更强；但系统中增加了一个控制模块，训练与验证都更复杂，且需要确认模式选择不会破坏可复现性和硬件效率。对芯片而言，一个动态稀疏图可能理论计算量更低，却因为不规则访存而变慢。

### 2.3 Low-rank / representative keys：压缩 key 或 attention matrix

`low-rank`（低秩）假设 attention matrix 可以由较少的基向量近似表示。Linformer 属于这类思路：用较少的代表性 key，减少 key 方向的计算量；进一步减少 query 数则会改变输出序列长度，这是课件第 18 页特别指出的边界。

不要把“减少 key”误解为“删除所有信息”。它是在假设许多列具有可压缩结构；若任务的注意力矩阵并不接近低秩，近似误差会增大。方法视角关心误差，系统视角关心压缩后的矩阵乘法和内存访问是否更规整。

### 2.4 Linear attention：改变矩阵乘法的结合顺序

标准 attention 可以写成：

$$
Y=\operatorname{softmax}(QK^T)V.
$$

其中 $QK^T$ 先产生 $N\times N$ 的中间矩阵。`linear attention`（线性注意力）尝试利用特征映射或分解改变计算顺序，避免显式构造完整的 $N\times N$ 矩阵。课件列出 Efficient Attention、Linear Transformer、Random Feature Attention 和 Performer。

**必须保留的限制**：softmax 在中间时会让分解更复杂；因此线性化通常需要改变归一化形式、使用核特征近似，或接受与标准 attention 不同的行为。它不是把任意 softmax attention 严格等价地“变快”，而是在某些假设下换取更好的长度扩展性。

#### 为什么“改变结合顺序”可能省掉 $N^2$

先忽略 softmax，把相似度写成核特征的内积：

$$
\operatorname{sim}(q_i,k_j)=\phi(q_i)^T\phi(k_j).
$$

标准写法会先形成所有 $i,j$ 的相似度，再乘 $V$：

$$
Y_i=\sum_{j=1}^{N}\bigl(\phi(q_i)^T\phi(k_j)\bigr)v_j.
$$

利用结合律，可以先累计 key-value：

$$
S=\sum_{j=1}^{N}\phi(k_j)v_j^T,
\qquad Y_i=\phi(q_i)^T S.
$$

这样不必显式保存 $N\times N$ 矩阵；但它依赖 $\phi$ 的选择和归一化设计。对于真正的 softmax attention，通常需要近似或改写，因此“线性”描述的是某种实现/近似下的长度缩放，不是所有 Transformer attention 的普遍等式。

### 2.5 Attention-free：不再用 Q/K 计算相关性

课件还展示两类更激进的选择：

- `Synthesizer`：不依赖传统 Q/K 的内容匹配，而用其他方式生成混合权重；
- `FNet`、`MLP-Mixer`：用 Fourier mixing 或 MLP mixing 在 token 之间交换信息。

它们回答的是“token mixing 是否一定要靠 Q/K 相似度”。好处是可以避开一部分 attention 计算；代价是失去或改变 input-dependent matching，表达能力和适用任务需要重新验证。不能只看 FLOPs 判断替代方案是否更好。

可以用一个最小对比来避免混淆：

| 机制 | 当前 query 是否根据输入内容选择 key | 混合权重从哪里来 |
|---|---|---|
| Standard self-attention | 是 | $QK^T$ 与 softmax |
| Synthesizer | 不一定 | 另一个 learned module 或固定生成器 |
| FNet | 否 | Fourier transform 的固定混合规则 |
| MLP-Mixer | 不是 Q/K 匹配 | token-mixing MLP 的参数 |

它们都可以让 token 之间交换信息，但“交换信息”不等于“根据当前内容动态检索”。这是比较 attention-free 方法时最重要的概念差异。

### 2.6 一张选择表

| 路线 | 主要保留/压缩对象 | 可能收益 | 主要边界 |
|---|---|---|---|
| Local / truncated | 附近连接 | $O(Nw)$、规则、易调度 | 远距离依赖可能被截断 |
| Stride | 规则的远距离采样 | 少量非局部连接 | 采样可能错过关键位置 |
| Global token | 少量全局汇总点 | 提供全局通道 | 汇总点可能形成瓶颈 |
| Clustering | 可能相关的组 | 跳过低价值区域 | 分组成本与误删风险 |
| Learnable pattern | 由模块决定的图 | 自适应 | 动态稀疏访存不规则 |
| Low-rank / Linformer | key 或矩阵的维度 | 降低矩阵规模 | 依赖低秩近似 |
| Linear / Performer | 中间矩阵与乘法顺序 | 避免显式 $N^2$ | 需要近似或改变归一化 |
| Synthesizer / MLP mixing | Q/K 匹配机制 | 更简单的 token mixing | 相关性机制发生改变 |

## 3. 同一个 Transformer block，三种任务角色

### 3.1 Encoder-only：先把输入变成有意义的表示

`encoder-only`（仅编码器）只保留 encoder。它的工作是“compute meaningful embeddings”：对整段输入建立表示，再交给分类、标注、检索等下游头部。它通常允许输入 token 双向互相看，因为目标不是按时间顺序生成下一个 token。

这解释了它和 decoder-only 的第一处关键差异：不是网络大小不同，而是**可见范围和训练目标不同**。一个适合填空和理解整句的模型，不一定适合逐 token 写出新句子。

### 3.2 BERT：MLM + 下游 fine-tuning

课件第 35–44 页以 BERT 为例。`BERT` 是 *Bidirectional Encoder Representations from Transformers*。它先把 token 输入 encoder，再从编码后的 embedding 预测目标。

#### MLM：遮住一部分词，再恢复它

`masked language model (MLM)`（遮挡语言模型）把句子中的部分 token 替换为 `[MASK]`，让 encoder 利用左右两侧上下文预测原词。课件示例中，`cute` 和 `reading` 被遮住，projection 分别预测它们。

完整路径是：

```text
[CLS] A [MASK] teddy bear is [MASK] [SEP]
       -> bidirectional encoder
       -> 两个被遮位置的 projection
       -> cute, reading
```

它训练的是“从完整上下文理解缺失内容”，因此适合学习表示。它不是自然的左到右生成过程：生成第一个词时不能把刚生成的词按同样方式无限延长。

#### 把一次 MLM 训练拆成 5 步

以句子 `A cute teddy bear is reading` 为例：

1. tokenizer 把句子变成 token id，并加上 `[CLS]` 和 `[SEP]`；
2. 选择若干位置，把原 token 替换为 `[MASK]`（课件图中是 `cute` 和 `reading`）；
3. 所有位置一起进入 encoder，因为输入侧允许双向 attention；
4. 只取被遮挡位置的 hidden representation，经过 projection 映射到词表大小；
5. 对每个被遮挡位置计算 cross-entropy，更新 encoder 和 prediction head。

因此，MLM 的监督信号不是“整句生成”，而是“在有上下文的条件下恢复选定位置”。没有被遮挡的位置仍参与上下文计算，但通常不直接贡献 MLM loss。

#### NSP：判断两句话是否相邻

`next sentence prediction (NSP)`（下一句预测）把两段句子放在一起，用 `[CLS]` 表示做 yes/no 判断。课件把它作为 BERT 的第二个 proxy task。此处要区分：MLM 预测 token，NSP 预测句子关系；它们是不同输出头。

#### Pretraining 到 fine-tuning

`pretraining`（预训练）先用大量无标签文本完成 MLM/NSP；`fine-tuning`（微调）再用较少的任务数据训练一个具体输出头，例如情感分类或序列标注。

课件给出的优缺点可以这样读：BERT 通常表示性能好，微调所需标注数据较少；但需要针对任务微调，而且不适合直接承担开放式文本生成。这里的“不适合”是任务匹配判断，不是说 encoder 完全不能输出任何文字。

![[assets/5 Transformer variants and applications/source/transformer-p39-39.png|BERT 的 MLM：被遮住的位置由 encoder 表示预测]]

### 3.3 Decoder-only：用 causal attention 逐步生成

`decoder-only`（仅解码器）只保留 decoder，用 causal attention 保证位置 $t$ 只能看见当前位置及以前的 token。GPT 的名字是 *Generative Pre-trained Transformer*，课件第 45–55 页用它说明 autoregressive language modeling。

最小例子：

```text
[BOS] A teddy bear is -> 预测下一个 token
[BOS] A teddy bear is reading -> 再预测下一个 token
```

训练时，一整段已知序列可以并行计算各个位置的 next-token loss；推理时，新 token 必须等上一个 token 生成后才能继续，所以“训练并行”不等于“生成并行”。这承接了 [[5 Attention and Transformer]] 中关于 causal mask 的解释。

对长度为 4 的输入，causal mask 的可见矩阵可以画成：

```text
query 0: 1 0 0 0
query 1: 1 1 0 0
query 2: 1 1 1 0
query 3: 1 1 1 1
```

其中 `1` 表示允许读取，`0` 表示在 softmax 前加上一个极小值或直接屏蔽。训练时虽然四行可以同时计算，但第 1 行不能读取 token 2、3；推理时第 4 行对应的新 token 还不存在，所以必须逐步追加。

**一个常见误会**：causal 不是“整个 decoder 只能看一个 token”。它表示第 $t$ 个位置能看见 $0$ 到 $t$ 的前缀；随着 $t$ 增加，可见上下文也增加。

### 3.4 Encoder-decoder：理解输入，再条件生成输出

`encoder-decoder` 同时保留两部分：encoder 对输入做双向表示，decoder 对输出做 causal generation，并通过 cross-attention 读取 encoder 表示。它适合翻译、摘要和图像描述等“输入与输出都重要、输出长度不固定”的任务。

| 架构 | 输入侧可见范围 | 输出侧可见范围 | 典型用途 |
|---|---|---|---|
| Encoder-only | 双向 | 不以生成作为核心 | 分类、匹配、理解 |
| Decoder-only | causal | causal | 续写、对话、代码生成 |
| Encoder-decoder | encoder 双向 | decoder causal + cross-attention | 翻译、摘要、条件生成 |

课件第 47–48 页的对比重点是 attention mask：encoder-decoder 的输入先双向理解，输出再按因果顺序生成；decoder-only 从一开始就使用 causal attention。

![[assets/5 Transformer variants and applications/source/transformer-p47-47.png|课件中的 encoder-decoder 与 decoder-only 对比]]

> **看图方法（课件 PDF p.47）**：左侧先计算输入的双向表示，再让 decoder 因果生成；右侧只有 decoder，输入本身也使用 causal attention。真正的区别在可见范围和信息流，不只是模块数量。

## 4. 从 GPT 到多模态：把模态变成 token 序列

### 4.1 Unsupervised learning 与 next-token prediction

课件先提出一个现实问题：高质量人工标签稀缺，而网络上有大量 unlabeled data。autoregressive generative model 可以用“预测下一个 token”构造自监督目标，不要求每个 token 都有人工标签。

这不是“预测下一个词就自动等于推理”的证明。更准确的说法是：要在多种文本上持续做好 next-token prediction，模型必须捕捉比局部词频更长程的结构；课件把它作为可能出现推理能力的动机性观察，不能当成能力保证。

### 4.2 GPT-2/GPT-3：从零样本到少样本提示

课件列出这些教学案例：

- GPT-2：zero-shot reading comprehension、summarization、translation；
- GPT-3：language model meta-learning，以及 few-shot arithmetic 和 general few-shot learning。

`zero-shot`（零样本）是不给任务示例，只给任务提示；`few-shot`（少样本）是在 prompt 中放少量输入-输出例子。它们展示的是同一 autoregressive 模型通过上下文改变任务行为，不代表所有任务都能可靠完成，也不替代独立评测。

### 4.3 iGPT 与 DALL-E：图像也可以离散成序列

`iGPT` 把图像表示为可预测的序列，课件用 completion 和 feature learning 说明 GPT 式建模可以扩展到像素/视觉 token。

`DALL-E` 的课件直觉是把 caption 和 image token 串接起来训练一个 Transformer：

```text
text caption tokens + image tokens -> one autoregressive sequence
```

这样，文本到图像、图像到图像等任务可以被看成条件序列建模。这里的“统一”是表示和训练接口上的统一；图像 token 的离散化、视觉质量和生成分辨率仍然是额外设计问题。

### 4.4 CodeX：代码是可执行、可测试的模态

课件提出一个很实用的观察：代码可以被 tokenizer 变成序列，但它还有自然语言没有的验证通道——解释器和 unit tests。模型生成的函数不只可以由人工阅读，还可以运行测试。

`pass@k` 衡量从 $k$ 个采样中至少有一个通过测试的概率。若某题生成 $n$ 个样本，其中 $c$ 个正确，课件给出的无偏估计为：

$$
\operatorname{pass@}k
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}},
\qquad n\geq k.
$$

**最小例子**：$n=10,c=2,k=1$ 时，估计为 $1-\frac{8}{10}=0.2$；若允许 $k=3$ 次尝试，则为 $1-\frac{\binom{8}{3}}{\binom{10}{3}}\approx0.533$。它随着尝试次数增大而上升，但这并不表示单个样本质量变高；它衡量的是“多次采样后至少成功一次”的机会。

课件还记录了 CodeX 的训练信息：159 GB 代码、来自 54 million repositories、从不同规模 GPT-3 模型 fine-tune，以及 tokenizer 对额外空格的处理。这里保留这些作为课件案例，不把它们推广成所有代码模型都必须采用的配置。

![[assets/5 Transformer variants and applications/source/transformer-p63-63.png|课件中的 HumanEval 评估页]]

> **看图方法（课件 PDF p.63）**：这里的评估对象是“根据 docstring 生成函数”，后续通过测试判断功能是否正确。要把它和普通文本相似度指标区分开。

**边界案例**：课件最后指出 binding & composition limitation。一个模型能分别生成看似合理的函数，不保证把多个函数、库接口和约束正确组合起来；可执行测试能发现一部分错误，但不能覆盖所有规格。

## 5. Vision Transformer：把图像变成 patch token

### 5.1 为什么图像能用 Transformer？

文本已经是 token 序列；图像原本是二维像素网格。ViT 的关键转换是把图像切成固定大小的 patch，每个 patch 展平并投影成一个 token embedding，然后把 patch 序列送入 Transformer。

若图像大小为 $H\times W$，patch 大小为 $P\times P$，不考虑 padding 时，patch 数为：

$$
N=\frac{H}{P}\cdot\frac{W}{P}.
$$

例如 $224\times224$ 图像使用 $16\times16$ patch，会得到 $14\times14=196$ 个 patch token，再加上可选的分类 token。这个例子解释了论文标题 “An Image is Worth 16x16 Words”：不是说一个 patch 真的是单词，而是把同一 Transformer 接口用于视觉块。

![[assets/5 Transformer variants and applications/diagrams/vit-patch-tokens.svg|1200]]

> **看图方法**：先看左侧的二维网格如何被切成小块，再看每个 patch 如何变成一个向量 token。patch 越小，token 数越多，attention 的序列长度成本也越高。

#### 从像素到 token 的 shape

设输入图像为 $x\in\mathbb{R}^{H\times W\times C}$，patch 大小为 $P\times P$。每个 patch 展平后有 $P^2C$ 个数，再通过一个线性投影映射到模型维度 $D$：

$$
\operatorname{patch}(x)\in\mathbb{R}^{P^2C}
\xrightarrow{\;W_E\in\mathbb{R}^{P^2C\times D}\;}
z_i\in\mathbb{R}^{D}.
$$

全部 patch 排成 $Z\in\mathbb{R}^{N\times D}$，其中 $N=HW/P^2$。通常还会加入位置表示 $E_{pos}\in\mathbb{R}^{N\times D}$，使模型知道两个外观相似的 patch 位于图像的不同位置；分类任务还可以在序列前加入一个 `[CLS]` token。

**最小算例**：$224\times224$、RGB（$C=3$）、$P=16$ 时，$N=196$，每个 patch 有 $16^2\times3=768$ 个像素值。如果模型维度为 $D=768$，投影矩阵的形状就是 $768\times768$。这一步把二维图像接口变成 Transformer 熟悉的二维 token tensor `[N, D]`。

### 5.2 ViT 的收益与代价

课件把 ViT 的动机描述为 NLP 与 CV 的统一建模、共享模块和促进联合建模。收益是架构接口简单、全局 self-attention 能直接建立远距离 patch 关系；代价是图像 patch 数量可能很大，$N^2$ attention 很快变贵，而且原始 ViT 没有自动注入 CNN 那样强的局部性、层次和 translation invariance 先验。

patch 大小是一个真实的设计旋钮：减小 $P$ 会让 $N=HW/P^2$ 增大，空间细节更丰富，但 full attention 的连接数按 $N^2$ 增长；增大 $P$ 会减少计算，却可能丢失小目标和局部纹理。这个 tradeoff 比“Transformer 能不能处理图像”更接近实际系统设计。

> **课件对照**：原始课件 p.72 仍保存在 `assets/5 Transformer variants and applications/source/transformer-p72-72.png`，但正文优先使用上面的 patch-to-token 教学图，因为它直接解释机制，而不是只展示结果页。

## 6. Swin Transformer：把视觉先验放回注意力

### 6.1 ViT 的视觉问题

课件第 76–78 页指出，文本和视觉信号并不完全相同；视觉任务经常需要层次（hierarchy）、局部性（locality）和 translation invariance（平移不变性）。目标检测、分割等任务也需要多尺度特征，不能只围绕一张全局分类向量设计。

### 6.2 Window attention 与 shifted window

最直接的视觉局部化是 sliding window（滑动窗口），像 CNN 一样让窗口逐位置移动。但课件第 79 页指出，逐位置滑动会让不同 query 使用不同 key set，真实实现较慢。

Swin 的选择是：

1. 在一层中使用不重叠的 non-overlapped windows，窗口内做 attention；
2. 下一层把窗口平移半个窗口大小；
3. 两层交替后，原本属于不同窗口的邻近 patch 可以交换信息。

![[assets/5 Transformer variants and applications/source/transformer-p79-79.png|课件中的 shifted-window 动机]]

这是一种很有代表性的工程折中：窗口内连接规则、容易批量化；移位让信息跨窗口传播；代价是单层不再全局可见，需要多层传播才能建立更远关系。

![[assets/5 Transformer variants and applications/diagrams/swin-shifted-windows.svg|1200]]

> **看图方法**：第一层的粗线把图像分成固定窗口；第二层把分割线移开，使第一层不同窗口边缘的 patch 在下一层进入同一个窗口。Swin 的重点是规则局部计算和跨窗口传播的组合。

若特征图有 $N=HW$ 个位置、窗口大小为 $M\times M$，窗口内 attention 的连接总量近似为：

$$
\frac{HW}{M^2}\cdot M^4=HWM^2.
$$

当 $M$ 固定时，它随图像位置数 $HW$ 线性增长，而不是全局 attention 的 $(HW)^2$。第二层 shift 不会把所有窗口重新变成一个全局窗口；它只改变分组，让信息在多层中逐步跨越边界。这就是 Swin 同时获得效率和更大感受野的机制。

### 6.3 用互动页面观察窗口如何连接

在本章开头的交互页面中选择 `Shifted windows`，先设序列长度 16、窗口 4，再切换到 `Local / truncated`。观察两点：

- 同样是局部限制，shifted window 的块状连接会在下一层改变分组；
- 连接数减少并不自动说明模型更好，它只说明单层需要处理的 query-key 对更少。

> **老师讲解重点**：先切换 `Full attention`，确认每个 query 都连到所有 key；再切换局部、步进、global 和 shifted 模式。每次只回答两件事：哪些连接消失了？远距离信息还有没有替代通路？最后才看连接数量和比例，不能把“更省”直接等同于“更好”。

交互页面是**教学示意**，不是 Swin 的完整二维实现；它把二维窗口压成一维网格，只保留“窗口分组与移位让信息跨组传播”的核心关系。

## 7. 一次完整判断：遇到新任务怎样选变体

可以按下面顺序思考，而不是先背模型名字：

1. **输出要不要逐 token 生成？** 要，就需要 causal decoder；不要，可先考虑 encoder 表示。
2. **输入和输出是否互相条件化？** 翻译/摘要等通常需要 encoder-decoder 的 cross-attention。
3. **序列长度是否让 $N^2$ 成为主要瓶颈？** 是，就检查局部、全局 token、低秩或线性化假设。
4. **输入有没有强局部/层次先验？** 图像通常有，所以要考虑 patch 大小、窗口、层次和移位。
5. **结果能否由外部程序验证？** 代码可以用 unit tests；其他模态可能需要不同的评估协议。
6. **理论连接数减少后，硬件是否真的更快？** 还要看访存是否规则、稀疏索引是否有额外开销、kernel 是否能融合。

## 8. 复盘：从全连接到跨模态

把本章串成一条链：

```text
full self-attention
  -> N² 个 query-key 对成为长序列瓶颈
  -> 设计 attention pattern 或低秩/线性近似
  -> 根据任务选择 encoder-only / decoder-only / encoder-decoder
  -> 用预训练目标塑造用途：MLM、NSP、next-token
  -> 把图像 patch、代码、文本都表示为 token 序列
  -> 为视觉重新加入局部性、层次和移位窗口等先验
```

现在你可以理解下一讲的入口：Transformer variants 还会继续讨论如何改变预测问题、缓存和长序列机制。暂时不必记住每个论文名字；先能回答“它减少了哪类连接、依赖什么假设、给硬件增加什么代价”就足够进入下一步。

## 9. External context：把课件的线索接到原始论文

这一节只补充能回答当前学习问题的在线资料。课件仍然是本章的主线；下面的论文用于确认术语、补足机制和理解实验边界。访问日期：2026-10-19。

### 9.1 精确 attention 也可以靠 IO 优化：FlashAttention

前面我们把“高效 attention”主要理解成减少连接数或做近似。FlashAttention 提供了一个重要的反例：它保持**精确 attention** 的数学结果，重点优化的是 GPU 高带宽显存（HBM）与片上 SRAM 之间的读写。

它使用 tiling（分块）和 online softmax，让计算分块进行，并尽量避免把完整的 $N\times N$ attention matrix 写回显存。这样，理论 FLOPs 没有简单地消失，但内存访问次数减少，实际 wall-clock time 可能下降。

这补上了一个容易混淆的工程层次：

```text
算法复杂度：要算多少个 query-key 对？
IO 复杂度：这些中间结果要在不同层级内存之间搬运多少次？
```

![[assets/5 Transformer variants and applications/diagrams/flashattention-io.svg|1200]]

> **看图方法**：左侧把完整的 $N\times N$ 中间矩阵写入并读回 HBM；右侧把 Q/K/V 分块放入片上 SRAM，边计算边更新 softmax 统计量。两者的目标结果可以相同，差别在中间数据搬运。

因此，看到一个方法声称“更高效”时，要分别问它是减少了连接、减少了近似误差，还是减少了内存访问。FlashAttention 论文还讨论了 block-sparse attention，但那属于在 IO-aware exact kernel 之上再次引入稀疏模式；不能把 FlashAttention 简化成“把 attention 变成稀疏 attention”。

**External context**：Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, arXiv:2205.14135。论文摘要报告了 tiling、HBM 访问分析和长序列实验；这里不把论文中的具体加速倍数当作所有硬件上的保证。

### 9.2 Longformer：local + global 的具体长文档方案

课件中把 local attention 和 global token 分开介绍。Longformer 把这两种想法组合成面向长文档的 attention：普通 token 使用局部窗口，少数任务相关位置使用 global attention，并以线性于序列长度的 attention 机制处理长文本。

这让“global token”更具体：全局位置不是抽象的万能开关，而是应由任务决定。例如文档分类可能需要一个汇总位置，问答任务可能需要让问题 token 或特殊标记访问更广范围。Longformer 还提出 Longformer-Encoder-Decoder（LED）来支持长文档生成，说明稀疏模式也可以放入 encoder-decoder，而不只服务于 encoder-only 模型。

**边界**：线性扩展描述的是 attention 机制的连接规模，不代表整个模型、数据预处理或生成过程都自动变成线性成本。

**External context**：Beltagy, Peters, and Cohan, *Longformer: The Long-Document Transformer*, arXiv:2004.05150。

### 9.3 GPT-3：few-shot 不是“无需训练”，而是无需任务级梯度更新

课件用 GPT-3 说明 few-shot learning。原始论文的精确说法是：GPT-3 在给定 prompt 和少量示例后，不进行任务特定的梯度更新或 fine-tuning，而是直接通过文本交互完成任务。

所以要区分三种情况：

| 方式 | 是否更新参数 | 示例如何进入模型 |
|---|---:|---|
| Fine-tuning | 是 | 示例用于梯度训练 |
| Few-shot prompting | 否 | 示例写入 prompt 上下文 |
| Zero-shot prompting | 否 | 只给任务说明或问题 |

这也解释了课件里“few-shot arithmetic”和“general few-shot learning”的教学意义：它们展示的是上下文条件化能力，而不是证明模型在每种任务上都可靠。GPT-3 原始论文同时报告了仍然困难的数据集和方法学问题，因此少样本成功案例必须和失败边界一起阅读。

**External context**：Brown et al., *Language Models are Few-Shot Learners*, arXiv:2005.14165。

### 9.4 ViT 与 Swin 的原始问题边界

课件把 ViT 讲成“patch token + Transformer”。ViT 原始论文的关键实验条件是：在大量数据上预训练，再迁移到 ImageNet、CIFAR-100 和 VTAB 等中小型任务；论文强调纯 Transformer 直接处理图像 patch 是可行的，但它的效果和训练资源依赖不能脱离数据规模讨论。

Swin 的原始论文则把课件中的三个视觉先验说得更具体：

- **hierarchy**：逐层形成不同空间尺度的表示；
- **locality**：在不重叠窗口中计算 attention；
- **cross-window connection**：下一层移位窗口，让邻近窗口交换信息。

因此，“Swin 的计算对图像尺寸近似线性”指它的窗口化 self-attention 机制和分层设计；它不意味着所有视觉任务的总成本都只与像素数做简单线性关系，也不意味着窗口化一定优于全局 attention。

**External context**：Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, arXiv:2010.11929；Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*, arXiv:2103.14030。

### 9.5 HumanEval：`pass@k` 衡量的是采样机会，不是单次正确率

课件给出了 `pass@k` 公式。HumanEval 原始论文进一步说明，这个评估集用 docstring 描述函数，让模型生成 Python 程序，再通过功能测试判断是否正确；因此它测量的是 functional correctness，而不是代码表面相似度。

论文还报告了重复采样的现象：增加 $k$ 可能显著提高“至少一个样本通过”的概率，但这不等于模型每次生成都可靠。原始研究同时指出，长操作链和变量绑定是典型困难。这个边界正好对应本章前面提到的 binding & composition limitation。

**External context**：Chen et al., *Evaluating Large Language Models Trained on Code*, arXiv:2107.03374。这里的 `pass@k` 解释与公式保持课件一致；具体数值属于该论文设置，不能直接外推到其他测试集。

### 9.6 在线扩展后的判断框架

把新增资料放回本章的工程问题，可以得到四个检查问题：

1. 连接数减少了吗，还是只是中间矩阵没有落到显存？
2. 方法是精确计算，还是用低秩、核特征或采样近似？
3. 稀疏模式是否规则到足以被硬件高效执行？
4. 论文中的数据规模、任务和硬件条件，是否与当前使用场景相同？

这四问能避免把“理论复杂度”“实际速度”“模型质量”和“任务泛化”混成一个指标。

**进一步查漏：近似算法和精确实现不是同一类优化。** Local/window attention 改变允许的信息连接，通常是稀疏或局部化的模型计算；low-rank、kernel/linear attention 可能改变数学近似；FlashAttention 则保持标准 attention 的精确结果，通过分块和减少高带宽内存读写来改进实现。比较论文时应分别核对“计算了哪些连接、是否近似、读写了哪些中间量”，不要只看到都叫 efficient attention 就当作同一种方法。

## 来源与截图说明

- 主要来源：课程课件 `Transformer.pdf`，Lecture 09，86 页，2026-10-14；本笔记中的页码指 PDF 页码。
- 课件第 8、39、47、63、72、79 页被渲染为本地截图，作为源材料核对和补充观察；正文解释仍然独立成立。
- 课件引用的论文与页面：Vaswani et al., *Attention Is All You Need*；Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*；Radford et al., *Improving Language Understanding by Generative Pre-Training*；Dosovitskiy et al., *An Image is Worth 16x16 Words*；Liu et al., *Swin Transformer*；Reformer、Routing Transformer、Linformer、Performer、FNet、MLP-Mixer、Synthesizer 等链接以课件原页为准。
- 本章新增的互动页是本地、确定性的教学插图，不调用网络，不代表任何模型的实测准确率。

### 在线资料（External context）

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al., arXiv:1706.03762。Transformer 原始论文，访问日期 2026-10-19。
- [FlashAttention](https://arxiv.org/abs/2205.14135), Dao et al., arXiv:2205.14135。精确 attention 的 IO-aware 实现，访问日期 2026-10-19。
- [Longformer](https://arxiv.org/abs/2004.05150), Beltagy, Peters, and Cohan, arXiv:2004.05150。local + global 长文档 attention，访问日期 2026-10-19。
- [BERT](https://arxiv.org/abs/1810.04805), Devlin et al., arXiv:1810.04805。encoder-only 预训练方法，访问日期 2026-10-19。
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), Brown et al., arXiv:2005.14165。GPT-3 与 prompt-based few-shot，访问日期 2026-10-19。
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929), Dosovitskiy et al., arXiv:2010.11929。ViT 原始论文，访问日期 2026-10-19。
- [Swin Transformer](https://arxiv.org/abs/2103.14030), Liu et al., arXiv:2103.14030。层次化 shifted-window 视觉 Transformer，访问日期 2026-10-19。
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374), Chen et al., arXiv:2107.03374。HumanEval 与代码功能正确性，访问日期 2026-10-19。
