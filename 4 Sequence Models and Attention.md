# Sequence Models and Attention

> [!tip] 互动入口
> [[互动实验入口]] · [[assets/3 Sequence Models and Attention/interactive/attention-alignment-explorer.html|打开注意力对齐互动]]
>
> 交互与下方固定权重例子使用同一套教学向量；页面改变输出位置时，输入 token、对齐权重和 context 向量会同步更新。脚本不可用时，正文中的权重表和手算仍是完整回退。

> Lecture 07｜面向 AI chip design 博士生的零基础讲义
>
> 本章的核心问题是：**输入和输出都是序列，甚至输入不是序列但输出是序列时，模型怎样逐步生成正确结果？** 课件以 machine translation（机器翻译）和 image captioning（图像描述）为主线，最后引出 attention（注意力机制），并连接到后续的 Transformer / GPT。

> **学习导航**：前置 [[3 RNN]]（hidden state、LSTM/GRU 与 BPTT）→ 当前：encoder-decoder、序列生成与动态读取 → 后续 [[5 Attention and Transformer]]（把 attention 推广为 self-attention 和 Transformer block）。

## 老师开场：为什么一个固定 context 不够

先想机器翻译：encoder 读完整句子，decoder 再生成另一种语言。如果 encoder 最后只交给 decoder 一个固定长度的 vector，那么长句子中早期的细节必须全部挤进这个瓶颈。短句可能没问题，长句就容易忘记主语、时间和修饰关系。

`Attention` 的想法不是把整个历史硬塞进一个摘要，而是让 decoder 在每个输出时间步重新查看 encoder 的所有 hidden states：

1. 当前 decoder state 充当问题（query）；
2. 每个 encoder state 提供可被匹配的索引（key）；
3. 根据 score 得到一组归一化权重；
4. 用这些权重对 encoder states 加权求和，形成当前的 context。

因此 attention 是一种 **content-based soft read**：它不是只挑一个位置，也不是人工规定词语对齐，而是学会“当前输出更需要哪些输入信息”。权重行和为 1，但亮的格子只能说明读取偏好，不能单独证明模型真的理解了语言。

### 先用三个数字看懂 context

如果 score 是 $[2,1,0]$，softmax 后约为 $[0.665,0.245,0.090]$；如果三个 encoder output 是 `[10, 0, 4]`，则：

$$
c_t=0.665\times10+0.245\times0+0.090\times4\approx7.01.
$$

这个例子最重要的不是小数，而是数据流：**score 决定读多少，value 决定读到的内容，weighted sum 产生 context**。下一步我们会把这个动态读取推广成一般 attention，再进入 Transformer。

从工程视角，attention 改善了信息瓶颈，却引入了对多个位置的读访问。序列越长，score、权重和 value 的计算与存储越需要分析；这是后面讨论并行、带宽和 KV cache 的入口。

## 0. 学习目标与先修知识

读完本章，你应该能够：

1. 区分 `sequence-to-sequence`、`X-to-sequence`、`encoder`、`decoder` 和 `autoregressive generation`；
2. 写出一个最小 RNN encoder-decoder 的数据流，说明 hidden state 在每个时间步的作用；
3. 解释为什么 greedy decoding 可能失败，以及 beam search 怎样用有限计算保留多个候选；
4. 解释 `teacher forcing`、`cross-entropy` 和 `<eos>` 在训练/推理中的角色；
5. 说明 fixed-size context vector 为什么形成 information bottleneck，以及 attention 如何按输出时间步重新读取输入；
6. 从 AI chip 角度辨认主要代价：sequential dependency、矩阵乘法、softmax、memory traffic、并行度与片上存储。

### 0.1 先修知识：把一句话看成一串向量

在前两篇笔记中，神经网络接收一个固定形状的 vector 或 image tensor。序列模型处理的是长度可变的集合：

$$
\mathbf{x}_{1:T}=(\mathbf{x}_1,\mathbf{x}_2,\ldots,\mathbf{x}_T)
$$

$T$ 是序列长度，$\mathbf{x}_t$ 是第 $t$ 个时间步的输入向量。对于句子，$\mathbf{x}_t$ 可以是第 $t$ 个 token 的 embedding；对于视频，$\mathbf{x}_t$ 可以是第 $t$ 帧的视觉特征。每个时间步的向量维度可以固定，但 $T$ 不必固定。

### 0.2 分层术语 roadmap

这里先给“路线图”，不要求现在背完。每一层的词只在后面真正需要时展开；读到对应章节时，再回到这一行复习。

```text
Level 0  序列的基本对象
  token -> embedding -> vector x_t -> sequence x_1:T
      |
Level 1  让模型读懂顺序（第 1–2 节）
  sequence-to-sequence / X-to-sequence
  RNN -> hidden state -> encoder / decoder -> context vector
      |
Level 2  让模型生成输出（第 3 节）
  autoregressive -> logit / probability -> greedy decoding
  -> beam search -> cumulative log-probability
      |
Level 3  让模型学会生成（第 4–5 节）
  teacher forcing -> cross-entropy -> exposure bias
  image captioning -> projection -> frozen-CNN / fine-tuning
      |
Level 4  解决固定摘要的瓶颈（第 6–8 节）
  information bottleneck -> attention score / alignment
  -> softmax weight alpha_t,i -> context c_t -> spatial feature
      |
Level 5  连接芯片和下一章（第 9–11 节）
  sequential dependency -> memory traffic -> beam reorder
  -> self-attention -> causal masking -> Transformer / GPT
```

| 什么时候 | 只需要认识这些词 | 学习检查 |
|---|---|---|
| 现在（Level 0） | `token`, `embedding`, `sequence`, `time step` | 能指出 $\mathbf{x}_t$ 和 $T$ 的含义 |
| 读第 1–2 节 | `seq2seq`, `RNN`, `hidden state`, `encoder`, `decoder`, `context` | 能画出“读入 source，再生成 target” |
| 读第 3 节 | `autoregressive`, `logit`, `probability`, `greedy`, `beam` | 能解释为什么要保留多个候选前缀 |
| 读第 4–5 节 | `teacher forcing`, `cross-entropy`, `exposure bias`, `image captioning` | 能区分训练时输入和推理时输入 |
| 读第 6–8 节 | `bottleneck`, `alignment`, `attention weight`, `spatial feature` | 能从 $\alpha_{t,i}$ 算出 $\mathbf{c}_t$ |
| 读第 9–11 节 | `sequential dependency`, `memory traffic`, `self-attention`, `causal masking` | 能把算法步骤对应到芯片代价 |

**使用规则**：一个词第一次在正文中出现时，我会同时给出英文名、中文直觉、它在当前小节解决的问题，以及一个输入/输出或公式例子。后文只保留标准英文名，避免重复背诵。

### 0.3 总体工程视角

**模型视角**关注语义、顺序和输出质量；**硬件视角**关注 RNN 的时间依赖是否限制吞吐、矩阵乘法能否复用、hidden state / attention map 是否能放入 SRAM；**系统视角**关注 decoder 的 token-by-token latency、beam size 带来的额外计算，以及 training 时保存中间 activation 的容量。以下每个机制都会沿着“问题 → 约束 → 设计 → 收益与代价”解释。

<svg viewBox="0 0 900 170" role="img" aria-label="Sequence to sequence pipeline from input encoding to autoregressive decoding and attention context" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="seq-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#475569"/></marker></defs>
<rect width="900" height="170" rx="12" fill="#f8fafc"/>
<g font-family="Arial, sans-serif" font-size="14" text-anchor="middle">
<g fill="#dbeafe" stroke="#2563eb" stroke-width="2"><rect x="25" y="55" width="135" height="58" rx="10"/><text x="92" y="80" fill="#1e3a8a">Input X</text><text x="92" y="99" fill="#1e3a8a">tokens / image</text></g>
<g fill="#ffffff" stroke="#64748b" stroke-width="1.5"><rect x="205" y="55" width="145" height="58" rx="10"/><text x="277" y="80" fill="#1f2937">Encoder RNN</text><text x="277" y="99" fill="#1f2937">hidden outputs</text><rect x="395" y="55" width="145" height="58" rx="10"/><text x="467" y="80" fill="#1f2937">Score + softmax</text><text x="467" y="99" fill="#1f2937">attention weights</text></g>
<g fill="#dcfce7" stroke="#16a34a" stroke-width="2"><rect x="585" y="55" width="135" height="58" rx="10"/><text x="652" y="80" fill="#14532d">Context c_t</text><text x="652" y="99" fill="#14532d">weighted read</text><rect x="765" y="55" width="110" height="58" rx="10"/><text x="820" y="80" fill="#14532d">Decoder</text><text x="820" y="99" fill="#14532d">next token</text></g>
<g stroke="#475569" stroke-width="2" marker-end="url(#seq-arrow)"><line x1="160" y1="84" x2="205" y2="84"/><line x1="350" y1="84" x2="395" y2="84"/><line x1="540" y1="84" x2="585" y2="84"/><line x1="720" y1="84" x2="765" y2="84"/></g>
<path d="M820 118 C820 150 652 150 652 118" fill="none" stroke="#16a34a" stroke-width="1.7" stroke-dasharray="6 4" marker-end="url(#seq-arrow)"/><text x="735" y="157" fill="#166534">autoregressive feedback</text>
</g></svg>

读图：encoder 产生输入的表示；attention 在当前 decoder 状态下从这些表示中读取 $\mathbf{c}_t$；decoder 生成一个 token，并把它反馈给下一步。没有 attention 时，context 可以退化为整个输入的单一固定摘要。

## 1. 为什么需要 sequence model？

> **本节新词**：`X-to-sequence` 表示任意输入到序列输出；`sequence-to-sequence (seq2seq)` 表示序列输入到另一条序列输出；`alignment` 表示输入位置与输出位置之间的对应关系。

### 1.1 从 fixed-size prediction 到 variable-length output

一个分类器可以把一张图映射为一个 label，但以下任务的输出长度事先不固定：

- **Machine translation**：英文句子输入，中文句子输出，两个长度通常不同；
- **Dialogue**：用户输入一句话，系统生成长度不固定的回复；
- **Image captioning**：图像输入，输出一句描述；
- **Video captioning**：视频输入，输出描述时间过程的文字。

这类任务统称为 **X-to-sequence**：输入 `X` 可以是 sequence，也可以是一张 image；输出是 sequence。若输入和输出都为序列，就是 **sequence-to-sequence (seq2seq)**。

```text
sequence x_1, ..., x_T  ──►  model  ──►  y_1, ..., y_U
image / video features  ──►  model  ──►  y_1, ..., y_U
```

输入和输出不要求同步：输入的第 3 个 token 不一定对应输出的第 3 个 token，甚至 alignment 可能是 non-monotonic（非单调）的。翻译时一个英文词可能对应多个中文词，也可能没有一一对应关系。

### 1.2 为什么不能只用一个普通 fully connected layer？

**学习者能看到的问题**：一句话的意思取决于词序和上下文。“not good” 与 “good” 不能只看词袋；而输出还要逐词生成。

**没有序列机制时的失败**：若直接把不同长度的输入 padding 到同一长度，模型仍需要知道哪些位置是真实 token；若把所有 token 一次展平，顺序结构和可变长度都变成额外负担。

**设计选择**：使用 recurrent neural network（RNN），让一个固定大小的 hidden state 随时间步更新，作为历史信息的可计算摘要。

**收益与边界**：RNN 能处理任意长度，但每个时间步依赖前一个 hidden state，因此推理有串行 latency，长距离信息还可能受到 vanishing / exploding gradient 影响。后文的 attention 会减少“所有信息必须塞进一个摘要”的压力。

## 2. RNN：让网络拥有“记忆”

> **本节新词**：`recurrent neural network (RNN)` 是会把上一步状态带到下一步的网络；`hidden state` 是当前对历史的固定大小摘要；`encoder` 读取输入，`decoder` 负责生成；`context vector` 是 decoder 使用的输入摘要。

### 2.1 单步更新的直觉

**Intuition**：把 hidden state 想成一张不断更新的便签。读入当前 token 后，RNN 用“旧便签 + 当前 token”写出新便签。便签大小固定，因此必须压缩信息。

RNN 的一个常见形式为：

$$
\mathbf{h}_t=\phi\left(W_{xh}\mathbf{x}_t+W_{hh}\mathbf{h}_{t-1}+\mathbf{b}_h\right)
$$

其中：

- $\mathbf{x}_t$：当前输入，shape 为 $d_x$；
- $\mathbf{h}_{t-1}$：上一个时间步的 hidden state，shape 为 $d_h$；
- $W_{xh}$：输入到 hidden 的权重，shape $d_h\times d_x$；
- $W_{hh}$：hidden-to-hidden 权重，shape $d_h\times d_h$；
- $\phi$：非线性函数，如 `tanh`；
- $\mathbf{h}_0$：序列开始前的初始状态，常设为零向量或可学习向量。

**AI chip lens**：单个时间步包含两个主要 matrix-vector multiplication。batch 化后可变成 matrix-matrix multiplication，但 $\mathbf{h}_{t}$ 仍必须等 $\mathbf{h}_{t-1}$，所以 sequence length 方向不能像普通 feed-forward layer 那样完全并行。

### 2.2 Encoder 与 decoder

一个最简单的翻译模型把任务拆成两个 RNN：

1. **Encoder** 按顺序读取 source tokens；最后一个 hidden state 作为输入句子的 fixed-size summary。
2. **Decoder** 使用这个 summary 初始化自己的 hidden state，然后逐步输出 target tokens。
3. decoder 产生 `<eos>`（end of sequence）时停止。

![[assets/3 Sequence Models and Attention/SeqAttn-7.png]]

> **截图导航（PDF p.7）**：左侧是 encoder 读取 `I ate an apple`，红色 `<eos>` 表示输入结束；右侧 decoder 从编码后的状态开始生成输出。要点不是图中的具体句子，而是“读完再生成”的延迟式 many-to-many 结构。

这就是 **encoder-decoder architecture**：encoder 负责理解输入，decoder 负责条件生成。

![[assets/3 Sequence Models and Attention/SeqAttn-8.png]]

> **截图导航（PDF p.8）**：decoder 每生成一个 token，就把它作为下一步输入；因此输出长度由模型生成的 `<eos>` 决定。

一个抽象的 decoder 更新可写成：

$$
\mathbf{s}_t=f_{\text{dec}}(\mathbf{y}_{t-1},\mathbf{s}_{t-1},\mathbf{c}),
\qquad
\mathbf{o}_t=W_{so}\mathbf{s}_t+\mathbf{b}_o
$$

$\mathbf{c}$ 是 encoder 提供的 context，$\mathbf{o}_t$ 是对整个 target vocabulary 的 logits。经 softmax 后得到：

$$
p(y_t\mid y_{<t},x)=\operatorname{softmax}(\mathbf{o}_t)
$$

### 2.3 `<bos>`、`<eos>` 与 embedding

decoder 通常需要两个特殊 token：`<bos>`（beginning of sequence）告诉它“开始生成”，`<eos>` 告诉系统“结束”。词本身不是可直接相乘的数字，因此先经过 **word embedding**：

$$
\text{token id } i \longmapsto E[i,:]\in\mathbb{R}^{d_e}
$$

`E` 是 embedding matrix，与其他参数一起训练。embedding 把离散索引转为连续空间中的向量，使相似语义可以由网络学习到相近表示。

### 2.4 变长 sequence 如何组成一个 batch？

RNN 的每个样本可以有不同长度，但矩阵计算通常希望一个 batch 具有规则 shape。常见做法是把同一 batch 中较短的 sequence 补成相同长度，并加入 **padding token**。例如：

```text
sample A: 我 喜欢 芯片 <eos> <pad>
sample B: 我 学习 AI   <eos> <pad>
sample C: 我 喜欢 AI   芯片  <eos>
```

`<pad>` 不是语义内容，不能被当成普通词参与 loss 或 attention。实现因此需要维护真实长度或 padding mask：loss mask 让 padding 不计入 cross-entropy，RNN packing/length information 避免 recurrent cell 把 padding 当成真实时间步，attention padding mask 则禁止 query 读取 padding 的 key/value。

这和 causal mask 是两种不同限制：padding mask 解决“这个位置根本不存在”，causal mask 解决“这个位置存在但未来不可见”。在芯片上，padding 会降低有效算力利用率；按长度分桶可以减少无效计算，但会增加 batch 调度和 data movement 的复杂度。

## 3. Autoregressive prediction：模型怎样生成序列

> **本节新词**：`autoregressive` 指“把已经生成的结果作为下一步输入”；`logit` 是 softmax 前的分数；`probability distribution` 是对 vocabulary 的归一化概率；`decoding` 是从这些分布中选择完整输出。

### 3.1 条件概率的乘积

给定输入 `x`，decoder 的完整输出序列概率由链式法则分解：

$$
p(\mathbf{y}\mid\mathbf{x})
=\prod_{t=1}^{U}p(y_t\mid y_1,\ldots,y_{t-1},\mathbf{x})
$$

每一个时间步只负责下一个 token，但全句概率是所有条件概率的乘积。实际计算常使用 log-probability：

$$
\log p(\mathbf{y}\mid\mathbf{x})
=\sum_{t=1}^{U}\log p(y_t\mid y_{<t},\mathbf{x})
$$

这解释了两个常见概念：`teacher forcing` 是训练时提供真实历史；`decoding` 是推理时根据模型分布选择历史。

### 3.2 Greedy decoding：快，但只看眼前

> **本节新词**：`greedy decoding` 每一步只选当前最大概率；它优化当前一步，不保证整句的 `MAP`（maximum a posteriori）结果。

在第 `t` 步，greedy decoding 选择：

$$
\hat y_t=\arg\max_{v\in\mathcal{V}}p(v\mid \hat y_{<t},\mathbf{x})
$$

它的优点是简单、内存少、延迟低；问题是**当前最可能的词不一定能带来全局最可能的句子**。一个略低概率的早期词可能使后续词的概率大幅提高。

![[assets/3 Sequence Models and Attention/SeqAttn-16.png]]

> **截图导航（PDF p.16）**：每个 decoder 时间步都有一个覆盖 vocabulary 的概率分布，选出的 token 以自回归方式反馈。

![[assets/3 Sequence Models and Attention/SeqAttn-18.png]]

> **截图导航（PDF p.18）**：课件指出 greedy 的局限：较早的局部选择会改变后续条件分布，不能保证整句概率最大。

### 3.3 Beam search：保留多个未来

> **本节新词**：`beam search` 保留最多 $K$ 条候选前缀；`cumulative log-probability` 把每一步的 log-probability 相加，用来比较不同长度的部分路径。

**局部问题**：如果只留一条前缀，早期错误无法恢复。

**设计选择**：在每一步扩展候选前缀，按累计 log-probability 只保留 top-$K$ 条。$K$ 称为 beam size。

对候选前缀 $y_{1:t}$ 评分：

$$
S(y_{1:t})=\sum_{i=1}^{t}\log p(y_i\mid y_{<i},x)
$$

流程是：

1. 给当前每条 beam 尝试 vocabulary 中的候选 token；
2. 得到新的前缀和对应 hidden state；
3. 计算累计分数；
4. 全部候选中保留分数最高的 $K$ 条；
5. 输出 `<eos>` 的路径停止扩展，最后从已完成路径中选最好的一条。

![[assets/3 Sequence Models and Attention/SeqAttn-22.png]]

![[assets/3 Sequence Models and Attention/SeqAttn-24.png]]

![[assets/3 Sequence Models and Attention/SeqAttn-27.png]]

> **截图导航（PDF p.22–27）**：树状图展示候选数的指数增长、剪枝以及累计 log-probability。beam search 仍是近似搜索，不保证找到真正的 MAP sequence；`K` 越大，质量潜力通常越高，但计算、缓存 hidden state 和 latency 也增加。

**AI chip lens**：beam search 会同时维护 $K$ 份 decoder state，要求更大的 activation buffer 和更复杂的 gather / reorder；若 beam 维度能与 batch 合并，矩阵乘法利用率会更好，但控制和内存访问变得不规则。

## 4. 训练 seq2seq：teacher forcing 与反向传播

> **本节新词**：`teacher forcing` 用真实历史 token 作为 decoder 输入；`cross-entropy` 惩罚目标 token 的低概率；`exposure bias` 指训练看到真实历史、推理却看到自身预测造成的分布差异。

### 4.1 训练时为什么不直接用模型自己的输出？

若训练初期模型在第一个词就错了，错误 token 会被喂给下一步，接着产生更大偏差，这叫 **error accumulation**。课件采用 **teacher forcing**：在时间步 $t$，decoder 输入真实的 $y_{t-1}$，而不是上一步的预测。

训练输入和目标是错位的：

```text
decoder input : <bos>   y1       y2       y3
target output :          y1       y2       y3       <eos>
```

因此一次 forward pass 可以在已知真实前缀的条件下得到每个位置的 probability distribution，再用 token-level cross-entropy：

$$
\mathcal{L}(x,y)=-\sum_{t=1}^{U+1}\log p_\theta(y_t\mid y_{<t},x)
$$

其中最后一个目标通常是 `<eos>`。对 batch 求平均后，通过 backpropagation 计算 encoder、decoder、embedding 的梯度，并用 optimizer 更新参数。

### 4.1.1 Sequence training 为什么尤其依赖 dataset？

在 sequence task 中，一个训练样本不是孤立的数字，而是带顺序的 `(source, target)` 对。dataset 同时决定：

- 哪些 source sequence 会被模型看到；
- target 是否是正确、完整且格式一致的答案；
- source 与 target 的 alignment、长度比例和词汇覆盖；
- `<bos>`、`<eos>`、padding、tokenization 是否遵循同一规则。

**最小例子：**如果 machine translation dataset 中同一句 source 有两个互相矛盾的 target，cross-entropy 会要求模型在同一条件下给两个 token 高概率；模型不是“不会学”，而是监督信号本身不一致。相反，如果训练集只有短句，decoder 可能学会生成短句，却没有足够经验处理长距离依赖。

因此，sequence dataset 要检查 source/target 配对、语言或模态覆盖、长短序列分布、重复样本、训练与测试的主题泄漏，以及特殊 token 的一致性。对于 image captioning，`(image, caption)` pairing 错一条就会把视觉 feature 与错误文字绑定；对于 video captioning，还要确认时间顺序和 caption 描述的时间范围。

**硬件视角**：dataset 的平均 sequence length 会影响 padding 浪费、batch 形状、attention 的 $U\times T$ 访问量和训练 activation memory；数据越长不只是“多几个 token”，还会改变 accelerator 的 buffer 和 bandwidth 压力。

### 4.1.2 可选现象展示：让 decoder 学会反转 sequence

运行 [04_seq2seq_attention.py](experiments/04_seq2seq_attention.py)：

```bash
cd "/Users/dongyangchen/Desktop/lecture note/deep learning/experiments"
python3 04_seq2seq_attention.py
```

这个 toy task 的 target 是 source 的逆序。它只使用数字 token，所以你可以把注意力放在机制上：

- training 使用 `teacher forcing`，把真实 target token 喂给下一步 decoder；
- inference 使用 greedy decoding，只把模型刚生成的 token 反馈回去；
- decoder 每一步输出一行 attention weights，最后打印 attention matrix 的 shape 和第一行权重。

默认运行即可看到正文中的 teacher forcing、greedy decoding 和 attention matrix。改变 `teacher_forcing` 只是可选的现象展示：它帮助观察训练输入与推理输入的差异，但不是完成本章的作业。attention weights 的行数对应 output steps，列数对应 input steps，这是由 context 的输入输出 shape 决定的。

这是教学用的最小实现，不是翻译 benchmark。要扩展到真实语言数据，可对照官方 [seq2seq translation tutorial](https://github.com/pytorch/tutorials/blob/main/intermediate_source/seq2seq_translation_tutorial.py)。

![[assets/3 Sequence Models and Attention/SeqAttn-29.png]]

![[assets/3 Sequence Models and Attention/SeqAttn-32.png]]

> **截图导航（PDF p.29–32）**：绿色/黄色 token 序列表示真实 source 和 target；红色的 `D` 表示 divergence/loss 对各时间步输出的反向梯度。训练时可并行计算所有 target 位置的 loss，推理时则必须逐步生成。

### 4.2 Training–inference mismatch

teacher forcing 的代价是 **exposure bias**：训练时 decoder 总看到真实历史，推理时只能看到自己的预测。一旦推理中出现错误，后续输入分布就偏离训练分布。课件没有展开具体缓解方法，但读者应记住这条边界：teacher forcing 让训练稳定、高效，却不会自动消除自回归误差。

**最小例子**：目标句子是真实 token `A → B → C`。训练时，预测 `B` 的输入是正确的 `A`，预测 `C` 的输入是正确的 `B`；推理时若模型先把 `A` 误生成成 `X`，下一步看到的是 `X`，而不是训练中熟悉的 `A`。错误可能因此传递，但不意味着每个早期错误必然导致整句失败。teacher forcing 是逐位置最大似然训练的输入安排，不是“推理时也知道正确答案”。

**External course-material extension**：D2L, *Dive into Deep Learning*, [Sequence-to-Sequence Learning](https://d2l.ai/chapter_recurrent-modern/seq2seq.html)，将 encoder、decoder、teacher forcing 和变长序列放在一个可执行教学例子中；本节仍以本课程 slide 的记号和顺序为主。访问日期：2026-10-22。

### 4.3 历史技巧：反转 source sequence

Sutskever 等人在早期 seq2seq 工作中反转 source sequence，同时在训练和推理使用相同反转。这样做不是改变语义，而是让 source 开头的信息在时间上更靠近 decoder 的早期状态，缩短部分依赖路径。

![[assets/3 Sequence Models and Attention/SeqAttn-33.png]]

## 5. X-to-sequence 应用：image captioning

> **本节新词**：`image captioning` 是图像到文字序列；`projection` 把视觉 feature 映射到 decoder 能使用的空间；`frozen-CNN` 冻结视觉编码器，`fine-tuning` 则进一步更新部分 CNN 参数。

### 5.1 从图像到句子

**问题**：输入没有天然的文字时间轴，但输出是一串 token。

**设计**：用一个预训练 CNN 把 image 转为 visual feature，再把该 feature 投影到 decoder 的初始状态或 context，之后由 RNN decoder 自回归生成 caption。

```text
image -> pretrained CNN -> visual feature -> RNN decoder -> word sequence
```

![[assets/3 Sequence Models and Attention/SeqAttn-37.png]]

> **截图导航（PDF p.37）**：左侧 CNN 提取图像 feature，右侧 RNN 负责“a bird ...”这样的序列生成。图像编码器和文字解码器是两个不同模态的网络。

训练数据是 `(image, caption)` pairs。课件给出两种实现边界：

这里的 pairing 是训练能否成立的前提：CNN 输出的 visual feature 必须对应这张图的 caption。数据集如果只包含一种构图或固定文字模板，decoder 可能学到模板和背景的 shortcuts，而不是图像与语言之间可迁移的关系。

- **Frozen-CNN baseline**：冻结预训练 CNN，只训练 projection 和 decoder；显存/算力要求较低，也减少破坏通用视觉特征的风险；
- **Fine-tuning**：选择性更新 CNN 层，使 visual feature 更适合 caption task，但反向传播和 activation 存储成本更高。

![[assets/3 Sequence Models and Attention/SeqAttn-44.png]]

![[assets/3 Sequence Models and Attention/SeqAttn-45.png]]

> **截图导航（PDF p.44–45）**：forward pass 使用 image 和 caption；backward pass 根据 caption loss 更新 decoder，是否更新 CNN 是一个明确的工程选择。

视频描述只是把 image feature 换成随时间变化的 video feature，再让 decoder 生成语言；核心仍然是 X-to-sequence。

## 6. 固定 context 的信息瓶颈

> **本节新词**：`information bottleneck` 指过多输入信息被压进固定大小的 vector，导致后续生成难以恢复早期细节。

### 6.1 为什么“只传最后一个 hidden state”会变困难？

简单 encoder-decoder 将任意长度输入压缩成一个固定大小的 vector $\mathbf{c}$：

$$
\mathbf{c}=\mathbf{h}^{\text{enc}}_T
$$

对短句它可能足够；对长句，早期 token 的信息必须穿过很多 recurrent updates，最后混在同一个向量中。课件称其为 **information bottleneck**。

**局部失败**：生成第一个输出词可能需要 source 开头的信息，生成后面的词又需要 source 末尾的信息；单一 $\mathbf{c}$ 要同时承载全部信息，部分内容会被 diluted downstream。

![[assets/3 Sequence Models and Attention/SeqAttn-50.png]]

> **截图导航（PDF p.50）**：输入序列最终汇聚到一个 fixed-size vector；这就是瓶颈的位置。

**连接下一概念**：既然不同输出时间步需要不同输入位置，就不应该只使用一个固定 context，而应让 decoder 每一步重新读取所有 encoder hidden outputs。

## 7. Attention：为每个输出步选择上下文

> **本节新词**：`attention score` 衡量当前 decoder state 与某个输入位置的匹配程度；`alignment` 是这些匹配关系；`attention weight` 是 score 经 softmax 后的可微权重；`context vector` 在本节变成随时间变化的 $\mathbf{c}_t$。

### 7.1 从固定摘要到动态读取

encoder 不再只留下最后的 hidden state，而是保留所有位置的输出：

$$
\mathbf{h}^{\text{enc}}_1,\ldots,\mathbf{h}^{\text{enc}}_T
$$

decoder 在输出时间步 $t$ 有当前状态 $\mathbf{s}_t$。attention 做三步：

1. **Score / alignment**：比较 $\mathbf{s}_t$ 与每个输入位置 $\mathbf{h}^{\text{enc}}_i$ 的相关性；
2. **Normalize**：对所有位置的 score 做 softmax，得到权重 $\alpha_{t,i}$；
3. **Weighted sum**：用这些权重对 encoder outputs 加权，得到当前 context $\mathbf{c}_t$。

$$
e_{t,i}=g(\mathbf{s}_{t-1},\mathbf{h}^{\text{enc}}_i)
$$

$$
\alpha_{t,i}=\frac{\exp(e_{t,i})}{\sum_{j=1}^{T}\exp(e_{t,j})},
\qquad
\mathbf{c}_t=\sum_{i=1}^{T}\alpha_{t,i}\mathbf{h}^{\text{enc}}_i
$$

由于 softmax，满足：

$$
0\leq\alpha_{t,i}\leq1,
\qquad
\sum_{i=1}^{T}\alpha_{t,i}=1
$$

这可以理解为“当前输出对输入各位置的注意力分布”，但它仍是可微的加权计算，不是人类意识意义上的注意。

![[assets/3 Sequence Models and Attention/SeqAttn-55.png]]

![[assets/3 Sequence Models and Attention/SeqAttn-57.png]]

> **截图导航（PDF p.55–57）**：每个 decoder 时间步都有一组不同的权重；权重依赖当前 decoder state，所以 context 随时间变化。

### 7.2 Score function 的选择

课件列出三类共享的 scoring function `g()`：

- **Dot product**：$e_{t,i}=\mathbf{s}_{t-1}^{T}\mathbf{h}^{\text{enc}}_i$。速度快，但通常要求两者维度相同；
- **Bilinear**：$e_{t,i}=\mathbf{s}_{t-1}^{T}W_a\mathbf{h}^{\text{enc}}_i$。增加一个可学习矩阵，表达能力更强；
- **Additive**：先拼接/变换两个向量，再经过非线性和向量投影，例如 $e_{t,i}=\mathbf{v}^{T}\tanh(W_s\mathbf{s}_{t-1}+W_h\mathbf{h}^{\text{enc}}_i)$。

三者共同的关键不是某个具体公式，而是：**同一个 score rule 被应用到所有 input-output position pairs**，从而学习 alignment。

![[assets/3 Sequence Models and Attention/SeqAttn-60.png]]

### 7.3 最小数值例子：softmax 权重怎样产生 context

#### 交互：按输出步查看 soft alignment

打开 [Attention alignment explorer](assets/3 Sequence Models and Attention/interactive/attention-alignment-explorer.html)，点击“下一步”，观察当前输出词读取哪一个或哪些输入词；右侧沿用本章 JS Engine 示范中的 value 向量，把权重乘到各向量上并实际算出 context。网页和 JS Engine 共用同一组固定教学数值；它们不是模型训练结果。正文下方的 [0.665, 0.245, 0.090] 与一维输入 [10, 0, 4] 是另一组手算例子，仍以正文公式为准。

Obsidian 原生打开：[[assets/3 Sequence Models and Attention/interactive/attention-alignment-explorer.html|打开 Attention alignment 互动页面]]

> [!tip]- Obsidian 内原生练习（不需要 HTML）
> 1. 先看本节的权重向量 [0.665, 0.245, 0.090]。
> 2. 指出哪个输入位置贡献最大，并说明为什么。
> 3. 验证三项权重加起来约等于 1。
> 4. 上面的 HTML 链接是可选的独立互动页；如果按钮不动，直接使用本节的数值例子和下方 attention 图。

静态检查方法：每一行权重应加总为 1；context 是 encoder outputs 的加权和。亮点只能说明模型在该步更偏向读取某些位置，不能单独证明存在唯一正确的词对齐。

> **老师讲解重点**：先读当前输出词，再沿这一行看它从哪些输入位置取信息，最后把权重乘到 value 上。颜色深浅只表示读取比例；真正的 context 是加权和，不是“选中了一个词”。

**Obsidian 内互动（JS Engine）**

点击“下一步”，观察不同输出位置的 attention weight 如何改变。

```js-engine
const box = container.createDiv();
box.createEl("b", { text: "老师讲解：权重决定读取比例，value 决定读到的内容" });
const next = box.createEl("button", { text: "下一步" });
const result = box.createEl("pre");
const outputs = ["我", "喜欢", "AI", "芯片"], inputs = ["I", "love", "AI", "chips"];
const weights = [[0.72, 0.16, 0.08, 0.04], [0.08, 0.70, 0.14, 0.08], [0.04, 0.10, 0.70, 0.16], [0.03, 0.07, 0.16, 0.74]];
const values = [[10, 0], [0, 8], [6, 2], [0, 4]];
let step = 0;
function show() {
  const row = weights[step];
  const context = values.reduce((sum, value, index) => [sum[0] + row[index] * value[0], sum[1] + row[index] * value[1]], [0, 0]);
  result.textContent = "输出：" + outputs[step] + "\n" + inputs.map((word, i) => {
    const bars = "█".repeat(Math.round(row[i] * 20));
    return word.padEnd(5, " ") + " " + bars + " " + row[i].toFixed(2);
  }).join("\n") + "\n权重和：" + row.reduce((a, b) => a + b, 0).toFixed(2) +
    "\nweighted sum context = [" + context.map(value => value.toFixed(2)).join(", ") + "]";
}
next.addEventListener("click", () => { step = (step + 1) % outputs.length; show(); });
show();
return;
```

假设某一 decoder 时间步对三个输入位置的 score 是：

$$
\mathbf{e}_t=[2,1,0]
$$

softmax 后约为：

$$
\boldsymbol{\alpha}_t
\approx[0.665,0.245,0.090]
$$

若三个 encoder output 是一维向量 `[10, 0, 4]`，当前 context 为：

$$
c_t=0.665\times10+0.245\times0+0.090\times4\approx7.01
$$

因此 context 主要由第一个位置贡献，但仍保留其他位置的一点信息。下一步 decoder state 改变后，score 和权重也会改变；这正是 attention 相比固定 context 的核心能力。

### 7.4 如何读 attention 图

课件中的 alignment matrix 通常以行表示 decoder output time，列表示 encoder input position；较亮/较深的格子表示更大的 $\alpha_{t,i}$。如果每一行的高权重大致沿对角线移动，说明 learned alignment 近似单调；如果高权重跳跃，则表示可能存在 reordering 或跨位置依赖。

![[assets/3 Sequence Models and Attention/SeqAttn-62.png]]

> **截图导航（PDF p.62）**：四张矩阵显示不同条件下的 attention pattern。不要把亮度当成“确定性证明”；它是模型产生的相对权重，可用于观察 alignment。

## 8. Attention 在 image captioning 中的作用

> **本节新词**：`spatial feature` 保留 CNN feature map 的空间位置；attention 可以在生成不同词时读取不同图像区域。

### 8.1 从一个 global feature 到 spatial features

简单 image captioning 把整张图压成一个 feature vector `c`，这同样形成 bottleneck。attention 版本保留 CNN feature map 的空间位置：

$$
\mathbf{H}\in\mathbb{R}^{H\times W\times d}
$$

把每个空间位置展平为 $N=H\times W$ 个 feature vectors，decoder 在生成每个词时对这些位置计算 attention。生成 `bird` 时可能集中看鸟的位置，生成 `water` 时可能转向水面区域。

![[assets/3 Sequence Models and Attention/SeqAttn-63.png]]

![[assets/3 Sequence Models and Attention/SeqAttn-65.png]]

![[assets/3 Sequence Models and Attention/SeqAttn-66.png]]

> **截图导航（PDF p.63–66）**：CNN 产生空间 feature map，attention map 随已生成词变化，再将 weighted feature 送入 decoder。课件展示的核心流程是“extract spatial features → compute alignment/attention → generate word”。

### 8.2 NMT 中的动态 context

在 neural machine translation 中，输入是 source token 序列。每个 decoder 时间步都从所有 encoder outputs 形成新的 context：

![[assets/3 Sequence Models and Attention/SeqAttn-71.png]]

![[assets/3 Sequence Models and Attention/SeqAttn-72.png]]

> **截图导航（PDF p.71–72）**：每个输出 token 都有自己的 context；“look at different parts”应理解成权重分布改变，而不是硬选择一个唯一位置。

## 9. 训练、推理与硬件实现的对照

> **本节新词**：`sequential dependency` 是必须等待前一步结果的数据依赖；`memory traffic` 是在 SRAM/DRAM 间搬运 activation、weight 和 attention data 的成本；`beam reorder` 是 beam 剪枝后重新排列候选状态。

| 阶段 | 输入历史 | 可并行程度 | 主要计算/存储压力 |
|---|---|---|---|
| encoder training | 完整 source sequence | RNN 时间方向受限；batch/feature 方向可并行 | 保存各时间步 hidden outputs 供反向传播 |
| decoder training + teacher forcing | 完整 target 的真实前缀 | token loss 可批量计算 | activation、softmax 和 cross-entropy |
| decoder inference | 自己刚生成的 token | 生成方向基本串行 | 每步 latency、KV/state buffer、beam reorder |
| attention | 所有 encoder outputs + 当前 decoder state | position pair 的 score 可并行 | 读大量 hidden states，产生 $U\times T$ 权重 |

**Hardware / implementation perspective（一般工程分析）**：

- RNN 的 $h_{t-1} \to h_t$ 是真实的数据依赖，决定了时间方向的最小 latency；
- attention 的 score 和 weighted sum 具有较好的 position-level parallelism，但会增加 memory bandwidth；
- beam search 复制 $K$ 份状态，可能让片上 buffer 和片外带宽成为瓶颈；
- softmax 需要 exponentiation、reduction 和 normalization，常需专门的近似或融合设计；
- training 需要保存 encoder outputs 和 decoder intermediates，而 inference 更关心低延迟和功耗。

这些是把算法映射到芯片时需要检查的约束，不是课件声称的某个特定硬件实现结果。

## 10. 常见混淆

### 10.1 encoder-decoder vs attention

`encoder-decoder` 是整体架构；`attention` 是 decoder 获取 encoder 信息的一种机制。最简单的模型只传最后一个 fixed context；attention 模型传递全部 encoder outputs，并为每个输出步计算动态 context。

### 10.2 teacher forcing vs greedy decoding

`teacher forcing` 是训练策略：输入真实历史 token。`greedy decoding` 是推理策略：每一步选择当前最大概率 token。一个模型可以用 teacher forcing 训练，再用 greedy 或 beam search 推理。

### 10.3 attention weight vs hard alignment

attention weight 是 `[0,1]` 的连续分布，通常对所有位置求和为 1；它不必等于唯一、离散、正确的词对齐。它是可学习的软读取机制。

### 10.4 probability vs log-probability

完整序列概率是很多小概率的乘积，数值容易下溢；beam search 通常累加 log-probability。因为 `log` 是单调函数，最大化概率和最大化 log-probability 等价，但加法更稳定。

## 11. 从本章通向 Transformer / GPT

> **本节新词**：`self-attention` 在同一序列内部计算依赖；`causal masking` 阻止生成位置读取未来 token；`Transformer` 以 attention 为核心，`GPT` 是采用自回归 Transformer 的生成模型家族。

课件最后把 attention 放到现代模型的入口处：

- **self-attention**：query、key、value 来自同一序列内部；
- **causal masking**：生成位置不能读取未来 token；
- **Transformer**：用 attention 为核心，减少 RNN 的顺序依赖；
- **GPT**：在大规模数据上 pretraining，再进行 post-training，并可结合 tool use。

本章先掌握一个最重要的抽象：**在每个输出时间步，根据当前状态对一组输入表示做可学习的加权读取**。后续学习 Transformer 时，变化主要是“谁产生 query/key/value、是否允许看未来、如何堆叠多头和多层”，而 attention 的“score → normalize → weighted sum”骨架仍然存在。

## 12. 回到完整的序列生成链

读完本章后，整条因果链是：可变长度输入先被 tokenizer/embedding 表示为向量序列；encoder 按顺序读取并形成 hidden outputs；decoder 以 `<bos>` 开始，在每个时间步根据历史状态产生 logits 和 token probability；训练时 teacher forcing 提供真实前缀，推理时模型使用自己生成的前缀；fixed-size context 只能提供一个摘要，attention 则在每个输出步根据当前 query 对全部 encoder outputs 做动态加权读取；在芯片上，这条链同时暴露了 RNN 的 sequential dependency、attention 的 position-wise parallelism 和两者的 memory traffic。

其中几个最重要的关系可以直接从正文公式读出：`<eos>` 是生成停止条件；beam size $K=1$ 退化为 greedy；$\alpha_{t,i}$ 的 $t$ 是输出时间步、$i$ 是输入位置；attention weight 是连续的软读取，不必等于离散词对齐。

## 12.5 从序列算法到可部署系统

### 12.5.1 Padding、mask 和 packed sequence

不同长度的 sequence 放进同一个 batch 时，常用 padding 补齐；padding 不是有效数据，必须用 padding mask 或 length 信息忽略它。packed sequence 则把有效时间步压紧，减少无效计算。二者表达的语义相近，但 memory layout、API 和 kernel 支持不同。

### 12.5.2 Teacher forcing 与 exposure bias

teacher forcing 用真实的上一 token 作为 decoder 输入，训练稳定而且容易并行；推理时只能使用模型自己生成的 token，因此训练分布和推理分布不同，这叫 exposure bias。scheduled sampling 等方法试图缓解它，但会带来稳定性或偏差问题。先记住：teacher forcing 是训练输入策略，不是模型结构。

### 12.5.3 Beam search 的系统代价

beam size 增大时，要维护更多候选前缀、hidden states、scores 和 token histories，还要做 top-k selection 与候选重排。使用 attention 时，beam 还会放大 key/value 读取和 cache 管理。更大的 beam 不保证质量线性提高，必须结合 latency budget 和评价指标选择。

### 12.5.4 Streaming 与 chunking

在线语音、传感器和翻译不能等待完整输入，通常使用 chunk、有限 context 或增量 state。这会引入边界效应、延迟和上下文截断。离线模型改成 streaming 模型，不只是把 batch 改成 1，还要重新定义可见范围、state reset、时间戳和 evaluation protocol。

### 12.5.5 序列模型不应只看一个分数

至少同时观察 token-level loss、sequence-level metric、长度偏差、重复、提前结束、长序列表现和 latency。beam size、tokenizer、padding、temperature 和 stop token 都会影响结果。部署还应记录首 token 延迟、每 token 延迟、峰值 memory 和能耗。

> **External context**：本节把 packed sequence、exposure bias、streaming 和系统级评估作为课件的工程扩展，解释“能生成”与“能稳定部署”的差别。

## 12.6 教授式完整算例：RNN 如何保存与遗忘信息

先用一个一维教学 RNN 看清 state，再回到真实矩阵公式。设：

$$
h_t=0.8h_{t-1}+x_t,\qquad h_0=0
$$

输入序列为 $[2,0,0]$：

| 时间步 | 计算 | state |
|---|---|---:|
| 1 | $h_1=0.8\times0+2$ | 2.00 |
| 2 | $h_2=0.8\times2+0$ | 1.60 |
| 3 | $h_3=0.8\times1.6+0$ | 1.28 |

后面输入都是 0 时，state 仍保留第一个输入的影响；但每一步乘以 0.8，所以影响逐渐衰减。若有效放大因子大于 1，则可能爆炸。真实 Vanilla RNN 中，这个因子来自 recurrent matrix 和 activation derivative 的组合。

### BPTT 为什么会出现 vanishing gradient？

若 loss 在时间 3，早期 state 的梯度要沿 $h_3\leftarrow h_2\leftarrow h_1$ 传回。简化后：

$$
\frac{\partial h_3}{\partial h_1}=0.8\times0.8=0.64
$$

经过 20 个时间步就是 $0.8^{20}\approx0.0115$。BPTT 就是把 RNN 沿时间展开后，对这个长 computation graph 使用 chain rule。

### LSTM gate 到底解决什么？

- forget gate：旧记忆保留多少；
- input gate：当前输入写入多少；
- output gate：cell state 暴露多少给 hidden state。

当 forget gate 接近 1、input gate 接近 0 时，cell state 可以近似原样通过，为长期梯度提供更平滑的路径。gate 不是拟人化的智能开关，而是由 sigmoid 产生的连续数值，训练中由数据学出来。

| 需求 | 应关注的选择 |
|---|---|
| 短序列、低参数量 | Vanilla RNN 可能足够 |
| 长期依赖、训练稳定 | LSTM 的 cell state 和 gates |
| 想减少门和参数 | GRU 的简化结构 |
| 在线 streaming | state memory、reset、chunk boundary 和 token latency |

> **讲授结论**：RNN 的记忆不是保存整句话，而是把任务需要的信息压进有限 state；vanishing/exploding gradient 是这条压缩路径在反向传播中的数学代价。

### 5. 这一步如何通向 attention

RNN 的瓶颈是：每个时间步都把过去压进一个有限的 hidden state；当输入很长时，早期信息可能被压缩或遗忘。Attention 的改变不是取消训练闭环，而是增加一条动态读取路径：当前 decoder state 产生 query，对多个 encoder outputs 计算权重，再把需要的 value 加权取回。下一篇会把这条路径推广成 self-attention，并分析它如何换取更高的并行度和更大的 memory traffic。

## 13. 一页学习链

```text
可变长度输出
  -> RNN hidden state 保存历史
  -> encoder-decoder 分离理解与生成
  -> autoregressive probability 分解
  -> greedy 过于短视
  -> beam search 保留候选前缀
  -> teacher forcing 稳定训练
  -> fixed context 成为信息瓶颈
  -> attention 为每个输出步动态读取输入
  -> self-attention / causal masking / Transformer
```

到这里，先不用把所有 score function 的变体当成新的模型。它们都服务于同一条骨架：用 query 与输入表示计算 score，normalize 成权重，再得到当前输出所需的 context。下一篇 Transformer 会把这条“动态读取”机制推广到同一序列内部。

## 来源与截图

- 主要来源：`Sequence Models and Attention.pdf`，Lan Xu，CS 280 Deep Learning，Lecture 07，2026 Fall，80 pages。
- 截图保存在 `assets/3 Sequence Models and Attention/`，文件名中的数字对应 PDF 页码。
- 本章的 softmax 数值计算、RNN 公式和 AI chip 工程视角是教学补充；课件中的模型结构、训练流程、beam search、attention 与应用案例均已在对应章节保留。

## 外部资料与进阶阅读

以下资料用于补充序列训练与部署边界。访问日期：2026-10-17。

- **External context：seq2seq、teacher forcing 与 attention**：PyTorch 官方 [Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)，适合把本章公式映射到一个完整训练流程。
- **External context：变长 sequence**：PyTorch 官方 [PackedSequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html)，用于理解 padding 与 packed layout 的区别。
- **External context：beam search 的理论背景**：D2L [Beam Search](https://d2l.ai/chapter_recurrent-modern/beam-search.html)，用于比较 greedy、beam size 和 cumulative log-probability；教学示例不等于所有翻译任务的最佳策略。
- **进阶路线**：sequence-level evaluation、streaming decoder、state management 和 beam/cache scheduling；这些是从模型算法进入在线系统的桥梁。
- **External course-material extension — attention pooling**：D2L, *Dive into Deep Learning*, [Attention Mechanisms and Transformers](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)，提供从 query/key/value 到加权读取的循序解释，可配合本章 §7 的数值例子复习；其公式约定与本课件可能略有不同，阅读时以变量含义而非字母名称对齐。访问日期：2026-10-22。
