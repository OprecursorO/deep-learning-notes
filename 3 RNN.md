# 3 RNN：Recurrent Neural Networks

> [!tip] 老师讲解
> 本章正文按“问题 → 直觉 → 公式 → 算例 → 工程代价”讲 RNN、BPTT、LSTM 和 GRU；互动只用来观察当前机制。

> [!tip] 互动入口
> [[互动实验入口]] · [[assets/2 CNN and RNN/interactive/rnn-slide-visual-explorer.html|打开 RNN 课件图视觉交互]] · [[assets/2 CNN and RNN/interactive/rnn-bptt-gradient-explorer.html|打开 BPTT 梯度链]] · [[assets/2 CNN and RNN/interactive/lstm-gate-state-explorer.html|打开 LSTM gate 与 state]]

> 本篇承接 [[1 Basic neural networks]]，与 [[2 CNN]] 并列；先学习序列、hidden state、Vanilla RNN 和 BPTT，再学习 LSTM/GRU。完成后进入 [[4 Sequence Models and Attention]]，理解 attention 如何补足固定 hidden state 的信息瓶颈。
>
> 课件来源：Lecture 5 **Recurrent Neural Networks I: Basics**（RNN 1.pdf 中 RNN 部分，slides 57–112）与 Lecture 6 **Recurrent Neural Networks II: LSTM, GRU**（RNN 2.pdf，65 页）。RNN 1 前半段的 semantic segmentation 已归入 [[2 CNN]]。

## 0. 分层术语路线

```text
序列对象 -> sequence / token / time step / hidden state
    -> Vanilla RNN / recurrent connection / parameter sharing
    -> unroll / computation graph / BPTT
    -> vanishing & exploding gradients / gradient clipping
    -> cell state / gates / LSTM / GRU
    -> streaming / state memory / latency
```

先理解“当前输入 + 过去摘要 → 新摘要”，再读矩阵公式；高级硬件视角放在相应机制之后。

## 老师开场：序列的难点是“顺序会改变意义”

图片的像素可以在同一层大体并行处理，但句子、语音和传感器信号有时间顺序。`dog bites man` 和 `man bites dog` 用的是相似的词，顺序却改变了意义。若每个时间步完全独立，模型就无法知道前面发生过什么；若把任意长度的历史全部原样保存，计算和存储又会失控。

RNN 的解决办法是维护一个固定宽度的 `hidden state`：

$$
h_t=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b_h).
$$

老师会把它说成“当前输入 $x_t$ 加上上一时刻的摘要 $h_{t-1}$，生成新的摘要 $h_t$”。这个摘要不是一段可以直接读出的文字，而是一组经过训练的数字。它能保留任务需要的信息，也可能忘掉不重要的信息。

### 为什么共享参数，又为什么不能完全并行

每个时间步调用同一个 recurrent cell，所以 $W_{xh}$、$W_{hh}$ 和 bias 只存一份；这让模型可以处理不同长度的序列。可是计算 $h_t$ 必须等 $h_{t-1}$，因此同一条序列的时间步存在 dependency chain。训练时把这条链展开做 BPTT；推理时还要保存每个请求自己的 state，并正确处理 reset、padding 和结束标记。

如果每一步的局部梯度因子大约是 $0.8$，跨 20 步的影响约为 $0.8^{20}\approx0.0115$；如果因子大于 1，又可能越来越大。这就是 vanishing/exploding gradient 的直觉来源。LSTM/GRU 并不是“更聪明的记忆”，而是用 gates 改造信息和梯度通过的路径。

从 AI chip 视角，RNN 的主要问题不是每个矩阵乘不能并行，而是同一条 stream 的下一步要等待上一步。硬件可以并行 batch、hidden dimension 或多个请求，但单条 sequence 的 latency 受到时间依赖限制。

> [!tip] 互动入口
> 本篇交互已整理为本地 HTML 视觉页面：[[assets/2 CNN and RNN/interactive/rnn-slide-visual-explorer.html|打开 RNN 课件图视觉交互]]。页面含时间展开、BPTT 梯度路径和 LSTM 门控三个主画布；互动数值是教学模型，不是训练结果，公式与课件静态图仍是依据。
>
> **按主题打开局部演示**：[[assets/2 CNN and RNN/interactive/rnn-bptt-gradient-explorer.html|BPTT 梯度链]] · [[assets/2 CNN and RNN/interactive/lstm-gate-state-explorer.html|LSTM gate 与 state]]。推荐先看主画布，再用局部页面复核一个机制。

## 1. Recurrent Neural Networks（RNN）

**动机梯度**：句子、声音和传感器信号的顺序不能打乱；固定窗口之外的历史又会消失。RNN 选择逐步读取 sequence，并将历史压进 hidden state。这样可以处理可变长度输入并共享 weights，但同一条 sequence 的 time steps 有依赖，限制并行度并形成 latency。

### 1.0 这一节的学习目标：读完你应该会什么？

如果你是第一次学 RNN，不要一开始背所有公式。请把目标分成三层：

课件给 RNN 设计提出了四个工程要求：支持 **variable-length sequence**（不同长度的序列）、追踪 **long-term dependencies**（长期依赖）、保留 **order**（顺序信息），并在不同样本和 time steps 间 **share parameters**（共享参数）。后面的 Vanilla RNN、LSTM、GRU 都是在这四个要求下做取舍。

1. **第一层：会讲故事**——RNN 按顺序读数据；每读一步，就更新一张“数字便签” $h_t$。
2. **第二层：会读图和公式**——知道 $x_t$ 是当前输入，$h_{t-1}$ 是旧便签，$h_t$ 是新便签；知道同一个 cell 在每个 time step 重复使用。
3. **第三层：会解释为什么需要 LSTM/GRU**——Vanilla RNN 的长链条会让 gradient 消失或爆炸，gates 用来控制哪些信息保留、写入和输出。

这一节不是要求你一次记住每个矩阵的形状，而是先建立下面这条主线：

$$
\text{sequence}\rightarrow\text{逐步读取}\rightarrow\text{更新 hidden state}\rightarrow\text{产生 output}
$$

> **学习提醒**：如果某个公式看不懂，先回到“当前输入 + 过去摘要 → 新摘要”这句话。公式只是把这句话写成可计算的形式。

### 1.1 RNN 入门：先把基础名词认清楚

如果你第一次接触 RNN，可以先只记住这一句话：**RNN 是一种按顺序读取一串数据，并把“到目前为止读到的信息”保存在 hidden state 里的 neural network。**

#### 1.1.1 什么是 sequence？

**Sequence（序列）** 是一串有顺序的数据。顺序很重要，因为后面的元素通常要结合前面的元素才能理解：

- sentence：`I love AI` 中 `AI` 出现在最后；
- speech：声音样本必须按时间顺序播放；
- sensor signal：温度或加速度会随时间变化；
- stock/time series：今天的数值与之前的趋势有关。

图片也有很多数字，但图片的重点是二维空间邻近关系；sequence 的重点是“先后顺序”。因此 CNN 常沿 height/width 看邻域，而 RNN 沿 time/order 读取数据。

#### 1.1.2 一个 sequence 由什么组成？

把一句话 `I love AI` 写成：

$$
x_1=\text{I},\qquad x_2=\text{love},\qquad x_3=\text{AI}
$$

这里的每一个位置叫一个 **time step（时间步）**。`time` 不一定真的表示钟表时间；它也可以表示句子中的第几个词、音频中的第几个采样点，或传感器记录中的第几个时刻。

| 名词 | 英文含义 | 对初学者的理解 |
|---|---|---|
| sequence | 有顺序的数据集合 | 整句话、整段音频或一串传感器数值 |
| time step $t$ | 序列中的一个位置 | 第 1 个词、第 2 个词…… |
| token | 序列中的一个基本元素 | 一个词、一个字、一个字符或一个音频片段 |
| $x_t$ | 第 $t$ 个输入 | RNN 当前正在读取的 token 的数字表示 |
| $T$ | sequence 长度 | 一共有多少个 time steps |

在实际计算中，模型不能直接处理文字 `love` 这种字符串，所以 token 会先转成数字向量，例如 one-hot vector 或 word embedding。你可以把 embedding 暂时理解为“用一串数字表示一个词”；embedding 的细节不影响先理解 RNN 的循环过程。

#### 1.1.3 hidden state 到底是什么？

**Hidden state（隐藏状态）** 是 RNN 在当前 time step 对“过去信息”的数字摘要。它不是一个人类可读的句子，也不是永久保存的 memory，而是一个会不断更新的 vector。

仍以 `I love AI` 为例：

1. 读入 `I`，得到 $h_1$：此时 state 只包含关于 `I` 的初步信息；
2. 读入 `love`，结合旧的 $h_1$，更新成 $h_2$：state 同时带有前两个 token 的信息；
3. 读入 `AI`，结合 $h_2$，更新成 $h_3$：state 携带整句到目前为止的摘要。

因此可以把 hidden state 想成一个**会随阅读过程改写的便签**：每读一个新 token，就根据“旧便签 + 当前 token”写出一张新便签。它之所以叫 hidden，是因为它通常不是最终要展示给用户的答案，而是 network 内部传递的中间表示。

#### 1.1.4 cell、input、output 分别是什么？

- **RNN cell**：执行“一次更新”的小网络。它接收当前 $x_t$ 和上一状态 $h_{t-1}$，输出新状态 $h_t$。同一个 cell 会在每个 time step 重复使用。
- **Input $x_t$**：当前这一步读到的数据，例如第 $t$ 个词的 embedding。
- **Previous hidden state $h_{t-1}$**：上一步留下的摘要，代表过去已经读过的内容。
- **New hidden state $h_t$**：当前步骤更新后的摘要，传给下一步。
- **Output $y_t$**：这一步对外产生的结果。某些任务每一步都要 output，某些任务只使用最后一个 output。

最核心的数据流可以写成：

```text
当前输入 x_t + 上一步状态 h_{t-1}
                 ↓
             RNN cell
                 ↓
       新状态 h_t +（可选）输出 y_t
```

这里的“循环”不是把一个数字送回过去，而是：第 $t$ 步产生的 $h_t$，会在第 $t+1$ 步作为 $h_{t-1}$ 使用。

#### 1.1.5 用一个极小例子走一遍

假设我们要处理 `I love AI`，并且每个 token 已经被表示成长度为 4 的向量：

$$
x_t\in\mathbb{R}^{4},\qquad h_t\in\mathbb{R}^{3}
$$

这表示：每个输入 token 有 4 个数字，RNN 的 hidden state 用 3 个数字保存历史摘要。计算过程是：

| 步骤 | 当前输入 | 进入 cell 的旧 state | cell 产生的新 state |
|---|---|---|---|
| $t=1$ | $x_1=\text{I}$ | $h_0$（通常初始化为全 0） | $h_1$ |
| $t=2$ | $x_2=\text{love}$ | $h_1$ | $h_2$ |
| $t=3$ | $x_3=\text{AI}$ | $h_2$ | $h_3$ |

注意：每一行都使用**同一个 RNN cell 和同一套 weights**，改变的只有当前输入和传入的 hidden state。$h_0$ 是序列开始前的初始状态；最简单的做法是把它设成全 0 向量。

如果任务是判断整句话情感，可以只使用最后的 $h_3$；如果任务是给每个词标注词性，就会使用 $h_1,h_2,h_3$ 的每一个 output。这就引出了后面的 many-to-one 和 many-to-many。

#### 1.1.6 符号和形状先记这一张表

| 符号 | shape（形状） | 意义 |
|---|---|---|
| $x_t$ | $d_x$ | 第 $t$ 步的输入向量 |
| $h_{t-1},h_t$ | $d_h$ | 旧的/新的 hidden state |
| $W_{xh}$ | $d_h\times d_x$ | 把输入维度变成 hidden 维度 |
| $W_{hh}$ | $d_h\times d_h$ | 把旧 hidden state 变成新 hidden 维度 |
| $W_{hy}$ | $d_y\times d_h$ | 把 hidden state 映射到 output 维度 |
| $y_t$ | $d_y$ | 当前步骤的 output 或 logits |

例如 $W_{xh}x_t$ 的形状是 $(d_h\times d_x)(d_x)=d_h$，所以它可以和 $W_{hh}h_{t-1}$ 相加。shape 的作用就是检查“这些数字能不能相乘、能不能相加”。

> **和前序笔记的连接**：weight、bias、activation、loss、gradient 等基础概念见 [[1 Basic neural networks]]。RNN 的新难点不是重新发明这些词，而是把同一个小网络沿 sequence 重复使用，并让 hidden state 在时间步之间传递。

### 1.2 为什么需要 RNN？从 image 到 sequence

CNN 主要处理有固定空间结构的 image；RNN 主要处理有顺序的 **sequence**，例如 sentence、speech signal、sensor readings 或 time series。sequence 中当前元素的含义常常依赖之前的元素：例如看到 “I am interested in AI” 时，最后一个词的含义与前文有关。

把 sequence 写成 $x_1,x_2,\ldots,x_T$：$T$ 是时间步（time steps）的数量，$x_t$ 是第 $t$ 个输入。RNN 的核心想法是维护一个会随时间更新的 **hidden state** $h_t$，把“到目前为止读到的信息”压缩在一个向量里。

固定窗口模型只看最近 $K$ 个 token；窗口之外的信息不会直接进入计算。RNN 通过 hidden state 把较早的信息带到后面的时间步，因此不需要预先规定一个固定窗口。

#### 1.2.1 从 autoregressive model 到 RNN

**Autoregressive model（自回归模型）** 的意思是：用已经出现的元素，预测下一个元素。对句子来说，它学习的是：

$$
p(\text{下一个词}\mid\text{前面已经出现的词})
$$

整句话的概率可以拆成每一步预测概率的乘积：

$$
p(x_1,x_2,\ldots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_1,\ldots,x_{t-1})
$$

初学者可以把它理解成“每次只回答下一个词是什么”，但模型要连续回答很多次。

最早的一种做法是 **n-gram language model**。例如 3-gram 只根据前两个词预测第三个词：

```text
I love -> ?
```

它可以统计训练语料中 `I love AI`、`I love chips` 等短片段出现的次数，再把次数转换成概率。这个方法直观，但有两个问题：

1. **context 太短**：如果答案依赖很早以前的词，固定窗口外的信息完全看不到；
2. **data sparsity**：词汇一多，可能的词组数量会快速增长，大多数组合在训练语料里从未出现，统计表就很稀疏。

![[assets/2 CNN and RNN/RNN1-57.png]]

> **来自 RNN 1 slide 57｜看图重点**：n-gram 用一个有限长度的词组统计条件概率。它的优点是简单直接，缺点是词组长度一增加，可能的组合数量会快速膨胀。

Neural language model 用 neural network 和 word embedding 代替巨大的查表，但如果它仍然只接收固定的前 $K$ 个词，context 仍有限。RNN 的改进是把过去的信息压缩到 hidden state 中，让新预测可以间接使用更长的历史。

![[assets/2 CNN and RNN/RNN1-62.png]]

> **来自 RNN 1 slide 62｜看图重点**：固定窗口模型只能直接看到选定窗口内的词；在窗口外的词不会直接参与当前计算。RNN 通过 hidden state 把更早的信息传到后面，这正是 recurrent connection 要解决的问题。

#### 1.2.2 Slides 中的两个小直觉例子

slides 用简单的 sequence 任务说明：RNN 不只会处理文字，它也能维护一个很小但有用的 state。

- **Summation network**：依次读入数字，把“目前为止的总和”放在 hidden state 中。读入新数字时，state 更新为旧总和加上新数字。
- **Parity-check network**：依次读入 0/1，判断 1 的个数是 even 还是 odd。它不需要记住全部输入，只要在 state 中保存当前 parity；每读入一个 1，就把 even/odd 翻转一次。

这些例子想表达的不是要手工设计这样的 network，而是：**hidden state 是任务需要的“压缩记忆”**。如果任务只需要总和，state 不必保存整串数字；如果任务需要判断奇偶，state 只需保存当前是 even 还是 odd。学习阶段会由 training data 调整 weights，让 network 自己学出合适的 state 更新规则。

### 1.3 Recurrent connection：同一个 cell 重复使用

普通 feedforward network 的信息主要向前流一次；RNN 在 hidden units 上增加 recurrent connection：上一个时间步的 hidden state 会反馈给当前时间步。

![[assets/2 CNN and RNN/RNN1-63.png]]

> **来自 RNN 1 slide 63｜看图重点**：左边是有 self-loop 的 compact 图；右边把它沿时间展开（unroll）成 time 1、time 2、time 3。每个时间步看到了不同的 input，但使用的是同一套 weights 和 biases，这就是 **parameter sharing through time**。

因此“循环”不是每一步都创建一个新网络，而是同一个 recurrent cell 被重复调用。对 AI chip design 来说，这意味着 weights 可以重复使用，减少反复从较慢 memory 读取同一组 parameters；但 time steps 之间又有依赖，不能像独立 image samples 那样完全并行。

![[assets/2 CNN and RNN/RNN-shared-weights-unrolled.svg]]

> **自制教学图｜读图方法**：水平方向是 hidden state 的时间传递，竖直方向是当前输入；上方只有一份参数 θ。三个 cell 是同一个 cell 的三次调用，不是三套独立 weights。

![[assets/2 CNN and RNN/RNN selected/RNN1-63.png]]

> **来自 RNN 1 slide 63｜看图重点**：左边的 self-loop 是“循环”的简写；右边是把它展开成 time 1、time 2、time 3。展开后看起来像多个网络，但它们共享同一套 weights。

### 1.4 Vanilla RNN 的计算

最基本的 Vanilla RNN 使用上一时刻的 hidden state $h_{t-1}$ 和当前输入 $x_t$ 计算新的 state：

$$
h_t=\tanh\left(W_{hh}h_{t-1}+W_{xh}x_t+b_h\right)
$$

再根据 hidden state 产生 output：

$$
y_t=W_{hy}h_t+b_y
$$

其中：

- $x_t\in\mathbb{R}^{d_x}$：当前输入，例如一个 token 的 embedding；
- $h_t\in\mathbb{R}^{d_h}$：RNN 的 memory/state；
- $y_t\in\mathbb{R}^{d_y}$：当前输出或 logits；
- $W_{xh},W_{hh},W_{hy}$：所有时间步共享的 weight matrices。

![[assets/2 CNN and RNN/RNN1-72.png]]

> **来自 RNN 1 slide 72｜公式阅读方法**：先把 $x_t$ 和 $h_{t-1}$ 做线性变换并相加，再经过 `tanh` 得到新的 $h_t$；最后把 $h_t$ 映射到 $y_t$。$W_{hh}h_{t-1}$ 是“保留过去信息”，$W_{xh}x_t$ 是“加入当前信息”。

#### 1.4.1 不带矩阵的手算直觉

为了先理解“状态如何移动”，暂时把真实 RNN 简化成一个教学模型：

$$
h_t=0.5h_{t-1}+x_t,
\qquad h_0=0
$$

若输入依次为 $x_1=2$、$x_2=4$、$x_3=6$：

| 时间步 | 计算 | 新状态 |
|---|---|---:|
| $t=1$ | $h_1=0.5\times0+2$ | $2$ |
| $t=2$ | $h_2=0.5\times2+4$ | $5$ |
| $t=3$ | $h_3=0.5\times5+6$ | $8.5$ |

你可以看到：$h_3$ 不是只等于当前输入 $6$，它还带着前面步骤留下的影响。这就是 RNN 的“记忆”直觉。真实 Vanilla RNN 会使用矩阵、bias 和 `tanh`，但信息流仍然是同一个模式。

对硬件来说，一次时间步的主要工作是 matrix-vector multiplication 和 activation。若 batch 中有多个 sequence，可以并行处理不同 sequence；但同一条 sequence 内通常必须等待 $h_{t-1}$ 才能计算 $h_t$。

### 1.5 RNN 的输入输出形式

根据任务，可以决定每个时间步是否需要 output：

- **Many-to-one**：输入一串 sequence，最后输出一个结果，例如 sentence sentiment classification。
- **Many-to-many**：每个时间步都输出，例如词性标注或 sequence labeling。
- **One-to-many**：一个输入产生一串输出，例如 image captioning 的 decoder。
- **Sequence-to-sequence**：先把 input sequence 编码成 state，再由另一个 RNN 生成 output sequence，例如 machine translation。

![[assets/2 CNN and RNN/RNN1-83.png]]

> **来自 RNN 1 slide 83｜many-to-many**：每个 $x_t$ 都更新一个 $h_t$，并产生对应的 $y_t$。所有时间步共用同一个 $f_W$，所以图中每个灰色 cell 虽然画了很多次，参数并没有复制很多份。

> **先用任务来记结构**：情感分类通常是 many-to-one；词性标注通常是 many-to-many；图像生成文字可以看作 one-to-many；机器翻译常用 sequence-to-sequence。

![[assets/2 CNN and RNN/RNN1-88.png]]

> **来自 RNN 1 slide 88｜sequence-to-sequence**：先用 many-to-one 部分把输入序列压缩成一个表示，再用 one-to-many 部分逐步生成输出序列。这个结构自然对应 encoder-decoder 思路。

### 1.6 BPTT：沿时间展开后反向传播

#### 为什么要把 RNN “展开”

前向运行时，RNN 看起来像一个带循环箭头的 cell：它每次读取 $x_t$ 和上一时刻的状态 $h_{t-1}$，产出 $h_t$。但普通 backpropagation 需要一张有方向的计算图来追踪“某个数影响了哪些结果”。因此训练时把循环按实际序列长度复制成时间上的调用链：第 1 次调用得到 $h_1$，它连到第 2 次调用得到 $h_2$，依此类推。图上出现多个 cell，表示同一个函数在不同时间调用，并不表示复制了多套参数。

**Backpropagation Through Time（BPTT，时间反向传播）** 就是：先按时间展开 RNN，再在这张展开图上执行链式法则。名字中的 “through time” 指梯度能从较晚时刻沿着隐藏状态依赖，回到较早时刻。

![[assets/2 CNN and RNN/RNN1-91.png]]

> **来自 RNN 1 slide 91｜先读左图，再读右图**：左边将 3 个时刻画成 $x^{(t)}\to h^{(t)}\to y^{(t)}$。字母 $u,w,v$ 标在重复连接上，表示跨时间共享的参数。右边进一步拆开 cell 内部的中间计算：每个时刻的输入和前一状态共同影响当前中间量、隐藏状态与输出，多个时刻的损失最后汇入同一个 $\mathcal L$。互动重绘了这条结构；请观察红色路径怎样从 $\mathcal L$ 倒走，并回到同一份 $u,w,v$。

#### 前向：每一步如何产生可训练的结果

先用本章的矩阵记法写 Vanilla RNN。对每个时刻 $t$：

$$
z_t=W_{xh}x_t+W_{hh}h_{t-1}+b_h,\qquad h_t=\tanh(z_t),\qquad y_t=W_{hy}h_t+b_y.
$$

$z_t$ 是 activation 之前的加权和；$h_t$ 是更新后的 hidden state；$y_t$ 是当前预测的 logits 或输出。三个矩阵和 bias 在每个时刻都重复使用。若每步都有监督信号，总损失可写为各时刻损失之和：

$$
\mathcal L=\sum_{t=1}^{T}\ell_t(y_t,\hat y_t).
$$

这不是说一定要每一步都计算 loss：many-to-many 任务通常如此，many-to-one 任务可能只在最后时刻计算 loss。BPTT 都适用；区别只是哪些 loss 节点向计算图送回梯度。课件的右图画出多个时间点连向一个总 $\mathcal L$，互动中用 $\mathcal L=\ell_1+\ell_2+\ell_3$ 作为三步教学情形。

#### 反向：梯度为什么会跨时间累积

从最晚时刻开始，先计算损失对当前输出的梯度，再沿 $y_t\to h_t\to h_{t-1}$ 等边反向应用链式法则。记 $\bar h_t=\partial\mathcal L/\partial h_t$，它包含两类影响：当前时刻输出损失直接给 $h_t$ 的信号，以及未来状态 $h_{t+1}$ 对 $h_t$ 的依赖所传回来的信号。以列向量约定表示，递推可写为：

$$
\bar h_t=\underbrace{W_{hy}^{\mathsf T}\frac{\partial\ell_t}{\partial y_t}}_{\text{当前输出的直接贡献}}+\underbrace{W_{hh}^{\mathsf T}\left(\bar h_{t+1}\odot(1-h_{t+1}^{\odot 2})\right)}_{\text{未来经循环连接传回的贡献}}.
$$

这里的 $\odot$ 读作 **逐元素相乘**（Hadamard product）：两个形状相同的向量按位置相乘，$(a\odot b)_i=a_i b_i$。例如 $(2,3)\odot(4,5)=(8,15)$；它不是矩阵乘法，也不是向量点积。类似地，$h^{\odot 2}$ 表示每个分量分别平方。公式中这个符号表示把传回来的梯度逐维乘上 `tanh` 的导数。最末时刻没有未来项，可把 $\bar h_{T+1}$ 看成 0。这个递推让“反向走图”变成可计算的步骤：先知道较晚时刻的梯度，再向左求较早时刻。

对 $h_k$ 的未来影响只看 recurrent path 时，局部导数矩阵会连乘：

$$
\frac{\partial h_T}{\partial h_k}
=J_TJ_{T-1}\cdots J_{k+1},\qquad
J_t=\frac{\partial h_t}{\partial h_{t-1}}
=\operatorname{diag}(1-h_t^{\odot2})W_{hh}.
$$

注意乘积顺序取决于向量和 Jacobian 的记法约定；含义不变：从 $h_k$ 到 $h_T$ 要经过每一个中间转移。$k$ 越早，因子越多。这解释了为什么最终 loss 对早期输入/状态的学习信号更容易变弱或变强。

#### 共享参数：同一份权重收到多处贡献

由于 $W_{hh}$ 在所有时间步重复出现，改变这一份矩阵会同时改变每次调用。因此总梯度必须把每个使用位置的局部贡献相加。令 $\delta_t=\partial\mathcal L/\partial z_t$，则：

$$
\frac{\partial\mathcal L}{\partial W_{hh}}=\sum_{t=1}^{T}\delta_t h_{t-1}^{\mathsf T},\qquad
\frac{\partial\mathcal L}{\partial W_{xh}}=\sum_{t=1}^{T}\delta_t x_t^{\mathsf T},\qquad
\frac{\partial\mathcal L}{\partial W_{hy}}=\sum_{t=1}^{T}\frac{\partial\mathcal L}{\partial y_t}h_t^{\mathsf T}.
$$

标量课件图把这些共享参数记作 $u,w,v$；它们接收从不同时间步回来的梯度分支，最终各自形成一份总梯度。求和不是把参数复制后分别更新，而是对同一参数在整条序列中造成的全部影响求导。随后优化器才用总梯度更新参数一次。

#### 用 3 步走一遍图

1. **前向开始**：读入 $x_1$ 与初始状态 $h_0$，算出 $h_1,y_1,\ell_1$；再将 $h_1$ 交给下一步。参数 $u,w,v$ 仍是同一份。
2. **前向结束**：第 2、3 步同样产生中间状态、输出与损失；把损失合成为 $\mathcal L=\ell_1+\ell_2+\ell_3$。此时图中的箭头大多从左向右或向上。
3. **反向开始**：从总损失出发，分别沿每个 $\ell_t$ 所在分支回到 $y_t,h_t$。回到 $h_2$ 时，既有第 2 步输出的直接贡献，也有第 3 步经 recurrent link 传回的贡献；到 $h_1$ 时还会再加上后续路径的影响。
4. **汇总参数梯度**：从第 1、2、3 步计算出的 $\partial\mathcal L/\partial w|_t$ 汇入同一个 $w$，形成 $\partial\mathcal L/\partial w=\sum_t\partial\mathcal L/\partial w|_t$。对 $u$ 和 $v$ 同理。完成这一步后，才用优化器更新参数。

#### 交互：沿老师的计算图前向与反向

**HTML 视觉交互**：[[assets/2 CNN and RNN/interactive/rnn-slide-visual-explorer.html#bptt|打开 BPTT 展开计算图]]。先按“前进一步”逐步点亮三个时间步，观察 $x_t,z_t,h_t,r_t,y_t$ 与损失怎样产生；再按“反传一步”，从总损失沿红色路径回到早期状态，最后观察三条分支如何汇入共享参数。选择 `t=1/2/3` 可聚焦比较不同时间步的路径长度。

页面的 $r$ 滑杆是**教学用标量近似**，不是老师给出的模型数值，也不是训练曲线。把跨一步的导数粗略设为同一个 $r$，则跨两步的梯度比例约为 $r^2$：$r=0.8$ 时为 $0.64$，$r=1$ 时保持，$r=1.2$ 时为 $1.44$。它只解释“连续乘法为何会衰减或放大”。真实网络在每一步乘的是 $J_t$ 这样的矩阵 Jacobian，方向不同可能放大或缩小，且各时刻的矩阵并不相同。

#### 常见误解与边界

- **“BPTT 是一种和 backpropagation 不同的更新规则。”** 不是。它把 recurrent graph 按时间展开，然后执行链式法则；更新仍由 optimizer 完成。
- **“每个时间步有一份独立权重。”** 图上有多个 cell，但参数共享；梯度贡献先求和，再更新那一份参数。
- **“只要用了 BPTT，就必然梯度消失。”** 不必然。它说明梯度如何计算；消失/爆炸是长链上的 Jacobian 连乘可能带来的数值问题，不是算法必然失败。
- **“每步导数小于 1 就严格等价于整个多维网络缩小。”** 固定标量只是直觉模型。真实矩阵的奇异值、方向与时变性共同影响梯度范数；即使某些方向缩小，另一些方向也可能放大。

vanishing gradient 的后果是早期输入几乎收不到来自较晚目标的有效学习信号，模型难以学 long-term dependency；exploding gradient 则可能让参数更新过大、训练不稳定。RNN 2 slide 27 展示了梯度在多次 recurrent transition 与 `tanh` 间传播，并指出最大 singular value 大于 1 时可能爆炸、小于 1 时可能消失；它也引出 gradient clipping 来限制过大的梯度范数。后面的 LSTM/GRU 会改变信息通路，但并不保证所有任务都不会遇到优化问题。

### 1.7 RNN language modeling：预测下一个 token

在 language modeling 中，目标是学习：给定前面的 token，预测下一个 token 的 probability distribution：

$$
p(x_t\mid x_1,\ldots,x_{t-1})
$$

每个 token 可以先表示成 one-hot vector 或 **word embedding**，再送入 RNN。输出 $y_t$ 通常经过 Softmax，得到整个 vocabulary 中每个候选 token 的 probability；训练时使用 cross-entropy loss。这个基础的 Softmax 和 cross-entropy 已在 [[1 Basic neural networks]] 中介绍，这里只关注 sequence 如何产生。

假设 vocabulary 只有 `{I, love, AI, chips, <EOS>}` 五个 token，那么在某一步，RNN 的 output 可能是一个长度为 5 的向量：

$$
y_t\in\mathbb{R}^{5}
$$

Softmax 会把它变成五个概率，例如 `I: 0.02, love: 0.03, AI: 0.70, chips: 0.20, <EOS>: 0.05`。这些概率加起来等于 1；模型可以选择概率最高的 token，也可以按照概率随机抽样。`<EOS>`（end of sequence）表示“句子结束”。

训练一整句话时，通常把当前 token 当作 input，把下一个 token 当作 target：

```text
input:  I      love       AI
target: love   AI         <EOS>
```

所以模型不是在一次计算中直接输出整句话，而是对每个 time step 都做一次“预测下一个 token”的分类。所有 time steps 的 cross-entropy 可以相加或取平均，作为这条 sequence 的 loss。

生成时，模型在 time 1 预测一个 token，把它作为下一步输入，再预测 time 2，如此循环。训练时常使用 **teacher forcing**：下一步输入使用真实的 previous token，而不是模型刚刚预测的 token。

![[assets/2 CNN and RNN/RNN1-105.png]]

> **来自 RNN 1 slide 105｜看图重点**：生成过程中，hidden state 沿时间传递；每一步的输出 token 会影响下一步。训练时 slide 使用 previous ground-truth tokens，这就是 teacher forcing。推理时没有 ground truth，只能把自己的输出反馈回来。

对 AI chip design 的一个重要区别是：训练通常需要保存许多时间步的 intermediate activations 供 BPTT 使用，带来 memory footprint；推理则可以逐步 streaming，只保留当前 hidden state 和必要的 weights，但每个新 token 仍需执行 recurrent matrix operations。

#### Slides 提到的 language-model 挑战

- **Vocabulary 很大**：如果要在数百万个候选词中预测下一个词，最后的 output layer 和 Softmax 都会很大，计算与 memory cost 都高。
- **Unknown words**：测试时可能出现训练语料没见过的人名、地点或新词。实际系统通常需要更细的 tokenization（例如拆成 subword），而不是假设所有词都在词表中。
- **Word segmentation**：英语通常用空格分隔单词，但中文没有天然空格；“什么算一个 word”本身就是 preprocessing 和 modeling 的问题。

这些是 slides 对基础 word-level language model 的限制说明；它们不是说 RNN 不能处理中文，而是说明输入 token 的定义会影响 vocabulary、sequence length 和计算量。

### 1.7.1 可选现象展示：比较 RNN、LSTM 和 GRU 的记忆

运行 [03_rnn_lstm_gru_memory.py](experiments/03_rnn_lstm_gru_memory.py)：

```bash
cd "/Users/dongyangchen/Desktop/lecture note/deep learning/experiments"
python3 03_rnn_lstm_gru_memory.py
```

任务很小：输入长度为 30 的 0/1 sequence，模型只需要预测第一个 bit。这个任务把 `long-term dependency` 单独拿出来，不让语言词表、语义和 preprocessing 抢走注意力。脚本分别训练 Vanilla RNN、LSTM、GRU，并打印 validation accuracy。

改变 sequence length、gradient clipping 或 hidden size，可以把正文讨论的三个因素分别显示出来：依赖距离、更新稳定性和模型容量。它们是可选的现象展示，不能被解读成“LSTM 永远优于 RNN”；当前 toy task 只验证一个边界：当信息必须跨越很多 recurrent steps 时，gated state 往往更容易保留有用信号。可以对照 PyTorch 的 [char-RNN classification tutorial](https://github.com/pytorch/tutorials/blob/main/intermediate_source/char_rnn_classification_tutorial.py)。

### 1.8 RNN 与 CNN 的硬件视角对比

| 维度 | CNN | Vanilla RNN |
|---|---|---|
| 数据结构 | image/grid，空间邻域 | sequence，时间顺序 |
| 参数复用 | 不同空间位置共享 filter | 不同时间步共享 recurrent weights |
| 主要并行性 | 空间位置和 channels 通常可大量并行 | 同一条 sequence 的 time steps 有依赖，parallelism 较少 |
| 主要 memory 问题 | feature maps、weights、activation traffic | hidden states、BPTT 保存的中间 state |
| 长距离信息 | 通过多层/receptive field 逐渐整合 | 通过 hidden state 传递，但 Vanilla RNN 容易梯度消失 |

这个对比不是说 CNN 或 RNN 谁“更好”，而是说明 accelerator 设计时要关注不同 bottleneck：CNN 常追求高吞吐的 spatial parallelism；RNN 需要在 sequential dependency、weight reuse、state buffering 和 latency 之间做权衡。

### 1.9 Vanilla RNN 为什么会忘记很久以前的信息？

RNN 的设计目标很诱人：让 $h_t$ 把过去一路带到未来。但在训练时，loss 的影响必须沿着很多个 time steps 反向传回去。每经过一步，都要乘上一个由 weights 和 activation derivative 构成的局部 Jacobian；很多个 Jacobian 连乘后，数值可能变得极小或极大。

这就是两个经典问题：

- **Vanishing gradient（梯度消失）**：gradient 越传越小，早期输入几乎收不到学习信号，模型难以学习 long-term dependency。
- **Exploding gradient（梯度爆炸）**：gradient 越传越大，parameter update 可能突然变得非常大，训练不稳定。

可以先用一个最简单的直觉理解：如果每一步都乘以 $0.5$，连续乘 20 次就是 $0.5^{20}$，已经非常接近 0；如果每一步都乘以 $2$，连续乘 20 次又会变得很大。这不是 BPTT 算错了，而是 recurrent computation graph 本身很长。

![[assets/2 CNN and RNN/RNN2 27--27.png]]

> **来自 RNN 2 slide 27｜看图重点**：Vanilla RNN 的 gradient flow 要穿过许多 recurrent steps。前向计算可能仍然正常，但反向传播的乘积会让 gradient 消失或爆炸。

更正式地说，若每个 time step 的 hidden-state transition 是 $h_t=f(h_{t-1},x_t)$，那么早期状态对后期状态的梯度包含一串 Jacobian（局部导数矩阵）的乘积：

$$
\frac{\partial h_T}{\partial h_k}=\prod_{t=k+1}^{T}\frac{\partial h_t}{\partial h_{t-1}}
$$

`Jacobian` 可以理解为“多维输入发生一点变化时，多维输出怎样变化”的矩阵。矩阵连乘的特征值若大多小于 1，梯度会逐步缩小；若大于 1，梯度会逐步放大。前向的 `tanh` 可能把 activation 压在有限范围内，但这不保证反向的 Jacobian 乘积稳定。

![[assets/2 CNN and RNN/RNN2-21-jacobian.png]]

> **截图导航**：读这页时不要把 `Eigen` 当成额外的网络层；它是在分析 Jacobian 连乘的放大/缩小方向。对硬件和数值实现来说，这对应 activation、gradient 和 update 的动态范围管理。

### 1.10 如何稳定 RNN training？

#### 1.10.1 Gradient clipping：先防止梯度爆炸

如果 gradient 的 norm 超过阈值 $\tau$，可以只缩小它的长度而保留方向：

$$
g\leftarrow g\cdot\min\left(1,\frac{\tau}{\lVert g\rVert}\right)
$$

这样做像给更新量装上“限速器”：gradient 很大时不让它把 weights 一步推得过远；gradient 正常时不改变它。它主要处理 **exploding gradient**，不能从根本上恢复已经消失的 gradient。

![[assets/2 CNN and RNN/RNN selected/RNN2-28.png]]

> **来自 RNN 2 slide 28｜看图重点**：clipping 把过大的 gradient 缩短到阈值以内，所以可以避免一次 update “冲飞”；它不会改变 gradient 的主要方向。

#### 1.10.2 改变 recurrent cell 的结构

如果 hidden state 默认可以把过去的信息近似原样传递，那么反向传播更容易保持稳定。LSTM 和 GRU 都采用了这个方向：它们不再让一个简单的 `tanh` 同时负责“记住、忘掉、加入新信息和输出”，而是增加可学习的 **gates**，让 network 自己决定信息流量。

### 1.11 LSTM：让记忆有“阀门”

**局部动机**：Vanilla RNN 每一步都把旧 state 和新输入混在一起经过非线性，长期信息容易在反复更新中衰减。LSTM 增加 cell state 与 gates，让旧记忆可沿较直接路径保留，再按需写入和输出。收益是更可控的长期依赖；代价是 gates 增加矩阵运算、parameters 和 state storage。

**Long Short-Term Memory（LSTM）** 可以看成 Vanilla RNN 的一个更精细的 recurrent cell。它的核心不是让模型记住所有东西，而是把记忆分成两条相关但不同的路径：

- **Cell state $C_t$**：沿着 cell 顶部的路径传递，像一条比较直接的长期信息通道。
- **Hidden state $h_t$**：当前 time step 对外暴露的状态，既可以传给下一步，也可以接 prediction head。

每个 gate 的输出通常经过 sigmoid，因此每个位置都在 $(0,1)$ 之间，可以理解为“保留多少”的 soft decision：0 表示几乎不通过，1 表示几乎全部通过。

![[assets/2 CNN and RNN/LSTM-cell-state-flow.svg]]

> **自制教学图｜读图方法**：左路是旧记忆经过 forget gate；下路是候选内容经过 input gate；两路在加法节点汇合成为新 $C_t$，最后 output gate 决定这一时刻暴露多少给 $h_t$。图中每个“×”都是逐元素相乘。

![[assets/2 CNN and RNN/RNN2/RNN2-34.png]]

> **来自 RNN 2 slide 34｜看图重点**：LSTM 用 multiplicative gates 判断信息是否重要。图中的 `×` 表示逐元素相乘，`+` 表示逐元素相加。

![[assets/2 CNN and RNN/RNN selected/RNN2-36.png]]

> **来自 RNN 2 slide 36｜看图重点**：Cell state 是一条较直接的历史通道。它不会每一步都被完整重写，而是只经过 gate 控制和新信息相加，因此比单纯反复经过 `tanh` 更有利于传递长期信息。

课件还提到 **peephole connection**：gate 不只读取 $x_t$ 与 $h_{t-1}$，也可以直接读取 cell state 的信息。这样 gate 能根据内部记忆的当前数值决定是否忘记、写入或输出。它是 LSTM 的可选变体，不是所有实现的默认配置。

![[assets/2 CNN and RNN/RNN2-45-peephole.png]]

#### 1.11.1 Forget gate：旧记忆保留多少？

$$
f_t=\sigma\left(W_f[h_{t-1},x_t]+b_f\right)
$$

$f_t$ 根据当前输入 $x_t$ 和旧状态 $h_{t-1}$，决定旧的 cell state $C_{t-1}$ 有多少继续留下。这里的“forget”不是一次性删除整段记忆，而是每个维度分别决定保留比例。

![[assets/2 CNN and RNN/RNN2/RNN2-38.png]]

#### 1.11.2 Input gate：当前信息值得写入多少？

LSTM 先计算一个写入比例 $i_t$，再计算当前输入提炼出的候选内容 $\widetilde{C}_t$：

$$
i_t=\sigma\left(W_i[h_{t-1},x_t]+b_i\right)
$$

$$
\widetilde{C}_t=\tanh\left(W_C[h_{t-1},x_t]+b_C\right)
$$

随后更新 cell state：

$$
C_t=f_t\odot C_{t-1}+i_t\odot\widetilde{C}_t
$$

其中 $\odot$ 是 **element-wise multiplication（逐元素相乘）**。这条公式可以用一句人话读出来：

> 新记忆 = 保留下来的旧记忆 + 值得写入的新信息。

#### 为什么 LSTM 要逐元素相乘？

LSTM 的 gate 不是一个单独的数字，而是一个和 state 维度相同的向量。例如，如果 cell state 有 3 个维度：

$$
C_{t-1}= [0.8,\,-0.4,\,0.6]
$$

forget gate 也会输出 3 个值：

$$
 f_t=[0.9,\,0.2,\,0.7]
$$

逐元素相乘表示：**第 1 个 gate 只控制第 1 个 memory dimension，第 2 个 gate 只控制第 2 个 dimension，第 3 个 gate 只控制第 3 个 dimension**：

$$
 f_t\odot C_{t-1}
 = [0.9\times0.8,\,0.2\times(-0.4),\,0.7\times0.6]
 = [0.72,\,-0.08,\,0.42]
$$

这意味着：

- 第 1 个记忆特征保留 90%；
- 第 2 个记忆特征只保留 20%；
- 第 3 个记忆特征保留 70%。

如果 gate 是矩阵乘法，就会让一个 gate 维度混合其他 memory 维度，失去“每个位置分别控制保留量”的清晰含义。LSTM 当然仍然在 gate 产生之前使用矩阵乘法来混合 $x_t$ 和 $h_{t-1}$；但 gate 产生之后，使用逐元素相乘来执行**按维度开关**。

同样，input gate 会逐元素控制 candidate 写入多少。设：

$$
 i_t=[0.1,\,0.8,\,0.5],
 \qquad
 \widetilde{C}_t=[0.6,\,-0.3,\,0.9]
$$

则：

$$
 i_t\odot\widetilde{C}_t
 = [0.1\times0.6,\,0.8\times(-0.3),\,0.5\times0.9]
 = [0.06,\,-0.24,\,0.45]
$$

因此新的 cell state 是：

$$
\begin{aligned}
C_t
&=f_t\odot C_{t-1}+i_t\odot\widetilde{C}_t\\
&=[0.72,\,-0.08,\,0.42]+[0.06,\,-0.24,\,0.45]\\
&=[0.78,\,-0.32,\,0.87]
\end{aligned}
$$

逐维解释如下：

| 维度 | 旧记忆保留 | 新内容写入 | 新 cell state |
|---|---:|---:|---:|
| 1 | $0.9\times0.8=0.72$ | $0.1\times0.6=0.06$ | $0.78$ |
| 2 | $0.2\times(-0.4)=-0.08$ | $0.8\times(-0.3)=-0.24$ | $-0.32$ |
| 3 | $0.7\times0.6=0.42$ | $0.5\times0.9=0.45$ | $0.87$ |

所以，LSTM 使用逐元素相乘，是为了让不同 gate 对不同的记忆特征进行独立控制。它不是把整个 $C_{t-1}$ 一次性全部保留或全部删除，而是可以做到：**这个维度保留多一点，另一个维度忘掉多一点，同时第三个维度写入新的信息**。

这里的“逐元素”也解释了为什么 gate 常被比喻成阀门或开关：

- gate 接近 $0$：对应维度几乎不通过；
- gate 接近 $1$：对应维度大部分通过；
- gate 介于 $0$ 和 $1$：对应维度部分通过。

![[assets/2 CNN and RNN/RNN2/RNN2-40.png]]

#### 1.11.3 Output gate：现在要展示多少记忆？

即使信息被保存在 $C_t$ 中，也不代表当前步骤要把全部信息暴露出去。Output gate 先决定展示比例，再将 cell state 经过 `tanh` 压缩：

$$
o_t=\sigma\left(W_o[h_{t-1},x_t]+b_o\right)
$$

$$
h_t=o_t\odot\tanh(C_t)
$$

因此，$C_t$ 更像“可持续维护的内部记忆”，$h_t$ 更像“这一刻对外报告的内容”。

![[assets/2 CNN and RNN/RNN2/RNN2-41.png]]

#### 1.11.4 用“记事本”理解 LSTM

假设模型正在读一句很长的话：

1. **Forget gate**：旧记事本中的哪些内容已经过时？
2. **Input gate**：当前读到的信息哪些值得写进去？
3. **Cell state**：把旧内容保留一部分，再加入新内容。
4. **Output gate**：这一刻需要把记事本中的哪些内容拿出来回答？

这不是说 LSTM 真的存储了可读的句子，而是一个帮助初学者理解信息流的类比。实际的 $C_t$ 和 $h_t$ 都是由很多数字组成的 vector。

#### 交互：分别调节遗忘、写入和输出

拖动三个 gate，观察旧 cell state 保留量、新候选内容写入量和对外 hidden state 怎样变化。教学例子固定 $C_{t-1}=0.8$、候选内容 $\widetilde{C}_t=0.6$，只展示一个维度；真实 LSTM 对 vector 的每个维度分别计算 gate，且 gate 值由 $x_t$ 和 $h_{t-1}$ 学得。

**HTML 视觉交互**：[[assets/2 CNN and RNN/interactive/rnn-slide-visual-explorer.html#lstm|打开 LSTM 三路门控图]]。拖动 sliders，主画布会同步改变保留、写入、cell state 合并和 hidden 输出。


静态理解仍以公式为准：$C_t=f_t\odot C_{t-1}+i_t\odot\widetilde{C}_t$，而 $h_t=o_t\odot\tanh(C_t)$；输出 gate 改变的是当前暴露的 hidden state，不会直接改写已更新的 cell state。

> **老师讲解重点**：先只调 `forget gate`，看旧记忆保留多少；再只调 `input gate`，看新内容写入多少；最后只调 `output gate`，确认它改变的是对外暴露的 $h_t$，不是直接擦掉 $C_t$。这三个动作对应三条不同的信息流。

### 1.12 GRU：更简化的 gated RNN

**Gated Recurrent Unit（GRU）** 也用 gates 解决 Vanilla RNN 的长期依赖问题，但结构比 LSTM 简单：它通常不单独维护 $C_t$，而是直接更新 hidden state $h_t$。在本讲义采用的记号中：

$$
z_t=\sigma\left(W_z[h_{t-1},x_t]\right),
\qquad
r_t=\sigma\left(W_r[h_{t-1},x_t]\right)
$$

$$
\widetilde{h}_t=\tanh\left(W[r_t\odot h_{t-1},x_t]\right)
$$

$$
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\widetilde{h}_t
$$

直觉上：

- $r_t$（reset gate）控制计算候选状态时要参考多少旧状态；
- $z_t$（update gate）控制最终更偏向旧状态，还是更偏向候选新状态；
- 最后一行是一个可学习的加权混合，因此旧信息可以比较顺畅地继续向前传。

不同教材有时会交换 gate 的命名或使用略有不同的公式。读公式时优先看它在做什么：**一个 gate 控制旧信息，一个 gate 控制新信息，最后混合成新的 hidden state**。

![[assets/2 CNN and RNN/RNN2/RNN2-58.png]]

> **来自 RNN 2 slide 58｜看图重点**：GRU 可以理解成对 LSTM 的简化：合并部分 gate，并且不再单独维护 compressed memory 和 regular hidden memory。

### 1.13 LSTM / GRU 的常见组合方式

讲义最后还展示了几种把 recurrent units 组合成 network 的方法：

- **Multi-layer RNN**：上一层在每个 time step 的输出，作为下一层同一 time step 的输入；层数增加后，底层可学习局部时间模式，上层整合更抽象的模式。
- **Bidirectional RNN / LSTM**：一条 RNN 从左到右读，另一条从右到左读，再把两个方向的表示结合起来。这样每个位置可以同时看到过去和未来的 context，适合 offline sequence labeling；但它不是 causal，不能直接用于必须实时输出的场景。
- **Encoder-decoder / sequence-to-sequence**：encoder 读取输入 sequence，decoder 根据编码后的状态逐步生成输出 sequence，适合 machine translation 等任务。

![[assets/2 CNN and RNN/RNN2-63-multilayer.png]]

> **截图导航**：multi-layer 不等于把同一个 RNN 在时间上重复更多次；它是在每个 time step 叠加网络深度。时间方向仍有 recurrent dependency，层方向则增加 representation depth。

![[assets/2 CNN and RNN/RNN2/RNN2-64.png]]

> **来自 RNN 2 slide 64｜看图重点**：Bidirectional LSTM 同时运行 forward 和 backward 两个方向；两个方向提供互补的 global context。

### 1.14 Vanilla RNN、LSTM 和 GRU 怎么选？

先不要把它们记成三个孤立的名词，可以按“记忆控制能力”来理解：

| 模型 | 记忆机制 | 优点 | 代价或限制 |
|---|---|---|---|
| Vanilla RNN | 一个 hidden state，通常经过 `tanh` 更新 | 结构最简单，适合入门和短 sequence | long-term dependency 容易遇到梯度消失/爆炸 |
| LSTM | hidden state + cell state + 三类 gates | 对保留、写入、输出分别控制，长期依赖更稳定 | 结构和参数更多，计算更重 |
| GRU | 一个 hidden state + 两类 gates | 比 LSTM 简洁，仍能控制信息更新 | 控制机制较少，具体效果取决于任务与数据 |

一个适合零基础的学习顺序是：

1. 先会读 Vanilla RNN 的 $h_t=f(x_t,h_{t-1})$；
2. 再理解 BPTT 为什么会让 gradient 连乘；
3. 然后只记住 LSTM 的三问：忘掉什么、写入什么、输出什么；
4. 最后把 GRU 看作“用更少组件实现类似的信息控制”。

#### 1.14.1 用一句连续解释把三种 recurrent unit 串起来

读 RNN 结构时，先抓住一条完整因果关系：第 $t$ 步的 Vanilla RNN 读取当前输入 $x_t$ 和上一时刻摘要 $h_{t-1}$，生成新的 $h_t$；它可以处理不同长度的 sequence，因为同一个 cell 会重复执行，直到输入结束。LSTM 在这条路径上增加 cell state 和 gates，分别控制旧信息保留、新信息写入和当前输出；GRU 去掉独立的 cell state，用较少的 gates 直接混合旧 hidden state 与候选状态。后面的矩阵形状和 BPTT 公式，都是在展开这条信息流。

## 2. RNN 的工程视角：时间依赖与 state 管理

CNN 与 RNN 都重复使用参数，但复用方向不同：CNN 把同一 filter 用在不同空间位置，通常能并行计算多个位置；RNN 把同一个 cell 用在不同时间步，而 $h_t$ 必须等 $h_{t-1}$ 产生后才能计算。这带来顺序延迟，但也让系统可以用较少的 recurrent weights 处理任意长度的序列。

部署 streaming RNN 时，要明确每条请求的 state 归属、序列开始时如何初始化、何时 reset，以及 padding/mask 如何阻止无效时间步更新 state。训练时 BPTT 还需要中间 state 或等价信息；因此训练内存与逐步推理的内存需求不能混为一谈。

## 3. 教授式完整算例：RNN 如何保存与遗忘信息

用一维教学 RNN 看清 state 的变化，设 $h_t=0.8h_{t-1}+x_t$、$h_0=0$，输入为 $[2,0,0]$：

| 时间步 | 计算 | state |
|---|---|---:|
| 1 | $h_1=0.8\times0+2$ | 2.00 |
| 2 | $h_2=0.8\times2+0$ | 1.60 |
| 3 | $h_3=0.8\times1.6+0$ | 1.28 |

历史输入仍影响后续 state，但每步乘以 $0.8$，影响逐渐衰减。对应的梯度也会连乘：$\partial h_3/\partial h_1=0.8^2=0.64$，20 步后 $0.8^{20}\approx0.0115$。真实网络的因子由 recurrent matrix 与 activation derivative 共同决定；LSTM 的 cell state 和 gates 提供更可控的信息通路。

### 可选互动：改变记忆保留系数

打开 [[assets/2 CNN and RNN/interactive/rnn-slide-visual-explorer.html#unroll|RNN 时间展开和状态路径]]，逐步读取 token，观察同一个 cell 的重复调用和 hidden state 接力。实际章节仍提供 $h_t=0.8h_{t-1}+x_t$ 的静态算例供对照；这个一维模型不代表真实训练曲线。

## 4. 一页串起 RNN

```text
sequence x_1...x_T
  -> 每步结合当前输入 x_t 与旧状态 h_(t-1)
  -> 产生新状态 h_t 和可选输出 y_t
  -> 训练时展开计算图并用 BPTT
  -> LSTM/GRU 通过门控改善长期信息通路
  -> 部署时管理每条序列的 state 与延迟
```

这条链的重点是：RNN 的“记忆”是固定宽度的可学习数字摘要，不是把过去原样保存；LSTM/GRU 改进信息通路，但不会自动保证任何任务都更好。

## 5. 外部资料与进阶阅读

以下资料补充框架实现与工程边界；访问日期：2026-10-19。

- **External context：RNN API 与变长输入**：PyTorch [RNN modules](https://pytorch.org/docs/stable/nn.html#recurrent-layers) 和 [PackedSequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html)。
- **External context：字符级 RNN 教程**：PyTorch [Classifying Names with a Character-Level RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)，把 recurrent cell、sequence 和训练循环对应到一个小任务。
- **进阶路线**：state reset、streaming inference、truncated BPTT、state buffering 与单 token latency；这些性能结果需结合真实 workload 和目标硬件测量。

## 来源与截图

- `RNN 1.pdf`：Lecture 5，sequence modeling、Vanilla RNN、BPTT 与 language modeling；本篇引用课件页码 57–112。
- `RNN 2.pdf`：Lecture 6，gradient instability、gradient clipping、LSTM、GRU 与 recurrent architectures，共 65 页。
- 截图继续使用 `assets/2 CNN and RNN/` 中以 `RNN1-`、`RNN2-` 开头的原文件，保留来源页码，未移动或覆盖素材。

### 本轮新增视觉覆盖

| 案例 | 课件来源 | 笔记位置 | 视觉材料 | 检查 |
|---|---|---|---|---|
| Recurrent cell 展开与参数共享 | RNN 1 slide 63 | §1.3 | `RNN-shared-weights-unrolled.svg` + 原课件截图 | 静态图明确一份参数重复使用；原截图继续提供课件出处 |
| BPTT 长链的梯度连乘 | RNN 1 slide 91；梯度不稳定见 RNN 2 slides 27–28 | §1.6、§1.9 | 原课件截图 + `interactive/rnn-slide-visual-explorer.html#bptt` | 理想化 scalar 演示已标注边界；公式和截图是静态回退 |
| LSTM 三类 gate 与 state 更新 | RNN 2 slides 34、38、40、41 | §1.11 | `LSTM-cell-state-flow.svg` + 原课件截图 + `interactive/rnn-slide-visual-explorer.html#lstm` | 图中 cell-state 路径与公式一致；互动限定为单维教学数值 |

