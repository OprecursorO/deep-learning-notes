# Basic Neural Networks

> [!tip] 互动入口
> [[互动实验入口]] · [[assets/1 Basic neural networks/interactive/gradient-descent-explorer.html|打开梯度下降互动]] · [[assets/1 Basic neural networks/interactive/perceptron-learning-explorer.html|打开 Perceptron 学习互动]] · [[assets/1 Basic neural networks/interactive/model-dataflow-comparison.html|打开模型数据流对照]]

> Course notes（中英结合，零基础版）
>
> 本笔记把两组 `Basic neural networks` slides 合并成一条学习主线：先补 `Math Review` 和 `Machine Learning Review`，再从一个 neuron 逐步走到 `Softmax`、`Loss`、`Regularization`、`Gradient Descent`、`Hidden Layers`、`Forward Pass` 和 `Backpropagation`。专业名词保留 English，方便后续阅读教材和代码。

> [!info] 与 Lecture 1 slides 的对应关系
> 原始课件是 Vault 中的 `lecture_note_01`（CS 280 Deep Learning, Lecture 1, 97 页）。本篇前半主线对应课件 p.15–26 的深度学习导论、p.48–70 的数学与机器学习复习、p.73–96 的人工神经元和 Perceptron；后面的 Softmax、反向传播及工程实现是既有笔记的后续教学扩展，不应误认为都在 Lecture 1 中讲完。原页截图放在 `assets/1 Basic neural networks/source/`，每张截图紧邻其解释。

### Lecture 1 的学习地图

| 课件段落 | 学习问题 | 笔记位置 |
|---|---|---|
| p.15–26：Introduction / data-driven approach | 如何从 examples 学到 input-to-output mapping？ | 1. Big Picture |
| p.48–52：Math review | gradient、minimum、probability、statistics 分别支持什么？ | 2. Math Review |
| p.53–70：ML review | 任务、监督信号、数据拆分、优化与泛化如何连起来？ | 3. Machine Learning Review |
| p.73–82：Artificial neuron | weighted sum、activation 和 hard decision 如何形成 classifier？ | 4. Artificial Neuron and Perceptron |
| p.83–96：Perceptron | 一次错误怎样改变参数；有限步收敛需要什么条件？ | 4.4 Perceptron Algorithm |
| p.97：Summary / next time | Lecture 1 怎样过渡到多层网络？ | 5. From Neurons to a Single-layer Network |

本篇的视觉层分成三种：**课件原页截图**保留教师的图示、具体案例与出处；**Markdown 解释和公式**承担完整讲解；**本地 HTML 互动**展示参数改变的因果过程。互动页面打开失败时，旁边的公式、静态图和表格仍然完整。

## 0. 面向 AI chip design 小白的学习目标

这两讲解决一个最基础但最重要的问题：**如何把输入数字变成可用的 prediction，以及如何让硬件和软件共同把这个过程高效地重复执行？** 你不需要先会芯片设计；先掌握下面这条因果链即可：

```text
data -> model computation -> prediction -> loss -> gradient -> parameter update
```

学完本篇，你应该能够：

1. 看懂一个 neuron、layer、MLP 的 input/output shape 和 weighted sum；
2. 解释为什么需要 activation、Softmax、loss、regularization 和 hidden layer；
3. 手算一个最小的 forward pass、loss、gradient 和 parameter update；
4. 区分 training、inference、parameter、hyperparameter、gradient 和 optimizer；
5. 从 AI chip 角度指出主要计算是 matrix multiplication，主要代价来自 data movement、memory capacity、bandwidth、latency 和 energy。

### 0.1 分层术语 roadmap

先看路线，不要求现在背完整词典。每一层只在对应章节出现时学习：

```text
Level 0  输入与答案（第 1 节）
  input -> feature -> label/target -> prediction
      |
Level 1  模型如何计算（第 2–4 节）
  function -> vector/matrix -> neuron -> weight/bias -> activation
      |
Level 2  模型如何评价自己（第 5–7 节）
  logit -> Softmax -> loss/objective -> regularization
      |
Level 3  模型如何学习（第 8–11 节）
  derivative -> gradient -> forward pass -> backpropagation
  -> Gradient Descent -> optimizer
      |
Level 4  从训练到部署（后续章节）
  parameter/hyperparameter -> training/inference
  -> representation -> generalization -> data movement
```

| 学到哪一层 | 本阶段只需回答的问题 |
|---|---|
| Level 0 | 输入是什么，正确答案是什么，prediction 如何与它比较？ |
| Level 1 | 每个 neuron 对输入做了什么加权计算？ |
| Level 2 | loss 如何把“错多少”变成可优化的数字？ |
| Level 3 | gradient 如何告诉每个 parameter 往哪边改？ |
| Level 4 | 训练好的 model 如何被硬件高效执行？ |

### 0.2 六篇笔记的连续学习路线

这四篇笔记不是四个互相独立的词典，而是一条逐层增加对象和依赖的主线：

```text
1 Basic neural networks
  数字输入 -> neuron -> loss -> gradient -> parameter update
        ↓
2 CNN
  image 的空间局部结构、参数共享与 dense prediction
        ↓
3 RNN
  sequence -> hidden state -> BPTT -> LSTM/GRU
        ↓
4 Sequence Models and Attention
  sequence 生成 -> encoder-decoder -> teacher forcing -> 动态读取 context
        ↓
5 Attention and Transformer
  Q/K/V -> self-attention -> Transformer block -> 大规模并行与 KV cache
        ↓
6 Transformer variants and applications
  长序列效率、BERT/GPT、ViT 与多模态应用
```

每一篇都建立在前一篇的 `tensor shape`、`forward pass`、`loss` 和 `backpropagation` 之上。AI chip design 的统一问题也逐步变清楚：模型到底做了哪些 arithmetic，哪些数据需要被复用，哪里出现 sequential dependency，哪里出现 memory capacity 或 bandwidth 压力。

**按顺序打开笔记**：[[1 Basic neural networks]] → [[2 CNN]] → [[3 RNN]] → [[4 Sequence Models and Attention]] → [[5 Attention and Transformer]] → [[6 Transformer variants and applications]]。这些笔记建立读懂模型计算并初步分析芯片代价的基础；要继续走向 AI 芯片工程，还需系统学习数字逻辑、计算机体系结构、存储层次、并行计算、编译器与 accelerator dataflow，不能把读完笔记等同于达到专家水平。

| 模型家族 | 输入怎样走 | 能并行的部分 | 最容易忽略的代价 |
|---|---|---|---|
| CNN | 局部空间窗口经共享 filters 变成 feature maps | 同一层多个空间位置 | feature maps 的读写、边缘与下采样造成的信息损失 |
| RNN | 当前输入与上一步 state 反复更新 | batch 与单步内部的矩阵运算 | 时间步必须依次完成，state 需要保存 |
| Transformer | token 投影为 Q/K/V，再按相关性读取 | 同层多个 token 的大部分计算 | 全局 attention 的 token-pair 工作量、KV cache 的容量和读取 |

这张表是静态回退：即使交互图没能打开，仍可用它比较三类模型的“数据流—并行性—成本”。下方对照图可切换模型，重点观察瓶颈如何从空间数据复用，转成时间依赖，再转成全局信息读取。

打开本地互动：[[assets/1 Basic neural networks/interactive/model-dataflow-comparison.html|切换 CNN、RNN、Transformer 的数据流对照图]]。它是教学示意，不是 benchmark；若 HTML/JS 不可用，直接使用上面的静态表和以下老师讲解。

**老师带你读图**：先选 CNN，看左上角 `2×2` patch 如何与同一个 filter 逐项相乘并相加，得到右侧 feature map 的一个格子：`1×1 + 2×0 + 5×0 + 6×1 = 7`。再选 RNN，沿箭头从 `x₁` 读到 `h₁`，再从 `h₁` 走到下一步；共享参数没有消除“必须先算出前一步状态”的依赖。最后选 Transformer，权重矩阵一行对应一个 query；第一行对三个 value 的权重为 `[0.70, 0.20, 0.10]`，因此 `0.70[1,0] + 0.20[0,2] + 0.10[2,1] = [0.90,0.50]`。这些 value 和权重只是为了展示“按比例读取再相加”的固定教学例子，不是实际模型预测。三种结构最大的区别不是“有没有乘法”，而是乘法围绕什么组织：CNN 围绕空间邻域，RNN 围绕时间状态，Transformer 围绕 token 间关系。

### 0.1.1 Reference glossary（查阅用，不是先修要求）

| Term | 对初学者的含义 | 在本篇中要观察什么 |
|---|---|---|
| `dataset / sample` | dataset 是训练数据集合；sample 是其中一条输入及其 label | 一次 update 处理多少 sample？ |
| `feature` | 能帮助区分任务的输入属性或中间表示 | 哪些 feature 是人工给的，哪些由 hidden layer 学到？ |
| `model` | 把 input 映射到 output 的可计算函数 | 输入和输出的 shape 是否匹配？ |
| `parameter` | 训练中由数据学到的 weight 和 bias | backpropagation 如何改变它？ |
| `hyperparameter` | 训练前由人设定的控制量，如 learning rate、batch size、depth | 改变它会怎样影响速度、稳定性和泛化？ |
| `logit / score` | output activation 之前的原始数值 | 它还不是 probability，是否需要 Softmax/Sigmoid？ |
| `loss / objective` | 衡量预测错误的数值；objective 是训练要最小化的整体目标 | 它是否可微、是否与任务目标一致？ |
| `optimizer` | 根据 gradient 计算 parameter update 的方法，如 SGD、Adam | update 的计算和额外 state 放在哪里？ |
| `inference` | 使用已训练 model 做 prediction 的阶段 | 是否只读 weights？延迟和能耗是多少？ |
| `backpropagation` | 沿 computation graph 反向计算 gradients 的算法 | 需要保存哪些 intermediate activations？ |
| `representation` | network 在中间层形成的、便于后续任务使用的表示 | 表示是否更容易分离类别或表达空间结构？ |

### 0.2 总体动机与工程视角

对初学者来说，神经网络首先是“许多小计算的组合”。如果只看最终 accuracy，你会看不到模型为什么能学、为什么会失败。对 AI chip design 来说，同一网络还必须被拆成可调度的 arithmetic 和 memory operations。

**Engineering perspective（通用工程视角）**：

- **算法/模型**优化 representation、loss 和 accuracy，担心 underfitting、overfitting 与数值不稳定；
- **硬件/实现**优化 matrix multiplication 的吞吐、片上 SRAM 的复用、外部 memory traffic、latency 和 energy，担心 bandwidth 不够、buffer 放不下和数据搬运压过计算；
- **软件/compiler**需要把 layer、batch 和 tensor layout 映射到 kernel 与 accelerator，担心 shape、算子融合和调度无法匹配硬件。

后文每次引入新机制时，都先说明它解决的局部问题，再说明它给硬件带来的计算或存储含义。

## 老师开场：神经网络不是魔法，而是一条可检查的计算链

我先把神经网络讲成一个你可以拿笔算的机器。机器接收输入数字，乘上一组可学习的权重，加上偏置，得到一个分数；多个分数组合起来，形成预测；预测和正确答案比较产生 `loss`；最后用 `gradient` 告诉每个权重应该增大还是减小。

最小例子只有一个输入：房间面积 $x=2$，模型想预测价格。设当前参数 $w=3,b=1$，那么：

$$
z=wx+b=3\times2+1=7.
$$

如果真实价格是 $y=10$，模型还差 3。用最简单的平方误差 $L=(z-y)^2$，此时 $L=9$。这一步只是在**评价**模型，还没有学习。学习发生在我们问：“如果把 $w$ 增大一点，loss 会变大还是变小？”这个变化率就是 `derivative`；对很多参数同时计算时，所有 derivative 组成 `gradient`。

### 一次 forward、一次 backward、一次 update

把上面的过程写成训练循环，就是：

```text
input x -> weighted sum z -> prediction ŷ -> loss L
                                      ↓
                         gradient ∂L/∂w, ∂L/∂b
                                      ↓
                         w,b <- w,b - learning_rate × gradient
```

这里最容易混淆的三件事要分开：

- `parameter` 是模型自己通过数据学到的数字，例如 $w,b$；
- `hyperparameter` 是我们在训练前设定的控制量，例如 learning rate、batch size 和 layer 数；
- `prediction` 是当前参数算出来的结果，不等于 label，也不等于 loss。

当你读后面的 `Softmax`、`cross-entropy` 和 `backpropagation` 时，始终回到这条链：**模型算什么、哪里变成错误、错误怎样传回参数**。不要把 activation、loss 和 optimizer 当成“神经网络的同义词”；它们在计算链中承担不同角色。

### 我会怎样把它翻译给芯片工程师

神经网络的很多 layer 最终都会落到矩阵乘加（multiply-accumulate, MAC）。算法教师关注 prediction 是否准确；硬件工程师还要关心：权重和 activation 是否能在片上复用、片外 memory traffic 是否过大、累加是否有足够的数值精度、不同 layer 的 shape 是否让计算阵列保持忙碌。

所以本章的目标不是让你背出一个 neuron 的定义，而是让你看到一条可以逐项验证的链：**数字如何经过算子变成结果，结果如何产生误差，误差如何改变参数，参数如何被硬件重复使用**。

## 1. Big Picture

课件的第一张问题地图是 **data-driven approach**：我们不再为每种输入手写所有规则，而是用带答案的 examples 估计一个 mapping function。对 AI chip 设计者，这意味着硬件执行的不是一条固定规则，而是同一组可编程参数和算子在大量 samples 上重复运行。

![[assets/1 Basic neural networks/nn1-data-driven-mapping.png]]

> **截图导航**：先看 input data 如何经过 mapping function 产生 expected output，再把 mapping function 理解成后文的 neural network；parameters 是使这个映射变好的可学习数值。

**Lecture 1 source case — p.21：red panda image classification**。课件把不同姿态和背景的小熊猫图片放在 mapping function 左侧，目标是输出类别名称。请检查截图中输入图片的变化：同一个类别不对应唯一像素图样，所以模型要从多种例子中学习可复用的映射，而不是记住某一张图片。

![[assets/1 Basic neural networks/source/lecture1-p21.png]]

`Deep Learning` 的核心任务是从 data 中学习一个 **mapping function**，把 input 转成 prediction：

```text
input x -> model / neural network -> prediction y-hat
                  ^
           learned parameters theta
```

- `x` (**input**)：输入，例如图片像素、句子中的词或房屋特征。
- `y` (**label / target**)：正确答案，也叫 ground truth。
- $\hat{y}$ (**prediction**)：model 的预测结果。
- $\theta$ (**parameters**)：model 从 data 中学到的 weights 和 biases。
- `model`：可计算 output 的 function，例如 $\hat{y}=f(\mathbf{x};\theta)$。

Traditional programming 是 `data + 人工写好的 rules -> result`；**Machine Learning** 是 `data + examples/answers -> 学出 rules`。Neural network 不是大脑的完整复制，而是一组可以组合的 mathematical units。

### 1.1 给完全零基础读者的学习路线

你可以把整门课先看成一个反复循环的问题：

1. **Input**：给模型一组数字，例如一张图片的 pixels。
2. **Model**：用 `weight` 和 `bias` 对数字做计算。
3. **Prediction**：模型给出答案 $\hat{y}$。
4. **Loss**：比较预测和正确答案 $y$，得到“错了多少”。
5. **Learning**：根据 loss 调整 parameters，让下一次预测更好。

这里不要求你先会 programming。只要先掌握“数字进入、数字经过计算、数字出来”的过程，再逐步理解 vector、matrix 和 derivative 即可。

## 2. Math Review

### 2.1 Function

Function 是一条确定的 input-output rule：

$$
f(x)=2x+1
$$

当 `x=3` 时，`f(3)=7`。Neural network 也是 function，只是把很多带 parameters 的 functions 组合起来。

### 2.2 Derivative and Gradient

**Motivation**：如果模型预测错了，我们需要知道“每个 parameter 应该向哪边改、改多少”。只知道总 loss 不足以更新大量 weights；derivative 和 gradient 把整体错误分解成每个 parameter 的局部责任。

**Derivative** 描述 input 稍微变化时 output 变化多快。例如：

$$
f(x)=x^2,\qquad f'(x)=2x
$$

在 `x=3` 附近，斜率是 `6`。

当 function 有多个 variables 时，用 **Gradient** 表示所有 partial derivatives：

$$
\nabla f(\mathbf{w})=
\begin{bmatrix}
\frac{\partial f}{\partial w_1}\\
\vdots\\
\frac{\partial f}{\partial w_d}
\end{bmatrix}
$$

Gradient 指向 function 增长最快的方向；沿 `-gradient` 方向走，通常会让 function 下降。

#### 为什么要用 Partial Derivative？

Neural network 通常有很多可学习的 **parameters**，例如 $w_1,w_2,\ldots,b$。这些 parameters 同时决定最终的 **loss**，所以我们需要分别回答：

> 只改变某一个 parameter，其他 parameters 暂时固定时，loss 会怎样变化？

这个问题由 **partial derivative** 回答：

$$
\frac{\partial L}{\partial w_j}
$$

它表示：在当前点附近，$w_j$ 增加一个很小的量，而其他 parameters 保持不变时，loss 的局部变化率。

- $\frac{\partial L}{\partial w_j}>0$：增大 $w_j$ 会使 loss 增大，因此 `Gradient Descent` 倾向于减小 $w_j$。
- $\frac{\partial L}{\partial w_j}<0$：增大 $w_j$ 会使 loss 减小，因此更新时会增大 $w_j$。
- $\frac{\partial L}{\partial w_j}\approx 0$：loss 对这个 parameter 在当前位置不敏感，单次更新通常很小。

如果只用一个总的 derivative，就无法区分每个 weight 和 bias 各自应该怎样调整。把所有 partial derivatives 按顺序组合起来，就得到 **gradient**：

$$
\nabla_{\theta}L(\theta)=
\begin{bmatrix}
\frac{\partial L}{\partial \theta_1}\\
\frac{\partial L}{\partial \theta_2}\\
\vdots\\
\frac{\partial L}{\partial \theta_m}
\end{bmatrix}
$$

其中 $\theta$ 表示全部 parameters。每一个分量告诉我们一个 parameter 的调整方向和敏感程度；gradient 合在一起则给出 loss 增长最快的方向。因此参数更新为：

$$
\theta\leftarrow\theta-\eta\nabla_{\theta}L(\theta)
$$

这里每一步的作用是：

1. 计算当前 prediction 和 target 之间的 **loss**，衡量模型错得多不多。
2. 对每个 parameter 求 partial derivative，判断各 parameter 对 loss 的影响方向和大小。
3. 将这些 partial derivatives 组成 gradient，得到完整的调整方向。
4. 乘以 **learning rate** $\eta$，控制每次移动的步长。
5. 减去这个调整量，沿着让 loss 下降的方向更新 parameters。

#### 数值例子：Partial Derivative 如何改变 Weight 和 Bias

考虑一个最简单的 linear neuron：

$$
\hat y=wx+b,\qquad L=\frac{1}{2}(\hat y-y)^2
$$

给定 $x=2$、target $y=5$、初始 $w=1$、$b=0$，learning rate 取 $\eta=0.1$。

1. **Forward pass**：

   $$
   \hat y=1\times2+0=2
   $$

2. **计算 loss**：prediction 比 target 小 $3$，因此

   $$
   L=\frac{1}{2}(2-5)^2=4.5
   $$

3. **分别求 partial derivatives**：由 chain rule，

   $$
   \frac{\partial L}{\partial w}=(\hat y-y)x=(-3)\times2=-6
   $$

   $$
   \frac{\partial L}{\partial b}=\hat y-y=-3
   $$

   $\frac{\partial L}{\partial w}=-6$ 表示在当前位置增大 $w$ 会降低 loss，而且 $w$ 的影响是 $b$ 的两倍，因为 input $x=2$。$\frac{\partial L}{\partial b}=-3$ 表示增大 $b$ 也会降低 loss。

4. **Gradient Descent update**：

   $$
   w_{\text{new}}=1-0.1(-6)=1.6
   $$

   $$
   b_{\text{new}}=0-0.1(-3)=0.3
   $$

5. **检查更新效果**：

   $$
   \hat y_{\text{new}}=1.6\times2+0.3=3.5
   $$

   新 prediction 从 $2$ 靠近 target $5$，新 loss 为

   $$
   L_{\text{new}}=\frac{1}{2}(3.5-5)^2=1.125
   $$

   loss 从 $4.5$ 降到 $1.125$。这就是 partial derivatives 的实际用途：它们把“模型错了多少”转换为“每个 weight 和 bias 应该怎样改”。在多层 neural network 中，**Backpropagation** 使用 `chain rule` 高效计算所有 parameters 的 partial derivatives，然后由 `Gradient Descent` 完成更新。

### 2.3 Minimum and Optimization

- **Local minimum**：附近没有更低的点。
- **Global minimum**：整个搜索空间中的最低点。
- 可微 function 在内部 minimum 通常满足 `gradient = 0`。
- **Hessian matrix** 是由二阶 derivatives 组成的 matrix，用来描述 loss surface 的弯曲方向；**positive definite** 表示任意非零方向上的二阶变化都为正，因此局部形状像碗，可以确认该点是 local minimum。

Neural network training 通常不会手算 minimum，而是使用 `Gradient Descent` 逐步逼近。

### 2.4 Vector, Matrix, and Dot Product

一个数字是 **scalar**；一列有序数字是 **vector**：

$$
\mathbf{x}=\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}
$$

**Dot product**：

$$
\mathbf{w}^{T}\mathbf{x}=w_1x_1+w_2x_2+w_3x_3
$$

例如 $\mathbf{w}=(4,5)$、$\mathbf{x}=(2,3)$，则 $\mathbf{w}^{T}\mathbf{x}=23$。Matrix 可以同时保存多个 neurons 的 weights，并进行 batch computation；看到 $W\mathbf{x}$ 时，可以先理解为“多个 weighted sums 同时进行”。

### 2.5 Probability and Statistics

- **Probability**：对不确定性的量化。
- **Conditional probability**：在已知条件下的 probability。
- **Distribution**：描述 random variable 可能取值及其 probability。
- **Expectation**：长期平均表现；可用 samples 的 mean 估计。
- **Maximum likelihood**：选择能让 observed data probability 最大的 parameters。
- **IID (independent and identically distributed)**：samples 相互独立，并来自同一 distribution，是许多 supervised learning 推导的基本假设。

常见 factorization：

$$
p(x,y)=p(y\mid x)p(x)
$$

## 3. Machine Learning Review

### 3.1 Common Tasks

- **Classification**：预测 categories，例如 cat / dog。
- **Regression**：预测 continuous value，例如房价。
- **Ranking**：按 relevance 对 items 排序。
- **Clustering**：无 labels 时，把相似 samples 分组。
- **Dimensionality reduction**：用较少 dimensions 保留重要 structure。

本课程 neural network 的主要例子是 classification。

### 3.2 Learning Settings

- **Supervised learning**：training data 有 `(x,y)` labels。
- **Unsupervised learning**：没有 labels，寻找 data structure。
- **Semi-supervised learning**：少量 labeled data 加大量 unlabeled data。
- **Reinforcement learning**：通过 reward 学习 action policy。

### 3.3 Training, Validation, and Test

1. **Training set**：调整 parameters。
2. **Validation set**：选择 architecture 和 **hyperparameters**。Hyperparameters 是 training 前由人设定的控制量，例如 learning rate、batch size 和 network depth；它们不是通过当前 sample 的 gradient 直接学出来的 parameters。
3. **Test set**：最后评估未见 data，不能反复用来调参。

在未见 samples 上的表现叫 **generalization**。

- **Underfitting**：model 太简单或 training 不足，连 training set 都做不好。
- **Overfitting**：model 记住 training samples，training error 很低但新 data 表现差。
- **Capacity**：model 表达复杂 patterns 的能力。

课件 p.68 的 supervised learning pipeline 将顺序写得很清楚：先收集数据和特征，再选择 hypothesis class 与 loss，最后通过 optimization 最小化 empirical loss。训练集用于参数拟合，validation 用于选择超参数，test 只在最后估计泛化表现。注意图中三种数据集承担不同角色，不能因为 test 数字不理想就反复调参再把它当独立评估。

![[assets/1 Basic neural networks/source/lecture1-p68.png]]

### 3.4 Dataset 为什么是 training 的基础？

**先记住一句话：模型不会从数据里“凭空发现真理”，它只能从 dataset 提供的输入、label 和分布中学习。** 训练循环再精确，如果 dataset 有系统性问题，模型也会稳定地学到错误规律。

可以把 dataset 看成 training 的“测量仪器”：它决定模型看到了哪些现象、哪些现象被标成了什么答案，以及哪些情况被完全遗漏。

| Dataset 属性 | 它影响什么 | 典型失败 |
|---|---|---|
| **数量（quantity）** | 参数能看到多少不同样本，gradient 是否只被少数例子支配 | 数据太少，模型容易记住训练样本，出现 overfitting |
| **质量（quality）** | 输入是否可靠、label 是否正确 | 错误或含糊 label 会把错误梯度反复传给模型 |
| **覆盖范围（coverage）** | 训练分布是否覆盖部署时的光照、噪声、设备、语言、边界情况 | 训练集准确率很高，但遇到 distribution shift 就失败 |
| **代表性（representativeness）** | dataset 中各类别、各场景的比例是否接近实际任务 | 类别不平衡时，模型可能只学会预测多数类 |
| **独立性与去重（independence/deduplication）** | train 与 validation/test 是否真正分离 | 数据泄漏让 test accuracy 虚高，部署表现却下降 |
| **标注定义（label policy）** | 不同标注者是否按同一规则给答案 | 同一个输入有互相矛盾的 labels，loss 的下限被抬高 |

这里的 **data distribution** 指输入与答案共同出现的规律。训练集上的 average loss 是 `empirical risk`；部署时真正关心的是未来样本上的 `expected risk`。两者之间的差距就是 dataset 与现实不匹配时暴露出的 generalization gap。

**最小例子：**假设要训练一个猫狗分类器，但训练集里的猫总在室内、狗总在室外。模型可能学到“背景是草地 → 狗”，而不是动物本身的视觉特征。即使 training accuracy 达到 100%，把室内的狗或草地上的猫交给它时仍会失败。问题不一定在 network architecture，而在 dataset 没有把“动物类别”和“背景”解耦。

因此，training 前至少要检查：

1. 输入和 label 的对应关系是否正确；
2. train/validation/test 是否按 sample 或 subject 正确切分，避免同一对象的近重复样本跨集合泄漏；
3. 每个类别和重要场景是否有足够覆盖；
4. preprocessing、augmentation 和部署输入是否使用同一套假设；
5. validation 指标是否按真实使用风险选择，而不是只看 overall accuracy。

**AI chip perspective**：dataset 不直接改变一次 MAC 的面积，但它决定训练要处理多少 samples、多少 epochs、多少 augmentation，以及模型是否需要更大容量来记住复杂分布。数据质量差导致的反复训练，会同时增加 accelerator 的计算、DRAM traffic 和 energy；因此 dataset 质量也是系统级效率问题。

**External course-material extension — Stanford CS231n, Generalization and dataset splits**：课程讲义强调 validation 用于模型选择、test 用于最终估计。实际执行时还要防止 preprocessing leakage：先切分数据，再只在 training split 上拟合 scaler、feature selector 或 vocabulary。时间相关数据按时间切分；同一患者、设备或视频片段的近重复样本应按 group 切分，避免同一来源同时进入 train 和 test。来源：[CS231n: Neural Networks Part 1, sections on train/val/test and model selection](https://cs231n.github.io/neural-networks-1/)，访问日期 2026-10-22。

**Lecture 1 source case — p.70：training 与 validation 曲线分离**。随 training time 或 capacity 增加，training error 持续下降，而 validation error 先下降后回升；回升区说明拟合训练样本并不等于对新样本更好。读图时看两条曲线之间的差距和 validation 最低点，它们比单独看 training accuracy 更能暴露 overfitting。

![[assets/1 Basic neural networks/source/lecture1-p70.png]]

### 3.5 Dataset、任务定义与 evaluation 的连接

一个 dataset 不只是一个文件夹，而是一个完整的 **measurement contract**：它规定输入如何采集、答案如何定义、哪些样本属于 training/validation/test，以及最终用什么指标判断“做得好”。这四件事必须一起看。

例如二分类模型先输出 logit $z$，再通过 sigmoid 得到 positive probability：

$$
p=\sigma(z)=\frac{1}{1+e^{-z}}
$$

如果系统用阈值 $0.5$ 决定类别，得到的是 `prediction`；如果阈值改成 $0.9$，同一个模型会产生不同的 precision/recall。这说明 **probability、threshold、metric 和 deployment policy 是不同层次的对象**：training loss 负责提供梯度，阈值和指标负责描述实际决策质量。

当 positive class 很少时，只看 accuracy 可能产生错觉。假设 1000 个样本中只有 10 个 positive，模型把所有样本都预测为 negative，accuracy 仍是 99%，但它没有找到任何 positive。此时应同时查看 confusion matrix 中的 TP、FP、FN、TN，并根据任务代价考虑 precision、recall 或 F1，而不是把单一 accuracy 当成完整结论。

从 AI chip 角度，dataset contract 也会改变硬件工作量：输入分辨率、feature dimension、batch size 和 padding policy 决定 tensor shape；label policy 和 metric 决定 output head、loss reduction 以及 validation pipeline。硬件增加 MAC 数量不能修复 label 错误、类别缺失或 train/test leakage。

## 4. Artificial Neuron and Perceptron

**Motivation**：最简单的分类器需要把一组输入数字压缩成一个可比较的 score。weighted sum 是最小的可学习计算单元；后面的 activation 决定这个 score 如何继续传播。

### 4.1 Weighted Sum

给定 input $x_1,\ldots,x_d$、weights $w_1,\ldots,w_d$ 和 bias $b$：

$$
z=\sum_{i=1}^{d}w_i x_i+b=\mathbf{w}^{T}\mathbf{x}+b
$$

- **Weight**：某个 input 的重要性和方向；正值推动 output 增大，负值推动 output 减小。
- **Bias**：整体平移项；即使 input 为 0，也允许 neuron 有非零响应。
- `z`：activation 前的 **pre-activation** 或 score。

### 4.2 Activation Function

Neuron output：

$$
a=\sigma(z)
$$

常见 functions：

- **Step function**：超过 threshold 才 output 1，适合解释 Perceptron。
- **Sigmoid**：
  $$
  \sigma(z)=\frac{1}{1+e^{-z}}
  $$
  output 在 `(0,1)`，可以解释为 binary classification probability；但极端区域 gradient 很小。
- **ReLU**：
  $$
  \operatorname{ReLU}(z)=\max(0,z)
  $$
  正数原样保留，负数变成 0，是 hidden layer 常用选择。

例：若 `x=(2,3)`、`w=(0.5,-1)`、`b=2`：

$$
z=0.5\times2-1\times3+2=0
$$

因此 `ReLU(z)=0`，`Sigmoid(z)=0.5`。

为了直观看出不同 **activation function** 如何改变 neuron output，可以比较下面三条曲线：`Step` 做硬阈值判断，`Sigmoid` 输出平滑的 probability-like value，`ReLU` 保留正值并把负值截为 0。

这是三种 function 的连续形状：

<svg viewBox="0 0 1200 430" role="img" aria-label="Step, Sigmoid, and ReLU activation functions" xmlns="http://www.w3.org/2000/svg">
<rect width="1200" height="430" fill="#ffffff"/>
<text x="600" y="35" text-anchor="middle" font-family="Arial" font-size="25" font-weight="700" fill="#0f172a">Activation Functions</text>
<g font-family="Arial" font-size="21" font-weight="700"><text x="200" y="75" text-anchor="middle" fill="#b91c1c">Step</text><text x="600" y="75" text-anchor="middle" fill="#1d4ed8">Sigmoid</text><text x="1000" y="75" text-anchor="middle" fill="#15803d">ReLU</text></g>
<g stroke="#cbd5e1"><path d="M60 155h280M60 275h280M460 155h280M460 275h280M860 155h280M860 275h280"/></g>
<g stroke="#334155" stroke-width="2"><path d="M60 335h280M200 110v240M460 335h280M600 110v240M860 335h280M1000 110v240"/></g>
<path d="M60 335H200V155H340" fill="none" stroke="#dc2626" stroke-width="4"/>
<path d="M460 300C510 298 545 280 570 250C588 228 600 215 615 200C640 170 680 152 740 150" fill="none" stroke="#2563eb" stroke-width="4"/>
<path d="M860 335H1000L1140 115" fill="none" stroke="#16a34a" stroke-width="4"/>
<g font-family="Arial" font-size="14" fill="#475569"><text x="65" y="355">-3</text><text x="195" y="355">0</text><text x="330" y="355">3</text><text x="465" y="355">-3</text><text x="595" y="355">0</text><text x="730" y="355">3</text><text x="865" y="355">-3</text><text x="995" y="355">0</text><text x="1130" y="355">3</text><text x="205" y="160">1</text><text x="605" y="160">1</text><text x="1005" y="120">6</text><text x="200" y="395" text-anchor="middle">a=0 (z&lt;0), a=1 (z&gt;=0)</text><text x="600" y="395" text-anchor="middle">a=1/(1+e^-z)</text><text x="1000" y="395" text-anchor="middle">a=max(0,z)</text></g>
</svg>

**Slide reference — activation functions**

![[assets/1 Basic neural networks/nn1-activation-functions.png]]

这张原始 slide 的重点是：activation function 决定 neuron 如何把 `pre-activation` $z$ 变成 output $a$。`Step` 适合说明 Perceptron 的硬阈值；`Sigmoid` 把数值压到 $(0,1)$；`ReLU` 让正值继续传播，同时把负值变成 0。


读图时横轴是 pre-activation $z$，纵轴是 neuron output。曲线的形状决定了 information 如何继续向下一层传播，也决定 gradient 在哪些区域可能变小。

**Lecture 1 source case — p.75：从多个输入到一个 neuron output**。原图将各 input feature 经 weight 汇入求和节点，再加 bias 并通过 non-linearity 得到 output。请沿着箭头读 `features → weighted sum + bias → activation → output`；它说明人工 neuron 的结构不是单纯的“阈值开关”，而是可组合的数值变换。

![[assets/1 Basic neural networks/source/lecture1-p75.png]]

### 4.3 Single Neuron as Linear Classifier

Binary classification 可以使用：

$$
\hat y=\operatorname{sign}(\mathbf{w}^{T}\mathbf{x}+b)
$$

$\mathbf{w}^{T}\mathbf{x}+b=0$ 是 **decision boundary**：一维是切分点，二维是一条直线，更高维是 **hyperplane**。如果一条 hyperplane 能完全分开两类，data 就是 **linearly separable**。

Dot product 也可以理解为一种 **pattern matching**：input direction 越接近 weight direction，score 越大，neuron 越容易 fire。Linear classifier 对 **outlier** 较敏感，一个异常 sample 可能明显改变 boundary。

#### 课件中的一维例子：为什么要先产生 score，再做分类

课件用一维 input 比较了两个 predictor：一个直接输出连续的 linear score，另一个把 score 经过 `Step` function 变成 0/1 分类结果。

![[assets/1 Basic neural networks/nn1-1d-predictors.png]]

把横轴理解成 neuron 的 score $s=\mathbf{w}^{T}\mathbf{x}$：

- 紫色直线 $y=\mathbf{w}^{T}\mathbf{x}$ 保留了“离边界有多远”的连续信息；
- 红色阶梯 $y=\operatorname{step}(\mathbf{w}^{T}\mathbf{x})$ 只回答“属于哪一边”，输出通常是 0 或 1；
- 因此，`Step` 很适合表达最终的 hard decision，但在训练时不方便使用 derivative；这也是后面引入 `Sigmoid`、`Softmax` 和 smooth loss 的重要背景。

这张图不是在说两个 predictor 都是最终答案，而是在说明：**linear score 是中间证据，activation / decision rule 把证据变成可解释的 output**。

课件原页 p.82 把线性 score 与 Step 输出画在同一横轴上。请比较两者：连续直线保留 score 大小，阶梯只保留类别边；因此训练需要可用的连续学习信号，而最终决策可以再使用阈值。

![[assets/1 Basic neural networks/source/lecture1-p82.png]]

### 4.4 Perceptron Algorithm

Perceptron 是早期可学习的 Artificial Neuron，常用于 `y in {-1,+1}` 的 binary classification。

对每个 training sample $(x_i,y_i)$：

1. 计算 $s=\mathbf{w}^{T}\mathbf{x}_i+b$。
2. 预测 $\hat{y}_i=\operatorname{sign}(s)$。
3. 如果 $y_i s\le 0$，update：

$$
\mathbf{w}\leftarrow\mathbf{w}+\eta y_i\mathbf{x}_i,
\qquad
b\leftarrow b+\eta y_i
$$

其中 $\eta>0$ 是 learning rate。若 prediction 正确，则不 update。真实 label 为 `+1` 时，沿 $\mathbf{x}_i$ 方向调整，使 score 变大；label 为 `-1` 时反方向调整。

**Perceptron Theorem**：若 data linearly separable，且 sample norm 与 separation margin 满足有限条件，Perceptron 会在有限次 updates 后找到正确分开 training set 的 hyperplane。若 data 不可线性分割，algorithm 可能永不停止，而且找到可行 boundary 不代表 generalization 最好。

#### 课件案例：用一次误分类修正边界

课件 p.84–86 将任务写成带标签样本 $(x_i,y_i)$、线性 score $w^T x_i$ 与符号 prediction。若 $y_i(w^T x_i+b)\le 0$，该点在错误一侧或恰好落在边界上，按 label 方向更新 weights，使下一次 score 朝正确符号移动。设一个教学样本 $x=(2,1)$、label $y=+1$、初始 $w=(0,0),b=0,\eta=1$，则初始 score 为 0，触发更新：$w'=(2,1), b'=1$；同一点的新 score 为 $2\cdot2+1\cdot1+1=6$，正号符合 label。这个例子演示更新方向，不是课件给出的数值结果。

![[assets/1 Basic neural networks/source/lecture1-p84.png]]
![[assets/1 Basic neural networks/source/lecture1-p85.png]]
![[assets/1 Basic neural networks/source/lecture1-p86.png]]

**交互：逐个追踪错误怎样移动边界。** 打开 [[assets/1 Basic neural networks/interactive/perceptron-learning-explorer.html|Perceptron learning explorer]]：默认数据可被直线分开；按“检查下一个样本”观察只有误分类触发 update，再切换到“含冲突点”检查定理边界。静态回退是更新式 $w\leftarrow w+\eta y_i x_i$、$b\leftarrow b+\eta y_i$ 与上方完整手算；页面中的样本坐标和参数标为 teaching example。

**Lecture 1 source case — p.93：Perceptron 收敛条件**。课件指出收敛界由 margin 决定，并特别标出不需要样本 i.i.d.，且错误次数上界不依赖数据序列长度 $n$。关键前提仍是存在一个正确分类的 $w^*$；因此这个结论不能推广成“任意数据都一定收敛”。

![[assets/1 Basic neural networks/source/lecture1-p93.png]]

## 5. From Neurons to a Single-layer Network

**Motivation**：一个 neuron 只能对一个 pattern 给出一个 score，无法同时表达多个 classes 或多个中间 features。把 units 组成 layer，才能让多组 weighted sums 并行产生 representation；代价是矩阵规模和参数存储一起增长。

一个 neuron 只能输出一个 score；多个 neurons 可以组成一个 **layer**。若 input 是 `D` 维，output 有 `M` 个 units：

$$
\mathbf{s}=W\mathbf{x}+\mathbf{b},
\qquad
\mathbf{h}=g(\mathbf{s})
$$

- `x`：`D` 维 input vector。
- `W`：通常是 `M x D` 的 weight matrix。
- `b`：`M` 维 bias vector。
- `s`：各 output unit 的 pre-activation scores。
- `h`：`M` 维 output vector。

这里的 **layer** 是一组同时处理 input 的 neurons；**unit** 是 layer 中的一个 neuron。一个 unit 可能对某种 input pattern 特别敏感，这种作用类似 **feature detector**：它把原始 input 转换成一个更容易被后续 layers 使用的 feature signal。

如果 `g` 是 **element-wise function**，每个 unit 独立计算：

$$
h_j=g(s_j)=g(\mathbf{w}_j^T\mathbf{x}+b_j)
$$

这种 output 可以作为多个 feature detectors。若 `g` 同时查看整个 vector，则 units 可以互相竞争，例如 `Softmax`。

## 6. Multiclass Classification and Softmax

**Motivation**：多个 class 的 raw scores 只能排序，不能说明“模型有多确定”，也不能直接作为可比较的 probability。Softmax 把竞争关系规范化，给 loss 一个可微的概率输入。

### 6.1 One-hot Encoding

`K` 个 classes 的 label 可以写成 **one-hot vector**：正确 class 的位置为 `1`，其他位置为 `0`。例如：

$$
\text{cat}\rightarrow[1,0,0]^T,\qquad
\text{dog}\rightarrow[0,1,0]^T,\qquad
\text{ship}\rightarrow[0,0,1]^T
$$

### 6.2 Winner-Take-All

最直接的 multiclass output 是 **Winner-Take-All (WTA)**：只让最高 score 的 unit output `1`：

$$
h_j=\begin{cases}
1,&j=\operatorname*{arg\,max}_i s_i\\
0,&\text{otherwise}
\end{cases}
$$

WTA 给出 class，但没有表达 confidence。`[0.51,0.49,0]` 与 `[0.99,0.01,0]` 会得到相同的 class output。

### 6.3 Softmax

**Softmax** 把 scores 转换为一个 **probability distribution**（一组总和为 1、可表示各 class 可能性的数值）。公式中的 **normalization** 指除以所有 exponential scores 的总和，使输出满足“总和为 1”的条件：

$$
p_k=P(Y=k\mid X=\mathbf{x})
=\frac{e^{s_k}}{\sum_{j=1}^{K}e^{s_j}}
$$

分子将 score 转为正数，分母负责 normalization。重要性质：

- $\operatorname*{arg\,max}_k p_k=\operatorname*{arg\,max}_k s_k$，最高分 class 不变；
- score 差距越大，probability 越接近 0 或 1；
- probability 表示 model 的 confidence，但不一定是校准良好的真实 confidence。

例：

$$
\mathbf{s}=[3.2,5.1,-1.7]^T
$$

取 exponential 并 normalization 后约为：

$$
\operatorname{Softmax}(\mathbf{s})\approx[0.13,0.87,0.00]^T
$$

prediction 是第二个 class，但完整 probability information 仍然保留。

下面用一个简化的 bar chart 对比同一组 scores 经过 Softmax 后的 class probabilities；第二个 class 的 bar 最高，因此是 prediction。

```chartsview
type: Column
data:
  - class: cat
    probability: 0.13
  - class: dog
    probability: 0.87
  - class: ship
    probability: 0.00
options:
  xField: class
  yField: probability
``` 

## 7. Objective, Log Loss, and Regularization

**Motivation**：只追求 training accuracy，模型可能对训练样本记忆过度；只看 accuracy，又无法区分“低置信度猜错”和“极度自信地猜错”。因此需要与任务一致的 loss，再用 regularization 控制复杂度。

**Engineering perspective**：L1 regularization 可能产生 sparse weights，便于剪枝和减少存储；L2 通常保持更多小权重，优化更平滑。算法上没有绝对赢家，硬件上还要看稀疏模式是否规则、零值是否真的能跳过 MAC，以及额外索引是否抵消收益。

![[assets/1 Basic neural networks/nn2-l1-l2-regularization.png]]

> **截图导航**：L1 的几何尖角更容易与 loss contour 相交在坐标轴上，因此产生 zero weights；L2 更倾向把所有 weights 缩小。读图时同时问“模型统计性质”和“硬件是否能有效利用这些零”。

### 7.1 What Is a Loss Function?

**Loss function** 是一个用来衡量 model prediction 与真实 label 之间差异的 function。对单个 training sample $(\mathbf{x}_i,y_i)$，model 先产生 prediction：

$$
\hat{y}_i=f(\mathbf{x}_i;\theta)
$$

然后用 loss function 比较 prediction 和 ground-truth label：

$$
\ell_i=\ell(\hat{y}_i,y_i)
$$

这里的 $\ell_i$ 是第 $i$ 个 sample 的 **single-example loss**：

- loss 越小，说明 prediction 越接近真实 label；
- loss 越大，说明 prediction 与真实 label 差距越大；
- training 的目标是调整 parameters $\theta$，使 average loss 尽可能小。

不同 task 可以使用不同的 loss function：

- **Regression** 常使用 **Mean Squared Error (MSE)**：
  $$
  \ell_i=(\hat{y}_i-y_i)^2
  $$
- **Classification** 常使用 **cross-entropy / negative log-likelihood**，因为它可以直接惩罚 model 对正确 class 给出的 probability。

Loss function 不是 prediction 本身，而是 training 时用来评价 prediction 的 numerical criterion。Neural network 根据 loss 计算 gradient，再用 `Gradient Descent` 更新 weights 和 biases。

这里的 **criterion** 可以理解为“训练时要最小化的评分标准”：它不直接告诉 model 最终答案，而是告诉 model 当前 prediction 离正确答案有多远。

### 7.2 Empirical Risk and Expected Risk

给定 training data：

$$
\mathcal{D}=\{(\mathbf{x}_i,y_i):1\le i\le N\}
$$

在 training samples 上的平均 loss 叫 **empirical loss**（只根据手头观察到的 samples 计算的平均误差）：

$$
\widehat{L}(f)=\frac{1}{N}\sum_{i=1}^{N}\ell(f,\mathbf{x}_i,y_i)
$$

真正关心的是从 data distribution 抽到新 sample 时的 **expected loss**（对潜在新 samples 的平均误差）：

$$
L(f)=\mathbb{E}_{(\mathbf{x},y)\sim D}[\ell(f,\mathbf{x},y)]
$$

Training loss 小不保证 expected loss 小，这就是 generalization 问题。

### 7.3 Logistic Regression and Negative Log-likelihood

在 binary classification 中，`linear score` 经过 Sigmoid 后得到 probability：

$$
p(y=1\mid\mathbf{x})=\sigma(\mathbf{w}^{T}\mathbf{x}+b)
$$

这种用 linear score 预测 probability 的 model 叫 **Logistic Regression**。在 `K` 个 classes 时，把 Sigmoid 的单一 probability 扩展为 Softmax probability vector，并使用相同的 negative log-likelihood 思路。

对于真实 class $y_i$，single-example loss 为 **negative log-likelihood**，也常称 **cross-entropy loss**：

$$
L_i=-\log P(Y=y_i\mid X=\mathbf{x}_i)
$$

代入 Softmax：

$$
L_i=-\log\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right)
$$

直觉：正确 class probability 越接近 1，loss 越接近 0；probability 很小，loss 会很大。因此 minimizing log loss 会迫使 model 把 probability 分配给正确 class。Accuracy 只记录对/错，无法区分“谨慎猜错”和“非常自信地猜错”，所以不适合作为主要 gradient-based objective。

### 7.4 Regularization

为减少 overfitting，在 data loss 后增加一个偏好简单 model 的 **regularization term**。**Regularization** 是一种对 model complexity 加惩罚的 training strategy：它牺牲一点 training fit，换取更好的 generalization。

$$
L(W)=\frac{1}{N}\sum_{i=1}^{N}L_i(f(\mathbf{x}_i,W),y_i)+\lambda R(W)
$$

- 第一项：data loss，要求预测匹配 labels。
- 第二项：regularization，限制 model complexity。
- `lambda >= 0`：regularization strength；太大可能造成 underfitting。

**L2 regularization**：

$$
R_{L2}(W)=\sum_{k,l}W_{k,l}^2
$$

倾向让 weights 变小，但通常不精确为 0；Bayesian interpretation 中对应 Gaussian prior。

**L1 regularization**：

$$
R_{L1}(W)=\sum_{k,l}|W_{k,l}|
$$

更容易产生 exact zeros，得到 **sparse model**，相当于选择部分 features；对应 Laplace prior。

**Elastic Net** 同时使用 L1 和 L2：

$$
R_{EN}(W)=\sum_{k,l}\left(\beta W_{k,l}^2+|W_{k,l}|\right)
$$

几何上，L1 constraint 像菱形，L2 constraint 像圆形；loss contours 更容易在 L1 的尖角相交，因此 L1 更容易产生 zero weights。

## 8. Optimization: Gradient Descent

**Motivation**：有了 loss 和 gradient 之后，仍然要决定每一步怎样移动 parameters。learning rate 太大可能发散，太小则浪费时间；batch 选择又会影响 gradient noise、并行度和 memory 使用。

**Engineering perspective**：mini-batch 让同一套 weights 被多个 samples 复用，适合矩阵乘法阵列；但 batch 越大，activation buffer 和 memory bandwidth 压力越大，单个样本的 latency 也未必更好。训练服务器关注 throughput，边缘设备推理则常更关心 latency 和 energy。

### 8.1 Basic Update

Training objective 是找到让 loss 最小的 parameters。对 parameter vector $\theta$：

$$
\theta\leftarrow\theta-\eta\nabla_{\theta}L(\theta)
$$

因为 gradient 指向上升最快方向，所以减去它就是下降方向。基本循环：

1. 用当前 parameters 做 prediction。
2. 计算 loss。
3. 计算 gradient。
4. 更新 weights 和 biases。

### 8.2 Learning Rate

#### 交互：看见 learning rate 如何改变下山步伐

先猜：如果把 eta 调大，参数会更快到达最低点，还是可能跨过最低点？打开 [Gradient descent explorer](assets/1 Basic neural networks/interactive/gradient-descent-explorer.html)，先用默认值点击“走一步”，再把 eta 分别调小和调大后点击“重新运行”。

Obsidian 原生打开：[[assets/1 Basic neural networks/interactive/gradient-descent-explorer.html|打开 Gradient descent 互动页面]]

> [!tip]- Obsidian 内原生练习（不需要 HTML）
> 1. 先写下你的预测：eta 变大后，loss 会怎样？
> 2. 查看下面的三种情况：eta 太小是“慢”，合适是“稳”，太大是“跨过最低点”。
> 3. 回到公式 theta ← theta − eta grad L(theta)，指出 eta 改变的是哪一部分。
> 4. 上面的 HTML 链接是可选的独立互动页；如果 Obsidian 只显示源码或按钮不动，直接使用本折叠练习和下方表格。

这是一个教学插图：它使用简单的 L(theta)=0.18 theta^2+0.15，不是课件中的真实训练曲线。静态结论仍然是：

| 现象 | 原因 |
|---|---|
| eta 太小 | 每次更新很短，收敛慢 |
| eta 合适 | 稳定地接近 loss minimum |
| eta 大于约 2.78 | 更新会跨过 minimum，开始来回摆动；大于约 5.56 时，这个教学函数上的误差会扩大 |

**Obsidian 内互动（JS Engine）**

调整 learning rate，再点击“运行 12 步”，观察参数路径如何变化。

```js-engine
const box = container.createDiv();
box.createEl("b", { text: "老师讲解：θ ← θ − η × 0.36θ；η 决定每步长度" });
const etaRow = box.createDiv(); etaRow.createEl("label", { text: "learning rate η " });
const eta = etaRow.createEl("input"); eta.type = "range"; eta.min = "0.02"; eta.max = "8"; eta.step = "0.02"; eta.value = "0.25";
const etaLabel = etaRow.createEl("span", { text: "0.25" });
const stepButton = box.createEl("button", { text: "走一步" });
const runButton = box.createEl("button", { text: "连续走 12 步" });
const resetButton = box.createEl("button", { text: "重置" });
const result = box.createEl("pre");
let theta = 3, history = [];
function render() {
  const gradient = 0.36 * theta, nextTheta = theta - Number(eta.value) * gradient;
  etaLabel.textContent = Number(eta.value).toFixed(2);
  result.textContent = "L(θ)=0.18θ²+0.15；∂L/∂θ=0.36θ\n当前 θ=" + theta.toFixed(3) + "，loss=" + (0.18 * theta * theta + 0.15).toFixed(3) +
    "，gradient=" + gradient.toFixed(3) + "\n走一步后 θ=" + nextTheta.toFixed(3) +
    (history.length ? "\n经过的 θ：" + history.map(value => value.toFixed(2)).join(" → ") : "\n先观察当前梯度的方向，再点击走一步。");
}
stepButton.addEventListener("click", () => { history.push(theta); theta -= Number(eta.value) * 0.36 * theta; render(); });
runButton.addEventListener("click", () => { history = []; theta = 3; for (let i = 0; i < 12; i++) { history.push(theta); theta -= Number(eta.value) * 0.36 * theta; } render(); });
resetButton.addEventListener("click", () => { theta = 3; history = []; render(); });
eta.addEventListener("input", render); render();
return;
```

- `eta` 大于约 2.78：会跨过 minimum；大于约 5.56：在这个教学函数上会发散。
- `eta` 太小：每步很短，training 很慢。
- 合适的 `eta`：loss 通常较快且稳定地降低。

### 8.3 Batch, SGD, and Mini-batch

完整 objective 的 gradient：

$$
\nabla_WL(W)=\frac{1}{N}\sum_{i=1}^{N}\nabla_WL_i(\mathbf{x}_i,y_i,W)+\lambda\nabla_WR(W)
$$

- **Batch Gradient Descent**：每次用全部 `N` 个 samples，稳定但单步昂贵。
- **SGD (Stochastic Gradient Descent)**：每次用一个 sample，单步便宜但 noise 大。
- **Mini-batch Gradient Descent**：每次用小批 samples，如 `32/64/128/256`，是 modern deep learning 最常见的折中。

这里的 **batch** 是一次计算所使用的一组 samples；**stochastic** 表示每次只用随机抽取的少量 data，因此得到的 gradient 是 noisy estimate，而不是完整 training set 的精确平均值。

SGD 的 loss 不一定每一步都下降；mini-batch 对多个 samples 求平均，通常更稳定，也适合 GPU 并行。

### 8.4 Optimizer 不只是 `step()`

最基本的 SGD 只需要保存当前 parameter 和 gradient。实际常见的 **Momentum** 会额外保存历史更新方向，使连续方向的更新累积、抑制来回震荡；**Adam** 则为每个 parameter 保存一阶梯度移动平均和二阶平方梯度移动平均，再用它们调整每个坐标的步长。它们都属于 optimizer，不是 network architecture。

对初学者，先区分三类数字：parameter 是模型要学的 weight/bias，gradient 是当前 batch 对 parameter 的局部建议，optimizer state 是为了决定下一步如何更新而保存的历史信息。换 optimizer 不会改变 forward pass 的 output shape，但可能改变收敛速度、稳定性和训练 memory。

**AI chip perspective**：如果用 Adam 训练，除了 weights、gradients 和 activations，还要读写 optimizer states；这些 states 会增加 memory capacity 和 bandwidth 压力。推理阶段通常不需要保存 optimizer states，因此 training accelerator 与 inference accelerator 的 memory budget 不能直接等同。

下面是一个理想化的 training curve：随着 iteration 增加，loss 总体下降。真实 SGD 或 mini-batch training 可能上下波动，但趋势通常应朝更低的 loss 前进。

```chartsview
type: Line
data:
  - iteration: 0
    loss: 4.5
  - iteration: 1
    loss: 3.2
  - iteration: 2
    loss: 2.4
  - iteration: 3
    loss: 1.8
  - iteration: 4
    loss: 1.35
  - iteration: 5
    loss: 1.05
  - iteration: 6
    loss: 0.82
  - iteration: 7
    loss: 0.68
options:
  xField: iteration
  yField: loss
  smooth: true
``` 

## 9. Capacity and Hidden Layers

**Motivation**：如果一条直线无法解决 XOR 等任务，继续调整同一个 neuron 的 weight 也没有用；需要先改变 representation，再让 output layer 在新的空间里做更容易的分类。Hidden layer 正是学习这种中间 representation 的机制。

### 9.1 Why One Neuron Is Not Enough

Single neuron 的 boundary 是一个 hyperplane，只能解决 linearly separable problems。经典反例是 **XOR**：正负 examples 交错，任何一条直线都不能把两类同时分开。

**Slide reference — XOR limitation**

![[assets/1 Basic neural networks/nn2-xor.png]]

如果两类 points 不能被一条直线分到两边，一个 neuron 就不够；这正是需要多个 neurons 或 hidden layer 的原因。

两种解决思路：

1. 手工把 input transform 到更好的 representation。
2. 加入 **hidden layer**，让 network 自己学习 representation。

### 9.2 Single Hidden Layer

不同 hidden units 可以学习不同的 linear boundaries；组合后把 input space 分成多个 regions，再由 output unit 组合这些 regions。这里的 **representation** 指 network 在中间层对原始 input 形成的新表达方式；好的 representation 会让后续 classifier 更容易区分 classes。对 Boolean problems，MLP 可以组合 AND、OR、NOT 等 gates，因此计算任意 Boolean function。

**Slide reference — hidden layer changes the representation**

![[assets/1 Basic neural networks/nn2-hidden-layer.png]]

这张 slide 的直觉是：hidden units 先把原始 input space 切成若干 regions，output unit 再组合这些 regions；network 学到的不只是最后的答案，也包括中间的 representation。

**Universal Approximation Theorem**：只要 hidden units 足够多，single-hidden-layer network 可以逼近很大范围内的 continuous functions。但“可以表示”不等于“容易学到”，实际仍需选择 units 数量并找到合适 parameters。

### 9.3 Why Deeper Networks?

更多 layers 可以用更少 units 表示某些复杂 functions。直观上：early layers 学简单 patterns，middle layers 组合 parts，later layers 组合 higher-level concepts。Deep network 不是每个 unit 更复杂，而是简单 operations 被多次 composition。

从 representation 的角度看，single hidden layer 也可能表示任意 continuous function，但所需 hidden units 数量可能非常大。某些 Boolean functions 或 multivariate polynomials 用 shallow network 表示时需要 exponentially many units，而增加 depth 后可以用 polynomial 或更少的 units 表示。这里的重点是 **representational efficiency**：同样的 function，deep architecture 可能用更紧凑的结构表达。

但 **Universal Approximation Theorem** 只说明“存在一个可以逼近的 network”，并没有告诉我们 hidden units 应该有多少，也不保证 optimization 一定容易。因此，理论上的 expressiveness 不等于实际 training 一定成功。

### 9.4 Connectivity

不一定所有 layers 都完全连接。常见结构包括：

- **Sequential connectivity**：information 依次通过各层。
- **Residual / highway connection**：提供更短的 information path，帮助 optimization。
- **Dense connection**：把早期 features 直接 concatenated 到后续 layers。

### 9.5 MLP as an Implicit Representation

Slides 还展示了 modern MLP 不只用于 classification 或 regression，也可以直接作为一个 **implicit representation**：network 接收 coordinates 或其他 query，输出该位置对应的 properties。例如在 neural rendering 中，输入 spatial position 和 viewing direction，MLP 可以输出 **color** 与 **density**，再通过 **volume rendering** 合成图像。

这里的 **query** 可以理解为“向 neural function 提问的输入”；**density** 表示一条光线在某个位置附近被物体占据、从而影响光线衰减的程度；**volume rendering** 则把同一条 ray 上多个位置的 color 和 density 按可见性加权组合成最终 pixel。它们分别回答“问哪里”“那里有多不透明”和“怎样合成看到的颜色”。

一个代表性例子是 **NeRF (Neural Radiance Fields)**。它把 3D scene 表示为一个可查询的 neural function，而不是只保存一张固定 image 或一个显式 voxel grid。NeRF 常结合 **positional encoding**，把低维 coordinates 映射到多频率 features，使 MLP 能表达更细的空间变化。这里的重点是：MLP 学到的不是“某个固定输入的答案”，而是一个可以在连续 coordinates 上查询的 function。

这说明 MLP 本身先对每个 query 做一次 differentiable computation，`volume rendering` 再把沿 ray 得到的多个 outputs 组合为 image。由于整个过程可微，最终 image 与 target image 之间的 loss 可以通过 Backpropagation 传回 MLP parameters。

这些 modern applications 延伸了课程中的基本观点：neural network 是 composed differentiable operations；只要定义好 query、output 和 loss，就可以通过 forward pass 与 Backpropagation 学习一个复杂 representation。

## 10. General Multi-layer Network

**Motivation**：单个 hidden layer 可以表示很多 function，但可能需要极多 units。多层 composition 让简单 operations 逐层组合，提升 representational efficiency；代价是 optimization、initialization 和 gradient flow 更难。

课程中的 `N-layer neural network` 通常指 `N-1` 个 hidden layers 加一个 output layer，不把 input units 算作一层。第 `l` 层：

$$
\mathbf{s}^{(l)}=W^{(l)}\mathbf{x}^{(l-1)}+\mathbf{b}^{(l)},
\qquad
\mathbf{x}^{(l)}=\sigma\left(\mathbf{s}^{(l)}\right)
$$

其中 $\mathbf{x}^{(0)}=\mathbf{x}$ 是原始 input。**Composition**（函数复合）表示一个 function 的 output 成为下一个 function 的 input；整个 network 因此是 functions 的 composition：

$$
\mathbf{x}^{(L)}=f_L\left(f_{L-1}\left(\cdots f_2(f_1(\mathbf{x}))\right)\right)
$$

没有 activation 时：

$$
h=W_1x+b_1,\qquad y=W_2h+b_2
$$

代入后仍是 $y=W'x+b'$。无论叠加多少 linear layers，仍等价于一个 linear model；activation function 带来 non-linearity，才能表达弯曲的 decision boundary 和复杂 patterns。

## 11. Inference: Forward Pass

### 11.1 Forward Pass 是什么

**Motivation**：训练好之后，用户只要求模型快速给出结果；inference 不再更新 weights，而是重复 forward computation。因此它是 AI accelerator 评估 throughput、latency、memory footprint 和 energy 的主要阶段。

**Inference** 是用已经学好的 parameters 对新 input 做 prediction，不修改 weights；也就是模型部署或测试时的“使用阶段”。从 input 到 output 依次计算，因此也叫 **forward pass**。

single hidden layer example：

$$
\mathbf{s}^{(1)}=W^{(1)}\mathbf{x}^{(0)}+\mathbf{b}^{(1)},
\qquad
\mathbf{x}^{(1)}=\sigma(\mathbf{s}^{(1)})
$$

$$
\mathbf{s}^{(2)}=W^{(2)}\mathbf{x}^{(1)}+\mathbf{b}^{(2)},
\qquad
\mathbf{o}=g(\mathbf{s}^{(2)})
$$

对于 regression，`g` 可以是 identity；对于 multiclass classification，`g` 通常是 Softmax。$\mathbf{x}^{(1)}$ 是 hidden representation，不是最终 prediction。

**Slide reference — forward pass**

![[assets/1 Basic neural networks/nn2-forward-pass.png]]

这张图可以当作“推理阶段”的路线图：已知 input 和 parameters，沿箭头逐层计算，最后得到 prediction。

### 11.2 Matrix Implementation

所有 units 可以用 matrix operations 并行计算：

```python
s1 = W1 @ x + b1
x1 = activation(s1)
s2 = W2 @ x1 + b2
out = output_activation(s2)
```

`@` 表示 matrix multiplication；`shape` 描述 array dimensions，例如 `(3,)` 是 3 个 numbers，`(2,3)` 是 2 x 3 matrix。

在真实 implementation 中，通常会同时处理一个 **batch** 的 samples。此时 input 可以写成矩阵 $X$，一次 matrix multiplication 就能并行完成多个 samples 的 forward computation。这样做不仅简洁，也能利用 GPU 的 parallel computation。

## 12. Parameter Learning: Backward Pass

### 12.0 Neural Network Training Flow

**Motivation**：训练阶段不仅要算 prediction，还要把错误信号传回每个 parameter。backward pass 能高效得到 gradients，但需要保存或重算 forward 的 intermediate values，这会直接影响训练 memory。

每一轮的含义是：先用当前 parameters 做 prediction，再与 target 计算 loss；然后从 loss 反向计算每个 parameter 的 gradient，最后更新 parameters。重复这一过程，通常会使 training loss 逐步降低。

**Slide reference — one training iteration**

![[assets/1 Basic neural networks/nn2-training-loop.png]]

训练不是一次完成，而是重复 `Forward pass → Loss → Backward pass → Parameter update` 这个循环。

### 12.0.1 可选现象展示：把训练循环真正跑起来

运行 [01_mlp_dataset.py](experiments/01_mlp_dataset.py)：

```bash
cd "/Users/dongyangchen/Desktop/lecture note/deep learning/experiments"
python3 01_mlp_dataset.py
```

这个实验生成一个两类的 noisy half-moons dataset，训练一个两层 MLP。它对应本章的完整链条：`dataset → mini-batch → forward → CrossEntropyLoss → backward → optimizer.step()`。终端会打印 `train_loss`、`train_acc` 和 `val_acc`。

这段程序只是把正文中的训练链路具体显示出来。默认运行时会看到 `train_loss`、`train_acc` 和 `val_acc`：训练集指标描述模型正在拟合什么，validation 指标描述它对未参与更新的数据是否仍然有效。把 epoch、learning rate 或数据噪声改动后，看到的差异分别对应“训练是否充分”“每步更新是否稳定”和“数据本身是否更难”；这些是可选的现象展示，不是完成本章的作业。

读懂这段代码时，最关键的因果关系是：`loss.backward()` 只计算 gradients，`optimizer.step()` 才根据 gradients 修改 parameters；validation set 只用于观察泛化，不参与参数更新。

实验代码是本笔记的最小教学实现，用来把正文已经解释的 training loop 变成可观察的数字；它不是完成本章的作业。如果要换成真实图像数据，可对照官方 [PyTorch MNIST example](https://github.com/pytorch/examples/tree/main/mnist)。

### 12.1 Training Objective

训练目标可写成：

$$
W^*=\operatorname*{arg\,min}_{W}\sum_{i=1}^{N}\operatorname{loss}\left(\mathbf{o}^{(i)},\mathbf{t}^{(i)}\right)
$$

`t` 是 target，`o` 是 output。Forward pass 计算 prediction；从 loss 出发向后计算 loss 对每个 parameter 的 derivative，叫 **backward pass**。

最终更新：

$$
W\leftarrow W-\eta\frac{\partial E}{\partial W}
$$

### 12.2 Chain Rule

若 quantity 通过多个 operations 依赖 parameter，就使用 **chain rule**。设：

$$
s=wx+b,\qquad y=\sigma(s),\qquad E=\frac{1}{2}(y-t)^2
$$

则：

$$
\frac{\partial E}{\partial w}
=\frac{\partial E}{\partial y}
\frac{\partial y}{\partial s}
\frac{\partial s}{\partial w}
$$

每个 factor 都是 local derivative，连乘后得到从 `w` 到 `E` 的完整影响。

### 12.3 Single-neuron Derivatives

令：

$$
\delta_y=\frac{\partial E}{\partial y},
\qquad
\delta_s=\frac{\partial E}{\partial s}
$$

则：

$$
\delta_s=\delta_y\,\sigma'(s)
$$

因为 `s=wx+b`：

$$
\frac{\partial E}{\partial w}=\delta_s x,
\qquad
\frac{\partial E}{\partial b}=\delta_s
$$

Bias derivative 不带 $x$，因为 $\frac{\partial s}{\partial b}=1$。

### 12.4 Multivariate Chain Rule

如果一个 node 的 output 被多个 children 使用（这种分叉叫 **fan-out**），来自各条路径的 derivative 需要相加：

$$
\frac{\partial E}{\partial t}
=\sum_j\frac{\partial E}{\partial z_j}\frac{\partial z_j}{\partial t}
$$

这叫 **distributed chain rule**。在 computation graph 中，一个 node 收到所有 children 的 derivative messages，aggregate 后再传给 parents。

## 13. Computation Graph and Backpropagation

**Motivation**：如果每个 parameter 都重新运行一次完整 network 来估计影响，训练成本会随 parameters 数量急剧增加。Computation graph 把复杂 function 拆成局部 operations，让 backward pass 复用中间结果和 local derivatives。

![[assets/1 Basic neural networks/nn2-backpropagation.png]]

> **截图导航**：沿图先看 forward 的 values 如何生成，再反向看 error signal 如何沿依赖关系传回。`fan-out` 处要把多条路径的 gradient 相加；这正是 distributed chain rule 的关键。

### 13.1 Computation Graph

**Computation graph** 把复杂 function 拆成简单 operations：

- **Nodes**：inputs 和 intermediate values。
- **Edges**：表示直接依赖关系。
- **Forward pass**：按依赖方向计算 values。
- **Backward pass**：按相反方向计算 derivatives。

例如：

$$
x,w,b\rightarrow s=wx+b\rightarrow y=\sigma(s)\rightarrow E
$$

### 13.2 Error Signal

用 shorthand：

$$
\delta_v=\frac{\partial E}{\partial v}
$$

$\delta_v$ 不是新的神秘变量，而是 loss 对 node $v$ 的 derivative，也叫 **error signal**。

### 13.3 General Backpropagation Algorithm

给定 computation graph 的 topological ordering $v_1,\ldots,v_N$：

1. **Forward pass**：从 parents 到 children 计算每个 node 的 value。
2. 令最终 loss node 的 error signal 为 $\delta_{v_N}=1$。
3. **Backward pass**：从后往前聚合 messages：

$$
\delta_{v_i}
=\sum_{j\in\operatorname{Ch}(v_i)}
\delta_{v_j}\frac{\partial v_j}{\partial v_i}
$$

4. 用得到的 gradients 更新 weights 和 biases。

### 13.4 Local Backward Patterns

**Multiplication node** $z=xy$：

$$
\frac{\partial z}{\partial x}=y,
\qquad
\frac{\partial z}{\partial y}=x
$$

因此 incoming gradient 分别乘上另一个 input。

**Max node** $y=\max_i z_i$：

$$
\frac{\partial y}{\partial z_i}=\begin{cases}
1,&i=\operatorname*{arg\,max}_j z_j\\
0,&\text{otherwise}
\end{cases}
$$

gradient 只传给取得 maximum 的 input。若多个 inputs 并列 maximum，implementation 需要规定 tie-breaking 或使用 **subgradient**；subgradient 是在不可微点用来表示可行局部下降方向的广义 derivative。

### 13.5 Why Backpropagation Is Efficient

Backpropagation 复用 forward pass 得到的 intermediate values，让每个 node 只计算自己的 local derivative，再把 message 传给前面的 nodes。这样一次 backward pass 就能高效得到大量 parameters 的 gradients，而且每种 operation 只需定义 local derivative。

对每条 connection，forward pass 大致需要一次 multiply-add；backward pass 通常需要计算并传播相应的 local derivative，因此总体计算量仍与 network 中的 connections 数量同阶，而不是为每个 parameter 重新计算一次完整的 loss。对 multilayer network，反向传播必须按照依赖关系反方向进行；涉及 vector-valued quantities 时，**Jacobian**（向量 function 对输入各分量的 partial derivatives 组成的 matrix）的乘法顺序也必须保持正确。

Backpropagation 已用于绝大多数 neural networks，包括许多 generative models 和更高级的 optimization methods，因为它提供了计算 parameter update 所需的 gradients。但它只是一种高效的数学计算方法，不是对 biological brain learning mechanism 的解释：目前没有充分证据表明大脑使用了与 error derivatives 完全对应的 signals。

## 14. CS / Python Supplement

```python
import numpy as np

x = np.array([2.0, 3.0])
w = np.array([0.5, -1.0])
b = 2.0
z = w @ x + b       # @ means dot product / matrix multiplication
a = max(0.0, z)     # scalar ReLU
```

- `array`：保存数字序列，可先理解为 vector。
- `@`：matrix multiplication 或 vector dot product。
- `shape`：array 的 dimensions。
- `for`：逐个处理 samples。
- `def`：定义 function。
- indentation：表示 code block。

公式 notation：

- 粗体小写 `x` 常表示 vector；大写 `W` 常表示 matrix。
- 上标 `(l)` 通常表示第 `l` 层，不是幂。
- 下标 `i` 常表示第 `i` 个 sample 或 feature。
- `T` 表示 transpose。
- `in` 或 `∈` 表示“属于”；`for all` 或 `forall` 表示“对所有”。

## 15. 回到完整训练链

读完本篇后，可以把所有组件放回同一个因果链中：Traditional programming 由人写出 rules；Machine Learning 用 dataset 中的 input/label 让 model 学 parameters；neuron 用 weighted sum 和 activation 产生 representation；output layer 产生 prediction；loss 把 prediction 与 target 的差异变成一个可微的标量；backpropagation 沿 computation graph 计算每个 parameter 的 gradient；optimizer 再更新 weights 和 biases。

最小手算例子也可以这样回看：$\mathbf{x}=(1,2)$、$\mathbf{w}=(3,-1)$、$b=2$ 时，$z=3\times1+(-1)\times2+2=3$，所以 $\operatorname{ReLU}(z)=3$。这个小计算已经包含 inference 中的 weighted sum 和 activation；training 只是再增加 target、loss、backward 与 update。

## 15.5 从会推导到会实现：数值、训练和硬件边界

前面的链条已经足够让你手算一个小网络；要达到工程级理解，还要回答三个问题：数学上正确，数值上稳定，系统上可执行。

### 15.5.1 数值稳定性

Softmax 的直接计算可能因为指数过大而 overflow。令 $m=\max_j z_j$，可以改写为：

$$
p_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}
$$

这就是 log-sum-exp trick。它不改变概率，只把最大指数输入平移到 0。分类训练中常直接使用 logits 版本的 cross-entropy，把 log-softmax 和 negative log-likelihood 合并，减少中间数值误差。

### 15.5.2 Initialization、normalization 与梯度传播

如果所有 hidden unit 的初始 weights 完全相同，它们会收到相同 gradient，始终学成相同的 unit，这叫 symmetry problem。随机 initialization 用来打破对称性；Xavier/Glorot 与 He initialization 则控制初始方差，避免信号逐层立刻爆炸或消失。

BatchNorm 和 LayerNorm 都会调整 activation 的尺度，但统计维度不同：BatchNorm 使用 batch 统计，LayerNorm 通常在单个 sample/token 的 feature 维度上归一化。normalization 不是自动解决 overfitting 的同义词。

### 15.5.3 Autograd 的工程真相

手推 chain rule 能解释 backpropagation，但框架还要保存 backward 所需的 intermediate activations，并处理 gradient accumulation、requires_grad 和训练/推理模式。训练时 activation 往往是显存大户；推理时使用 no-grad/inference 模式可以避免创建不需要的反向图。

算法工程师关心 gradient 是否正确，框架工程师关心 graph 和 memory 是否可管理，芯片工程师关心 activation 是否必须写回片外 memory。数学正确不自动等于系统可部署。

### 15.5.4 可复现性与结果审计

至少记录数据版本、代码版本、随机种子、依赖版本、设备、precision、batch size、learning rate、训练步数和 evaluation protocol。即使 seed 固定，某些并行 reduction 仍可能有 nondeterminism，因此应区分“可重复配置”和“唯一结果”。

| 审计层 | 必须问的问题 | 常见失败 |
|---|---|---|
| 数学 | shape、loss、gradient 是否一致？ | 维度或目标定义错误 |
| 数值 | 指数、归一化、累加是否稳定？ | NaN、overflow、梯度消失 |
| 软件 | train/eval、no-grad、seed 是否正确？ | 训练和推理行为不一致 |
| 硬件 | weight、activation、partial sum 在哪里？ | 带宽或片上 SRAM 不够 |
| 结果 | validation、test、精度和 latency 是否分开？ | 把一次 accuracy 当系统结论 |

> **External context**：本节把课件公式连接到真实实现；它是工程背景，不是对某款芯片 benchmark 的声明。

## 15.6 教授式完整推导：一个 neuron 如何真正学会

前面分别讲了 weighted sum、activation、loss、gradient 和 update。现在把它们连成一次完整推导。设输入 $x=2$，真实标签 $t=1$，模型为：

$$
z=wx+b,\qquad y=\sigma(z),\qquad E=\frac{1}{2}(y-t)^2
$$

初始参数取 $w=0.5, b=-0.5$，learning rate 取 $\eta=0.1$。

### 1. Forward pass

$$
z=0.5\times2-0.5=0.5
$$

$$
y=\sigma(0.5)=\frac{1}{1+e^{-0.5}}\approx0.6225
$$

模型预测正类的概率约为 0.6225。因为真实答案是 1，方向正确但还不够自信。

### 2. Loss

$$
E=\frac{1}{2}(0.6225-1)^2\approx0.0713
$$

loss 不是 prediction，而是训练时用来产生学习信号的标量。

### 3. Backward pass

计算图是：

```text
w, x, b -> z -> sigmoid -> y -> squared error -> E
```

先算：

$$
\frac{\partial E}{\partial y}=y-t=-0.3775
$$

$$
\frac{\partial y}{\partial z}=y(1-y)\approx0.2350
$$

所以：

$$
\frac{\partial E}{\partial z}=\frac{\partial E}{\partial y}\frac{\partial y}{\partial z}\approx-0.0887
$$

又因为 $z=wx+b$：

$$
\frac{\partial E}{\partial w}=\frac{\partial E}{\partial z}x\approx-0.1774,
\qquad
\frac{\partial E}{\partial b}=\frac{\partial E}{\partial z}\approx-0.0887
$$

### 4. Parameter update

$$
w_{new}=0.5-0.1(-0.1774)=0.51774
$$

$$
b_{new}=-0.5-0.1(-0.0887)=-0.49113
$$

更新后 $z_{new}\approx0.54435$，sigmoid 输出会更接近 1。这就是训练最小闭环：forward 产生 prediction，loss 产生误差，backward 产生方向，optimizer 改参数。

### 最容易错的地方

| 错误 | 为什么错 |
|---|---|
| 把 y 当成 label | y 是 prediction，t 才是 target |
| 只算 partial E / partial y 就更新 w | weight 的影响还要经过 z，必须使用 chain rule |
| gradient 为负就把 weight 减小 | 更新是 w 减去 eta 乘 gradient，负 gradient 会让 weight 增大 |
| loss 下降就认为泛化好 | 一个 sample 的 loss 不能替代 validation/test |

> **讲授结论**：如果你能解释上面四步，就真正理解了神经网络如何学习；CNN、RNN、Transformer 改变的是信息连接方式，不是这个训练闭环。

### 5. 这一步如何连接后面的三篇课

- **CNN**：把普通 neuron 的全连接改成局部连接和权重共享，但仍然执行 `forward → loss → backward → update`。
- **RNN**：把同一个小网络沿时间重复使用，新增 hidden state；反向传播仍然是 chain rule，只是计算图沿时间展开。
- **Attention / Transformer**：把“一个输出读取哪些输入”的权重变成可学习矩阵；训练闭环不变，变化的是连接方式和数据搬运代价。

因此，后面看到新模型时，先问两个问题：**它改变了谁和谁连接？它的梯度从哪里沿计算图传回？**

## 16. Takeaway

Neural network 是由许多 simple computational units 组合成的 **parameterized function**。完整训练循环是：

$$
\text{input}
\xrightarrow{\text{Forward Pass}}
\text{prediction}
\xrightarrow{\text{Loss}}
\text{error}
\xrightarrow{\text{Backpropagation}}
\text{gradients}
\xrightarrow{\text{Gradient Descent}}
\text{updated parameters}
$$

最重要的理解顺序是：

```text
Math
-> neuron
-> single-layer network
-> multiclass / Softmax
-> log loss and regularization
-> Gradient Descent
-> hidden layers
-> forward pass
-> chain rule
-> Backpropagation
```

这条链路说明了 neural network 如何从 input 产生 prediction，如何衡量错误，以及如何把错误有效地分配回每个 weight 和 bias。

### Lecture 1 原页案例索引

| Case ID | 来源页 | 笔记位置 | 本地截图 | 覆盖与可读性 |
|---|---:|---|---|---|
| L1-C1 | 21 | 1. Big Picture | `source/lecture1-p21.png` | 已解释 red panda 图像映射；清晰 |
| L1-C2 | 68 | 3.3 Training, Validation, and Test | `source/lecture1-p68.png` | 已说明三类数据职责；清晰 |
| L1-C3 | 70 | 3.3 Training, Validation, and Test | `source/lecture1-p70.png` | 已解释 validation U 形走势与过拟合；清晰 |
| L1-C4 | 75 | 4.2 Activation Function | `source/lecture1-p75.png` | 已跟读 neuron 结构与公式；清晰 |
| L1-C5 | 82 | 4.3 Single Neuron as Linear Classifier | `source/lecture1-p82.png` | 已比较 score 与 Step decision；清晰 |
| L1-C6 | 84–86 | 4.4 Perceptron Algorithm | `source/lecture1-p84.png` 至 `p86.png` | 已解释任务、迭代与误分类更新；清晰 |
| L1-C7 | 93 | 4.4 Perceptron Algorithm | `source/lecture1-p93.png` | 已说明 margin 前提与次数界；清晰 |

截图按 144 dpi 导出，在正常阅读宽度可辨识正文和公式。其他 slide 多为课程行政信息、历史照片或重复目录；它们不增加本篇核心计算链的理解，因此未逐页嵌入。原始 PDF 留在 Vault 根目录作为完整出处。

## 外部资料与进阶阅读

以下资料只补充实现缺口，不替代课件主线。访问日期：2026-10-17。

- **External context：autograd 和反向图**：PyTorch 官方 [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)，帮助你把手推 backpropagation 对应到 computation graph、saved tensors 和 no-grad。
- **External context：初始化与优化器**：PyTorch 官方 [Neural network modules](https://pytorch.org/docs/stable/nn.html) 和 [Optimizer](https://pytorch.org/docs/stable/optim.html)，用于查 API 行为，不把 API 当成数学解释。
- **External context：数值稳定性**：PyTorch 官方 [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)，注意它接收 logits，而不是已经 softmax 的概率。
- **进阶路线**：读完本章后再学习 numerical precision、autograd memory、kernel fusion 和 accelerator dataflow；这些是从“会训练”走向“会设计系统”的桥梁。
