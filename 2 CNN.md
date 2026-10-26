# 2 CNN

> [!tip] 互动入口
> [[互动实验入口]] · [[assets/2 CNN and RNN/interactive/cnn-slide-visual-explorer.html|打开 CNN 课件图视觉交互]]
>
> **按主题打开局部演示**：[[assets/2 CNN and RNN/interactive/convolution-window-explorer.html|卷积窗口]] · [[assets/2 CNN and RNN/interactive/pooling-receptive-field-explorer.html|Pooling 窗口]]。这两个页面是同一组 $4\times4$ 输入与 $2\times2$ filter 的局部教学演示；主交互优先用于读课件结构，局部页面用于核对算式和窗口位置。

> **互动阅读方式**：新交互以老师课件中的结构图为主，不是截图浏览。先看主画布里的 input volume、filter、feature map 或 residual shortcut，再用窗口位置、stride、pooling window 和路径步骤去改变同一张图。每个颜色和箭头都对应一个具体关系：橙色是当前局部操作，绿色是当前输出或合并点，蓝色是输入/主路径。课件图中的原始结构来自 CNN 1 p.14、p.16、p.29、p.43 与 CNN 2 p.73、p.75；数值、颜色和运动是标注为 interactive teaching annotation 的教学叠加。

> 本笔记整合自 Lecture 3: **CNNs I - Architecture & Training**、Lecture 4: **CNNs II - Network Regularization & Architecture**，以及 Lecture 5 中的 semantic segmentation 部分（Lan Xu，CS 280 Deep Learning，Fall 2026）。RNN 内容已独立整理到 [[3 RNN]]。

> **Prerequisite**：本笔记默认你已经读过 [[1 Basic neural networks]]。其中已经解释了 neuron、weight、bias、parameter、hyperparameter、vector/matrix、loss、gradient、Softmax、training/validation/test、Gradient Descent 和 Backpropagation。这里只在 CNN 需要的位置补充新的图像与张量背景。

> **学习导航**：前置 [[1 Basic neural networks]] → 当前：图像中的空间计算（CNN）→ 后续 [[3 RNN]] 学习时间递归，再进入 [[4 Sequence Models and Attention]]。CNN 在空间位置复用 filter；RNN 在时间步复用 recurrent weights，但受到 state dependency 限制。

## 老师开场：为什么图像不能先粗暴地拉平成一长串数字

假设一张 $224\times224$ 的 RGB 图片被拉平成长度为 $150528$ 的向量，再接一个 fully connected layer。数学上当然可以算，但这个做法会丢掉一个非常有用的事实：**相邻像素通常比相隔很远的像素更有关系，而且同一种边缘可能出现在图片的任何位置**。如果每个位置都使用一套独立权重，参数量和数据搬运都会迅速膨胀。

CNN 的设计回答了两个局部问题：

1. 我只需要先看一个小邻域，不必一次连接整张图；
2. 我在左上角寻找“竖直边缘”的方法，在右下角也可以复用。

于是一个 `filter/kernel` 在空间上滑动，同一组 weights 对每个 patch 做 dot product，得到一个 `feature map`。浅层可能响应边缘，中层组合纹理和局部形状，深层再把它们组合成更抽象的 object feature。这不是预先写死“边缘是什么”，而是训练从数据中学出 filter。

### 先手算一个窗口

取一个 $4\times4$ 灰度输入和一个 $2\times2$ filter：

$$
X=\begin{bmatrix}1&1&0&0\\1&1&0&0\\0&0&1&1\\0&0&1&1\end{bmatrix},
\quad K=\begin{bmatrix}1&-1\\1&-1\end{bmatrix}.
$$

左上角 patch 与 filter 的逐元素乘积和为 $1-1+1-1=0$。窗口向右移动后，patch 改变，但 filter 没有改变；这就是 **parameter sharing**。一个位置产生一个 scalar，所有位置的 scalar 按空间位置排回去，才叫 feature map。

注意课件和深度学习框架常把这个操作叫 convolution，但实现通常是 **cross-correlation**：不先把 kernel 上下/左右翻转。入门时先把“窗口、逐元素乘法、求和、滑动”讲清楚，再处理命名差异。

### CNN 的工程含义

CNN 的优势不只是参数少，还在于 spatial locality 和 weight reuse：同一个 filter 可以被许多窗口重复使用，硬件可以尝试把 filter 或输入 tile 留在片上 SRAM，减少反复从 DRAM 搬运。代价是 feature map 仍可能很大，边界 padding、stride、pooling 和 layout 会影响有效计算量与内存访问。

因此我讲 CNN 时不会只问“卷积公式是什么”，还会追问：这个 output pixel 看到了原图多大区域？同一个 weight 被复用了多少次？stride 让输出缩小了多少？哪些空间细节因此丢失？

## 0.1 分层术语 roadmap

先按学习顺序认识词，详细词典放在本节后面查阅：

```text
Level 0  CNN 的空间对象（第 1–2 节）
  tensor -> channel -> kernel/filter -> convolution -> feature map
  -> receptive field -> stride/padding/pooling
      |
Level 1  CNN 的训练与泛化（第 3–7 节）
  activation -> forward/backprop -> normalization
  -> generalization -> overfitting -> regularization
      |
Level 2  CNN 的结构选择（第 8–10 节）
  skip connection -> residual mapping -> dense prediction
  -> encoder-decoder -> upsampling -> dilation -> Dice loss
      |
Level 3  AI chip 映射（CNN 分析之后）
  MAC -> FLOPs -> throughput/latency -> bandwidth
  -> memory hierarchy -> dataflow -> quantization
```

> 阅读规则：先学当前层的输入、动作和输出；后层的词只在需要时引入。下方词典是 `Reference glossary`，用于回查，不是开篇要背的清单。

### 0.1.1 Reference glossary（按主题查阅）

下面这张词典按“读图时会遇到的顺序”排列。不要只背中文翻译；每个词都要能回答“它接收什么、做什么、输出什么”。

### CNN 词典：从像素到 feature

| Term | 中文解释 | 读图时要问什么 |
|---|---|---|
| `tensor` | 多维数字数组；图像通常写成 $H\times W\times C$，训练时再加 `batch` 维度 | 每一维代表 height、width、channel 还是 batch？ |
| `channel` | 同一空间位置上的一组数；RGB 图像有 red、green、blue 三个 channels | 这个 layer 有多少个 input/output channels？ |
| `kernel / filter` | 一组会学习的 weights，在局部区域寻找一种 pattern | 它的形状是否覆盖所有 input channels？ |
| `convolution` | filter 在图像上滑动，对每个 patch 做逐元素乘法、求和并加 bias | 一个位置产生的 scalar 表示什么响应？ |
| `feature map` | 一个 filter 在所有位置产生的响应集合，也叫 `activation map` | 数值大在哪里？对应的 pattern 在哪里？ |
| `receptive field` | 某个 neuron 能“看到”的输入区域 | 这个 unit 依赖原图的多大范围？ |
| `stride` | filter 每次移动几格；越大，输出位置越少 | 输出尺寸为什么变小？ |
| `padding` | 在边缘外补数值，常见是 zero padding | 边缘像素有没有被充分处理？ |
| `pooling` | 在局部窗口聚合 activation；`max pooling` 取最大值 | 它丢掉了哪些精细位置，保留了什么强响应？ |
| `activation` | 非线性函数，例如 `ReLU`，让多层网络不只是一个线性变换 | 为什么卷积后不能只继续做线性计算？ |
| `equivariance` | 输入平移时，表示也按可预测方式平移 | pattern 的位置变化是否仍被保留？ |
| `invariance` | 输入发生指定变化时，最终输出基本不变 | 分类结果是否不在乎小幅位置变化？ |

### CNN dense prediction 词典：从“整张图是什么”到“每个像素是什么”

| Term | 中文解释 | 读图时要问什么 |
|---|---|---|
| `dense prediction` | 对图像中大量空间位置同时输出预测，而不是只给整张图一个 label | 输出是否仍保留 $H\times W$ 的空间位置？ |
| `semantic segmentation` | 给每个像素分配 object category；同类的不同实例不必区分 | 这是 pixel-wise classification，还是 instance segmentation？ |
| `encoder-decoder` | encoder 压缩并提取语义，decoder 恢复空间分辨率 | 哪些信息负责“是什么”，哪些信息负责“在哪里”？ |
| `upsampling` | 把低分辨率 feature map 变回更高分辨率 | 是固定插值、unpooling，还是 learnable operator？ |
| `transpose convolution` | 带可学习 weights 的上采样算子；不是普通 convolution 的简单逆运算 | 输出尺寸由 kernel、stride、padding 如何决定？ |
| `dilated / atrous convolution` | 在 kernel 元素之间留空洞，以较少下采样扩大 receptive field | 如何扩大 context 又保留更密的空间网格？ |
| `Dice loss` | 用区域重叠衡量预测 mask 与真实 mask 的一致程度 | 类别不平衡时，像素级 loss 是否被背景主导？ |

### Training 与 architecture 词典：从会算到会泛化

| Term | 中文解释 | 作用 |
|---|---|---|
| `forward pass` | 输入沿网络向前计算 prediction | 产生 activation、prediction 和 loss 所需的中间值 |
| `backpropagation` | 用 chain rule 从 loss 反向计算每个 parameter 的 gradient | 告诉每个 weight 应该向哪边改 |
| `mini-batch` | 一次共同处理的一小组 training samples | 在计算成本和 gradient 稳定性之间折中 |
| `epoch` | 整个 training set 被完整看过一次 | 用来描述训练进度，不是一次 update |
| `normalization` | 调整 activation 的数值尺度，例如 BatchNorm | 让 optimization 更稳定；不等于自动解决 overfitting |
| `generalization` | 把训练中学到的规律用于 unseen data | 真正关心的是 validation/test 上的表现 |
| `overfitting` | 记住 training set 的偶然细节，遇到新数据反而变差 | 需要 regularization、augmentation 或更合适的模型 |
| `regularization` | 有意限制模型对训练样本的过度依赖 | 包括 early stopping、weight decay、dropout 等 |
| `skip connection` | 把输入 $x$ 沿 shortcut 直接送到后面，与 $F(x)$ 相加 | ResNet 用它改善深层网络的 optimization |
| `residual mapping` | ResNet 学习的变化量 $F(x)$，输出为 $F(x)+x$ | 让 layer block 可以从 identity mapping 开始 |

### AI chip design 词典：把模型翻译成计算和数据搬运

| Term               | 中文解释                                             | 读模型/看硬件时要问什么                     |
| ------------------ | ------------------------------------------------ | -------------------------------- |
| `MAC`              | multiply-accumulate，一次乘法并把结果累加；卷积与矩阵乘法大量由 MAC 构成 | 每层需要多少 MAC？PE 每周期能做多少？           |
| `FLOPs`            | floating-point operations，浮点运算次数；常用来估计模型计算量      | 这是总运算量还是每秒性能 FLOP/s？精度格式是什么？     |
| `throughput`       | 单位时间处理多少图片、token 或样本                             | workload 的 batch 和并行度够不够？        |
| `latency`          | 一次输入从开始到结果返回的时间                                  | 是否有必须串行等待的步骤？                    |
| `bandwidth`        | 单位时间可搬运的数据量                                      | weights/activations 是否来得及送到计算单元？ |
| `memory hierarchy` | registers、片上 SRAM/cache、外部 DRAM 等不同容量与速度层级       | 数据能否留在更快、更近的存储中复用？               |
| `data movement`    | 在计算单元和不同存储层之间搬运 weights、activations、gradients    | 搬运成本是否比 arithmetic 本身更突出？        |
| `PE / MAC array`   | processing element / 乘加阵列；执行大量规则乘加的计算结构          | tensor shape 能否持续填满阵列？           |
| `dataflow`         | 计算时安排 weights、activations、partial sums 如何流动和复用   | 哪类数据固定、广播、累加或重用？                 |
| `quantization`     | 用较低位宽表示 weights/activations，例如从 FP32 到 INT8      | 精度、动态范围、带宽与吞吐之间怎样取舍？             |

这些是通用芯片工程词汇，不是本课 slides 对某款芯片的指标。它们帮助你从模型公式走到实现问题：先数运算，再看数据在哪里，最后判断计算阵列是否能被持续喂满。

> **学习纪律**：第一次出现的 term 先用上表建立直觉，再回到小节中的公式。看到 `shape` 时先写出每一维含义；看到 `gate` 时先问“它控制哪条信息流”。这两步比死记 architecture 名字更接近 AI 芯片工程中的真实读图方式。

## 0. CNN 的学习阶梯

如果你对机器学习还不熟，不要把 CNN 看成一堆新名词。按下面四层学习，每一层只解决一个问题：

1. **先回答“它为什么存在？”**：普通 fully connected network 处理图片时参数太多，而且看不见像素的局部空间关系；CNN 用局部连接和共享参数解决这两个问题。
2. **再回答“它每一步做什么？”**：`convolution` 用 filter 在图片上滑动，`activation` 加入非线性，`pooling` 压缩空间尺寸。
3. **再回答“它怎么学？”**：先 forward 得到 prediction 和 loss，再 backpropagation 得到 gradient，最后更新 parameters。
4. **最后回答“为什么还要这么多 architecture 和 regularization？”**：有的结构解决 optimization，有的方法减少 overfitting，有的设计减少计算量或扩大表达能力。

读完 CNN 部分，你不必马上记住所有 architecture 名字；至少应该能用自己的话回答：

- 一个 filter 为什么能在不同位置寻找同一种 pattern？
- `stride`、`padding`、`pooling` 分别怎样改变 feature map？
- training loss 下降而 validation error 上升时，发生了什么？
- ResNet 的 skip connection 为什么可能让深层网络更容易训练？

> **阅读方法**：每个小节先看“它解决什么问题”，再看图和最小例子，最后才看公式。公式看不懂时，先回到这三个动作：**看局部 → 做加权计算 → 把结果传给下一层**。

## 1. 为什么需要 Convolutional Neural Network？

两份 slides 的主线是：先说明 fully connected image model 的问题，再介绍 convolution、activation、pooling 和 CNN 的数学性质；随后解释 CNN training、regularization，以及 LeNet 到现代 CNN architecture 的演化。

### 1.1 全连接网络处理图像的问题

一张 RGB 图像可以表示为一个三维数组：高度 `H`、宽度 `W`、颜色通道数 `C=3`。例如 $200\times200\times3$ 的图像有 $120{,}000$ 个输入数值。

如果第一层使用 fully connected layer，并连接到 1000 个 hidden units，参数量大约是：

$$
200\times200\times3\times1000=120{,}000{,}000
$$

这带来三个问题：

- **参数太多**：内存、计算量和过拟合风险都很高。
- **忽略二维结构**：相邻像素通常比远处像素更相关，但 fully connected layer 把它们当成普通的一维向量处理。
- **位置变化不鲁棒**：物体稍微平移后，原来的像素位置全部改变，网络可能把同一个物体当成不同模式。

![[assets/2 CNN and RNN/CNN1-03-motivation.png]]

> **来自 CNN 1 slide 3｜截图导航**：先看输入图像的 $200\times200\times3$，再看全连接第一层的 `120 million` 参数，最后看“物体稍微平移怎么办”的问题。这个页面适合在读 1.1 时回看，因为它把 CNN 的动机压缩成一个工程约束：输入很大、像素有空间结构、目标位置会变化。

### 1.2 CNN 的核心设计思想

**局部动机**：图像分类只需要一个类别，但驾驶感知、医学影像或机器人还要知道物体边界和位置；如果网络过早把空间位置压成一个向量，边界细节就找不回来了。后面的 dense prediction 章节会用 encoder-decoder、skip connections 和 upsampling 处理这个限制。

CNN 利用图像的 spatial topology，并把预期的视觉变化纳入结构设计：

1. **Local connectivity**：一个 hidden unit 只连接输入图像的一个小区域（patch），称为它的 **local receptive field**。
2. **Parameter sharing**：同一个 feature map 中不同空间位置的 unit 使用同一组 weights；同一个 pattern 在不同位置出现时可以被同一个 filter 检测到。
3. **Pooling / subsampling**：在局部邻域聚合 activation，减少空间分辨率并忽略不重要的精细位置。
4. **交替堆叠 feature extraction 与 pooling**：浅层学习边缘/纹理，中层组合成局部形状，深层形成更抽象的 semantic concepts。这就是一种 **representation learning**，类似从像素到概念的 artificial visual pathway。

这里的“忽略位置”需要区分两个术语：

- **Equivariance**：输入发生变换，输出表示也以可预测的相同方式变换。
- **Invariance**：输入发生某种变换，输出表示基本不变。

CNN 通常希望卷积层对 translation 保持 equivariance，而分类结果对小幅 translation 具有 robustness（接近 invariance）。

## 2. Convolution layers

### 2.1 一个 filter 如何产生一个数

> **External course-material extension — 术语边界**：在数学定义中，convolution 会先翻转 kernel，再滑动相乘；主流深度学习库（包括 PyTorch `Conv2d`）的实现通常不翻转 kernel，严格说是 cross-correlation（互相关）。神经网络文献和 API 仍普遍把这个层叫 convolution，因为可学习 kernel 时翻转可被参数重新吸收，表达能力不变。手算时要遵循课件/框架展示的权重方向，不能把图像处理里的固定卷积核定义直接套过来。来源：Stanford CS231n, [Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/), section “Convolution Layer”；PyTorch, [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)，访问日期 2026-10-22。

**Motivation**：如果每个位置都单独学习一套参数，同一种边缘移动一格就要重新学一次；全连接层也会把远近像素一视同仁。CNN 选择局部连接和参数共享，让同一个小型 detector 在整张图上重复使用。收益是参数少且保留位置结构；代价是滑动窗口会产生大量 activation 和 memory traffic。

假设输入是 $32\times32\times3$ 的 RGB 图像，使用一个 $5\times5\times3$ 的 filter $\mathbf{w}$。filter 的第三维必须覆盖所有输入 channels。

在图像上的某个位置取出同样大小的 patch $\mathbf{x}$，将 patch 展平成长度 $5\times5\times3=75$ 的向量，然后做 dot product：

$$
z=\mathbf{w}^{T}\mathbf{x}+b
$$

其中 $b$ 是 bias。这个结果是一个 scalar，表示该 filter 在这个位置检测到目标 pattern 的强弱。随后把 filter 在所有可能位置滑动（slide），每个位置都得到一个数，所有数排列成一个 **activation map / feature map**。

> 注：深度学习实现中常把 cross-correlation 也称为 convolution；这里重点是“局部 dot product + 滑动”，不必纠结数学信号处理里是否翻转 kernel。

下面这张图把一次最简单的滑动过程抽象出来：同一个 filter 在不同位置重复使用，每个位置产生一个 output scalar，最后这些 scalar 组成 feature map。

<svg viewBox="0 0 920 150" role="img" aria-label="Convolution extracts a patch, computes a weighted sum, and places one output value in a feature map" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="arrow-conv" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#64748b"/></marker></defs>
<rect width="920" height="150" rx="12" fill="#f8fafc"/>
<g font-family="Arial, sans-serif" font-size="14" text-anchor="middle">
<g fill="#dbeafe" stroke="#2563eb" stroke-width="2"><rect x="20" y="48" width="120" height="52" rx="10"/><text x="80" y="70" fill="#1e3a8a">Input image</text><text x="80" y="88" fill="#1e3a8a">H x W</text></g>
<g fill="#ffffff" stroke="#64748b" stroke-width="1.5"><rect x="175" y="48" width="135" height="52" rx="10"/><text x="242" y="70" fill="#1f2937">Local patch</text><text x="242" y="88" fill="#1f2937">K x K</text><rect x="345" y="48" width="145" height="52" rx="10"/><text x="417" y="70" fill="#1f2937">Multiply</text><text x="417" y="88" fill="#1f2937">and sum</text><rect x="525" y="48" width="105" height="52" rx="10"/><text x="577" y="70" fill="#1f2937">Add bias</text></g>
<g fill="#dcfce7" stroke="#16a34a" stroke-width="2"><rect x="665" y="48" width="115" height="52" rx="10"/><text x="722" y="70" fill="#14532d">One output</text><text x="722" y="88" fill="#14532d">scalar</text><rect x="815" y="48" width="85" height="52" rx="10"/><text x="857" y="70" fill="#14532d">Feature</text><text x="857" y="88" fill="#14532d">map</text></g>
<g stroke="#64748b" stroke-width="2" marker-end="url(#arrow-conv)"><line x1="140" y1="74" x2="175" y2="74"/><line x1="310" y1="74" x2="345" y2="74"/><line x1="490" y1="74" x2="525" y2="74"/><line x1="630" y1="74" x2="665" y2="74"/><line x1="780" y1="74" x2="815" y2="74"/></g>
<path d="M857 105 C857 135 242 135 242 105" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#arrow-conv)"/><text x="550" y="137" fill="#64748b">slide the same filter to the next patch</text>
</g></svg>

图中 `Slide filter to next position` 是 convolution 的关键：filter 的 weights 不变，只改变它读取的 input patch。

![[assets/2 CNN and RNN/CNN1-14.png]]

> **来自 CNN 1 slide 14｜看图重点**：输入是 `32×32×3`，所以 filter 也必须是 `5×5×3`，第三个维度覆盖 RGB 三个 channels。filter 在空间位置上滑动，每个位置产生一个 scalar；这些 scalar 合在一起才是 feature map。

### 2.1.1 手算一个最小例子

> **Background / worked example**：下面的 $4\times4$ 灰度矩阵不是 slides 中的原始实验结果，而是帮助理解 convolution 运算的教学例子。

先暂时忽略 RGB channel，只看一张 $4\times4$ 的灰度图。把较亮的像素写成 1，较暗的像素写成 0：

$$
X=\begin{bmatrix}
1&1&0&0\\
1&1&0&0\\
0&0&1&1\\
0&0&1&1
\end{bmatrix}
$$

我们设计一个 $2\times2$ filter：

$$
K=\begin{bmatrix}
1&-1\\
1&-1
\end{bmatrix},\qquad b=0
$$

这个 filter 的直觉是：左边的值乘以正数、右边的值乘以负数。因此：

- 左亮右暗时，结果偏正；
- 左暗右亮时，结果偏负；
- 左右差不多时，结果接近 0。

所以它像一个非常简单的 **vertical edge detector**，用来检测左右方向的亮度变化。实际 CNN 不会由我们手工指定这种 filter，而是从训练数据中学出来。

#### 第一步：放在左上角

filter 覆盖输入的左上角 patch：

$$
P_{1,1}=\begin{bmatrix}1&1\\1&1\end{bmatrix}
$$

逐元素相乘，再把所有结果相加，最后加 bias：

$$
z_{1,1}=1\cdot1+1\cdot(-1)+1\cdot1+1\cdot(-1)+0=0
$$

这里得到的 0 表示：这个 $2\times2$ 小区域左右没有明显差异。

#### 第二步：向右移动一个像素

stride 为 1，所以 filter 向右移动一格，覆盖：

$$
P_{1,2}=\begin{bmatrix}1&0\\1&0\end{bmatrix}
$$

计算：

$$
z_{1,2}=1\cdot1+0\cdot(-1)+1\cdot1+0\cdot(-1)=2
$$

这一次左边是 1、右边是 0，所以 filter 的响应很强，得到正数 2。这个数字不是类别，而是“这个位置有多像该 filter 正在寻找的 pattern”。

#### 第三步：继续滑动

filter 继续向右、再回到下一行向右滑动。因为没有 padding，$4\times4$ 输入上只能放置 $3\times3$ 个 $2\times2$ patch，所以最终得到 $3\times3$ 的 feature map：

$$
Z=X*K=\begin{bmatrix}
0&2&0\\
0&0&0\\
0&-2&0
\end{bmatrix}
$$

其中：

- 正数表示“左亮右暗”的 pattern；
- 负数表示相反方向的 pattern；
- 接近 0 表示没有明显匹配。

如果后面接 ReLU：

$$
\operatorname{ReLU}(Z)=\max(0,Z)=\begin{bmatrix}
0&2&0\\
0&0&0\\
0&0&0
\end{bmatrix}
$$

负响应会被置为 0，于是这个 channel 主要保留 filter 关心的正向 pattern。

#### 这个例子中每一步到底在做什么？

1. **选一个小窗口**：只看输入的一小块，而不是一次看整张图（local connectivity）。
2. **逐元素相乘**：比较 patch 和 filter 在每个位置上的数值是否“同意”。相同方向的权重贡献正值，相反方向贡献负值。
3. **求和**：把局部比较结果压缩成一个 scalar，表示当前位置的匹配程度。
4. **加 bias**：给这个 neuron 一个可学习的整体偏移。
5. **滑动并重复**：用完全相同的 filter 检查每个位置（parameter sharing）。
6. **排列成 feature map**：把每个位置的响应放回原来的空间布局，因此我们既知道“检测到了什么”，也知道“大概在哪里”。
7. **经过 activation**：例如 ReLU 引入非线性，让多层 CNN 能学习复杂 pattern。

### 2.1.2 从灰度图到 RGB 图像

> **Background**：slides 使用 RGB 图像；这里把灰度例子扩展到 3 个 channels，帮助理解 filter 的第三个维度。

真实 RGB 输入有 3 个 channels。此时一个 $2\times2$ filter 不是普通的 $2\times2$ 矩阵，而是：

$$
K\in\mathbb{R}^{2\times2\times3}
$$

在一个位置，输入 patch 也有 $2\times2\times3=12$ 个数。计算时会把三个 channel 的乘积全部加起来，再加一个 bias：

$$
z=\sum_{c=1}^{3}\sum_{u=1}^{2}\sum_{v=1}^{2}
K_{u,v,c}X_{i+u,j+v,c}+b
$$

所以一个 filter 在一个位置仍然只产生一个 scalar，但它可以同时利用 red、green、blue 三个 channel 的信息。若有 64 个不同 filters，就会得到 64 个 feature maps，输出形状为 $H_{out}\times W_{out}\times64$。

### 2.1.3 三个最容易混淆的词

- **Kernel / filter**：一组可学习的 weights，例如上面的 $K$；它负责寻找一种 pattern。
- **Neuron / output unit**：filter 放在某一个具体位置后，做完 dot product 和 bias 得到的一个 scalar。
- **Feature map / activation map**：同一个 filter 在所有位置得到的 scalar 集合，记录该 pattern 在哪里出现、响应有多强。

因此可以用一句话记忆：**一个 filter 在不同位置产生许多 neurons，这些 neurons 共同组成一个 feature map。**

### 2.1.4 Stride：filter 每次移动几格？

#### 交互：让 filter 真正滑起来

打开 [Convolution window explorer](assets/2 CNN and RNN/interactive/convolution-window-explorer.html)，点击“下一个位置”，先观察红框覆盖的 patch 和 feature map 中对应的响应，再切换 stride 与 padding。这是一个教学插图，使用本节的 4x4 输入和 2x2 filter；它不是课件 benchmark。

Obsidian 原生打开：[[assets/2 CNN and RNN/interactive/convolution-window-explorer.html|打开 Convolution window 互动页面]]

> [!tip]- Obsidian 内原生练习（不需要 HTML）
> 1. 用本节的 4x4 输入和 2x2 filter，先标出左上角 patch。
> 2. 把 filter 向右移动一格，写出新的 patch 和响应值。
> 3. 将 stride 从 1 改成 2，预测输出位置会变多还是变少。
> 4. 上面的 HTML 链接是可选的独立互动页；如果按钮不动，使用本节已有的手算过程和尺寸公式。

没有互动时，记住同一条静态规律：N_out=floor((N+2P-K)/S)+1。stride 改变窗口跳跃距离，padding 改变边界外可见范围；它们不改变 filter 本身的 weights。

> **老师讲解重点**：先看红框覆盖了哪些输入数字，再看同一个位置在 feature map 中产生了哪个响应；最后只改变一个变量。你要把“窗口移动”说成“同一个 filter 在不同空间位置复用”，把“输出变小”说成“stride 减少了被计算的位置”，而不是笼统地说“图片被压缩了”。

**Obsidian 内互动（JS Engine）**

点击“下一个窗口”，观察同一个 filter 如何移动，以及输出尺寸如何变化。

```js-engine
const box = container.createDiv();
box.createEl("b", { text: "老师讲解：同一个 filter 在空间位置间复用" });
const input = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]];
const kernel = [[1, -1], [1, -1]];
const controls = box.createDiv();
const stride = controls.createEl("select");
stride.createEl("option", { text: "stride = 1", value: "1" });
stride.createEl("option", { text: "stride = 2", value: "2" });
const previous = controls.createEl("button", { text: "上一个窗口" });
const next = controls.createEl("button", { text: "下一个窗口" });
const result = box.createEl("pre");
let position = 0;
function show() {
  const step = Number(stride.value), n = Math.floor((4 - 2) / step) + 1, total = n * n;
  position = Math.min(position, total - 1);
  const row = Math.floor(position / n) * step, col = (position % n) * step;
  const patch = input.slice(row, row + 2).map(values => values.slice(col, col + 2));
  const products = patch.flatMap((values, i) => values.map((value, j) => value + "×" + kernel[i][j]));
  const output = [];
  for (let i = 0; i < n; i++) {
    const values = [];
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let u = 0; u < 2; u++) for (let v = 0; v < 2; v++) sum += input[i * step + u][j * step + v] * kernel[u][v];
      values.push(sum);
    }
    output.push(values.join("  "));
  }
  result.textContent = "窗口 " + (position + 1) + "/" + total + "；输出位置 [" + Math.floor(position / n) + "," + (position % n) + "]\npatch=" + JSON.stringify(patch) +
    "\n逐元素乘积：" + products.join(" + ") + " = " + products.reduce((sum, item) => { const pair = item.split("×").map(Number); return sum + pair[0] * pair[1]; }, 0) +
    "\nfeature map：\n" + output.join("\n") + "\n输出尺寸：" + n + "×" + n;
}
previous.addEventListener("click", () => { position = Math.max(0, position - 1); show(); });
next.addEventListener("click", () => { position = Math.min(position + 1, Math.pow(Math.floor((4 - 2) / Number(stride.value)) + 1, 2) - 1); show(); });
stride.addEventListener("change", () => { position = 0; show(); }); show();
return;
```

**Stride** 是 filter 在输入上滑动时，每次移动的步长。

仍使用上面的 $4\times4$ 输入和 $2\times2$ filter，并先设 `padding=0`：

- `stride=1`：每次向右移动 1 格，走到行末后向下移动 1 格。可以放置 $3\times3$ 个 patch，所以输出是 $3\times3$。
- `stride=2`：每次向右移动 2 格，向下也移动 2 格。只能放置左上、右上、左下、右下四个 patch，所以输出是 $2\times2$。

对于任意一维空间尺寸，输出大小为：

$$
N_{out}=\left\lfloor\frac{N+2P-K}{S}\right\rfloor+1
$$

其中 $N$ 是输入大小，$K$ 是 kernel size，$P$ 是 padding 大小，$S$ 是 stride。代入 $N=4,K=2,P=0$：

$$
S=1:\quad \left\lfloor\frac{4-2}{1}\right\rfloor+1=3
$$
$$
S=2:\quad \left\lfloor\frac{4-2}{2}\right\rfloor+1=2
$$

所以 stride 越大，filter 检查的位置越少，输出空间尺寸越小，计算也越少；但代价是可能跳过一些细节。

### 2.1.5 Padding：在边缘外面补什么？

如果不 padding，$2\times2$ filter 不能以输入边缘上的每个像素为中心，而且每经过一层 convolution，空间尺寸通常都会变小。**Padding** 就是在输入的边缘外补上一圈或多圈数值。

最常见的是 **zero padding**：补 0。例如原来的 $4\times4$ 输入使用 `padding=1` 后变成 $6\times6$：

$$
\begin{bmatrix}
0&0&0&0&0&0\\
0&1&1&0&0&0\\
0&1&1&0&0&0\\
0&0&0&1&1&0\\
0&0&0&1&1&0\\
0&0&0&0&0&0
\end{bmatrix}
$$

现在 $2\times2$ filter 可以在更大的范围内滑动。对 $N=4,K=2,P=1,S=1$：

$$
N_{out}=\frac{4+2(1)-2}{1}+1=5
$$

因此输出是 $5\times5$。注意，padding=1 并不意味着输出一定和输入一样大；输出大小还取决于 kernel size 和 stride。

对常见的奇数 kernel，例如 $K=3$、`stride=1`：

$$
P=1\quad\Longrightarrow\quad N_{out}=N
$$

这叫 **same convolution**，因为输入和输出的空间尺寸相同。相反，`padding=0` 通常称为 **valid convolution**，表示只使用完全落在输入内部的 patch。

### 2.1.6 Padding 和 stride 的直觉对比

| 设置 | filter 的行为 | 输出空间尺寸 | 主要效果 |
|---|---|---:|---|
| `padding=0` | 不看输入外部 | 通常变小 | 保留真实输入区域，边缘信息较少 |
| `padding=1` | 边缘外补一圈 0 | 可以保持或扩大尺寸 | 让边缘像素也被充分处理 |
| `stride=1` | 每个位置都检查 | 较大 | 保留更多空间细节 |
| `stride=2` | 隔一个位置检查 | 约缩小一半 | 同时完成下采样，计算更少 |

一个简单记忆方法是：**padding 决定“边界外补多少”，stride 决定“每次跳多远”。** 它们共同决定 filter 能放置多少个位置，也就决定 feature map 的大小。

![[assets/2 CNN and RNN/CNN1-26.png]]

> **来自 CNN 1 slide 26｜看图重点**：输入 `7×7`、`3×3 filter`、`stride=1`、边缘补 1 圈 0 时，输出仍是 `7×7`。灰色区域是补出来的边界；它让原本靠近边缘的像素也能参与完整的卷积窗口。

#### 用 NumPy 复现 convolution、padding 和 stride

> **Background / code example**：这是一个最小教学实现，用来对应上面的手算过程。它只处理单 channel、没有 batch；真实框架还会处理 RGB channels、多个 filters 和 GPU 计算。

```python
import numpy as np


def conv2d(x, kernel, bias=0.0, stride=1, padding=0):
    # Add zeros around the input boundary.
    x_pad = np.pad(
        x,
        ((padding, padding), (padding, padding)),
        mode="constant",
    )

    height, width = x_pad.shape
    kernel_height, kernel_width = kernel.shape
    out_height = (height - kernel_height) // stride + 1
    out_width = (width - kernel_width) // stride + 1
    output = np.zeros((out_height, out_width))

    for i in range(out_height):
        for j in range(out_width):
            row = i * stride
            col = j * stride
            patch = x_pad[row:row + kernel_height, col:col + kernel_width]
            output[i, j] = np.sum(patch * kernel) + bias

    return output


x = np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 1],
])
kernel = np.array([
    [1, -1],
    [1, -1],
])

print(conv2d(x, kernel, stride=1, padding=0))
# [[ 0.  2.  0.]
#  [ 0.  0.  0.]
#  [ 0. -2.  0.]]

print(conv2d(x, kernel, stride=2, padding=0).shape)
# (2, 2)

print(conv2d(x, kernel, stride=1, padding=1).shape)
# (5, 5)
```

代码中最重要的三行分别对应三个概念：`patch` 是当前 receptive field，`patch * kernel` 是逐元素相乘，`np.sum(...) + bias` 是产生一个 output neuron。外层两个 `for` 循环负责按照 `stride` 移动 filter。

### 2.2 为什么 parameter sharing 有效

如果每个位置都学习一套不同的 weights，参数量会随图像大小增长。CNN 让同一 feature map 的所有位置共享同一个 filter：

- 参数量只取决于 filter 大小和输入 channels，而不是图像的 $H,W$。
- 同一条边缘或纹理无论出现在哪里，都可被同一个 filter 检测。
- 输出仍保留 pattern 的空间位置，因此后续层可以组合空间结构。

一个 convolution layer 通常有多个 filters。若输入 channel 数为 $C_{in}$，有 $C_{out}$ 个大小为 $K\times K$ 的 filters，则每个 filter 的形状是 $K\times K\times C_{in}$，输出有 $C_{out}$ 个 feature maps（即 output channels）。

### 2.3 1x1 convolution

$1\times1$ convolution 在每个空间位置单独作用，但会跨越所有 input channels。它等价于对每个像素位置的 channel vector 做一个小型 fully connected transformation，因此可以：

- 减少或增加 channel dimensionality；
- 在不改变空间分辨率的情况下做 feature pooling / mixing；
- 用于 Network-in-Network、GoogleNet 等结构。

## 3. Convolution 的尺寸与复杂度

**Motivation**：模型设计不能只问“能不能识别”，还要问输出 feature map 多大、参数多少、每次 inference 要做多少乘加。尺寸公式把 architecture choice 翻译成 memory footprint 和 compute cost。

设输入空间尺寸为 $N\times N$，kernel size 为 $K$，padding 为 $P$，stride 为 $S$。每个空间维度的输出尺寸为：

$$
N_{out}=\left\lfloor\frac{N+2P-K}{S}\right\rfloor+1
$$

如果没有 padding（`valid convolution`）、stride 为 1，则 $N_{out}=N-K+1$；输出会变小。

如果选择合适的 zero padding（`same` 的常见设置是 $S=1, P=(K-1)/2$，例如奇数 $K$），可以保持空间尺寸不变。padding 也能处理公式中的非整数情况，或主动控制输出大小。

### 3.1 参数量与计算量

若输入 channels 为 $C_{in}$，输出 channels（filters 数量）为 $C_{out}$：

$$
\text{参数量}=K^2C_{in}C_{out}+C_{out}
$$

![[assets/2 CNN and RNN/CNN1-26-complexity.png]]

> **来自 CNN 1 slide 26｜截图导航**：这页适合在尺寸公式之后看，重点是为什么 `padding` 不只是“补边”，还可以控制输出尺寸；输出不能整除时，实际实现会按框架规则处理边界。

最后一项是每个 filter 一个 bias。注意参数量与输入图像的 $N$ 无关，这是 parameter sharing 带来的关键优势。

输出 activation map 的形状为 $N_{out}\times N_{out}\times C_{out}$。每个输出元素大约需要 $K^2C_{in}$ 次乘加，因此一层的计算量与输出空间位置数、kernel 面积、输入 channels 和 filters 数量共同相关。

#### 一个尺寸和参数量算例

> **Background / worked example**：下面的数字用于演示尺寸和参数量公式，不代表某个具体 CNN 的实验配置。

假设输入是 $32\times32\times3$，使用 $16$ 个 $5\times5\times3$ filters，`stride=1`、`padding=0`：

$$
N_{out}=\frac{32-5}{1}+1=28
$$

因此输出形状是 $28\times28\times16$。每个 filter 有 $5\times5\times3=75$ 个 weights 和 1 个 bias，整层参数量为：

$$
(5\times5\times3+1)\times16=1{,}216
$$

注意输出有 $28\times28\times16=12{,}544$ 个 activation，但它们并不各自拥有一套新 weights；同一个 filter 在不同位置重复使用。这正是 parameter sharing 让参数量远小于 fully connected layer 的地方。

#### 从一个 output element 到整层 MAC 与存储

上面的例子还可以继续向硬件展开。一个 output element 需要 $5\times5\times3=75$ 次乘法和大约 75 次加法；整层共有 $28\times28\times16=12{,}544$ 个 output elements，因此乘加规模约为：

$$
12{,}544\times75=940{,}800\text{ MACs}
$$

这里的 MAC（multiply-accumulate）是“乘法后累加一次”的基本工作单位。参数量只有 1,216，但 activation 计算量接近百万 MAC，说明**参数少不等于计算和带宽都少**。如果用 FP16 保存 output，仅 output activation 就有 $12{,}544\times2\approx25$ KiB；实际 layer 还要读取 input feature map、weights，并在 training 时保存更多 intermediate activations。

这也是 CNN accelerator 需要 data reuse 的原因：同一组 filter weights 会被许多空间位置重复使用，同一个 input activation 也会参与多个相邻 output。硬件 dataflow（例如 weight-stationary、output-stationary 或 row-stationary）本质上是在决定哪类数据尽量留在近端 buffer，减少 DRAM traffic。

### 3.2 Stride 与 receptive field

- `stride=1`：filter 每次移动一个像素，保留更多空间细节。
- `stride>1`：跳着移动，输出更小，也同时完成下采样。

对于 kernel size 为 $K$ 的单个 convolution output element，它只依赖输入中的 $K\times K$ 区域，这个区域叫 **receptive field**。堆叠多层 convolution 后，深层 unit 的 receptive field 会覆盖输入中更大的区域，因此可以从局部边缘逐渐组合出整体形状。

## 4. Activation functions

**Motivation**：如果每层只有 linear transform，多层仍可合并成一个 linear model，无法表达复杂 pattern；如果 activation 饱和，gradient 又会变小。选择 activation 是表达能力、训练稳定性和硬件算力之间的取舍。

卷积得到的 $z$ 如果层层只做线性变换，多层仍可合并为一次线性变换，无法表达复杂关系。因此 convolution layer 后通常加入 nonlinear activation $a=f(z)$。

### 4.1 Sigmoid 与 Tanh

$$
\operatorname{sigmoid}(z)=\frac{1}{1+e^{-z}},\qquad
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$

Sigmoid 输出在 $(0,1)$，Tanh 输出在 $(-1,1)$。当输入绝对值很大时，两者都会 saturation，导数接近 0，深层网络中容易产生 vanishing gradients。slides 提到 Tanh 仍常见于 recurrent neural networks，例如 LSTM、GRU。

### 4.2 ReLU 及其变体

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

ReLU 的正半轴梯度为 1，计算简单，通常比 sigmoid/tanh 更容易训练。但当 unit 长期落在负半轴时，梯度为 0，可能出现 “dead ReLU”。

Leaky ReLU 给负半轴一个小斜率 $\alpha$：

$$
\operatorname{LeakyReLU}(z)=\begin{cases}z,&z>0\\\alpha z,&z\le0\end{cases}
$$

ELU 在负半轴使用平滑的指数形式，试图保留非零梯度并改善均值偏移。slides 还列出较平滑的 Swish 和 Mish 作为现代 activation functions 的例子。

下面的连续 SVG 曲线是一个理想化的教学比较，不是 slides 中的实验结果。它帮助你看到：`Sigmoid` 和 `Tanh` 在两端会趋于饱和，`ReLU` 在负半轴直接为 0。

![[assets/1 Basic neural networks/activation-functions-cnn.svg|601]]

图中横轴是输入 $z$，纵轴是 activation output。蓝色的 Sigmoid 和红色的 Tanh 在两端变平，说明导数接近 0；绿色的 ReLU 在 $z<0$ 时保持 0，在 $z>0$ 时近似线性增长。

### 4.3 如何选择

- Internal layers of CNN：ReLU 及其变体是常见默认选择。
- Output layer：取决于 prediction task 和 loss function。`Softmax`、logits、binary classification 的 sigmoid，以及 output activation 与 loss 的配套关系，见 [[1 Basic neural networks]] 的 Softmax 和 Objective 章节。这里需要记住的是：CNN 的 convolution blocks 负责提取 visual features，最后的 output layer 才把 features 转换成具体任务的 prediction。

## 5. Pooling layers

**Motivation**：经过多层卷积后，feature map 仍可能很大，后续计算和存储成本会快速累积；同时分类常不需要知道强 feature 精确落在哪个相邻像素。Pooling 用局部聚合降低空间尺寸，但会丢失精确位置，因此 dense prediction 任务通常要谨慎使用或用 skip connection 恢复细节。

Pooling 在每个 activation map 上独立操作，用局部邻域聚合来降低 spatial size。低分辨率意味着更少细节、更少后续计算，并对小幅形变更鲁棒。

![[assets/2 CNN and RNN/pooling-max-window.svg]]

> **自制教学图｜读图方法**：先看橙色窗口如何覆盖输入，再看同一窗口只留下一个最大值。这个关系比直接背 `max` 公式更重要：pooling 把“附近是否出现强 feature”保留下来，同时把精确坐标变粗。

![[assets/2 CNN and RNN/CNN selected/CNN1-47.png]]

> **来自 CNN 1 slide 47｜看图重点**：pooling 把空间尺寸变小，例如从 `224×224×64` 变成 `112×112×64`；空间信息减少了，但 channel 数在这个例子中不变。它的作用不是学习新的 filter，而是压缩和保留局部强响应。

### 5.1 Max pooling

给定窗口区域 $\mathcal{N}$：

$$
y_{i,j}=\max_{(u,v)\in\mathcal{N}}x_{i+u,j+v}
$$

它保留局部最强 activation，可理解为“这个 feature 在附近是否出现”。Average pooling 则取平均。

Pooling layer 没有 learnable parameters；它只由 window size 和 stride 等超参数决定。代价是精确位置信息被丢弃，因此现代 CNN 也会用 strided convolution 或其他 downsampling design 替代部分 pooling。

#### 交互：逐窗观察 max pooling

在 Obsidian 中调整 window 与 stride，再用“下一个窗口”逐步移动。观察输入中的橙色区域如何对应到输出中的蓝色格子，并核对窗口内最大值。这个 4×4 activation map 是教学示例，不是课件测量数据。

**Obsidian JS Engine 互动**

```js-engine
const box = container.createDiv();
box.createEl("b", { text: "老师讲解：max pooling 保留局部最大值，不保留精确坐标" });
const controls = box.createDiv();
const windowSelect = controls.createEl("select");
windowSelect.createEl("option", { text: "2 × 2 window", value: "2" });
windowSelect.createEl("option", { text: "3 × 3 window", value: "3" });
const strideSelect = controls.createEl("select");
strideSelect.createEl("option", { text: "stride = 1", value: "1" });
strideSelect.createEl("option", { text: "stride = 2", value: "2" });
const previous = controls.createEl("button", { text: "上一个" });
const next = controls.createEl("button", { text: "下一个" });
const output = box.createEl("pre");
const input = [[1, 3, 2, 0], [4, 6, 1, 2], [5, 2, 7, 8], [3, 1, 9, 4]];
let position = 0;
function render() {
  const size = Number(windowSelect.value), stride = Number(strideSelect.value);
  const n = Math.floor((4 - size) / stride) + 1, total = n * n;
  position = Math.min(position, total - 1);
  const row = Math.floor(position / n), col = position % n;
  const out = [];
  for (let i = 0; i < n; i++) {
    const values = [];
    for (let j = 0; j < n; j++) {
      let max = -Infinity;
      for (let u = 0; u < size; u++) for (let v = 0; v < size; v++) max = Math.max(max, input[i * stride + u][j * stride + v]);
      values.push(max);
    }
    out.push(values.join("  "));
  }
  const patchValues = input.slice(row * stride, row * stride + size).map(values => values.slice(col * stride, col * stride + size));
  const patch = patchValues.map(values => values.join(" ")).join(" / ");
  output.textContent = "输入：\n" + input.map((values, i) => values.map((value, j) => i >= row * stride && i < row * stride + size && j >= col * stride && j < col * stride + size ? "[" + value + "]" : " " + value + " ").join(" ")).join("\n") +
    "\n\n输出：\n" + out.join("\n") + "\n\n窗口 " + (position + 1) + "/" + total + "；当前 patch = " + patch + "；max = " + Math.max(...patchValues.flat()) + "；输出尺寸 = " + n + " × " + n +
    "\n解释：每个输出只留最大 activation；空间位置更粗，但 feature map 更小。";
}
next.addEventListener("click", () => { position = Math.min(position + 1, Math.pow(Math.floor((4 - Number(windowSelect.value)) / Number(strideSelect.value)) + 1, 2) - 1); render(); });
previous.addEventListener("click", () => { position = Math.max(0, position - 1); render(); });
windowSelect.addEventListener("change", () => { position = 0; render(); });
strideSelect.addEventListener("change", () => { position = 0; render(); });
render();
```

无插件时可打开 [max pooling 与 receptive field explorer](assets/2 CNN and RNN/interactive/pooling-receptive-field-explorer.html)。静态结论不依赖互动：`2 × 2`、`stride=2` 会把 `4 × 4` 压成 `2 × 2`，结果为 `[[6, 2], [5, 9]]`；每个输出只保留相应窗口的最大值。这里“附近有强响应”比“强响应的精确坐标”更重要，因此下采样省计算却丢位置细节。课件 CNN 1 slide 47 展示了 feature map 下采样前后的尺寸变化；注意课件图示是 `224 × 224 × 64 → 112 × 112 × 64`，不是本互动的小矩阵算例。

> **老师讲解重点**：pooling 不是“学习出来的 filter”。它没有 weights，只按窗口做聚合；它保留局部最强响应，却把精确坐标变粗。你要同时说出收益（更少的空间计算和存储）与代价（位置细节损失）。

#### Max pooling 手算例子

> **Background / worked example**：下面用小矩阵演示 slides 中的 max pooling；真实 CNN 的窗口会作用在每个 feature map 上。

对下面的 $4\times4$ activation map 使用 $2\times2$ window、`stride=2`：

$$
X=\begin{bmatrix}
1&3&2&0\\
4&6&1&2\\
5&2&7&8\\
3&1&9&4
\end{bmatrix}
$$

四个窗口分别取最大值：

$$
\operatorname{MaxPool}(X)=\begin{bmatrix}
\max(1,3,4,6)&\max(2,0,1,2)\\
\max(5,2,3,1)&\max(7,8,9,4)
\end{bmatrix}
=\begin{bmatrix}6&2\\5&9\end{bmatrix}
$$

这里没有学习任何新 weight；pooling 只是把每个局部区域压缩成一个代表值。它保留“附近有没有强烈 feature”的信息，但不再精确记录这个最大值位于窗口的哪个像素。

对应的 NumPy 版本如下。与 convolution 不同，函数中没有 kernel weights，也没有 bias 或 gradient update：

```python
def max_pool2d(x, window_size=2, stride=2):
    height, width = x.shape
    out_height = (height - window_size) // stride + 1
    out_width = (width - window_size) // stride + 1
    output = np.zeros((out_height, out_width))

    for i in range(out_height):
        for j in range(out_width):
            row = i * stride
            col = j * stride
            window = x[row:row + window_size, col:col + window_size]
            output[i, j] = np.max(window)

    return output
```

## 6. CNN 的数学性质

**Motivation**：同一个 object 发生小幅平移时，底层 feature 的位置应该跟着变，但分类结果不应完全改变。`equivariance` 保留可预测的空间变化，`invariance` 则在最终任务层面忽略不重要的变化；两者混淆会导致错误的 architecture 设计。

令图像表示函数为 $\Phi$，输入变换为 $T$。

### 6.1 Equivariance

若存在输出空间中的对应变换 $T'$，使得：

$$
\Phi(Tx)=T'\Phi(x)
$$

则称 $\Phi$ 对 $T$ equivariant。普通 convolution 对 translation 是 equivariant：输入图像平移，feature map 也相应平移。

### 6.2 Invariance

若：

$$
\Phi(Tx)=\Phi(x)
$$

则称表示对该变换 invariant。图像分类希望最终表示对小幅 translation 具有 robustness，但不一定希望底层 feature map 完全不变。

普通 CNN 对 rotation 等 sampling lattice 的其他 isometry 并不天然 equivariant。让网络学习 rotated copies 的 filter 只能近似解决；exact equivariance 需要 architectural constraints。slides 提到的方向包括 **Group Equivariant Networks**（重新定义 convolution/pooling，使其对某个 transformation group $G$ equivariant）和 Capsule Networks（用其他 network design 替代部分 pooling）。

**边界条件也重要**：平移等变的直觉最干净地成立于无限/周期边界或不受边界影响的内部区域。有限图像上的 zero padding 会在边缘引入特殊值；stride 大于 1 还会让小幅平移遇到采样相位变化。因此普通离散 CNN 的“平移等变”不是对任意图像、任意平移都毫无例外的保证。分类中的 pooling 可以增加对小位移的容忍度，但也可能丢掉定位所需的信息。

## 7. CNN 结构例子

- **LeNet-style CNN**：早期用于 handwritten digit recognition，体现 convolution、pooling、分类层的基本组合。
- **AlexNet**：更深的 CNN structure，推动了大规模 visual recognition；slides 标注作者 Ilya Sutskever、Alex Krizhevsky、Geoffrey Hinton。

二者共同体现典型思路：空间尺寸逐渐减小，feature channels 和语义抽象程度逐渐增加，最后连接到 prediction head。

## 8. CNN training：把训练看成 optimization

**Motivation**：有了 CNN architecture 仍不代表它能训练成功；输入尺度、初始化、optimizer、learning-rate schedule 和 normalization 会决定 gradient 是否能稳定传播。每项技术解决的是不同 failure mode，不能统称为“让模型更强”。

### 8.1 Mini-batch SGD 的基本循环

监督学习中，每个样本有 input 和 label。训练过程可以概括为：

1. 从 training set 采样一个 mini-batch；
2. **Forward propagation**：将 batch 经过网络，计算 prediction 和 loss；
3. **Backpropagation**：利用链式法则计算每个参数的 gradient；
4. 用 gradient 更新 parameters。

训练同时包含两个目标：

- **Optimization**：如何有效降低 training loss？
- **Generalization**：如何避免只记住 training set（overfitting），在新数据上也表现好？训练集、validation set、test set 的职责，以及 loss、gradient 的基础定义，见 [[1 Basic neural networks]]。本节重点放在 CNN 特有的训练困难：输入尺度、初始化、loss surface 和 normalization。

### 8.1.1 CNN training 中 dataset 的作用

CNN 不会自动知道哪些视觉变化应该忽略、哪些变化代表类别改变；这些规律要由 dataset 中的样本和 label 提供。训练图片的数量重要，但**覆盖范围和标注一致性同样重要**：如果所有猫都在同一种背景中，CNN 可能学到背景；如果边界框或 segmentation mask 标注不一致，gradient 会把互相冲突的视觉模式推给同一组 filters。

对视觉任务，dataset 至少要覆盖：

- 不同位置、尺度、姿态、光照、遮挡和传感器噪声；
- 每个类别的难例与相似类别，而不只是容易分类的样本；
- 与部署场景相近的 camera、resolution、color pipeline 和 preprocessing；
- train/validation/test 之间按 image、video 或 subject 去重，避免同一场景的近重复帧泄漏。

这解释了为什么 `data augmentation` 不是“无条件制造更多图片”：它只能表达我们相信 label 不变的变化。错误的 flip、rotation 或 crop 会制造错误监督信号，模型会认真学习这个错误。

**工程视角**：算法工程师关心 dataset 是否覆盖目标分布；验证工程师关心切分、去重和 label policy 是否可审计；AI chip 工程师则要估计 augmentation、训练轮数和重训次数带来的额外计算与 memory traffic。dataset 设计错误，通常不能靠增加 MAC 数量补救。

### 8.1.2 可选现象展示：看见 CNN 的 tensor shape

运行 [02_cnn_shapes.py](experiments/02_cnn_shapes.py)：

```bash
cd "/Users/dongyangchen/Desktop/lecture note/deep learning/experiments"
python3 02_cnn_shapes.py
```

脚本生成带 vertical/horizontal bar 的小图像，训练一个两层 CNN，并打印：

```text
input -> Conv2d -> MaxPool2d -> Conv2d -> MaxPool2d
```

打印出的 shape 可以与网络定义逐项对应：最后的 `16 * 4 * 4` 是 classifier 接收的 flatten dimension。修改 `padding`、`stride` 或 pooling 只是观察输出 shape 如何变化的可选方式，正文的尺寸公式已经给出判断方法。

![[assets/2 CNN and RNN/CNN selected/CNN1-61.png]]

> **来自 CNN 1 slide 61｜看图重点**：一次 mini-batch 更新包含四步：取数据、forward 计算 loss、backpropagation 计算 gradient、用 gradient 更新 parameters。训练不是“看一眼图片就变聪明”，而是重复这个闭环很多次。

### 8.2 Data preprocessing

若不同输入维度的尺度差异很大，loss surface 可能出现狭长的 ravines：某些方向梯度很陡，另一些方向很平，普通 gradient descent 会来回摆动、收敛缓慢。常见的基础处理是将输入 center 到 zero mean 并缩放到 unit variance：

$$
x'_{j}=\frac{x_j-\mu_j}{\sigma_j}
$$

对 images 应遵循具体 model 的 preprocessing convention。slides 的 CIFAR-10 例子包括：减去 mean image（AlexNet）、减去 per-channel mean（VGGNet），或再除以 per-channel standard deviation（ResNet）。PCA/whitening 在现代 visual recognition 中并不总是常规步骤。

![[assets/2 CNN and RNN/CNN1-70.png]]

> **来自 CNN 1 slide 70｜CIFAR-10 preprocessing**：这张 slide 的重点不是要求三种方法同时使用，而是提醒我们：**输入处理必须和模型训练时采用的 convention 保持一致**。如果预训练模型原来接收的是“减去 per-channel mean”的图片，推理时却改成“除以 255 后再标准化”，输入分布就可能和模型学到的分布不一致。

对一张 CIFAR-10 RGB 图片，可以把像素看成形状为 $32\times32\times3$ 的数组。三种方法可以这样理解：

1. **Subtract the mean image（AlexNet）**：先为每个像素位置和颜色 channel 计算平均值，得到一个同样是 $32\times32\times3$ 的 `mean image`，然后逐元素相减：
   $$
   x'_{h,w,c}=x_{h,w,c}-\mu_{h,w,c}
   $$
   它不仅考虑“红、绿、蓝的平均亮度”，还保留了平均图像中每个空间位置的信息。
2. **Subtract the per-channel mean（VGGNet）**：只为 R、G、B 三个 channel 各计算一个平均值，所以只有 3 个 numbers：
   $$
   x'_{h,w,c}=x_{h,w,c}-\mu_c
   $$
   同一个 channel 的所有像素都减去同一个均值；例如所有 red pixels 都减去 $\mu_R$。
3. **Subtract mean and divide by per-channel standard deviation（ResNet）**：先减去每个 channel 的均值，再除以该 channel 的标准差：
   $$
   x'_{h,w,c}=\frac{x_{h,w,c}-\mu_c}{\sigma_c}
   $$
   这样每个 channel 大致被调整到 zero mean、unit variance，使不同 channel 的数值尺度更接近。

`PCA`（Principal Component Analysis）和 `whitening` 会进一步旋转或重新缩放输入特征，使它们之间的相关性降低；但正如 slide 所说，在现代 visual recognition pipeline 中，它们通常不是默认必做步骤。真正使用某个 pretrained model 时，应优先查看该模型的官方 preprocessing 文档。

### 8.3 Weight initialization 与 symmetry breaking

如果同一层所有 weights 都初始化为 0，所有 hidden units 产生完全相同的输出和梯度，无法打破 symmetry；它们永远学成相同的 unit，网络容量被浪费。

> **Background / worked example**：假设同一层有两个 neuron，它们的 weights 都是全 0、bias 也都是 0。无论输入是 $(1,2)$ 还是 $(3,4)$，两个 neuron 都得到相同的 $z=0$ 和相同的 activation；反向传播时它们收到的 gradient 也相同。更新后两个 neuron 仍然完全一样，相当于只用了一个 neuron。加入不同的小随机值后，它们从一开始就观察到略有不同的 pattern，后续 gradient 才能把它们训练成不同的 features。

最简单的方法是 small random initialization（例如 zero-mean Gaussian，标准差约 $10^{-2}$），但深层网络可能出现：

- activation 逐层变得接近 0；
- sigmoid/tanh saturation；
- local gradients 接近 0，导致 vanishing gradients。

**Xavier / Glorot initialization** 让激活尺度在层间更稳定，slides 给出的标准差近似为：

![[assets/2 CNN and RNN/CNN1-78.png]]

> **来自 CNN 1 slide 78｜看图重点**：如果激活进入 sigmoid/tanh 的饱和区，曲线局部斜率（也就是 local gradient）接近 0，反向传播的信号会变得很弱。因此初始化不是随便填数字，而是要控制每层 activation 的尺度，避免一开始就“卡死”。

$$
\operatorname{std}(W)=\frac{1}{\sqrt{\text{fan\_in}}}
$$

对 convolution layer，

$$
\text{fan\_in}=K^2C_{in}
$$

ReLU 会将负值截断为 0，因此需要修正尺度。slides 给出 He/MSRA initialization：

$$
\operatorname{std}(W)=\sqrt{\frac{2}{\text{fan\_in}}}
$$

### 8.4 SGD 与改进算法

最基本的 SGD 更新为：

$$
\theta_{t+1}=\theta_t-\eta\,\nabla_{\theta}L_{B_t}(\theta_t)
$$

其中 $\theta$ 是所有 parameters，$\eta$ 是 learning rate，$L_{B_t}$ 是当前 mini-batch loss。SGD 可能在狭长 ravine 中震荡，也可能在 high-dimensional space 的 saddle point 附近变慢。

**SGD + Momentum** 累积过去的更新方向：

$$
v_t=\mu v_{t-1}-\eta\nabla L(\theta_t),\qquad
\theta_{t+1}=\theta_t+v_t
$$

它能抑制反复摆动并加速稳定方向，但仍需要调参。**Nesterov Momentum** 会先“look ahead”到 velocity 将带来的位置，再在那里计算 gradient。

**AdaGrad** 为每个参数维护历史 squared gradients，使 steep directions 的步长变小、flat directions 的步长相对变大；但累积量不断增加，learning rate 可能最终变得过小。

若 $g_t=\nabla_\theta L_t$，AdaGrad 的典型形式是：

$$
r_t=r_{t-1}+g_t\odot g_t,\qquad
\theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{r_t}+\epsilon}
$$

其中 $\odot$ 表示 element-wise multiplication。某个 parameter 的历史 gradient 越大，它的分母越大，之后的步长就越小。

**RMSProp** 用 exponential moving average 平滑历史 squared gradients，避免 AdaGrad 的无限累积。

$$
r_t=\rho r_{t-1}+(1-\rho)(g_t\odot g_t),\qquad
\theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{r_t}+\epsilon}
$$

与 AdaGrad 的区别是：RMSProp 会逐渐忘记很久以前的 gradients，因此 $r_t$ 不会只增不减。

**Adam** 近似结合 Momentum（一阶矩）与 RMSProp（二阶矩）：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$
$$
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

这里的除法和平方根通常按 element-wise 进行。bias correction 在训练早期很重要。

### 8.5 Learning rate 与 decay

SGD、Momentum、AdaGrad、RMSProp、Adam 都包含 learning rate 这一关键 hyperparameter。学习率过大可能使 loss 爆炸，过小则训练极慢。

常见 schedule：

- **Step decay**：在指定 epochs 将 learning rate 乘以一个比例，例如 ResNet 在 30、60、90 epochs 后乘 $0.1$。
- **Cosine decay**、**linear decay**、**inverse-square-root decay**。
- **Linear warmup**：训练初期从接近 0 逐渐增大 learning rate，避免高初始学习率导致 loss explosion。slides 给出的经验规则是：batch size 放大 $N$ 倍时，初始 learning rate 也可近似放大 $N$ 倍，但实际仍需验证。

## 9. Normalization layers

### 9.1 Batch Normalization

**Motivation**：每层参数不断变化时，后续层接收到的 activation 尺度可能跟着变化，导致优化不稳定。BatchNorm 在训练时标准化 mini-batch statistics，再通过可学习的 scale/shift 保留表达能力。收益是较稳定的优化；边界是 batch size 和 train/test statistics 的处理会影响实现。

深层网络中，前面层的参数改变会导致后面层输入 activation 的分布不断变化。Batch Normalization 对一个 mini-batch 的 activation 做标准化：

$$
\mu_B=\frac{1}{m}\sum_{i=1}^{m}x_i,\qquad
\sigma_B^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2
$$
$$
\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad
y_i=\gamma\hat{x}_i+\beta
$$

$\gamma,\beta$ 是 learnable scale 和 shift，因此 BN 不会强迫网络只能使用 zero mean、unit variance 的表示。

训练时使用当前 mini-batch 的 statistics，并维护 running mean/variance；test time 通常使用 running statistics，而不是依赖当前测试 batch。对 convolutional network，常按 channel 在 batch 和 spatial positions 上统计。

![[assets/2 CNN and RNN/CNN1-121.png]]

> **来自 CNN 1 slide 121｜看图重点**：BatchNorm 的顺序是“算 mini-batch mean → 算 variance → normalize → 用可学习的 `γ` 和 `β` 做 scale/shift”。训练时统计当前 batch；测试时使用训练过程中积累的 running statistics，这样单张图片也能稳定推理。

如果你想把 BatchNorm 和其他 normalization 放在一起比较，可再看这张 CNN 1 的总结页：它把 BatchNorm、LayerNorm、InstanceNorm、GroupNorm 放在同一张表里。先确认“统计量是跨 batch、跨 channel 还是跨 spatial positions 计算”，再记名称。

![[assets/2 CNN and RNN/CNN1-124-normalization.png]]

slides 总结的 benefits 包括改善优化稳定性、允许更大的 learning rate、缓解 activation scale 的变化；但它会依赖 batch size，且 normalization 不等同于自动解决 overfitting。

### 9.2 Layer / Instance / Group Normalization

- **Layer Normalization**：对单个 sample 的 layer features 做 normalization，不依赖 batch 维度。
- **Instance Normalization**：通常对每个 sample、每个 channel 的 spatial positions 做 normalization，常用于 style transfer 等任务。
- **Group Normalization**：将 channels 分组后做 normalization，在小 batch 场景中是 BatchNorm 的替代方向。

## 10. CNN2：从训练误差到泛化

CNN2 的核心问题是：模型在 training set 上表现很好，不代表它能处理没有见过的新图像。**Generalization** 指模型把学到的规律迁移到 unseen data 的能力；如果模型把训练图片中的偶然细节也记住了，就会出现 **overfitting**。

> **训练检查**：每次看到 training loss 下降，都要同时问“dataset 代表的是什么分布？”和“validation/test 是否真的来自未见过的对象与场景？”只有在这两个问题都清楚时，loss 曲线才有可解释性。

### 10.1 Batch 是什么？

一个 **batch** 是一次送进网络、一起计算的若干个 training samples。假设一张 RGB 图像的 shape 是 $32\times32\times3$，如果一次取 $B=64$ 张图，那么这一批输入可以写成：

$$
X_B\in\mathbb{R}^{64\times32\times32\times3}
$$

这里第一个维度 $B$ 就是 **batch size**。网络会对这 64 张图分别做 forward pass，然后把它们的 loss 或 gradient 汇总，进行一次 parameter update。

- **Batch size = 1**：一次只看一张图，更新噪声较大，但占用内存少。
- **Mini-batch**：常见的折中，例如 32、64、128 张图。
- **Full batch**：一次使用整个 training set，gradient 更稳定，但内存和计算开销很大。

注意：batch 不是新的数据类别，而是“这一次一起处理的数据小组”。一个 epoch 表示整个 training set 被完整看过一次；如果有 10,000 张图、batch size 为 100，那么一个 epoch 大约包含 100 次 parameter updates。

### 10.2 Early stopping

训练时同时观察 training loss 和 validation performance。若 training loss 继续下降，但 validation error 开始上升，说明模型正在记忆 training set 的细节。**Early stopping** 会在 validation performance 最好时保存 checkpoint，并在之后变差时停止训练或恢复最佳 checkpoint。

![[assets/2 CNN and RNN/CNN selected/CNN2-4.png]]

> **来自 CNN 2 slide 4｜看图重点**：training error 继续下降并不代表模型继续变好；当 validation error 在最低点之后反而上升，说明模型开始过拟合。Early stopping 选择 validation 表现最好的 checkpoint，而不是盲目训练到最后。

它是一种 regularization：我们不一定要把 training loss 优化到最低，因为那个模型可能已经过拟合。由于 validation error 会受到 mini-batch stochasticity 影响，可以使用 patience（连续若干轮没有改善才停止）或 temporal smoothing。

### 10.3 Data augmentation

**Motivation**：有限训练图片可能让模型把背景、位置等偶然细节当成类别线索。augmentation 用保持 label 的变换让模型看到更多合理变化；收益是减少对固定外观的依赖，边界是每种变换都必须符合任务语义，否则会给模型错误标签。

**Data augmentation** 通过对训练图像做保持标签的变换，制造更多看起来不同但语义相同的样本，从而减少模型对某个固定外观的依赖。视觉任务中常见变换包括 translation、horizontal/vertical flip、rotation、smooth warping 和 noise。

变换必须符合任务的语义假设：object recognition 通常可以使用 horizontal flip，但 handwritten digit recognition 中翻转可能改变数字含义。AutoAugment 的思路是从很多候选变换组合中自动搜索适合数据集的 policy；slides 给出的历史结果只用于说明这种搜索方向，不能理解成所有任务都能获得同样的 accuracy。

![[assets/2 CNN and RNN/CNN2-10-augmentation.png]]

> **来自 CNN 2 slide 10｜截图导航**：把每个变换都当成一个假设来检查：变换后 label 是否仍然正确？如果答案是否定的，就不能为了“增加数据量”而使用它。

![[assets/2 CNN and RNN/CNN selected/CNN2-10.png]]

> **来自 CNN 2 slide 10｜看图重点**：data augmentation 不是随便把图片改得越多越好，而是制造“标签仍然正确”的变化。能不能翻转、旋转多少，取决于任务本身。

这张补充截图把变换组合放大。先看每列输入如何变化，再检查变换后 label 是否仍然成立；这一步把 augmentation 从“图片特效”变成可验证的建模假设。

![[assets/2 CNN and RNN/CNN2-10-augmentation-detail.png]]

### 10.4 Weight regularization 与 transfer learning

**L2 regularization / weight decay** 在 loss 中惩罚过大的 weights：

$$
L_{total}=L_{data}+\frac{\lambda}{2}\lVert\theta\rVert_2^2
$$

因此 gradient 中多出 $\lambda\theta$，SGD 更新可以写成：

$$
\theta_{t+1}=(1-\eta\lambda)\theta_t-\eta\nabla_\theta L_{data}
$$

直觉上，每次更新都会轻微地把 weights 拉回 0，避免模型使用过大的、过于专门化的参数。

**Transfer learning** 是先使用在大数据集上训练好的 CNN，再把它迁移到目标任务。常见做法是：

1. 保留预训练 network 的 convolution layers，因为它们已经学到边缘、纹理等通用 visual features；
2. 替换或重新训练最后的 classification head；
3. 目标数据较少时，可以冻结大部分 layers；数据较多时，再用较小 learning rate fine-tune 全网。

### 10.5 Dropout：随机关闭部分 unit

**Motivation**：若多个 units 总是共同出现，模型可能过度依赖固定组合；Dropout 训练时随机移除部分 activation，迫使其学习更稳健的替代路径。它的代价是训练计算带随机 mask，推理语义需要用 inverted dropout 对齐 activation 尺度。

**Dropout** 是 stochastic regularization：training 时随机把一部分 activation 设为 0，使网络不能过度依赖某几个固定 features。设保留概率为 $p$，inverted dropout 在 training 时使用：

$$
\tilde{h}_i=\frac{m_i h_i}{p},\qquad m_i\sim\operatorname{Bernoulli}(p)
$$

除以 $p$ 是为了让 $\mathbb{E}[\tilde h_i]=h_i$，这样 test time 可以直接使用完整网络，不需要再额外缩放。slides 还介绍了 DropConnect（随机删除 connections）、fractional pooling、Cutout（随机遮住图像区域）和 Mixup（训练随机混合的图像与 labels）等相关方法。

![[assets/2 CNN and RNN/CNN2-20.png]]

上图是 slides 中 Dropout 的原始示意；阅读时重点看 training 和 test 的区别：training 使用随机子网络，test 使用完整网络。

![[assets/2 CNN and RNN/CNN selected/CNN2-24.png]]

> **来自 CNN 2 slide 24｜看图重点**：Dropout 的核心是 training 时随机改变网络，让模型不能依赖某几个固定 unit；test 时不再随机关闭 unit，而是使用完整网络来预测。

把 training 和 test 两张图并排看，最容易理解 inverted dropout：训练时是许多随机子网络，测试时恢复完整网络；训练期除以 $p$ 已经把平均 activation 对齐，因此测试期不再额外缩放。

![[assets/2 CNN and RNN/CNN2-21-dropout-training.png]]

![[assets/2 CNN and RNN/CNN2-24-dropout-test.png]]

### 10.6 Hyperparameter optimization

**Hyperparameter** 是训练前由我们设置的量，例如 learning rate、batch size、weight decay、dropout probability 和 network depth；它们不是像 weights 那样通过 backpropagation 直接学出来的。用 training set 训练、validation set 比较不同设置、最后只用 test set 做一次 unbiased evaluation。

Grid search 会枚举预先指定的组合；random search 在同一个预算下通常更容易探索高影响的 hyperparameters。更复杂的方法包括 Bayesian optimization。无论使用哪种方法，都不能根据 test set 反复调参，否则 test set 也会被间接“记住”。

## 11. CNN2：现代 CNN architecture

### 11.1 经典 CNN architecture cases：LeNet、AlexNet、ZFNet、VGGNet、ResNet

先把几个名字放在同一条历史线上：**LeNet** 展示基本的 convolution + pooling 组合；**AlexNet** 证明更深的 CNN 可以处理大规模 ImageNet；**ZFNet** 用 visualization 分析 AlexNet 的中间 feature；**VGGNet** 把结构进一步规则化。它们都是 sequential architecture，信息主要沿着一条主干从 input 流向 prediction head。

#### 11.1.1 AlexNet：让深层 CNN 在大规模图像上工作

AlexNet 的关键改变包括：更多 convolution layers、`ReLU` 替代 `tanh`、local contrast normalization，以及在 fully connected layers 中使用 `Dropout`。对初学者来说，可以把它理解为：AlexNet 同时改进了 **representation**（更深的 feature extraction）和 **training/generalization**（ReLU、normalization、dropout）。

在它之前的 **LeNet-style CNN** 是一个更小的参照物：它把 handwritten digit 的图像依次送入 convolution、pooling 和分类层。先看这张结构图，再看 AlexNet/VGG 的大网络，就能把“更深、更宽、更多 feature channels”理解成逐步扩展，而不是突然出现的一堆层名。

![[assets/2 CNN and RNN/CNN1-57-lenet.png]]

> **截图导航**：LeNet 图中重复出现的 convolution/pooling block 是 CNN 的基本积木；后来的 architecture 主要是在积木数量、连接方式、计算效率和训练稳定性上做设计。

![[assets/2 CNN and RNN/CNN2-45.png]]

> **来自 CNN 2 slide 45｜看图重点**：输入从 $227\times227\times3$ 开始，经过 convolution 和 max pooling 后，空间尺寸逐渐变小、channel 数逐渐增加；最后进入 fully connected layers 和 linear classifier。这里的 `[55×55×96]` 等标注表示 feature map 的空间大小和 channel 数。

AlexNet 使用的 preprocessing 也与前面 CNN 1 的 CIFAR-10 slide 相关：当 slides 说“subtract the mean image（AlexNet）”时，意思是要按照该 model 训练时的 convention 处理输入，而不是把任何 normalization 方案随意替换进去。

#### 11.1.2 VGGNet：小 filter、深网络、规则设计

VGGNet 的设计原则非常整齐：主要使用 `3×3 convolution`、`stride=1`、`padding=1`，配合 `2×2 max pooling`、`stride=2`。因此它容易阅读和复现：连续堆叠多个小的 convolution blocks，再逐步下采样。VGG16 和 VGG19 的数字表示它们大致包含 16 或 19 个带参数的 layers。

![[assets/2 CNN and RNN/CNN2-49.png]]

> **来自 CNN 2 slide 49｜看图重点**：右侧比较了 AlexNet、VGG16 和 VGG19。VGG 用很多个小的 `3×3` filters 替代较大的 early filters，让 network 更深；代价是 parameters 和 computation 都较大。

VGGNet 的 preprocessing 对应前面的 **per-channel mean**：每个 RGB channel 使用一个训练集平均值，例如 $\mu_R,\mu_G,\mu_B$，然后对整张图片中对应 channel 的每个 pixel 做减法。

#### 11.1.3 ResNet：用 skip connection 解决深度优化问题

**局部动机**：堆叠更深的 plain layers 后，训练误差反而可能变差；这不一定是表达能力不足，而可能是优化困难。ResNet 让 block 学“相对输入要改多少”，并保留 identity shortcut。收益是信息与 gradient 有较直接的路径；代价是 shortcut 的 shape 必须匹配，必要时需要 projection，此外相加操作仍要搬运和对齐 activation。

当 plain network 继续堆叠 convolution layers 时，困难可能不是“层数多所以表达能力不够”，而是 **optimization 变难**。ResNet 让一组 layers 学习 residual mapping $F(x)$，并把输入 $x$ 沿 shortcut 直接加回去：

$$
y=F(x)+x
$$

如果新增 layers 暂时没有学到有用变换，只要让 $F(x)\approx0$，这个 block 就近似 identity mapping $y\approx x$。shortcut 也提供了较短的 gradient path，使信息和 gradient 更容易穿过深层网络。

![[assets/2 CNN and RNN/CNN2-73.png]]

> **来自 CNN 2 slide 73｜看图重点**：左边的 plain layers 只能顺序传递 $x$；右边的 residual block 把 convolution 计算出的 $F(x)$ 与原始 $x$ 相加，得到 $F(x)+x$。这条弯曲的 identity shortcut 是 ResNet 最重要的结构。

这张补充截图把 residual mapping 的计算路径单独放大。沿两条路读：主路计算 $F(x)$，shortcut 保留 $x$；相加节点的输出才是下一层看到的 $y$。对 AI 芯片来说，shortcut 还意味着缓存、对齐和 element-wise add 的 data movement。

![[assets/2 CNN and RNN/CNN2-73-residual-mapping.png]]

ResNet 的 preprocessing 对应 **per-channel mean + per-channel standard deviation**：先对 R、G、B 各自减去 mean，再除以各自的 standard deviation。注意，ResNet 的核心贡献（skip connection）和 preprocessing 是两个不同层次的设计：前者改变 network architecture，后者改变输入数值尺度。

这三个 case 可以这样记：

| Network | 主要解决的问题 | 代表性结构/做法 | 与 preprocessing 的 slide 对应 |
|---|---|---|---|
| AlexNet | 让较深 CNN 能用于大规模 visual recognition | ReLU、更多 convolution、Dropout | subtract mean image |
| VGGNet | 用规则、可复现的方式堆叠更深网络 | repeated `3×3` convolution blocks | subtract per-channel mean |
| ResNet | 让 very deep network 更容易 optimization | residual mapping + skip connection | subtract mean，再除 per-channel std |

不要把表格最后一列理解成“只有这个模型能用这种 preprocessing”；它是 slides 给出的 historical examples。实际项目中应以具体 implementation 或 pretrained checkpoint 的 preprocessing documentation 为准。

### 11.2 GoogLeNet / Inception：并行分支

普通 sequential network 在每一层选择一种 kernel size；**Inception module** 在同一个局部单元中并行使用不同的 branches，例如 $1\times1$、$3\times3$、$5\times5$ convolution 和 pooling，再把结果 concatenate。不同 branch 能同时观察不同 spatial scale。

Inception 使用 **bottleneck layer** 和 $1\times1$ convolution 降低 channel 数。若输入每个空间位置有 $C_{in}$ 个 channels，$1\times1$ filter 不看邻近像素，但会把这个位置的 channel vector 映射成新的 channel vector；因此它等价于在每个 pixel 位置共享同一个小型 fully connected layer。

![[assets/2 CNN and RNN/CNN2-54.png]]

这张补充截图专门突出 Inception 的并行分支：同一输入同时经过不同 receptive field 的操作，最后在 channel 维度 concatenate。对硬件来说，不同 branch 可以映射到并行计算单元，但还要支付 branch synchronization 和 memory traffic 的代价。

![[assets/2 CNN and RNN/CNN2-55-inception-branches.png]]

### 11.3 DenseNet、SENet 与高效网络

- **DenseNet**：把早期 layers 的 outputs 直接传给后续 layers，鼓励 feature reuse，但会增加连接和 memory 开销。
- **Squeeze-and-Excitation Network (SENet)**：学习每个 channel 的重要性，再对 channels 做重新加权。
- **MobileNet / ShuffleNet**：面向移动设备，重点减少 parameters、FLOPs 和 memory access。
- **EfficientNet**：同时考虑 network depth、width 和 input resolution，在固定 compute budget 下寻找较好的 compound scaling。

![[assets/2 CNN and RNN/CNN2-90-mobilenet.png]]

> **截图导航**：MobileNet 的重点不是单纯“层更少”，而是重新拆分 convolution 的计算，优先降低乘加量与 memory access；这正是从模型结构走向 AI accelerator 约束的入口。

#### 11.3.1 低位宽表示：为什么 quantization 会影响芯片设计？

**Engineering perspective / background**：模型中的 weights 和 activations 通常以 FP32、FP16 或 BF16 表示；**quantization**（量化）用更少的 bits 近似它们，例如 INT8。这样做的直接收益是每个元素占用更少 storage，weight/activation bandwidth 降低，很多 accelerator 也能在同样面积内放置更多低位宽 MAC；代价是数值范围和精度变窄，模型的 accuracy 可能下降。

一个线性量化的直觉形式是：把浮点数 $x$ 映射到整数 $q$，再用 scale $s$ 近似恢复：

$$
q=\operatorname{clip}\left(\operatorname{round}\left(\frac{x}{s}\right),q_{\min},q_{\max}\right),
\qquad \hat{x}=s q
$$

其中 `clip` 防止超出整数可表示范围。$s$ 太大，量化步长粗，细节丢失；$s$ 太小，大值会饱和到边界。实际工程还要决定 per-tensor 还是 per-channel scale、如何校准 activation range，以及 accumulator 是否使用更高位宽。

对 CNN 来说，量化不仅改变 weight 文件大小，还会改变 dataflow：输入 feature map、weights、partial sums 的位宽不同，PE array 的 packing、累加器宽度、DMA burst 和验证方法都要相应调整。量化因此是模型精度、吞吐、带宽、面积和能耗之间的系统 trade-off，不是简单的“把 float 换成 int”。

### 11.4 Neural Architecture Search（NAS）

手工设计 architecture 需要大量经验和试验。**NAS** 先定义可搜索的 building blocks、连接方式和 compute budget，再自动搜索一个可堆叠的 cell 或 network。NASNet、AmoebaNet 和 ENAS 是 slides 提到的代表方向。NAS 并不是“完全不需要设计”：search space、目标指标和资源预算仍然由人决定。

## 12. CNN 的 dense prediction：让模型回答“每个像素是什么”

前面的 CNN 分类只输出一个 image-level label，例如“这张图里有车”。`Dense prediction`（密集预测）要求模型对每个空间位置输出结果；`semantic segmentation`（语义分割）就是其中的典型任务：输入一张图，输出同样覆盖整张图的 pixel-wise class map。它不区分同一类别的不同实例，例如两辆车都可以标成 `car`。

![[assets/2 CNN and RNN/RNN1-08-semantic-segmentation.png]]

> **截图导航**：左边是输入 image，右边是每个像素的类别标注。看图时先确认输出仍然有空间坐标，再理解为什么分类网络最后的 fully connected layer 不够用。

### 12.1 为什么不能直接把分类网络照搬过来？

如果把整张图压缩成一个向量再分类，模型会丢掉精确的边界位置；逐像素使用独立的小分类器又无法共享足够的全局语义，计算量也很大。课件给出的主线是：用 **fully convolutional network** 保留卷积结构，把 fully connected layers 改写为等价的 convolution，使网络可以在整张图上滑动计算。

更完整的结构是 **encoder-decoder**：encoder 逐步下采样，扩大 receptive field 并提取“这是什么”；decoder 逐步 upsample，恢复“它在哪里”。

![[assets/2 CNN and RNN/RNN1-26-encoder-decoder.png]]

### 12.2 Upsampling、U-Net 与 dilated convolution

`Upsampling`（上采样）可以用 unpooling、插值或 learnable `transpose convolution` 实现。transpose convolution 的 weights 通过训练学习如何把低分辨率 feature map 映射回更密的网格；它不是把普通 convolution 的公式简单倒过来。

![[assets/2 CNN and RNN/RNN1-29-transpose-convolution.png]]

**U-Net** 在 encoder 和 decoder 之间加入 skip connections，把浅层的边界/位置细节直接送到 decoder，再和深层语义融合。这与 ResNet 的 shortcut 有相似的“缩短信息路径”思想，但目标不同：ResNet 主要帮助优化，U-Net 的 skip connection 还负责恢复空间细节。

![[assets/2 CNN and RNN/RNN1-41-unet.png]]

`Dilated convolution`（空洞卷积，也叫 atrous convolution）在 kernel 元素之间留出间隔，因此不显著减少 feature map 分辨率也能扩大 receptive field。它适合需要较大上下文、又不能过度下采样的 dense prediction 任务。

![[assets/2 CNN and RNN/RNN1-42-dilated-convolution.png]]

### 12.3 Segmentation 的 loss

最直接的方法是对每个像素做 cross-entropy，再对像素求和或平均；当背景像素远多于目标像素时，模型可能只学会预测背景。课件还介绍了 **soft Dice loss**，它直接关注预测区域与真实区域的重叠程度：

$$
\operatorname{Dice}(p,y)=\frac{2\sum_i p_i y_i+\epsilon}{\sum_i p_i+\sum_i y_i+\epsilon}
$$

其中 $p_i$ 是模型对第 $i$ 个像素属于目标类的 soft probability，$y_i$ 是真实的 0/1 mask，$\epsilon$ 防止分母为 0。训练时常最小化 $1-\operatorname{Dice}$ 或把 Dice 与 pixel-wise cross-entropy 组合。

![[assets/2 CNN and RNN/RNN1-52-dice-loss.png]]

> **AI 芯片视角**：dense prediction 的难点是高分辨率 feature map 带来更大的 activation memory 和 memory traffic；encoder-decoder、skip connection、dilated convolution 都是在精度、空间细节与硬件成本之间做取舍。

## 13. 从结构理解到硬件分析

### 13.1 同一个 convolution 有不同实现

同一个 convolution 可以用 direct sliding window、im2col 加 GEMM，或专用 dataflow 实现。数学结果应一致，但 workspace、cache reuse 和 memory traffic 不同。

卷积层的 MAC 近似为：

$$
H_{out}W_{out}C_{out}(K_hK_wC_{in})
$$

这个数量只描述 arithmetic work，不等于真实 latency。真实 latency 还取决于 tile 是否填满 PE array、weights 是否能留在 SRAM、外部 memory 是否成为瓶颈，以及算子之间能否 fusion。

### 13.2 Depthwise、pointwise 与模型效率

Depthwise convolution 对每个 input channel 单独做空间卷积，pointwise convolution 用 $1\times1$ kernel 混合 channels。两者组合可以减少 MAC，但可能降低计算阵列利用率、增加 kernel launch 和 memory overhead。少 FLOPs 不保证更快，必须在目标设备和目标 batch 上实测。

### 13.3 BatchNorm folding 与量化边界

推理时，固定的 BatchNorm 统计量可以与前面的 convolution 合并，减少独立算子和中间 activation 写回，这叫 BatchNorm folding。PTQ 在训练后校准量化范围，QAT 在训练中模拟量化误差；二者都要检查 accuracy、动态范围、累加精度和实际 kernel 支持。

### 13.4 不要只数 FLOPs

可以用 arithmetic intensity 做第一轮判断：单位 memory traffic 能完成多少 arithmetic。intensity 低时更可能受 bandwidth 限制；intensity 高但阵列利用率不足时，可能受 tile shape 或调度限制。这是系统分析入口，不是完整性能模型。

> **External context**：本节的 direct/im2col、BatchNorm folding、PTQ/QAT 和 arithmetic intensity 是工程扩展，用来连接课件的 CNN 数据流与部署分析。

## 14. 教授式完整算例：一层 CNN 的参数、感受野与硬件代价

设输入为 32×32×3，kernel 为 5×5，输出 channels 为 16，stride 为 1，padding 为 0。

### 1. 输出尺寸与参数量

$$
H_{out}=W_{out}=\left\lfloor\frac{32-5}{1}\right\rfloor+1=28
$$

输出 shape 是 28×28×16。每个 filter 必须覆盖 3 个 input channels，因此参数量为：

$$
(5\times5\times3+1)\times16=1216
$$

参数量不随输入的 32×32 直接增加，因为同一个 filter 被空间位置共享。

### 2. 一个 output neuron 如何产生

输出位置 $(i,j)$、第 $c$ 个 output channel 的一个数为：

$$
y_{i,j,c}=b_c+\sum_{u=0}^{4}\sum_{v=0}^{4}\sum_{r=0}^{2}K_{u,v,r,c}X_{i+u,j+v,r}
$$

它读取一个 5×5×3 patch，共 75 个数字，逐元素乘上 filter weights，再求和。单个 output unit 的 receptive field 是 5×5 空间区域，但覆盖全部 3 个颜色 channels。

### 3. 一层总计算量

输出元素数量是 $28\times28\times16=12544$。每个元素需要 75 次乘加，因此约有：

$$
12544\times75=940800\text{ MACs}
$$

参数只有 1216 个，但计算接近百万 MAC。原因是参数被大量复用：同一个 filter 要在 28×28 个位置重复使用。

### 4. 感受野逐层扩大

若连续三层都是 kernel=3、stride=1、没有 dilation，则第一层看到 3×3，第二层看到 5×5，第三层看到 7×7 的原图区域。更一般地，若当前 jump 为 $j$、receptive field 为 $r$，新层 kernel 为 $k$、stride 为 $s$、dilation 为 $d$：

$$
r_{new}=r+(k-1)d\,j,\qquad j_{new}=js
$$

### 5. 为什么硬件不能只看参数量

硬件还要处理 weights、input activations 和 output/partial sums。若每个 output 都从 DRAM 重新读取 weights，memory traffic 可能先限制性能。weight-stationary 尽量保留 weights，output-stationary 尽量保留 partial sums，row-stationary 同时利用多类局部复用；它们是数据搬运策略，不是三种不同的 convolution。

> **讲授结论**：CNN 的核心是局部连接、参数共享、空间保留和逐层扩大 receptive field；芯片的核心是让这些复用真正发生在近端 memory，而不是被无效搬运抵消。

### 6. 这层 CNN 如何回到第一篇的训练闭环

这层卷积只说明了 **forward** 的数据流：patch 与 filter 做 MAC，得到 feature map。训练时，每个 filter weight 还会收到来自所有输出位置的梯度；因为同一组 weight 被重复使用，这些位置的梯度会累加后再更新同一个 filter。于是 CNN 的参数共享同时带来两个结果：参数量减少、梯度也必须聚合。完整的 `loss → backward → update` 见 [[1 Basic neural networks]] 的 neuron 推导。

对芯片工程师来说，这个“同一 weight 被很多位置复用”的事实既是算法优势，也是 dataflow 设计的核心机会。

## 15. 总结：从输入到可训练 CNN

一个可工作的 CNN pipeline 可以理解为：

<svg viewBox="0 0 980 125" role="img" aria-label="CNN pipeline from image to prediction" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="arrow-pipe" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#64748b"/></marker></defs><rect width="980" height="125" rx="12" fill="#f8fafc"/>
<g font-family="Arial, sans-serif" font-size="13" text-anchor="middle"><g fill="#dbeafe" stroke="#2563eb" stroke-width="2"><rect x="15" y="36" width="115" height="50" rx="10"/><text x="72" y="57" fill="#1e3a8a">Input image</text><text x="72" y="74" fill="#1e3a8a">H x W x C</text></g><g fill="#fff" stroke="#64748b" stroke-width="1.5"><rect x="155" y="36" width="145" height="50" rx="10"/><text x="227" y="57">Convolution</text><text x="227" y="74">local + shared</text><rect x="325" y="36" width="125" height="50" rx="10"/><text x="387" y="57">Activation</text><text x="387" y="74">nonlinearity</text><rect x="475" y="36" width="145" height="50" rx="10" fill="#fef3c7" stroke="#d97706"/><text x="547" y="57" fill="#78350f">Pooling / stride</text><text x="547" y="74" fill="#78350f">smaller map</text><rect x="645" y="36" width="130" height="50" rx="10"/><text x="710" y="57">Deeper blocks</text><text x="710" y="74">abstract features</text></g><g fill="#dcfce7" stroke="#16a34a" stroke-width="2"><rect x="800" y="36" width="160" height="50" rx="10"/><text x="880" y="57" fill="#14532d">Prediction head</text><text x="880" y="74" fill="#14532d">class / regression</text></g><g stroke="#64748b" stroke-width="2" marker-end="url(#arrow-pipe)"><line x1="130" y1="61" x2="155" y2="61"/><line x1="300" y1="61" x2="325" y2="61"/><line x1="450" y1="61" x2="475" y2="61"/><line x1="620" y1="61" x2="645" y2="61"/><line x1="775" y1="61" x2="800" y2="61"/></g></g></svg>

可以把每个 block 想成一个“观察和压缩”的步骤：`Convolution` 在局部区域寻找 pattern，`Activation` 加入非线性，`Pooling` 减少空间细节。网络越深，通常越关注高层语义，而不是单个像素。

训练时还存在另一个闭环：

<svg viewBox="0 0 700 310" role="img" aria-label="Mini-batch training loop: forward pass, loss, backpropagation, optimizer, and parameter update" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="arrow-train" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#64748b"/></marker></defs><rect width="700" height="310" rx="12" fill="#f8fafc"/>
<g font-family="Arial, sans-serif" font-size="14" text-anchor="middle"><g fill="#dbeafe" stroke="#2563eb" stroke-width="2"><rect x="240" y="18" width="220" height="48" rx="10"/><text x="350" y="47" fill="#1e3a8a">Mini-batch images + labels</text></g><g fill="#fff" stroke="#64748b" stroke-width="1.5"><rect x="255" y="92" width="190" height="46" rx="10"/><text x="350" y="120">Forward propagation</text><rect x="255" y="164" width="190" height="46" rx="10" fill="#fee2e2" stroke="#dc2626"/><text x="350" y="192" fill="#7f1d1d">Loss: prediction vs label</text><rect x="255" y="236" width="190" height="46" rx="10" fill="#fff7ed" stroke="#ea580c"/><text x="350" y="264" fill="#7c2d12">Backprop + optimizer update</text></g><g stroke="#64748b" stroke-width="2" marker-end="url(#arrow-train)"><line x1="350" y1="66" x2="350" y2="92"/><line x1="350" y1="138" x2="350" y2="164"/><line x1="350" y1="210" x2="350" y2="236"/></g><path d="M255 259 C110 259 110 115 255 115" fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#arrow-train)"/><text x="112" y="184" fill="#64748b" transform="rotate(-90 112 184)">repeat each update</text></g></svg>

这个闭环的核心问题是：如何让 `Loss` 逐步变小，同时让模型在没有见过的新图像上也能工作（generalization）。

$$
\text{image}
\xrightarrow{\text{local convolution + shared filters}}
\text{feature maps}
\xrightarrow{\text{activation}}
\text{nonlinear features}
\xrightarrow{\text{pooling/downsampling}}
\text{compact representations}
\xrightarrow{\text{deeper layers}}
\text{semantic prediction}
$$

本讲 CNN slides 最后总结了 CNN 的结构性质：convolution 带来 translation equivariance；层层下采样和组合产生 compositionality；localized filters 带来 deformation stability；每个 filter 的参数量与输入图像大小无关，单层复杂度随图像大小近似线性增长。

slides 还给出一个更抽象的复杂度视角：在特定的 downsampling 假设下，构建覆盖不同尺度的表示所需的网络深度可以近似为 $O(\log n)$；这里的 $n$ 指输入空间规模。这个结论不是说所有 CNN 都严格拥有 $\log n$ 层，而是说明逐层下采样可以高效扩大 receptive field。

训练方面，CNN 是 non-convex optimization：需要合理的 data preprocessing、weight initialization、first-order parameter updates、learning-rate schedule，以及必要时的 normalization。理解这些组件如何配合，比死记某一个 model 名称更重要。

从 AI chip design 的角度，CNN 的关键线索是空间位置之间的并行、filter/activation 复用，以及高分辨率 feature map 的存储与带宽需求。RNN 的时间依赖和 state buffering 已独立放在 [[3 RNN]]。

## 16. 外部资料与进阶阅读

以下资料用于补充实现和部署边界。访问日期：2026-10-17。

- **External context：Conv2d 的 shape 与参数语义**：PyTorch 官方 [torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)，核对 stride、padding、dilation、groups 和 output shape。
- **External context：BatchNorm 与训练/推理差异**：PyTorch 官方 [BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)，重点看 running statistics 和 train/eval 行为。
- **External context：CNN dataflow 背景**：D. W. et al. [Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow](https://arxiv.org/abs/1602.00567)，用于理解 weight/activation/partial-sum reuse；论文中的硬件结果不能直接当成你的设备 benchmark。
- **进阶路线**：depthwise/pointwise、operator fusion、PTQ/QAT、roofline 和 accelerator mapping；先能读懂 shape 与 MAC，再读这些系统主题。
- **External course-material extension — 卷积层的定义和 shape**：Stanford CS231n，[Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)，核对局部连接、参数共享与输出尺寸；PyTorch，[`torch.nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)，核对 `stride`、`padding`、`dilation`、`groups`。两者分别是课程解释与 API 语义，不是某个 accelerator 的性能保证。访问日期：2026-10-22。
- Eyeriss 原始论文作者为 Yu-Hsin Chen、Joel Emer、Vivienne Sze，*Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks*, ISCA 2016, [arXiv:1602.00567](https://arxiv.org/abs/1602.00567)。本篇使用它讨论数据复用与 dataflow；原论文的硬件结果不可直接外推到其他芯片。访问日期：2026-10-22。

## 来源与截图

- Lecture 3：**CNNs I - Architecture & Training**；Lecture 4：**CNNs II - Network Regularization & Architecture**。
- Lecture 5（`RNN 1.pdf`）中的 semantic segmentation 部分属于 CNN dense prediction，相关图像保留原课件页码并继续放在 `assets/2 CNN and RNN/`。
- RNN 1 中从 sequence modeling 开始的内容已移入 [[3 RNN]]；截图继续沿用原文件，避免重复拷贝或断开引用。

### 本轮新增视觉覆盖

| 案例 | 课件来源 | 笔记位置 | 视觉材料 | 检查 |
|---|---|---|---|---|
| Pooling 下采样 | CNN 1 slide 47 | §5 Pooling layers | `CNN selected/CNN1-47.png` + `pooling-max-window.svg` + 窗口互动 | 来源页保留原尺寸变化；教学图用小矩阵解释 max 操作；两者未混作同一算例 |
| 窗口/步幅对输出的影响 | 教学演示，非课件实验数据 | §5 Pooling layers | `interactive/pooling-receptive-field-explorer.html` 与 JS Engine 版本 | 本地离线；保留静态手算和课件截图回退 |

