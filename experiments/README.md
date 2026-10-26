# Deep Learning Hands-on Experiments

这些脚本把当前三篇 lecture notes 中的核心机制变成最小可运行实验。它们使用合成数据，目的是隔离机制，避免下载数据集和复杂 preprocessing 把初学者带偏。

## 给完全初学者的阅读方式

你不需要先理解所有代码。下面只是帮助你把正文中的现象看见，不是作业要求：

1. **先读正文**：先理解笔记中对应的概念，再把脚本当成一个可视化例子。
2. **直接运行默认版本**：观察终端里的 shape、数值和生成的 PNG。
3. **对照输入和输出**：找到脚本中的 `q/k/v`、`weights`、`output`，把它们和笔记中的符号对应起来。
4. **需要时再改参数**：改变 sequence length 或 head 数，只是为了观察正文中已经解释的变化。
5. **联系硬件含义**：把图中的矩阵和权重理解成 GEMM、reduction、buffer 和 memory traffic。

实验中常见的 shape 记法：`B` 是 batch size，`N` 是 token 数，`D` 是 feature dimension，`H` 是 head 数。`[B,N,D]` 表示“有 B 个样本，每个样本 N 个 token，每个 token 有 D 个数字”。

## 环境

```bash
python3 -m pip install --user torch torchvision matplotlib
```

`torchvision` 和 `matplotlib` 是可选依赖；这四个脚本本身只要求 `torch`，后续扩展到 MNIST/CIFAR-10 或画图时再使用它们。

## 顺序

```bash
python 01_mlp_dataset.py
python 02_cnn_shapes.py
python 03_rnn_lstm_gru_memory.py
python 04_seq2seq_attention.py
python 05_attention_reading.py
python 06_position_and_causal_mask.py
python 07_multihead_attention.py
python 08_attention_system_cost.py
```

每个脚本都固定随机种子。建议第一次只观察终端输出，第二次修改一个变量，再解释输出为什么改变：`batch_size`、`learning rate`、sequence length、CNN pooling 或 teacher-forcing ratio。

## 每个实验回答什么问题？

- `01_mlp_dataset.py`：完整训练循环如何降低 loss？为什么要分 train/validation？
- `02_cnn_shapes.py`：卷积和 pooling 怎样改变 tensor shape？参数量与计算结构在哪里？
- `03_rnn_lstm_gru_memory.py`：最后的 hidden state 能否记住序列开头？LSTM/GRU 的门控解决什么困难？
- `04_seq2seq_attention.py`：teacher forcing 如何训练 decoder？greedy inference 如何生成？attention weights 是否随输出步改变？
- `05_attention_reading.py`：Q/K 如何决定“读谁”，V 如何决定“读到什么”？观察 `A` 和 `Y=AV`。
- `06_position_and_causal_mask.py`：位置编码如何区分 token 位置？causal mask 如何阻断未来位置？
- `07_multihead_attention.py`：为什么一个 attention layer 有多个 head？检查 `[B,H,Q,K]` 的每个 head 权重图。
- `08_attention_system_cost.py`：sequence length、层数、head 数和 dtype 如何转化为 score matrix 与 KV cache 的存储压力？

## Attention / Transformer 可视化实验

这四个实验对应本章的学习顺序。第一次运行会在当前目录生成 `outputs/*.png`；图像用于理解机制，不是模型精度 benchmark。

### 05：Attention 是一次加权读取

```bash
python 05_attention_reading.py
```

终端会打印 `scores -> weights -> context` 的 shape 和数值，图片左侧是 attention heatmap，右侧是 Value 的加权读出。修改 `queries` 或 `values`，观察输出如何变化。

小白检查点：`weights` 的每一行加起来应接近 1；如果把某个 query 改得更接近某个 key，对应那一列通常会变亮；`context` 是 Value 的混合结果，不是 Key 的混合结果。

### 06：位置编码与因果 mask

```bash
python 06_position_and_causal_mask.py
```

左图展示不同位置的 sinusoidal feature；右图展示 decoder 只能读取当前和过去位置。把 `diagonal=1` 改成 `diagonal=0`，观察连当前位置也被屏蔽时会发生什么。

小白检查点：mask 的白色/黑色区域表示可见性，不是模型已经学会的“注意力”；它是人为施加的约束。每一行 softmax 后仍应加和为 1。

### 07：Multi-head attention

```bash
python 07_multihead_attention.py
```

脚本直接调用 PyTorch 的 `MultiheadAttention`，输出 `[batch, heads, query, key]`，并画出每个 head 的权重。随机初始化的图只用于理解数据结构；训练后再观察不同 head 是否形成不同模式。

小白检查点：`heads=4` 不表示有 4 层 Transformer，而是同一个 attention layer 同时有 4 个子空间。每个 head 的矩阵都是 `[query, key]`，最后才合并回 model dimension。

### 08：从模型配置估算硬件存储

```bash
python 08_attention_system_cost.py --bytes-per-element 2
python 08_attention_system_cost.py --layers 40 --heads 40 --head-dim 128 --bytes-per-element 2
```

表格和 log-log 图同时显示一个 attention layer 的 score weights 与全部层 KV cache。修改 batch、layers、heads、head dimension 和 dtype，练习把 Transformer 配置转换成 SRAM/DRAM 容量和带宽问题。

小白检查点：把 `N` 从 2048 改成 4096，score weights 不只是翻倍，而是约变成 4 倍；KV cache 约变成 2 倍。这是“计算并行”仍可能被 memory capacity/bandwidth 限制的原因。

## GitHub examples 已下载

这些文件保留了官方教程的原始内容，建议先阅读 README 中的最小实验，再用下面的完整例子做可视化扩展。原始文件不要直接覆盖；复制成自己的 `*_viz.py` 或 notebook 后修改。

### 1. MLP / CNN：从 loss 到图像预测

目录：`github_examples/pytorch_examples_mnist/main.py`

来源：[pytorch/examples](https://github.com/pytorch/examples/tree/main/mnist)。它使用 MNIST，适合画：

- training loss 和 test accuracy 曲线；
- 预测正确/错误的手写数字网格；
- CNN 第一层卷积核或 activation map；
- batch size、learning rate 改变后的收敛速度。

建议先把 `test()` 改成返回每个样本的 prediction、label 和 image，再用 `matplotlib.pyplot.subplots()` 显示 16 张图片。观察错误样本时问：模型错的是笔画、形状，还是数据本身含糊？

### 2. RNN：序列输入与分类结果

目录：`github_examples/pytorch_tutorials/char_rnn_classification_tutorial.py`

来源：[PyTorch char-RNN classification tutorial](https://github.com/pytorch/tutorials/blob/main/intermediate_source/char_rnn_classification_tutorial.py)。适合画：

- 不同 nationality/category 的 confusion matrix；
- 每个类别的预测概率条形图；
- sequence length 改变后 accuracy 和 inference time 的变化；
- 把 `hidden` 在 time step 上保存下来后，用 PCA 或 heatmap 看 hidden state 如何变化。

先不要改模型结构，先把一个输入名字的每个字符和对应 hidden state 保存下来。你要观察的是：RNN 每读入一个字符，hidden state 如何积累前文信息。

### 3. Seq2seq + attention：最值得做的可视化

目录：`github_examples/pytorch_tutorials/seq2seq_translation_tutorial.py`

来源：[PyTorch seq2seq translation tutorial](https://github.com/pytorch/tutorials/blob/main/intermediate_source/seq2seq_translation_tutorial.py)。它已经包含 encoder-decoder、teacher forcing 和 Bahdanau attention。建议完成三张图：

1. 一条 source sentence 对应一条 target sentence 的 attention heatmap；
2. teacher forcing ratio 为 `1.0`、`0.5`、`0.0` 时的训练 loss 曲线；
3. greedy decoding 输出与 target 的逐 token 对照表。

画 attention 时，横轴放 input words，纵轴放 output words。先问“亮点是否大致对应当前要生成的词”，再问“错误翻译是否伴随 attention 分散或错位”。这比只看最终翻译字符串更能理解 attention。

### 4. D2L：按概念查阅和改实验

目录：`github_examples/d2l/`。

- `cnn/why-conv.md`、`conv-layer.md`、`padding-and-strides.md`、`pooling.md`、`lenet.md`：卷积、尺寸、pooling 和 LeNet；
- `recurrent/gru.md`、`lstm.md`、`seq2seq.md`、`beam-search.md`、`encoder-decoder.md`、`machine-translation-and-dataset.md`：RNN、门控、seq2seq 和 beam search；
- `attention/attention-scoring-functions.md`、`bahdanau-attention.md`、`attention-pooling.md`、`queries-keys-values.md`：attention score、Bahdanau attention 和 query/key/value。

D2L 文件是 Markdown 版本的教程原文，里面的代码单元可以复制到 Jupyter notebook。它们适合在你已经跑通本目录四个最小脚本后，逐步换成真实 dataset 和更完整的训练流程。

## 推荐的可视化顺序

```text
MNIST prediction grid
  -> CNN feature map
  -> RNN hidden-state heatmap
  -> seq2seq loss curve
  -> attention alignment heatmap
  -> beam-search candidate tree
```

每次只改变一个因素，并记录：输入 shape、模型参数量、训练时间、validation/test 指标、图像或 heatmap 的变化。这样你看到的是设计选择带来的因果效果，而不是一次改很多设置后无法解释的结果。

## 与 slides 的关系

这些是教学实验，不是 slides 中的 benchmark 结果。它们刻意使用小型合成数据；不要把最终 accuracy 当成真实任务性能。实验的目标是验证数据流、shape、loss、state、decoder 和 attention 的机制。

## GitHub 参考

- [PyTorch examples](https://github.com/pytorch/examples)：官方示例集合，适合对照完整 training loop。
- [PyTorch MNIST example](https://github.com/pytorch/examples/tree/main/mnist)：将 MLP/CNN 实验替换为真实图像数据集时使用。
- [PyTorch char-RNN classification tutorial](https://github.com/pytorch/tutorials/blob/main/intermediate_source/char_rnn_classification_tutorial.py)：对照 RNN 的序列输入和分类输出。
- [PyTorch seq2seq translation tutorial](https://github.com/pytorch/tutorials/blob/main/intermediate_source/seq2seq_translation_tutorial.py)：对照 encoder-decoder、teacher forcing 和 attention。
- [Dive into Deep Learning](https://github.com/d2l-ai/d2l-en)：查阅 CNN、RNN、LSTM、GRU 和 attention 的更完整实验。
