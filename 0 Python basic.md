# Python 基础：从零到深度学习

> [!tip] 老师讲解
> 本章正文就是从零开始的讲课内容：我会先解释每个对象，再把它连接到后面的神经网络代码。

> 目标：即使没有编程基础，也能读懂并改写常见的深度学习代码。
>
> 学习方式：每一节都先运行代码，再修改一个地方观察结果。不要只背语法。

## 0. 你最终要看懂什么

深度学习代码通常可以拆成这条流水线：

```text
读取数据 -> 清洗/变形 -> 放入模型 -> 计算 loss -> 反向传播 -> 更新参数 -> 验证
```


对应的 Python 工具大致是：

| 任务 | 常用工具 |
|---|---|
| 基础逻辑 | Python 变量、函数、循环、条件判断 |
| 数值计算 | NumPy |
| 表格数据 | pandas |
| 画图 | matplotlib |
| 深度学习 | PyTorch |

## 老师开场：为什么我不让你直接背 Python

你现在看到的不是编程考试，而是以后读懂神经网络的字母表。深度学习代码表面上有很多库和术语，真正反复出现的动作只有四个：**拿到一批数字、改变数字的形状、把数字送进函数、检查输出是否合理**。如果这四件事没有建立直觉，后面的 `CNN`、`RNN` 和 `Transformer` 看起来就会像魔法。

我先用一个极小的例子讲完整路径。假设我们有三个人的身高：`[1.60, 1.75, 1.82]`。这三个数可以放进 Python 的 `list`，也可以放进 NumPy 的 `array`，还可以变成 PyTorch 的 `tensor`。数字本身没有变，变的是我们能否一次对它们做向量化计算、能否记录梯度、能否交给 GPU 或 AI accelerator。

```python
import torch

height = torch.tensor([1.60, 1.75, 1.82])
centered = height - height.mean()
print(centered.shape)   # torch.Size([3])
print(centered)
```

这里 `shape` 的意思不是“数据有多大”这么笼统，而是**每一维分别有多少个元素**。`[3]` 表示一条长度为 3 的向量；如果有 4 个人、每个人 3 个特征，shape 就可能是 `[4, 3]`。后面神经网络里最容易出错的地方，往往不是公式，而是把 `[batch, feature]` 错当成 `[feature, batch]`。

### 老师怎样带你读一段深度学习代码

遇到陌生代码时，我会要求你按下面顺序读，而不是从第一行开始逐词翻译：

1. **先找数据**：输入变量是什么？一批有多少条样本？每条样本有多少个特征？
2. **再找 shape**：每个张量的每一维代表什么？`batch`、`channel`、`height`、`width` 不能混叫。
3. **再找模型**：模型接收什么 shape，返回什么 shape？
4. **最后找训练动作**：哪里计算 `loss`，哪里调用 `backward()`，哪里更新参数？

例如，下面这段代码不是四个互不相关的 API，而是一条完整的数据流：

```python
prediction = model(x)       # forward：输入经过模型
loss = loss_fn(prediction, y)
optimizer.zero_grad()       # 清除上一步残留的梯度
loss.backward()             # backward：沿计算图求 gradient
optimizer.step()            # update：用 gradient 改参数
```

`zero_grad()`、`backward()`、`step()` 的顺序很重要：如果不清除旧梯度，PyTorch 默认会把梯度累加；如果没有 `backward()`，优化器没有方向可走；如果没有 `step()`，模型只是在计算 loss，并没有学习。

### 这和 AI 芯片有什么关系

在软件里，一行 `y = x @ W` 看起来很短；在芯片里，它会变成大量乘法、加法、读写和同步。你以后看到 `tensor` 时，要同时问两件事：

- **语义问题**：这些数字分别代表什么？
- **系统问题**：这些数字放在哪里，怎样被重复使用？

这就是本章的真正任务：先把 Python、NumPy 和 PyTorch 的对象读懂，再把每个对象翻译成 shape、内存布局和计算操作。等你能看出这条数据流，再学习神经网络公式就不会只剩下背诵。

## 1. 第一个 Python 程序

### 1.1 什么是 `.py` 文件

`.py` 就是 Python 源代码文件。本质上是一个普通文本文件，只是文件名以 `.py` 结尾。例如：

```text
hello.py
train_model.py
data_prepare.py
```

不要把代码保存成 `hello.py.txt`。在文件管理器中如果看不到扩展名，需要打开“显示文件扩展名”。

### 1.2 用 VS Code 创建并运行

1. 新建一个文件夹，例如 `python-learning`。
2. 用 VS Code 打开这个文件夹。
3. 点击“新建文件”，命名为 `hello.py`。
4. 输入下面的代码并保存（`Command + S`）：

```python
print("我的第一个 Python 文件")
name = input("你叫什么名字？")
print(f"你好，{name}！")
```

5. 点击右上角的运行按钮，或在文件中右键选择“Run Python File”。

第一次运行时，如果 VS Code 提示选择 Python 解释器，选择已经安装的 Python 3（通常显示为 `Python 3.x`）。

### 1.3 用终端创建并运行

打开“终端”（macOS 可按 `Command + Space` 搜索 Terminal），进入项目文件夹：

```bash
cd ~/Desktop/python-learning
python3 hello.py
```

如果终端显示 `你好，...！`，说明运行成功。`cd` 是切换文件夹，`python3 hello.py` 是让 Python 解释器执行这个文件。

也可以先确认 Python 是否安装：

```bash
python3 --version
```

有些 Windows 环境使用 `python hello.py`，Linux/macOS 通常使用 `python3 hello.py`。看到 `command not found` 时，先安装 Python 3，并重新打开终端。

### 1.4 从最简单的脚本开始

建议按下面的目录组织学习代码：

```text
python-learning/
├── hello.py            # 入门练习
├── basics.py           # 语法练习
├── data_prepare.py     # 数据处理
├── train_model.py      # 模型训练
└── data/               # 数据文件
```

`.py` 文件中的代码从上到下执行。把可重复使用的代码写成函数，把真正启动程序的部分放在最后：

```python
def main():
    numbers = [1, 2, 3, 4]
    print(sum(numbers) / len(numbers))


if __name__ == "__main__":
    main()
```

`if __name__ == "__main__":` 的意思是：只有直接运行这个文件时，才执行 `main()`。以后从其他文件导入函数时，不会意外启动整个程序。

### 1.5 创建项目专用的虚拟环境

深度学习项目会用到很多第三方库。虚拟环境可以避免不同项目的库相互冲突。在项目文件夹中运行：

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# Windows PowerShell：.venv\\Scripts\\Activate.ps1
```

激活后，终端前面通常会出现 `(.venv)`。之后安装的库只属于这个项目：

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib
```

用完可以退出环境：

```bash
deactivate
```

VS Code 中也要选择 `.venv` 作为 Python 解释器，否则可能出现“明明安装了库却无法导入”的问题。

### 1.6 `.py`、交互式窗口和 Notebook 的区别

- `.py` 文件：适合保存完整、可重复运行的程序，例如 `train_model.py`。
- Python 交互式窗口：适合临时试一行代码。
- Jupyter Notebook（`.ipynb`）：适合把文字、图表和分段代码放在一起，常用于数据探索。

学习时可以先在交互式窗口试算式，再把确认可用的代码整理进 `.py` 文件。训练深度学习模型时，建议最终保留一个可以从头运行的 `.py` 脚本。

```python
print("Hello, deep learning!")

name = "小明"
age = 20
height = 1.75
is_student = True

print(name, age, height, is_student)
print(f"{name}今年{age}岁")  # f-string：把变量放进字符串
```

常见数据类型：

- `int`：整数，如 `3`
- `float`：小数，如 `3.14`
- `str`：字符串，如 `"cat"`
- `bool`：真假值，`True` / `False`
- `None`：暂时没有值

用 `type(x)` 查看类型，用 `#` 写注释。

```python
x = 3
print(type(x))       # <class 'int'>
print(x + 2, x ** 2) # 加法、乘方
```

### 1.1 数据类型详解

#### 数字类型：`int`、`float`、`complex`

```python
count = 10          # int，整数
temperature = 36.5  # float，浮点数
z = 2 + 3j          # complex，复数，初学阶段很少用

print(7 // 2)       # 3，整除
print(7 % 2)        # 1，余数
print(2 ** 3)       # 8，乘方
```

深度学习中的特征和模型参数通常是 `float`。类别标签常见为整数；在 PyTorch 中，分类标签经常需要 `torch.long` 类型。

#### `str`：字符串

```python
text = "Deep Learning"
print(len(text))
print(text.lower())       # 小写
print(text.upper())       # 大写
print(text[0:4])          # 切片：Deep
print("learning" in text.lower())

label = 3
message = "类别编号：" + str(label)  # 不同类型拼接前要转换
print(message)
```

字符串不可直接修改；需要修改时通常创建一个新字符串。`f"...{变量}..."` 是最推荐的格式化方式。

#### `bool`：布尔值

```python
is_ready = True
has_error = False
print(is_ready and not has_error)

print(bool(0), bool(1))       # False True
print(bool(""), bool("cat")) # False True
```

在 `if` 中，`0`、空字符串 `""`、空列表 `[]`、空字典 `{}` 和 `None` 通常会被当作 `False`。

#### `None`：没有结果

```python
result = None
if result is None:
    print("还没有计算结果")
```

判断是否为空值时使用 `is None`，不要写成 `result == None`。

### 1.2 类型转换与输入

```python
age_text = input("请输入年龄：")  # input 返回的永远是 str
age = int(age_text)
height = float("1.75")
print(age + 1, height)
```

常见转换函数：`int()`、`float()`、`str()`、`bool()`。转换前要确认内容合法，否则会出现 `ValueError`。

```python
text = "12.5"
number = float(text)
print(number)
```

### 1.3 比较运算与逻辑运算

```python
a, b = 3, 5
print(a == b)  # 等于
print(a != b)  # 不等于
print(a < b, a >= b)

score = 85
print(60 <= score < 90)  # Python 支持连续比较
print(score >= 60 and score <= 100)
```

注意：`=` 是赋值，`==` 才是比较。`and` 表示并且，`or` 表示或者，`not` 表示取反。

### 1.4 可变与不可变

`int`、`float`、`str`、`tuple` 通常看作不可变类型；`list`、`dict`、`set` 是可变类型。函数传入列表后，函数内部修改列表可能影响外部变量：

```python
def add_score(scores):
    scores.append(100)

my_scores = [80, 90]
add_score(my_scores)
print(my_scores)  # [80, 90, 100]
```

需要独立副本时，可以使用 `scores.copy()`。理解这一点有助于避免数据预处理时“改了原始数据”的问题。

### 1.5 列表推导式与解包

```python
numbers = [1, 2, 3, 4, 5]
squares = [n ** 2 for n in numbers]
even = [n for n in numbers if n % 2 == 0]
print(squares, even)

point = (10, 20)
x_coord, y_coord = point
print(x_coord, y_coord)
```

列表推导式适合简短的“遍历并生成新列表”操作；逻辑复杂时，使用普通 `for` 循环更容易阅读。

### 1.6 常用字典操作

```python
sample = {"image": "cat.jpg", "label": 0}
print(sample.get("label", -1))       # key 不存在时返回默认值
sample["size"] = 224

for key, value in sample.items():
    print(key, value)
```

`get()` 比直接写 `sample["missing"]` 更安全；后者在 key 不存在时会触发 `KeyError`。

## 2. 容器：list、tuple、dict、set

### 2.1 list：最常用的有序数据

```python
scores = [80, 92, 75, 88]
print(scores[0])      # 第一个元素，索引从 0 开始
print(scores[-1])     # 最后一个元素
print(scores[1:3])    # 切片：索引 1 到 2

scores.append(95)
scores[0] = 85
print(len(scores), sum(scores) / len(scores))
```

注意：`scores[1:3]` 不包含索引 3。图片、标签、文件名在预处理时经常先存在 list 中。

### 2.2 dict：用名字查值

```python
sample = {"image": "cat.jpg", "label": 0, "width": 224}
print(sample["label"])
sample["split"] = "train"
```

深度学习中的配置、一个样本的多个字段，常用 dictionary 表示。

### 2.3 tuple 与 set

```python
image_size = (224, 224)  # tuple 通常表示不想修改的组合
classes = {"cat", "dog", "cat"}
print(classes)            # set 自动去重
```

## 3. 条件判断与循环

```python
loss = 0.42
if loss < 0.5:
    print("目前还不错")
elif loss < 1.0:
    print("还需要训练")
else:
    print("检查数据或模型")
```

Python 用缩进表示代码块，通常使用 4 个空格。不要混用 tab 和空格。

```python
for epoch in range(3):
    print(f"第 {epoch + 1} 轮")

predictions = [0.2, 0.7, 0.9]
for index, prediction in enumerate(predictions):
    print(index, prediction)
```

`range(3)` 产生 `0, 1, 2`。训练模型时，`for epoch` 和 `for batch` 就是这种循环。

## 4. 函数：把重复逻辑封装起来

```python
def mean(values):
    """返回一组数字的平均值。"""
    if len(values) == 0:
        raise ValueError("values 不能为空")
    return sum(values) / len(values)

print(mean([1, 2, 3]))
```

函数的核心是：输入（参数）→处理→输出（`return`）。深度学习代码中，模型的 `forward`、数据预处理函数、评估函数都是函数。

```python
def accuracy(predicted, target):
    correct = sum(p == t for p, t in zip(predicted, target))
    return correct / len(target)

print(accuracy([1, 0, 1], [1, 1, 1]))
```

## 5. 模块、导入与可复现随机数

```python
import math
import random

print(math.sqrt(16))
random.seed(42)  # 固定随机种子，方便复现实验
print([random.random() for _ in range(3)])
```

`import` 是把别人写好的功能拿来用。项目中常见：

```python
import numpy as np
import torch
```

## 6. NumPy：理解数组、shape 和向量化

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
print(x.shape, x.dtype)       # (3,) 和 float64
print(x + 10, x * 2)

matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.shape)           # (2, 3)：2 行、3 列
print(matrix[:, 1])           # 所有行的第 2 列
print(matrix.mean(axis=0))    # 沿行方向聚合，结果有 3 个数
```

### 6.1 深度学习最重要的 shape 思维

```text
灰度图片：       (height, width)
一批灰度图片：   (batch, height, width)
彩色图片：       (channels, height, width)
一批彩色图片：   (batch, channels, height, width)
文本 token：     (batch, sequence_length)
```

遇到错误时，先打印：

```python
print("x.shape =", x.shape)
print("x.dtype =", x.dtype)
```

### 6.2 矩阵乘法

```python
weights = np.array([[0.5, -1.0], [1.0, 0.2]])  # (2, 2)
features = np.array([2.0, 3.0])                # (2,)
print(weights @ features)                      # (2,)
```

`@` 是矩阵乘法；`*` 是逐元素相乘。神经网络中的线性层本质上大量使用矩阵乘法。

### 6.3 广播：为什么不同 shape 有时也能相加

NumPy/PyTorch 会按从最右侧开始对齐维度。两个维度相等，或其中一个是 `1`，就可以广播（broadcast）到共同形状；否则会报 shape 错误。比如 `(3, 1)` 和 `(1, 4)` 相加，结果是 `(3, 4)`，相当于把列向量复制到 4 列、把行向量复制到 3 行。广播通常不需要真的复制整块数据，但逻辑结果确实覆盖了所有组合。

```python
import numpy as np

rows = np.array([[10], [20], [30]])  # shape (3, 1)
cols = np.array([[1, 2, 3, 4]])      # shape (1, 4)
grid = rows + cols                    # shape (3, 4)
print(grid)
```

输出每个格子都是 `rows` 中一个数加上 `cols` 中一个数。初学时遇到 shape 错误，先把两个 shape 从右往左对齐；不要把 broadcasting 和矩阵乘法混为一谈，`+` 通常逐元素运算，`@` 才表示矩阵乘法。

## 7. pandas 与 matplotlib：先看懂数据

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({"epoch": [1, 2, 3], "loss": [1.2, 0.7, 0.4]})
print(df.head())
print(df["loss"].mean())

plt.plot(df["epoch"], df["loss"], marker="o")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
```

训练时把每轮的 loss 保存下来再画图，可以判断模型是否真的在学习。

## 8. 文件与异常：让实验更可靠

```python
from pathlib import Path

data_dir = Path("data")
print(data_dir.exists())

try:
    number = int("abc")
except ValueError as error:
    print("转换失败：", error)
```

常见异常：

- `NameError`：变量名不存在，检查拼写和执行顺序。
- `TypeError`：类型不匹配，例如字符串和数字直接相加。
- `IndexError`：索引超出范围。
- `KeyError`：字典中没有这个 key。
- `FileNotFoundError`：路径写错，或当前工作目录不是你以为的目录。
- `RuntimeError: size mismatch`：模型层或数据的 shape 对不上。

## 9. PyTorch 张量：NumPy 的深度学习版本

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape, x.dtype, x.device)
print(x.mean(), x @ x.T)
```

张量（tensor）和 NumPy array 很像，但它可以放到 GPU，并记录自动求导信息。

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
print(x.grad)  # dy/dx = 2x + 3，在 x=2 时为 7
```

`requires_grad=True` 表示需要记录计算过程；`backward()` 计算梯度。

## 10. 用 PyTorch 写第一个可训练模型

下面的模型学习一条直线 `y = 2x + 1`。它展示了训练的完整骨架。

```python
import torch
from torch import nn

torch.manual_seed(42)
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

model = nn.Linear(in_features=1, out_features=1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for epoch in range(200):
    prediction = model(x)       # forward：前向计算
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()       # 清空上一步梯度
    loss.backward()             # backward：反向传播
    optimizer.step()            # 更新 weight 和 bias

    if (epoch + 1) % 50 == 0:
        print(epoch + 1, loss.item())

print(model.weight.item(), model.bias.item())
print(model(torch.tensor([[5.0]])).item())
```

记住这 5 步：

1. `prediction = model(x)`
2. `loss = loss_fn(prediction, y)`
3. `optimizer.zero_grad()`
4. `loss.backward()`
5. `optimizer.step()`

## 11. Dataset、DataLoader 与 batch

```python
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_x, batch_y in loader:
    print(batch_x.shape, batch_y.shape)
```

`batch_size` 表示一次送进模型多少个样本；`shuffle=True` 每轮打乱训练数据。真实项目通常把读取、预处理写进自定义 `Dataset`。

## 12. 分类模型的最小模板

```python
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

model = BinaryClassifier(input_dim=4)
loss_fn = nn.BCEWithLogitsLoss()  # 内部包含 sigmoid，更稳定
```

二分类标签通常是 `0/1`，多分类常用 `nn.CrossEntropyLoss()`，其标签是类别编号 `0, 1, 2, ...`。

## 13. 训练与验证模式

```python
model.train()  # 训练模式
for batch_x, batch_y in loader:
    ...

model.eval()   # 验证/测试模式
with torch.no_grad():
    prediction = model(x)
```

`eval()` 会改变 Dropout、BatchNorm 等层的行为；`no_grad()` 节省显存并避免记录梯度。验证集不要调用 `optimizer.step()`。

### 13.1 数据切分：防止答案偷偷进入训练

训练集（training set）用于更新参数；验证集（validation set）用于选择模型和超参数；测试集（test set）留到最后，用来估计模型在未参与开发的数据上的表现。如果反复根据 test 分数改模型，test 就间接参与了开发，分数会变得乐观，这叫 test-set leakage（测试集泄漏）。

另一个常见泄漏是先对完整数据计算均值/标准差、词表或特征选择，再切分数据。即使没有标签，验证/测试集的分布信息也可能泄漏进训练流程。通常先切分，再只用 training set 拟合预处理器，并把同一变换应用到验证和测试数据。时间序列还必须按时间先后切分，不能随机打乱后让未来信息进入过去。

**读分数时的边界**：一个 test score 只代表所用数据、切分方式、指标和评估流程，不保证部署环境表现。类别不平衡时，accuracy 可能掩盖少数类失败；要结合任务选择 precision、recall、F1 或其他合适指标。

## 14. 建议的学习顺序与练习

### 第 1 阶段：Python 语法

- 手写一个函数，输入一组分数，返回平均分、最高分和最低分。
- 读取一个字典列表，筛选出 `label == 1` 的样本。
- 用循环打印 1 到 100 中所有能被 3 整除的数字。

### 第 2 阶段：NumPy 与数据

- 创建形状为 `(3, 4)` 的随机矩阵，分别计算每行和每列的平均值。
- 把一维数组 reshape 成 `(2, 5)`，并解释每个维度代表什么。
- 画出一条下降的 loss 曲线。

### 第 3 阶段：PyTorch

- 修改第 10 节代码，让模型学习 `y = -3x + 2`。
- 把 `batch_size` 改成 1、4，观察训练速度和 loss。
- 给模型增加一层 `nn.Linear` 和一个 `nn.ReLU()`。

### 第 4 阶段：小项目

用 20 个二维点做二分类：

1. 生成坐标 `(x1, x2)` 和标签 `0/1`。
2. 用 `DataLoader` 分 batch。
3. 写一个含 `Linear + ReLU + Linear` 的模型。
4. 记录 train loss、validation accuracy。
5. 画 loss 曲线和分类结果散点图。

## 15A. Python 基础练习（建议逐题创建 `.py` 文件）

练习时建议新建 `practice_01.py`、`practice_02.py` 等文件。先自己写，再对照提示；不要一开始就看答案。

### 练习 1：个人信息

创建 `practice_01.py`：定义姓名、年龄、身高、是否为学生 4 个变量，并用 f-string 打印一句完整介绍。

示例输出：

```text
我叫小明，今年20岁，身高1.75米，学生=True
```

检查点：年龄应是 `int`，身高应是 `float`，学生状态应是 `bool`。

### 练习 2：温度转换

创建 `practice_02.py`：把摄氏温度转换成华氏温度。

公式：`华氏 = 摄氏 × 9 / 5 + 32`。输入 `0` 时应输出 `32`。

```python
celsius = float(input("摄氏温度："))
fahrenheit = celsius * 9 / 5 + 32
print(f"华氏温度：{fahrenheit:.1f}")
```

### 练习 3：成绩判断

创建 `practice_03.py`：输入一个 0 到 100 的分数，根据分数输出等级：

- 90 及以上：A
- 80–89：B
- 60–79：C
- 60 以下：D

额外挑战：如果输入小于 0 或大于 100，输出“分数无效”。

### 练习 4：列表统计

给定：

```python
scores = [72, 88, 91, 65, 100, 58]
```

分别计算总分、平均分、最高分、最低分，并统计及格人数（分数大于等于 60）。

提示：使用 `sum()`、`len()`、`max()`、`min()`，及格人数可以用循环或列表推导式完成。

### 练习 5：字符串处理

输入一行英文句子，完成以下操作：

1. 输出字符数；
2. 转成小写；
3. 统计单词数量；
4. 判断是否包含 `python`。

提示：`sentence.lower()`、`sentence.split()`、`"python" in sentence.lower()`。

### 练习 6：字典中的样本信息

创建一个字典表示一张图片：

```python
sample = {"filename": "cat_001.jpg", "label": 0, "width": 224, "height": 224}
```

完成以下操作：

- 打印文件名和标签；
- 增加 `split: "train"`；
- 计算图片像素总数 `width * height`；
- 使用 `get()` 读取可能不存在的 `channels`，默认值为 3。

### 练习 7：循环与损失值

给定每轮训练的损失：

```python
losses = [1.20, 0.95, 0.72, 0.60, 0.48]
```

使用 `enumerate()` 打印：

```text
Epoch 1: loss=1.20
Epoch 2: loss=0.95
...
```

额外挑战：判断 loss 是否一直下降；如果某一轮比上一轮大，打印警告。

### 练习 8：写一个可复用函数

编写函数 `summarize_scores(scores)`，返回一个字典：

```python
{"mean": ..., "max": ..., "min": ..., "passed": ...}
```

其中 `passed` 是及格人数。传入空列表时，抛出 `ValueError("scores 不能为空")`。

### 练习 9：NumPy 形状练习

```python
import numpy as np

data = np.arange(12).reshape(3, 4)
```

完成以下操作：

- 打印 `data.shape`；
- 取出第 2 行；
- 取出第 3 列；
- 计算每一行的平均值；
- 把它 reshape 成 `(4, 3)`。

写一句话说明：在机器学习中，`(3, 4)` 可以表示 3 个样本、每个样本有 4 个特征。

### 练习 10：第一个 PyTorch 预测

用 `torch.tensor` 创建：

```python
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])
```

创建 `nn.Linear(1, 1)`，让模型学习 `y = 2x`。训练 100 轮后，输入 `4.0`，预测结果应接近 `8.0`。

检查点：每轮都要按顺序执行 `zero_grad()`、`backward()`、`step()`。

### 练习的自我检查

- 能否不看笔记解释 `=` 和 `==` 的区别？
- 能否说出一个 list、dict 和 NumPy array 的使用场景？
- 能否根据 `x.shape` 判断 batch 维度在哪里？
- 能否独立写出 PyTorch 训练循环的 5 个关键步骤？

## 15. 写深度学习代码时的检查清单

- 数据和标签的数量是否相同？
- `print(x.shape, y.shape)` 后，维度是否符合模型输入？
- 分类标签是否是正确的 dtype（`CrossEntropyLoss` 通常需要 `torch.long`）？
- 训练前是否调用 `model.train()`，验证前是否调用 `model.eval()`？
- 每个 batch 是否执行了 `zero_grad → backward → step`？
- loss 是下降还是出现 `nan`？学习率是否太大？
- 是否误把测试集当成训练集使用？

## 16. 下一步连接到现有课程笔记

- 先读 [[1 Basic neural networks]]，把这里的 `Linear`、`ReLU`、`loss`、`backward` 对上数学概念。
- 再读 [[2 CNN]]，重点观察图片张量 shape 和 batch 维度。
- 接着读 [[3 RNN]]，重点理解序列顺序与 hidden state。
- 学习 [[5 Attention and Transformer]] 时，重点关注 token、embedding、sequence length 这些张量维度。
- 做实验时，保留随机种子、训练曲线和验证结果，便于复现与比较。

## External context：官方入门资料

- Python Software Foundation，[The Python Tutorial](https://docs.python.org/3/tutorial/)，第 3、4、5 章分别讲控制流、数据结构和模块；本笔记为零基础读者增加了逐步示例。访问日期：2026-10-22。
- PyTorch，[Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)，覆盖 tensor、Dataset/DataLoader、建模、autograd、优化和保存模型；适合把第 9–13 节映射到框架工作流。访问日期：2026-10-22。

---

## 一句话记忆

Python 负责表达“怎么处理数据”，NumPy/PyTorch 负责高效计算，神经网络负责学习参数；当你能看懂 `shape → forward → loss → backward → step`，就已经掌握了深度学习代码的主干。
