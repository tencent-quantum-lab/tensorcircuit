# MacOS Tensorcircuit 安装教程

[_Mark (Zixuan) Song_](https://marksong.tech) 撰写

由于苹果更新了Tensorflow，因此M系列（直到M2）和英特尔系列Mac上的安装可以遵循完全相同的过程。

## 从头开始

对于全新的Macos或未安装Xcode和Homebrew的Macos。

### 安装Xcode命令行工具

<font color=gray><em>需要对机器的图形访问。</em></font>

如果网络良好，请运行`xcode-select --install`进行安装。

或者，如果网络连接较弱，请从[苹果](https://developer.apple.com/download/more/)下载命令行工具安装映像，然后进行安装。

## 安装Miniconda

由于MacOS和软件包的限制，因此建议安装miniconda以解决问题。

```bash
curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
source ~/miniconda/bin/activate
conda install -c apple tensorflow-deps
```

## 安装TC后端

有四个后端可供选择，Numpy，Tensorflow，Jax和Torch。

### 安装Jax，Pytorch，Qiskit，Cirq（可选）

```bash
pip install [Package Name]
```

### 安装Tensorflow（可选）

#### 安装步骤

Tensorflow版本2.13或之后：
```bash
pip install tensorflow
pip install tensorflow-metal
```

Tensorflow版本2.12或之前：
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

#### 验证Tensorflow安装

```python
import tensorflow as tf

cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64)
```

## 安装Tensorcircuit

```bash
pip install tensorcircuit
```

直到2023年7月，这已在运行Ventura的英特尔i9 Mac、运行Ventura的M1 Mac、运行Ventura的M2 Mac、运行Sonoma测试版的M2 Mac上进行了测试。