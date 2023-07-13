# Tensorcircuit Installation Guide on MacOS

Contributed by [_Mark (Zixuan) Song_](https://marksong.tech)

Apple has updated Tensorflow (for MacOS) so that installation on M-series (until M2) and Intel-series Mac can follow the exact same procedure.

## Starting From Scratch

For completely new Macos or Macos without Xcode and Homebrew installed.

### Install Xcode Command Line Tools

<font color=gray><em>Need graphical access to the machine.</em></font>

Run `xcode-select --install` to install if on optimal internet.

Or Download it from [Apple](https://developer.apple.com/download/more/) Command Line Tools installation image then install it if the internet connection is weak.

## Install Miniconda

Due to the limitation of MacOS and packages, the latest version of Python does not always function as desired, thus miniconda installation is advised to solve the issues.

```bash
curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
source ~/miniconda/bin/activate
conda install -c apple tensorflow-deps
```

## Install TC Backends

There are four backends to choose from, Numpy, Tensorflow, Jax, and Torch.

### Install Jax, Pytorch, Qiskit, Cirq (Optional)

```bash
pip install [Package Name]
```

### Install Tensorflow (Optional)

#### Installation

For Tensorflow version 2.13 or later:
```bash
pip install tensorflow
pip install tensorflow-metal
```

For Tensorflow version 2.12 or earlier:
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

#### Verify Tensorflow Installation

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

## Install Tensorcircuit

```bash
pip install tensorcircuit
```

Until July 2023, this has been tested on Intel Macs running Ventura, M1 Macs running Ventura, M2 Macs running Ventura, and M2 Macs running Sonoma beta.