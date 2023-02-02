# Tensorcircuit Installation Guide on MacOS

Contributed by Mark (Zixuan) Song

## Starting From Scratch
For completely new macos or macos without xcode and brew
### Install Xcode Command Line Tools
<font color=gray><em>Need graphical access to the machine.</em></font> 

Run `xcode-select --install` to install if on optimal internet.

Or Download from [Apple](https://developer.apple.com/download/more/)  Command Line Tools installation image then install if internet connection is weak.
## Install Miniconda
Due to the limitation of MacOS and packages, the lastest version of python does not always function as desired, thus miniconda installation is advised to solve the issues.
```
curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
source ~/miniconda/bin/activate
conda install -c apple tensorflow-deps
```
## Install TC Prerequisites
```
pip install numpy scipy tensornetwork networkx
```
## Install TC Backends
There are four backends to choose from, Tensorflow, Jax, Torch.
### Install Jax, Pytorch, Qiskit, Cirq (Optional)
```
pip install [Package Name]
```
### Install Tensorflow (Optional)
#### Install Tensorflow (Recommended Approach)
❗️ Tensorflow with MacOS optimization would not function correctly in version 2.11.0 and before. Do not use this version of tensorflow if you intented to train any machine learning model.

FYI:  Error can occur when machine learning training or gpu related code is involved.

⚠️ Tensorflow without macos optimization does not support Metal API and utilizing GPU (both intel chips and M-series chips) until at least tensorflow 2.11. Tensorflow-macos would fail when running `tc.backend.to_dense()`
```
conda config --add channels conda-forge 
conda config --set channel_priority strict
conda create -n tc_venv python tensorflow=2.7
```
#### Verify Tensorflow Installation
```
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
```
pip install tensorcircuit
```

Testing Platform
- Platform 1:
	- MacOS Ventura 13.1 (Build version 22C65)
	- M1 Ultra
- Platform 2:
	- MacOS Ventura 13.2 (Build version 22D49)
	- M1 Ultra (Virtual)