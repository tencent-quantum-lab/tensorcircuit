# Tensorcircuit Installation Guide on MacOS

Contributed by [_Mark (Zixuan) Song_](https://marksong.tech)

Apple has updated Tensorflow (for MacOS) so that installation on M-series (until M2) and Intel-series Mac can follow the exact same procedure.

## Starting From Scratch

For completely new Macos or Macos without Xcode installed.

If you have Xcode installed, skip to Install TC backends.

### Install Xcode Command Line Tools

<font color=gray><em>Need graphical access to the machine.</em></font>

Run `xcode-select --install` to install if on optimal internet.

Or Download it from [Apple](https://developer.apple.com/download/more/) Command Line Tools installation image then install it if the internet connection is weak.

## Install TC Backends

There are four backends to choose from, Numpy, Tensorflow, Jax, and Torch.

### Install Jax, Pytorch (Optional)

```bash
pip install [Package Name]
```
### Install Tensorflow (Optional - Recommended)

#### Install Miniconda (Optional - Recommended)

If you wish to install Tensorflow optimized for MacOS (`tensorflow-macos`) or Tensorflow GPU optimized (`tensorflow-metal`) please install miniconda.

If you wish to install Vanilla Tensorflow developed by Google (`tensorflow`) please skip this step.

```bash
curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
source ~/miniconda/bin/activate
conda install -c apple tensorflow-deps
```

#### Installation

```bash
pip install tensorflow
```

If you wish to use tensorflow-metal PluggableDevice, then continue install (not recommended):

```bash
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

## Benchmarking

This data is collected by running `benchmarks/scripts/vqe_tc.py` 10 times and average results.

<table>
  <tr>
    <th></th>
    <th>Vanilla Tensorflow</th>
    <th>Apple Tensorflow</th>
    <th>Apple Tensorflow with Metal Plugin</th>
  </tr>
  <tr>
    <td>Construction Time</td>
    <td>11.49241641s</td>
    <td>11.31878941s</td>
    <td>11.6103961s</td>
  </tr>
  <tr>
    <td>Iteration time</td>
    <td>0.002313011s</td>
    <td>0.002333004s</td>
    <td>0.046412581s</td>
  </tr>
  <tr>
    <td>Total time</td>
    <td>11.72371747s</td>
    <td>11.55208979s</td>
    <td>16.25165417s</td>
  </tr>
</table>


Until July 2023, this has been tested on Intel Macs running Ventura, M1 Macs running Ventura, M2 Macs running Ventura, and M2 Macs running Sonoma beta.