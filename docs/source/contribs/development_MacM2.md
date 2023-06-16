# Tensorcircuit Installation Guide on MacOS

Contributed by [Hong-Ye Hu](https://github.com/hongyehu)

The key issue addressed in this document is **how to install both TensorFlow and Jax on a M2 chip MacOS without conflict**. 

## Starting From Scratch

### Install Xcode Command Line Tools

<font color=gray><em>Need graphical access to the machine.</em></font>

Run `xcode-select --install` to install if on optimal internet.

Or Download from [Apple](https://developer.apple.com/download/more/) Command Line Tools installation image then install if internet connection is weak.

## Install Miniconda

Due to the limitation of MacOS and packages, the lastest version of python does not always function as desired, thus miniconda installation is advised to solve the issues. And use anaconda virtual environment is always a good habit.

```
curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
source ~/miniconda/bin/activate
```

## Install Packages
First, create a virtual environment, and make sure the python version is 3.8.5 by
```
conda create --name NewEnv python==3.8.5
conda activate NewEnv
```
Then, install the TensorFlow from `.whl` file (file is attached in *contribs* folder). This will install TensorFlow version 2.4.1
```
pip install ~/Downloads/tensorflow-2.4.1-py3-none-any.whl
```
Next, one need to install **Jax** and **Optax** by
```
conda install jax==0.3.0
conda install optax==0.1.4
```
Now, hopefully, you should be able to use both Jax and TensorFlow in this environment. But sometimes, it may give you an error "ERROR: package Chardet not found.". 
If that is the case, you can install it by `conda install chardet`.
Lastly, install tensorcircuit
```
pip install tensorcircuit
```
This is the solution that seems to work for M2-chip MacOS. Please let me know if there is a better solution!


