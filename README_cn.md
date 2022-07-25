<h1 align="center"> TENSORCIRCUIT </h1>

<p align="center">
  <!-- tests (GitHub actions) -->
  <a href="https://github.com/tencent-quantum-lab/tensorcircuit/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/workflow/status/tencent-quantum-lab/tensorcircuit/ci/master?logo=github&logo=github" />
  </a>
  <!-- docs -->
  <a href="https://tensorcircuit.readthedocs.io/">
    <img src="https://img.shields.io/badge/docs-link-green.svg?logo=read-the-docs"/>
  </a>
  <!-- PyPI -->
  <a href="https://pypi.org/project/tensorcircuit/">
    <img src="https://img.shields.io/pypi/v/tensorcircuit.svg?logo=pypi"/>
  </a>
  <!-- binder -->
  <a href="https://mybinder.org/v2/gh/refraction-ray/tc-env/master?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Ftencent-quantum-lab%252Ftensorcircuit%26urlpath%3Dlab%252Ftree%252Ftensorcircuit%252F%26branch%3Dmaster">
    <img src="https://mybinder.org/badge_logo.svg"/>
  </a>
  <!-- License -->
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?logo=apache"/>
  </a>
</p>

<p align="center"> <a href="README.md">English</a> |  简体中文 </p>

TensorCircuit 是下一代量子电路模拟器，支持自动微分、即时编译、硬件加速和向量并行化。

TensorCircuit 建立在现代机器学习框架之上，并且与机器学习后端无关。 它特别适用于量子经典混合范式和变分量子算法的高效模拟。

## 入门

请从 [快速上手](/docs/source/quickstart.rst) 和 [Jupyter 教程](/docs/source/tutorials) 开始。

有关更多信息和介绍，请参阅有用的 [示例脚本](/examples) 和 [完整文档](https://tensorcircuit.readthedocs.io/zh/latest/)。 [测试](/tests) 中的 API docstring 和测试用例也提供了丰富的信息。

以下是一些最简易的演示。

- 电路操作:

```python
import tensorcircuit as tc
c = tc.Circuit(2)
c.H(0)
c.CNOT(0,1)
c.rx(1, theta=0.2)
print(c.wavefunction())
print(c.expectation_ps(z=[0, 1]))
print(c.sample())
```

- 运行时特性定制:

```python
tc.set_backend("tensorflow")
tc.set_dtype("complex128")
tc.set_contractor("greedy")
```

- 使用即时编译 + 自动微分:

```python
def forward(theta):
    c = tc.Circuit(2)
    c.R(0, theta=theta, alpha=0.5, phi=0.8)
    return tc.backend.real(c.expectation((tc.gates.z(), [0])))

g = tc.backend.grad(forward)
g = tc.backend.jit(g)
theta = tc.array_to_tensor(1.0)
print(g(theta))
```

## 安装

该包是用纯 Python 编写的，可以通过 pip 直接获取：

```python
pip install tensorcircuit
```

我们推荐安装时同时安装 TensorFlow，这可以通过以下安装可选项实现：

```python
pip install tensorcircuit[tensorflow]
```

其他安装选项包括： `[torch]`, `[jax]` and `[qiskit]`。

此外我们有每日发布的最新版本 pip package，可以尝鲜开发的最新功能，请通过以下方式安装:

```python
pip uninstall tensorcircuit
pip install tensorcircuit-nightly
```

我们也有 [Docker 支持](/docker)。

## 优势

- 基于张量网络模拟引擎

- 即时编译、自动微分、向量并行化兼容，GPU 支持

- 效率

  - 时间：与 TFQ 或 Qiskit 相比，加速 10 到 10^6 倍

  - 空间：600+ qubits 1D VQE 工作流（收敛能量误差：< 1%）

- 优雅

  - 灵活性：自定义张量收缩、多种 ML 后端/接口选择、多种数值精度

  - API 设计：人类可理解的量子，更少的代码，更多的可能

## 引用

该项目由[腾讯量子实验室](https://quantum.tencent.com/)发布，现阶段由 [Shi-Xin Zhang](https://github.com/refraction-ray) 维护。

如果该软件对您的研究有帮助, 请引用我们的白皮书文章:

[TensorCircuit: a Quantum Software Framework for the NISQ Era](https://arxiv.org/abs/2205.10091).

## 贡献

有关贡献指南和说明，请参阅 [贡献](/CONTRIBUTING.md)。

我们欢迎大家提出 [issues](https://github.com/tencent-quantum-lab/tensorcircuit/issues), [PR](https://github.com/tencent-quantum-lab/tensorcircuit/pulls), 和 [讨论](https://github.com/tencent-quantum-lab/tensorcircuit/discussions)，这些都托管在 GitHub 上。

## 研究和应用

### DQAS

可微量子架构搜索的应用见 [应用](/tensorcircuit/applications)。
参考论文：https://arxiv.org/pdf/2010.08561.pdf

### VQNHE

关于变分量子神经混合本征求解器的应用，请参见 [应用](tensorcircuit/applications)。
参考论文：https://arxiv.org/pdf/2106.05105.pdf 和 https://arxiv.org/pdf/2112.10380.pdf 。

### VQEX - MBL

VQEX 在 MBL 相位识别上的应用见 [教程](/docs/source/tutorials/vqex_mbl.ipynb)。
参考论文: https://arxiv.org/pdf/2111.13719.pdf 。

```

```
