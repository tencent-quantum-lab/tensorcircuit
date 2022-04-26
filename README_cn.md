<h1 align="center"> TENSORCIRCUIT </h1>

<p align="center">
  <!-- tests (GitHub actions) -->
  <a href="https://github.com/quclub/tensorcircuit-dev/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/workflow/status/quclub/tensorcircuit-dev/ci/master?logo=github&style=flat-square&logo=github" />
  </a>
  <!-- docs -->
  <a href="">
    <img src="https://img.shields.io/badge/docs-link-green.svg?style=flat-square&logo=read-the-docs"/>
  </a>
  <!-- PyPI -->
  <a href="https://pypi.org/project/tensorcircuit/">
    <img src="https://img.shields.io/pypi/v/tensorcircuit.svg?style=flat-square&logo=pypi"/>
  </a>
  <!-- License -->
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square&logo=apache"/>
  </a>
</p>

<p align="center"> <a href="README.md">English</a> |  简体中文 </p>

TensorCircuit 是下一代量子电路模拟器，支持自动微分、即时编译、硬件加速和矢量化并行。

TensorCircuit 建立在现代机器学习框架之上，并且与机器学习后端无关。 它特别适用于量子经典混合范式和变分量子算法的高效模拟。

## 入门

请从 [快速上手](/docs/source/quickstart.rst) 和 [Jupyter 教程](/docs/source/tutorials) 开始。

有关更多信息和介绍，请参阅有用的 [示例脚本](/examples) 和 [完整文档](/docs/source)。 [测试](/tests) 中的 API docstring 和测试用例也提供了丰富的信息。

以下是一些最简易的演示。

- 电路操纵:

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

- 使用 jit 进行自动微分:

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

该包纯粹是用python编写的，可以通过pip获取：

```python
pip install tensorcircuit
```

我们也有 [Docker 支持](/docker)。

## 优势

* 基于张量网络仿真引擎

* JIT、AD、矢量化并行兼容，GPU 支持

* 效率

   * 时间：与 tfq 或 qiskit 相比，加速 10 到 10^6 倍

   * 空间：600+ qubits 1D VQE 工作流程（融合能量不准确度：< 1%）

* 优雅

   * 灵活性：自定义收缩、多种 ML 后端/接口选择、多种 dtype 精度

   * API 设计：人类可理解的量子，更少的代码，更多的力量

## 贡献

有关贡献指南和说明，请参阅 [贡献](/CONTRIBUTING.md)。

我们欢迎大家提出问题、PR 和讨论，这些都托管在 GitHub 上。

## 研究和应用

### DQAS

可微量子架构搜索的应用见 [应用](/tensorcircuit/applications)。
参考论文：https://arxiv.org/pdf/2010.08561.pdf

### VQNHE

关于变分量子神经混合本征求解器的应用，请参见 [应用](tensorcircuit/applications)。
参考论文：https://arxiv.org/pdf/2106.05105.pdf 和 https://arxiv.org/pdf/2112.10380.pdf 。

### VQEX - MBL

VQEX 在 MBL 相位识别上的应用见 [教程](https://github.com/quclub/tensorcircuit-tutorials/blob/master/tutorials/vqex_mbl.ipynb)。
参考论文: https://arxiv.org/pdf/2111.13719.pdf 。

