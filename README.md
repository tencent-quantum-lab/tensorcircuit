<p align="center">
  <a href="https://github.com/tencent-quantum-lab/tensorcircuit">
    <img width=90% src="docs/source/statics/logov2.jpg">
  </a>
</p>

<p align="center">
  <!-- tests (GitHub actions) -->
  <a href="https://github.com/tencent-quantum-lab/tensorcircuit/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/tencent-quantum-lab/tensorcircuit/ci.yml?branch=master" />
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

<p align="center"> English | <a href="README_cn.md"> 简体中文 </a></p>

TensorCircuit is the next generation of quantum software framework with support for automatic differentiation, just-in-time compiling, hardware acceleration, and vectorized parallelism.

TensorCircuit is built on top of modern machine learning frameworks: Jax, TensorFlow, and PyTorch. It is specifically suitable for highly efficient simulations of quantum-classical hybrid paradigm and variational quantum algorithms in ideal, noisy and approximate cases. It also supports real quantum hardware access and provides CPU/GPU/QPU hybrid deployment solutions since v0.9.

## Getting Started

Please begin with [Quick Start](/docs/source/quickstart.rst) in the [full documentation](https://tensorcircuit.readthedocs.io/).

For more information on software usage, sota algorithm implementation and engineer paradigm demonstration, please refer to 70+ [example scripts](/examples) and 30+ [tutorial notebooks](https://tensorcircuit.readthedocs.io/en/latest/#tutorials). API docstrings and test cases in [tests](/tests) are also informative.

The following are some minimal demos.

- Circuit manipulation:

```python
import tensorcircuit as tc
c = tc.Circuit(2)
c.H(0)
c.CNOT(0,1)
c.rx(1, theta=0.2)
print(c.wavefunction())
print(c.expectation_ps(z=[0, 1]))
print(c.sample(allow_state=True, batch=1024, format="count_dict_bin"))
```

- Runtime behavior customization:

```python
tc.set_backend("tensorflow")
tc.set_dtype("complex128")
tc.set_contractor("greedy")
```

- Automatic differentiations with jit:

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

<details>
  <summary> More highlight features for TensorCircuit (click for details) </summary>

- Sparse Hamiltonian generation and expectation evaluation:

```python
n = 6
pauli_structures = []
weights = []
for i in range(n):
    pauli_structures.append(tc.quantum.xyz2ps({"z": [i, (i + 1) % n]}, n=n))
    weights.append(1.0)
for i in range(n):
    pauli_structures.append(tc.quantum.xyz2ps({"x": [i]}, n=n))
    weights.append(-1.0)
h = tc.quantum.PauliStringSum2COO(pauli_structures, weights)
print(h)
# BCOO(complex64[64, 64], nse=448)
c = tc.Circuit(n)
c.h(range(n))
energy = tc.templates.measurements.operator_expectation(c, h)
# -6
```

- Large-scale simulation with tensor network engine

```python
# tc.set_contractor("cotengra-30-10")
n=500
c = tc.Circuit(n)
c.h(0)
c.cx(range(n-1), range(1, n))
c.expectation_ps(z=[0, n-1], reuse=False)
```

- Density matrix simulator and quantum info quantities

```python
c = tc.DMCircuit(2)
c.h(0)
c.cx(0, 1)
c.depolarizing(1, px=0.1, py=0.1, pz=0.1)
dm = c.state()
print(tc.quantum.entropy(dm))
print(tc.quantum.entanglement_entropy(dm, [0]))
print(tc.quantum.entanglement_negativity(dm, [0]))
print(tc.quantum.log_negativity(dm, [0]))
```

</details>

## Install

The package is written in pure Python and can be obtained via pip as:

```python
pip install tensorcircuit
```

We recommend you install this package with tensorflow also installed as:

```python
pip install tensorcircuit[tensorflow]
```

Other optional dependencies include `[torch]`, `[jax]`, `[qiskit]` and `[cloud]`.

For the nightly build of tensorcircuit with new features, try:

```python
pip uninstall tensorcircuit
pip install tensorcircuit-nightly
```

We also have [Docker support](/docker).

## Advantages

- Tensor network simulation engine based

- JIT, AD, vectorized parallelism compatible

- GPU support, quantum device access support, hybrid deployment support

- Efficiency

  - Time: 10 to 10^6+ times acceleration compared to TensorFlow Quantum, Pennylane or Qiskit

  - Space: 600+ qubits 1D VQE workflow (converged energy inaccuracy: < 1%)

- Elegance

  - Flexibility: customized contraction, multiple ML backend/interface choices, multiple dtype precisions, multiple QPU providers

  - API design: quantum for humans, less code, more power

- Batteries included

  <details>
  <summary> Tons of amazing features and built in tools for research (click for details) </summary>

  - Support **super large circuit simulation** using tensor network engine.

  - Support **noisy simulation** with both Monte Carlo and density matrix (tensor network powered) modes.

  - Support **approximate simulation** with MPS-TEBD modes.

  - Support **analog/digital hybrid simulation** (time dependent Hamiltonian evolution, **pulse** level simulation) with neural ode modes.

  - Support **Fermion Gaussian state** simulation with expectation, entanglement, measurement, ground state, real and imaginary time evolution.

  - Support **qudits simulation**.

  - Support **parallel** quantum circuit evaluation across **multiple GPUs**.

  - Highly customizable **noise model** with gate error and scalable readout error.

  - Support for **non-unitary** gate and post-selection simulation.

  - Support **real quantum devices access** from different providers.

  - **Scalable readout error mitigation** native to both bitstring and expectation level with automatic qubit mapping consideration.

  - **Advanced quantum error mitigation methods** and pipelines such as ZNE, DD, RC, etc.

  - Support **MPS/MPO** as representations for input states, quantum gates and observables to be measured.

  - Support **vectorized parallelism** on circuit inputs, circuit parameters, circuit structures, circuit measurements and these vectorization can be nested.

  - Gradients can be obtained with both **automatic differenation** and parameter shift (vmap accelerated) modes.

  - **Machine learning interface/layer/model** abstraction in both TensorFlow and PyTorch for both numerical simulation and real QPU experiments.

  - Circuit sampling supports both final state sampling and perfect sampling from tensor networks.

  - Light cone reduction support for local expectation calculation.

  - Highly customizable tensor network contraction path finder with opteinsum interface.

  - Observables are supported in measurement, sparse matrix, dense matrix and MPO format.

  - Super fast weighted sum Pauli string Hamiltonian matrix generation.

  - Reusable common circuit/measurement/problem templates and patterns.

  - Jittable classical shadow infrastructures.

  - SOTA quantum algorithm and model implementations.

  - Support hybrid workflows and pipelines with CPU/GPU/QPU hardware from local/cloud/hpc resources using tf/torch/jax/cupy/numpy frameworks all at the same time.

  </details>

## Contributing

### Status

This project is created and maintained by [Shi-Xin Zhang](https://github.com/refraction-ray) with current core authors [Shi-Xin Zhang](https://github.com/refraction-ray) and [Yu-Qin Chen](https://github.com/yutuer21). We also thank [contributions](https://github.com/tencent-quantum-lab/tensorcircuit/graphs/contributors) from the open source community.

### Citation

If this project helps in your research, please cite our software whitepaper to acknowledge the work put into the development of TensorCircuit.

[TensorCircuit: a Quantum Software Framework for the NISQ Era](https://quantum-journal.org/papers/q-2023-02-02-912/) (published in Quantum)

which is also a good introduction to the software.

Research works citing TensorCircuit can be highlighted in [Research and Applications section](https://github.com/tencent-quantum-lab/tensorcircuit#research-and-applications).

### Guidelines

For contribution guidelines and notes, see [CONTRIBUTING](/CONTRIBUTING.md).

We welcome [issues](https://github.com/tencent-quantum-lab/tensorcircuit/issues), [PRs](https://github.com/tencent-quantum-lab/tensorcircuit/pulls), and [discussions](https://github.com/tencent-quantum-lab/tensorcircuit/discussions) from everyone, and these are all hosted on GitHub.

### License

TensorCircuit is open source, released under the Apache License, Version 2.0.

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="16.66%"><a href="https://re-ra.xyz"><img src="https://avatars.githubusercontent.com/u/35157286?v=4?s=100" width="100px;" alt="Shixin Zhang"/><br /><sub><b>Shixin Zhang</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=refraction-ray" title="Code">💻</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=refraction-ray" title="Documentation">📖</a> <a href="#example-refraction-ray" title="Examples">💡</a> <a href="#ideas-refraction-ray" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-refraction-ray" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-refraction-ray" title="Maintenance">🚧</a> <a href="#research-refraction-ray" title="Research">🔬</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/pulls?q=is%3Apr+reviewed-by%3Arefraction-ray" title="Reviewed Pull Requests">👀</a> <a href="#translation-refraction-ray" title="Translation">🌍</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=refraction-ray" title="Tests">⚠️</a> <a href="#tutorial-refraction-ray" title="Tutorials">✅</a> <a href="#talk-refraction-ray" title="Talks">📢</a> <a href="#question-refraction-ray" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/yutuer21"><img src="https://avatars.githubusercontent.com/u/83822724?v=4?s=100" width="100px;" alt="Yuqin Chen"/><br /><sub><b>Yuqin Chen</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=yutuer21" title="Code">💻</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=yutuer21" title="Documentation">📖</a> <a href="#example-yutuer21" title="Examples">💡</a> <a href="#ideas-yutuer21" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-yutuer21" title="Research">🔬</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=yutuer21" title="Tests">⚠️</a> <a href="#tutorial-yutuer21" title="Tutorials">✅</a> <a href="#talk-yutuer21" title="Talks">📢</a></td>
      <td align="center" valign="top" width="16.66%"><a href="http://jiezhongqiu.com"><img src="https://avatars.githubusercontent.com/u/3853009?v=4?s=100" width="100px;" alt="Jiezhong Qiu"/><br /><sub><b>Jiezhong Qiu</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=xptree" title="Code">💻</a> <a href="#example-xptree" title="Examples">💡</a> <a href="#ideas-xptree" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-xptree" title="Research">🔬</a></td>
      <td align="center" valign="top" width="16.66%"><a href="http://liwt31.github.io"><img src="https://avatars.githubusercontent.com/u/22628546?v=4?s=100" width="100px;" alt="Weitang Li"/><br /><sub><b>Weitang Li</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=liwt31" title="Code">💻</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=liwt31" title="Documentation">📖</a> <a href="#ideas-liwt31" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-liwt31" title="Research">🔬</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=liwt31" title="Tests">⚠️</a> <a href="#talk-liwt31" title="Talks">📢</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/SUSYUSTC"><img src="https://avatars.githubusercontent.com/u/30529122?v=4?s=100" width="100px;" alt="Jiace Sun"/><br /><sub><b>Jiace Sun</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=SUSYUSTC" title="Code">💻</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=SUSYUSTC" title="Documentation">📖</a> <a href="#example-SUSYUSTC" title="Examples">💡</a> <a href="#ideas-SUSYUSTC" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-SUSYUSTC" title="Research">🔬</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=SUSYUSTC" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/Zhouquan-Wan"><img src="https://avatars.githubusercontent.com/u/54523490?v=4?s=100" width="100px;" alt="Zhouquan Wan"/><br /><sub><b>Zhouquan Wan</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=Zhouquan-Wan" title="Code">💻</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=Zhouquan-Wan" title="Documentation">📖</a> <a href="#example-Zhouquan-Wan" title="Examples">💡</a> <a href="#ideas-Zhouquan-Wan" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-Zhouquan-Wan" title="Research">🔬</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=Zhouquan-Wan" title="Tests">⚠️</a> <a href="#tutorial-Zhouquan-Wan" title="Tutorials">✅</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/ls-iastu"><img src="https://avatars.githubusercontent.com/u/70554346?v=4?s=100" width="100px;" alt="Shuo Liu"/><br /><sub><b>Shuo Liu</b></sub></a><br /><a href="#example-ls-iastu" title="Examples">💡</a> <a href="#research-ls-iastu" title="Research">🔬</a> <a href="#tutorial-ls-iastu" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/YHPeter"><img src="https://avatars.githubusercontent.com/u/44126839?v=4?s=100" width="100px;" alt="Hao Yu"/><br /><sub><b>Hao Yu</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=YHPeter" title="Code">💻</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=YHPeter" title="Documentation">📖</a> <a href="#infra-YHPeter" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=YHPeter" title="Tests">⚠️</a> <a href="#tutorial-YHPeter" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/SexyCarrots"><img src="https://avatars.githubusercontent.com/u/63588721?v=4?s=100" width="100px;" alt="Xinghan Yang"/><br /><sub><b>Xinghan Yang</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=SexyCarrots" title="Documentation">📖</a> <a href="#translation-SexyCarrots" title="Translation">🌍</a> <a href="#tutorial-SexyCarrots" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/JachyMeow"><img src="https://avatars.githubusercontent.com/u/114171061?v=4?s=100" width="100px;" alt="JachyMeow"/><br /><sub><b>JachyMeow</b></sub></a><br /><a href="#tutorial-JachyMeow" title="Tutorials">✅</a> <a href="#translation-JachyMeow" title="Translation">🌍</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/Mzye21"><img src="https://avatars.githubusercontent.com/u/86239031?v=4?s=100" width="100px;" alt="Zhaofeng Ye"/><br /><sub><b>Zhaofeng Ye</b></sub></a><br /><a href="#design-Mzye21" title="Design">🎨</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/erertertet"><img src="https://avatars.githubusercontent.com/u/41342153?v=4?s=100" width="100px;" alt="erertertet"/><br /><sub><b>erertertet</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=erertertet" title="Code">💻</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=erertertet" title="Documentation">📖</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=erertertet" title="Tests">⚠️</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/yicongzheng"><img src="https://avatars.githubusercontent.com/u/107173985?v=4?s=100" width="100px;" alt="Yicong Zheng"/><br /><sub><b>Yicong Zheng</b></sub></a><br /><a href="#tutorial-yicongzheng" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://marksong.tech"><img src="https://avatars.githubusercontent.com/u/78847784?v=4?s=100" width="100px;" alt="Zixuan Song"/><br /><sub><b>Zixuan Song</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=MarkSong535" title="Documentation">📖</a> <a href="#translation-MarkSong535" title="Translation">🌍</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=MarkSong535" title="Code">💻</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=MarkSong535" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/buwantaiji"><img src="https://avatars.githubusercontent.com/u/25216189?v=4?s=100" width="100px;" alt="Hao Xie"/><br /><sub><b>Hao Xie</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=buwantaiji" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/pramitsingh0"><img src="https://avatars.githubusercontent.com/u/52959209?v=4?s=100" width="100px;" alt="Pramit Singh"/><br /><sub><b>Pramit Singh</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=pramitsingh0" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/JAllcock"><img src="https://avatars.githubusercontent.com/u/26302022?v=4?s=100" width="100px;" alt="Jonathan Allcock"/><br /><sub><b>Jonathan Allcock</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=JAllcock" title="Documentation">📖</a> <a href="#ideas-JAllcock" title="Ideas, Planning, & Feedback">🤔</a> <a href="#talk-JAllcock" title="Talks">📢</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/nealchen2003"><img src="https://avatars.githubusercontent.com/u/45502551?v=4?s=100" width="100px;" alt="nealchen2003"/><br /><sub><b>nealchen2003</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=nealchen2003" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/eurethia"><img src="https://avatars.githubusercontent.com/u/84611606?v=4?s=100" width="100px;" alt="隐公观鱼"/><br /><sub><b>隐公观鱼</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=eurethia" title="Code">💻</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=eurethia" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/WiuYuan"><img src="https://avatars.githubusercontent.com/u/108848998?v=4?s=100" width="100px;" alt="WiuYuan"/><br /><sub><b>WiuYuan</b></sub></a><br /><a href="#example-WiuYuan" title="Examples">💡</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://www.linkedin.com/in/felix-xu-16a153196/"><img src="https://avatars.githubusercontent.com/u/61252303?v=4?s=100" width="100px;" alt="Felix Xu"/><br /><sub><b>Felix Xu</b></sub></a><br /><a href="#tutorial-FelixXu35" title="Tutorials">✅</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=FelixXu35" title="Code">💻</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=FelixXu35" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://scholar.harvard.edu/hongyehu/home"><img src="https://avatars.githubusercontent.com/u/50563225?v=4?s=100" width="100px;" alt="Hong-Ye Hu"/><br /><sub><b>Hong-Ye Hu</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=hongyehu" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/PeilinZHENG"><img src="https://avatars.githubusercontent.com/u/45784888?v=4?s=100" width="100px;" alt="peilin"/><br /><sub><b>peilin</b></sub></a><br /><a href="#tutorial-PeilinZHENG" title="Tutorials">✅</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=PeilinZHENG" title="Code">💻</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=PeilinZHENG" title="Tests">⚠️</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=PeilinZHENG" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://emilianog-byte.github.io"><img src="https://avatars.githubusercontent.com/u/57567043?v=4?s=100" width="100px;" alt="Cristian Emiliano Godinez Ramirez"/><br /><sub><b>Cristian Emiliano Godinez Ramirez</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=EmilianoG-byte" title="Code">💻</a> <a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=EmilianoG-byte" title="Tests">⚠️</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/ztzhu1"><img src="https://avatars.githubusercontent.com/u/111620128?v=4?s=100" width="100px;" alt="ztzhu"/><br /><sub><b>ztzhu</b></sub></a><br /><a href="https://github.com/tencent-quantum-lab/tensorcircuit/commits?author=ztzhu1" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

## Research and Applications

### DQAS

For the application of Differentiable Quantum Architecture Search, see [applications](/tensorcircuit/applications).

Reference paper: https://arxiv.org/abs/2010.08561 (published in QST).

### VQNHE

For the application of Variational Quantum-Neural Hybrid Eigensolver, see [applications](/tensorcircuit/applications).

Reference paper: https://arxiv.org/abs/2106.05105 (published in PRL) and https://arxiv.org/abs/2112.10380 (published in AQT).

### VQEX-MBL

For the application of VQEX on MBL phase identification, see the [tutorial](/docs/source/tutorials/vqex_mbl.ipynb).

Reference paper: https://arxiv.org/abs/2111.13719 (published in PRB).

### Stark-DTC

For the numerical demosntration of discrete time crystal enabled by Stark many-body localization, see the Floquet simulation [demo](/examples/timeevolution_trotter.py).

Reference paper: https://arxiv.org/abs/2208.02866 (published in PRL).

### RA-Training

For the numerical simulation of variational quantum algorithm training using random gate activation strategy by us, see the [project repo](https://github.com/ls-iastu/RAtraining).

Reference paper: https://arxiv.org/abs/2303.08154 (published in PRR as a Letter).

### TenCirChem

[TenCirChem](https://github.com/tencent-quantum-lab/TenCirChem) is an efficient and versatile quantum computation package for molecular properties. TenCirChem is based on TensorCircuit and is optimized for chemistry applications.

Reference paper: https://arxiv.org/abs/2303.10825 (published in JCTC).

### EMQAOA-DARBO

For the numerical simulation and hardware experiments with error mitigation on QAOA, see the [project repo](https://github.com/sherrylixuecheng/EMQAOA-DARBO).

Reference paper: https://arxiv.org/abs/2303.14877.

### NN-VQA

For the setup and simulation code of neural network encoded variational quantum eigensolver, see the [demo](/docs/source/tutorials/nnvqe.ipynb).

Reference paper: https://arxiv.org/abs/2308.01068.

### More works

 <details>
  <summary> More research works and code projects using TensorCircuit (click for details) </summary>

- Neural Predictor based Quantum Architecture Search: https://arxiv.org/abs/2103.06524 (published in Machine Learning: Science and Technology).

- Quantum imaginary-time control for accelerating the ground-state preparation: https://arxiv.org/abs/2112.11782 (published in PRR).

- Efficient Quantum Simulation of Electron-Phonon Systems by Variational Basis State Encoder: https://arxiv.org/abs/2301.01442 (published in PRR).

- Variational Quantum Simulations of Finite-Temperature Dynamical Properties via Thermofield Dynamics: https://arxiv.org/abs/2206.05571.

- Understanding quantum machine learning also requires rethinking generalization: https://arxiv.org/abs/2306.13461.

- Decentralized Quantum Federated Learning for Metaverse: Analysis, Design and Implementation: https://arxiv.org/abs/2306.11297. Code: https://github.com/s222416822/BQFL.

- Non-IID quantum federated learning with one-shot communication complexity: https://arxiv.org/abs/2209.00768 (published in Quantum Machine Intelligence). Code: https://github.com/JasonZHM/quantum-fed-infer.

- Quantum generative adversarial imitation learning: https://doi.org/10.1088/1367-2630/acc605 (published in New Journal of Physics).

- GSQAS: Graph Self-supervised Quantum Architecture Search: https://arxiv.org/abs/2303.12381.

- Practical advantage of quantum machine learning in ghost imaging: https://www.nature.com/articles/s42005-023-01290-1 (published in Communications Physics).

- Zero and Finite Temperature Quantum Simulations Powered by Quantum Magic: https://arxiv.org/abs/2308.11616.

- Comparison of Quantum Simulators for Variational Quantum Search: A Benchmark Study: https://arxiv.org/abs/2309.05924.

- Statistical analysis of quantum state learning process in quantum neural networks: https://arxiv.org/abs/2309.14980 (Pubilshed in NeurIPS).

- Generative quantum machine learning via denoising diffusion probabilistic models: https://arxiv.org/abs/2310.05866.

- Quantum imaginary time evolution and quantum annealing meet topological sector optimization: https://arxiv.org/abs/2310.04291.

- Google Summer of Code 2023 Projects (QML4HEP): https://github.com/ML4SCI/QMLHEP, https://github.com/Gopal-Dahale/qgnn-hep, https://github.com/salcc/QuantumTransformers.

  </details>

If you want to highlight your research work or projects here, feel free to add by opening PR.
