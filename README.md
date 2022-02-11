# TENSORCIRCUIT

TensorCircuit is the next generation of quantum circuit simulator with support for automatic differentiation, just-in-time compiling, hardware acceleration, and vectorized parallelism.

TensorCircuit is built on top of modern machine learning frameworks, and has the beautiful backend agnostic feature. It is specifically suitable for simulations of quantum-classical hybrid paradigm and variational quantum algorithms.

## Basic Usage

Please begin with [Quick Start](/docs/source/quickstart.rst) and [Jupyter Tutorials](https://github.com/quclub/tensorcircuit-tutorials/tree/master/tutorials).

For more information and introductions, please refer to helpful scripts [examples](/examples) and [documentations](/docs/source). API docstrings (incomplete for now) and test cases in [tests](/tests) are also informative.

The following are some minimal demos.

Circuit manipulation:

```python
import tensorcircuit as tc
c = tc.Circuit(2)
c.H(0)
c.CNOT(0,1)
c.rx(1, theta=0.2)
print(c.wavefunction())
print(c.expectation((tc.gates.z(), [1])))
print(c.perfect_sampling())
```

Runtime behavior customization:

```python
tc.set_backend("tensorflow")
tc.set_dtype("complex128")
tc.set_contractor("greedy")
```

Automatic differentiations with jit:

```python
def forward(theta):
    c = tc.Circuit(2)
    c.R(0, theta=theta, alpha=0.5, phi=0.8)
    return tc.backend.real(c.expectation((tc.gates.z(), [0])))

g = tc.backend.grad(forward)
g = tc.backend.jit(g)
theta = tc.gates.num_to_tensor(1.0)
print(g(theta))
```

## Contributing

For contribution guidelines and notes, see [CONTRIBUTING](/CONTRIBUTING.md).

### Cautions

Please open issues or PRs.

NEVER directly push to this repo!

Keep the codebase private!

### Install

For development workflow, we suggest to first configure a good conda environment. The versions of dependecy package may vary in terms of development requirements. The minimum requirement is the [TensorNetwork](https://github.com/google/TensorNetwork) package (pip install suggested).

### Docs

```bash
cd docs
make html
```

### Tests

```bash
pytest
```

### Formatter

```bash
black .
```

### Linter

```bash
pylint tensorcircuit tests
```

### Type checker

```bash
mypy tensorcircuit
```

### Integrated script

For now, we introduce one for all checker for development:

```bash
./check_all.sh
```

## Research projects and application codes

### DQAS

For application of Differentiable Quantum Architecture Search, see [applications](/tensorcircuit/applications). Reference paper: https://arxiv.org/pdf/2010.08561.pdf.

### VQNHE

For application of Variational Quantum-Neural Hybrid Eigensolver, see [applications](/tensorcircuit/applications). Reference paper: https://arxiv.org/pdf/2106.05105.pdf.
