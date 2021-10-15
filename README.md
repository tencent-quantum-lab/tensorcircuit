# TENSORCIRCUIT

Build on top of [TensorNetwork](https://github.com/google/TensorNetwork), differentiable quantum circuit simulator gains benefits from swift implementation to auto differentiation infrastructure.

## Basic Usage

```python
import tensorcircuit as tc
c = tc.Circuit(2)
c.H(0)
c.CNOT(0,1)
print(c.wavefunction())
print(c.expectation((tc.gates.z(), [1])))
```

Runtime behavior customization:

```python
tc.set_backend("tensorflow")
tc.set_dtype("complex128")
tc.set_contractor("greedy")
```

Auto differentiations with jit (tf and jax backend currently supported):

```python
@tc.backend.jit
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

### Guidelines

Please open issues or PRs.

NEVER directly push to this repo!

Keep the codebase private!

### Install

For development workflow, we suggest to first configure a good conda environment. The versions of dependecy package may vary in terms of development requirements. The minimum requirement is the [TensorNetwork](https://github.com/google/TensorNetwork) package (pip install suggested).

Secondly, fork this repo to your GitHub account (make sure keeping the repo **private**!), and [setup](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) the SSH access to your GitHub account.

Lastly

```bash
git clone git@github.com:yourgithub/tensorcircuit-dev.git
export PYTHONPATH=/path/for/tensorcircuit/
```

or a better approach for the last step

```bash
python setup.py develop
```

Now, you are ready to `import tensorcircuit` and enjoy coding.

### Docs

```bash
cd docs
make html
```

### Tests

```bash
pytest
```

### Linter

```bash
black .
```

### Type checker

```bash
mypy tensorcircuit
```

### CI

We currently use GitHub Action for test CI, but it has limited quota for free private repo.

## Research projects and application codes

### DQAS

For application of Differentiable Quantum Architecture Search, see [applications](/tensorcircuit/applications). Reference paper: https://arxiv.org/pdf/2010.08561.pdf.

### VQNHE

For application of Variational Quantum-Neural Hybrid Eigensolver, see [applications](/tensorcircuit/applications). Reference paper: https://arxiv.org/pdf/2106.05105.pdf.
