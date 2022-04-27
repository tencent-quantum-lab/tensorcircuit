# Run TensorCircuit on Windows Machine with Docker
This note is only a step-by-step tutorial to help you build and run a Docker Container for Windows Machine users with the given dockerfile. 
If you want to have a deeper dive in to Docker, please check the official [Docker Orientation](https://docs.docker.com/get-started/)
and free courses on YouTube.
## Why We Can't Run TensorCircuit on Windows Machine

Due to the compatability issue with the [JAX](https://jax.readthedocs.io/en/latest/index.html) backend on Windows,
we could not directly use most of TensorCircuit features on Windows machines. Please be aware that it is possible to [install 
JAX on Windows](https://jax.readthedocs.io/en/latest/developer.html), but it is very tricky and not recommended unless
you have solid understanding of Windows environment and C++ tools. Virtual machine is also an option for development if
you are familiar with it. In this tutorial we would discuss the deployment of Docker for TensorCircuit since it us 
the most convenient and workable solution for most beginners.

## What Is Docker
Docker is an open platform for developing, shipping, and running applications. Docker enables you to separate your applications from your infrastructure so you can deliver software quickly. With Docker, you can manage your infrastructure in the same ways you manage your applications. By taking advantage of Docker’s methodologies for shipping, testing, and deploying code quickly, you can significantly reduce the delay between writing code and running it in production.

(Source: https://docs.docker.com/get-started/overview/) 

For more information and tutorials on Docker, you could check the [Docker Documentation](https://docs.docker.com/get-started/overview/).

## Install Docker and Docker Desktop
[Download Docker Desktop]() for Windows and install it by following its instructions.
- Circuit manipulation:

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

## Install

The package is purely written in Python and can be obtained via pip as:

```python
pip install tensorcircuit
```

We also have [Docker support](/docker).

## Advantages

* Tensor network simulation engine based

* JIT, AD, vectorized parallelism compatible, GPU support

* Efficiency

  * Time: 10 to 10^6 times acceleration compared to tfq or qiskit

  * Space: 600+ qubits 1D VQE workflow (converged energy inaccuracy: < 1%)

* Elegance

  * Flexibility: customized contraction, multiple ML backend/interface choices, multiple dtype precisions

  * API design: quantum for humans, less code, more power

## Contributing

For contribution guidelines and notes, see [CONTRIBUTING](/CONTRIBUTING.md).

We welcome issues, PRs, and discussions from everyone, and these are all hosted on GitHub.

## Researches and Applications

### DQAS

For the application of Differentiable Quantum Architecture Search, see [applications](/tensorcircuit/applications).
Reference paper: https://arxiv.org/pdf/2010.08561.pdf.

### VQNHE

For the application of Variational Quantum-Neural Hybrid Eigensolver, see [applications](/tensorcircuit/applications).
Reference paper: https://arxiv.org/pdf/2106.05105.pdf and https://arxiv.org/pdf/2112.10380.pdf.

### VQEX - MBL

For the application of VQEX on MBL phase identification, see the [tutorial](https://github.com/quclub/tensorcircuit-tutorials/blob/master/tutorials/vqex_mbl.ipynb).
Reference paper: https://arxiv.org/pdf/2111.13719.pdf.

