# TENSORCIRCUIT

With [TensorNetwork](https://github.com/google/TensorNetwork) project announced by Google, quantum circuit simulator based on it may gain benefits from swift implementation to auto differentiation abilities.

See `tensorcircuit.applications` for relevant code on so-call differentiable quantum architecture search.

## Baisc Usage

```python
import tensorcircuit as tc
c = tc.Circuit(2)
c.H(0)
c.CNOT(0,1)
print(c.perfect_sampling())
print(c.wavefunction())
print(c.measure(1))
print(c.expectation((tc.gates.z(), [1])))
```

Runtime behavior changing:

```python
tc.set_backend("tensorflow")
tc.set_dtype("complex128")
tc.set_contractor("greedy")
```

Auto differentiations with jit (tf and jax supported):

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

## DQAS

For application of Differentiable Quantum Architecture Search, see [applications](/tensorcircuit/applications)
