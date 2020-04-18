# TENSORCIRCUIT

This project is partially inspired by [mpsim](https://github.com/grmlarose/mpsim) which builds the quantum circuit model on top of tensornetwork setups instead of directly matrix manipulations.

With [TensorNetwork](https://github.com/google/TensorNetwork) project announced by Google, such setup may gain benefits from swift implementation to auto differentiation abilities.

This is only a toy project and it is at very early stage. Try it on your own risk.

## Baisc Usage

```python
import tensorcircuit as tc
c = tc.Circuit(2)
c.H(0)
c.CNOT(0,1)
print(c.perfect_sampling())
print(c.wavefunction())
print(c.measure(1))
print(c.expectation(tc.gates.z(), 1))
```

Runtime behavior changing:

```python
tc.set_backend("tensorflow")
tc.set_dtype("complex128")
tc.set_contractor("greedy")
```