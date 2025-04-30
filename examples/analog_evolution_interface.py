"""
jax backend is required, experimental built-in interface for parameterized hamiltonian evolution
"""

import optax
import tensorcircuit as tc
from tensorcircuit.experimental import evol_global, evol_local

K = tc.set_backend("jax")


def h_fun(t, b):
    return b * tc.gates.x().tensor


hy = tc.quantum.PauliStringSum2COO([[2, 0]])


def h_fun2(t, b):
    return b[2] * K.cos(b[0] * t + b[1]) * hy


@K.jit
@K.value_and_grad
def hybrid_evol(params):
    c = tc.Circuit(2)
    c.x([0, 1])
    c = evol_local(c, [1], h_fun, 1.0, params[0])
    c.cx(1, 0)
    c.h(0)
    c = evol_global(c, h_fun2, 1.0, params[1:])
    return K.real(c.expectation_ps(z=[0, 1]))


opt = K.optimizer(optax.adam(0.1))
b = K.implicit_randn([4])
for _ in range(50):
    v, gs = hybrid_evol(b)
    b = opt.update(gs, b)
    print(v, b)
