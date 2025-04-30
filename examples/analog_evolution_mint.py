"""
jax backend analog evolution targeting at minimizing evolution time
"""

import optax
import tensorcircuit as tc
from tensorcircuit.experimental import evol_global

K = tc.set_backend("jax")

hx = tc.quantum.PauliStringSum2COO([[1]])


def h_fun(t, b):
    return K.sin(b) * hx


def fast_evol(t, b):
    lbd = 0.08
    c = tc.Circuit(1)
    c = evol_global(c, h_fun, t, b)
    loss = K.real(c.expectation_ps(z=[0]))
    return loss + lbd * t**2, loss
    # l2 regularization to minimize t while target z=-1


vgf = K.jit(K.value_and_grad(fast_evol, argnums=(0, 1), has_aux=True))

opt = K.optimizer(optax.adam(0.05))
b, t = tc.array_to_tensor(0.5, 1.0, dtype=tc.rdtypestr)

for i in range(500):
    (v, loss), gs = vgf(b, t)
    b, t = opt.update(gs, (b, t))
    if i % 20 == 0:
        print(v, loss, b, t)
