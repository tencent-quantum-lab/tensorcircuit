"""
Parameterized Hamiltonian (Pulse control/Analog simulation) with AD/JIT support using jax ode solver
"""

import optax
from jax.experimental.ode import odeint
import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex128")

hx = tc.quantum.PauliStringSum2COO([[1]])
hz = tc.quantum.PauliStringSum2COO([[3]])


# psi = -i H psi
# we want to optimize the final z expectation over parameters params
# a single qubit example below


def final_z(b):
    def f(y, t, b):
        h = b[3] * K.sin(b[0] * t + b[1]) * hx + K.cos(b[2]) * hz
        return -1.0j * K.sparse_dense_matmul(h, y)

    y0 = tc.array_to_tensor([1, 0])
    y0 = K.reshape(y0, [-1, 1])
    t = tc.array_to_tensor([0.0, 10.0], dtype=tc.rdtypestr)
    yf = odeint(f, y0, t, b)
    c = tc.Circuit(1, inputs=K.reshape(yf[-1], [-1]))
    return K.real(c.expectation_ps(z=[0]))


vgf = K.jit(K.value_and_grad(final_z))


opt = K.optimizer(optax.adam(0.1))
b = K.implicit_randn([4])
for _ in range(50):
    v, gs = vgf(b)
    b = opt.update(gs, b)
    print(v, b)
