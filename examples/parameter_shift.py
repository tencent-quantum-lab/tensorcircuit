"""
Demonstration on the correctness and efficiency of parameter shift gradient implementation
"""

import numpy as np
import tensorcircuit as tc
from tensorcircuit import experimental as E

K = tc.set_backend("tensorflow")

n = 6
m = 3


def f1(param):
    c = tc.Circuit(n)
    for j in range(m):
        for i in range(n - 1):
            c.cnot(i, i + 1)
        for i in range(n):
            c.rx(i, theta=param[i, j])
    return c.expectation_ps(y=[n // 2])


g1f1 = K.jit(K.grad(f1))

r1, ts, tr = tc.utils.benchmark(g1f1, K.ones([n, m], dtype="float32"))

g2f1 = K.jit(E.parameter_shift_grad(f1))

r2, ts, tr = tc.utils.benchmark(g2f1, K.ones([n, m], dtype="float32"))

np.testing.assert_allclose(r1, r2, atol=1e-5)
print("equality test passed!")

# mutiple weights args version


def f2(paramzz, paramx):
    c = tc.Circuit(n)
    for j in range(m):
        for i in range(n - 1):
            c.rzz(i, i + 1, theta=paramzz[i, j])
        for i in range(n):
            c.rx(i, theta=paramx[i, j])
    return c.expectation_ps(y=[n // 2])


g1f2 = K.jit(K.grad(f2, argnums=(0, 1)))

r1, ts, tr = tc.utils.benchmark(
    g1f2, K.ones([n, m], dtype="float32"), K.ones([n, m], dtype="float32")
)

g2f2 = K.jit(E.parameter_shift_grad(f2, argnums=(0, 1)))

r2, ts, tr = tc.utils.benchmark(
    g2f2, K.ones([n, m], dtype="float32"), K.ones([n, m], dtype="float32")
)

np.testing.assert_allclose(r1[0], r2[0], atol=1e-5)
np.testing.assert_allclose(r1[1], r2[1], atol=1e-5)
print("equality test passed!")
