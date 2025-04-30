"""
comparison between expectation evaluation with/wo lightcone simplification
"""

import numpy as np
import tensorcircuit as tc

K = tc.set_backend("tensorflow")


def brickwall_ansatz(c, params, gatename, nlayers):
    n = c._nqubits
    params = K.reshape(params, [nlayers, n, 2])
    for j in range(nlayers):
        for i in range(0, n, 2):
            getattr(c, gatename)(i, (i + 1) % n, theta=params[j, i, 0])
        for i in range(1, n, 2):
            getattr(c, gatename)(i, (i + 1) % n, theta=params[j, i, 1])
    return c


def loss(params, n, nlayers, enable_lightcone):
    c = tc.Circuit(n)
    for i in range(n):
        c.h(i)
    c = brickwall_ansatz(c, params, "rzz", nlayers)
    expz = K.stack(
        [c.expectation_ps(z=[i], enable_lightcone=enable_lightcone) for i in range(n)]
    )
    return K.real(K.sum(expz))


vg1 = K.jit(K.value_and_grad(loss), static_argnums=(1, 2, 3))


def efficiency():
    for n in range(6, 40, 4):
        for nlayers in range(2, 6, 2):
            print(n, nlayers)
            print("w lightcone")
            (v2, g2), _, _ = tc.utils.benchmark(
                vg1, K.ones([nlayers * n * 2]), n, nlayers, True
            )
            if n < 16:
                print("wo lightcone")
                (v1, g1), _, _ = tc.utils.benchmark(
                    vg1, K.ones([nlayers * n * 2]), n, nlayers, False
                )
                np.testing.assert_allclose(v1, v2, atol=1e-5)
                np.testing.assert_allclose(g1, g2, atol=1e-5)


## further correctness check
def correctness(n, nlayers):
    for _ in range(5):
        v1, g1 = vg1(K.implicit_randn([nlayers * n * 2]), n, nlayers, False)
        v2, g2 = vg1(K.implicit_randn([nlayers * n * 2]), n, nlayers, True)
        np.testing.assert_allclose(v1, v2, atol=1e-5)
        np.testing.assert_allclose(g1, g2, atol=1e-5)


if __name__ == "__main__":
    efficiency()
    correctness(7, 3)
