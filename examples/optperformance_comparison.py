"""
Optimization for performance comparison for different densities of two-qubit gates
(random layouts averaged).
"""
import sys

sys.path.insert(0, "../")
import tensorflow as tf
import numpy as np
import cotengra as ctg
import tensorcircuit as tc

K = tc.set_backend("tensorflow")
optr = ctg.ReusableHyperOptimizer(
    methods=["greedy"],
    parallel=True,
    minimize="flops",
    max_time=30,
    max_repeats=512,
    progbar=True,
)
tc.set_contractor("custom", optimizer=optr, preprocessing=True)

eye4 = K.eye(4)
cnot = tc.array_to_tensor(tc.gates._cnot_matrix)


def energy_p(params, p, seed, n, nlayers):
    g = tc.templates.graphs.Line1D(n)
    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)
    for i in range(nlayers):
        for k in range(n):
            c.ry(k, theta=params[2 * i, k])
            c.rz(k, theta=params[2 * i + 1, k])
            c.ry(k, theta=params[2 * i + 2, k])
        for k in range(n // 2):  # alternating entangler with probability
            c.unitary_kraus(
                [eye4, cnot],
                2 * k + (i % 2),
                (2 * k + (i % 2) + 1) % n,
                prob=[1 - p, p],
                status=seed[i, k],
            )

    e = tc.templates.measurements.heisenberg_measurements(
        c, g, hzz=1, hxx=0, hyy=0, hx=-1, hy=0, hz=0
    )  # TFIM energy from expectation of circuit c defined on lattice given by g
    return e


vagf = K.jit(
    K.vectorized_value_and_grad(energy_p, argnums=0, vectorized_argnums=(0, 2)),
    static_argnums=(3, 4),
)

energy_list = []


if __name__ == "__main__":

    n = 12
    nlayers = 12
    nsteps = 250
    sample = 10
    debug = True

    for a in np.arange(0.1, 1, 0.1):
        energy_sublist = []
        params = K.implicit_randn(shape=[sample, 3 * nlayers, n])
        seeds = K.implicit_randu(shape=[sample, nlayers, n // 2])
        opt = tc.backend.optimizer(tf.keras.optimizers.Adam(2e-3))
        for i in range(nsteps):
            p = (n * nlayers) ** (a - 1)
            p = tc.array_to_tensor(p, dtype="float32")
            e, grads = vagf(params, p, seeds, n, nlayers)
            params = opt.update(grads, params)
            if i % 50 == 0 and debug:
                print(a, i, e)
        energy_list.append(K.numpy(e))

    print(energy_list)
