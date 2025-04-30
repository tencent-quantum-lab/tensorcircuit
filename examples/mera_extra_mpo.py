"""
MERA VQE example with Hamiltonian expectation in MPO representation
"""

import time
import logging
import numpy as np
import tensornetwork as tn
import optax
import cotengra
import tensorflow as tf
import tensorcircuit as tc

logger = logging.getLogger("tensorcircuit")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


K = tc.set_backend("tensorflow")
tc.set_dtype("complex128")


optc = cotengra.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    parallel="ray",
    minimize="combo",
    max_time=90,
    max_repeats=1024,
    progbar=True,
)

tc.set_contractor("custom", optimizer=optc, preprocessing=True)


def MERA_circuit(params, n, d):
    c = tc.Circuit(n)

    idx = 0

    for i in range(n):
        c.rx(i, theta=params[2 * i])
        c.rz(i, theta=params[2 * i + 1])
    idx += 2 * n

    for n_layer in range(1, int(np.log2(n)) + 1):
        n_qubit = 2**n_layer
        step = int(n / n_qubit)

        for _ in range(d):
            # even
            for i in range(step, n - step, 2 * step):
                c.exp1(i, i + step, theta=params[idx], unitary=tc.gates._xx_matrix)
                c.exp1(i, i + step, theta=params[idx + 1], unitary=tc.gates._zz_matrix)
                idx += 2

            # odd
            for i in range(0, n, 2 * step):
                c.exp1(i, i + step, theta=params[idx], unitary=tc.gates._xx_matrix)
                c.exp1(i, i + step, theta=params[idx + 1], unitary=tc.gates._zz_matrix)
                idx += 2

        for i in range(0, n, step):
            c.rx(i, theta=params[idx])
            c.rz(i, theta=params[idx + 1])
            idx += 2

    return c, idx


def MERA(params, n, d, hamiltonian_mpo):
    c, _ = MERA_circuit(params, n, d)
    return tc.templates.measurements.mpo_expectation(c, hamiltonian_mpo)


MERA_vvag = K.jit(K.vectorized_value_and_grad(MERA), static_argnums=(1, 2, 3))


def train(opt, j, b, n, d, batch, maxiter):
    Jx = j * np.ones([n - 1])  # strength of xx interaction (OBC)
    Bz = -b * np.ones([n])  # strength of transverse field
    hamiltonian_mpo = tn.matrixproductstates.mpo.FiniteTFI(Jx, Bz, dtype=np.complex128)
    # matrix product operator
    hamiltonian_mpo = tc.quantum.tn2qop(hamiltonian_mpo)
    _, idx = MERA_circuit(K.ones([int(1e6)]), n, d)
    params = K.implicit_randn([batch, idx], 0, 0.05)
    times = []
    times.append(time.time())
    for i in range(maxiter):
        e, g = MERA_vvag(params, n, d, hamiltonian_mpo)
        params = opt.update(g, params)
        times.append(time.time())
        if i % 100 == 99:
            print("energy: ", e)
            print(
                "time analysis: ",
                times[1] - times[0],
                (times[-1] - times[1]) / (len(times) - 2),
            )
    return K.min(e)


if __name__ == "__main__":
    if K.name == "jax":
        exponential_decay_scheduler = optax.exponential_decay(
            init_value=1e-2, transition_steps=500, decay_rate=0.9
        )
        opt = K.optimizer(optax.adam(exponential_decay_scheduler))
    elif K.name == "tensorflow":
        exponential_decay_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2, decay_steps=500, decay_rate=0.9
        )
        opt = K.optimizer(tf.keras.optimizers.Adam(exponential_decay_scheduler))
    e = train(opt, 1, -1, 64, 2, 2, 5000)
    print("optimized energy:", e)

# backend: n, d, batch: compiling time, running time
# jax: 16, 2, 8: 730s, 0.033s
# tf: 16, 2, 8: 140s, 0.043s
# tf: 32, 2, 2: 251s, 0.9s
