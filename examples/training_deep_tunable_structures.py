"""
An integrated script demonstrating:
1. shortcut setup of cotengra contractor (with correct interplay with multiprocessing);
2. jit scan acceleration for deep structured circuit with multiple variables;
3. tensor controlled tunable circuit structures all in one jit;
4. batched trainable parameters via vmap/vvag;
and yet anonther demonstration of infras for training with incremental random activation
"""

import time
import numpy as np
import jax
import optax
import tensorcircuit as tc


def main():
    tc.set_contractor("cotengra-40-64")
    K = tc.set_backend("jax")
    tc.set_dtype("complex128")

    ii = tc.gates._ii_matrix
    xx = tc.gates._xx_matrix
    yy = tc.gates._yy_matrix
    zz = tc.gates._zz_matrix

    n = 12
    nlayers = 7
    g = tc.templates.graphs.Line1D(n)
    ncircuits = 10
    heih = tc.quantum.heisenberg_hamiltonian(
        g, hzz=1.0, hyy=1.0, hxx=1.0, hx=0, hy=0, hz=0
    )

    def energy(params, structures, n, nlayers):
        def one_layer(state, others):
            params, structures = others
            # print(state.shape, params.shape, structures.shape)
            l = 0
            c = tc.Circuit(n, inputs=state)
            for i in range(1, n, 2):
                matrix = structures[3 * l, i] * ii + (1.0 - structures[3 * l, i]) * (
                    K.cos(params[3 * l, i]) * ii + 1.0j * K.sin(params[3 * l, i]) * zz
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )

            ### YY
            for i in range(1, n, 2):
                matrix = structures[3 * l + 1, i] * ii + (
                    1.0 - structures[3 * l + 1, i]
                ) * (
                    K.cos(params[3 * l + 1, i]) * ii
                    + 1.0j * K.sin(params[3 * l + 1, i]) * yy
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )

            ### XX
            for i in range(1, n, 2):
                matrix = structures[3 * l + 2, i] * ii + (
                    1.0 - structures[3 * l + 2, i]
                ) * (
                    K.cos(params[3 * l + 2, i]) * ii
                    + 1.0j * K.sin(params[3 * l + 2, i]) * xx
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )

            ### Even layer
            ### ZZ
            for i in range(0, n, 2):
                matrix = structures[3 * l, i] * ii + (1.0 - structures[3 * l, i]) * (
                    K.cos(params[3 * l, i]) * ii + 1.0j * K.sin(params[3 * l, i]) * zz
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )
            ### YY

            for i in range(0, n, 2):
                matrix = structures[3 * l + 1, i] * ii + (
                    1.0 - structures[3 * l + 1, i]
                ) * (
                    K.cos(params[3 * l + 1, i]) * ii
                    + 1.0j * K.sin(params[3 * l + 1, i]) * yy
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )

            ### XX
            for i in range(0, n, 2):
                matrix = structures[3 * l + 2, i] * ii + (
                    1.0 - structures[3 * l + 2, i]
                ) * (
                    K.cos(params[3 * l + 2, i]) * ii
                    + 1.0j * K.sin(params[3 * l + 2, i]) * xx
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )
            s = c.state()
            return s, s

        params = K.cast(K.real(params), dtype="complex128")
        structures = (K.sign(structures) + 1) / 2  # 0 or 1
        structures = K.cast(structures, dtype="complex128")

        c = tc.Circuit(n)

        for i in range(n):
            c.x(i)
        for i in range(0, n, 2):
            c.H(i)
        for i in range(0, n, 2):
            c.cnot(i, i + 1)
        s = c.state()
        s, _ = jax.lax.scan(
            one_layer,
            s,
            (
                K.reshape(params, [nlayers, 3, n]),
                K.reshape(structures, [nlayers, 3, n]),
            ),
        )
        c = tc.Circuit(n, inputs=s)
        # e = tc.templates.measurements.heisenberg_measurements(
        #     c, g, hzz=1, hxx=1, hyy=1, hx=0, hy=0, hz=0
        # )
        e = tc.templates.measurements.operator_expectation(c, heih)
        return K.real(e)

    vagf = K.jit(K.vvag(energy, argnums=0, vectorized_argnums=0), static_argnums=(2, 3))

    structures = tc.array_to_tensor(
        np.random.uniform(low=0.0, high=1.0, size=[3 * nlayers, n]), dtype="complex128"
    )
    structures -= 1.0 * K.ones([3 * nlayers, n])
    params = tc.array_to_tensor(
        np.random.uniform(low=-0.1, high=0.1, size=[ncircuits, 3 * nlayers, n]),
        dtype="float64",
    )

    # opt = K.optimizer(tf.keras.optimizers.Adam(1e-2))
    opt = K.optimizer(optax.adam(1e-2))

    for _ in range(50):
        time0 = time.time()
        e, grads = vagf(params, structures, n, nlayers)
        time1 = time.time()
        params = opt.update(grads, params)
        print(K.numpy(e), time1 - time0)


if __name__ == "__main__":
    main()
