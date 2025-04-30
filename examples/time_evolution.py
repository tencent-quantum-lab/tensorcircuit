"""
A simple static Hamiltonian evolution benchmark
"""

import time
from functools import partial
import numpy as np
from scipy.integrate import solve_ivp
import tensorcircuit as tc
from tensorcircuit.experimental import hamiltonian_evol

K = tc.set_backend("jax")
tc.set_dtype("complex128")


@partial(K.jit, static_argnums=1)
def total_z(psi, N):
    return K.real(
        K.sum(K.stack([tc.expectation([tc.gates.z(), i], ket=psi) for i in range(N)]))
    )


@K.jit
def naive_evol(t, h, psi0):
    return K.reshape(K.expm(-1j * t * h) @ K.reshape(psi0, [-1, 1]), [-1])


@K.jit
def hpsi(h, y):
    return K.reshape(-1.0j * h @ K.reshape(y, [-1, 1]), [-1])


def main(N):
    psi0 = np.zeros([2**N])
    psi0[0] = 1
    psi0 = tc.array_to_tensor(psi0)
    g = tc.templates.graphs.Line1D(N, pbc=False)
    h = tc.quantum.heisenberg_hamiltonian(g, hzz=1, hxx=0, hyy=0, hx=1, sparse=False)
    tlist = K.arange(0, 3, 0.1)
    time0 = time.time()
    for t in tlist:
        psit = naive_evol(t, h, psi0)
        psit /= K.norm(psit)
        print(total_z(psit, N))
    time1 = time.time()
    r = hamiltonian_evol(1.0j * tlist, h, psi0, callback=partial(total_z, N=N))
    print(r)
    time2 = time.time()

    def fun(t, y):
        y = tc.array_to_tensor(y)
        return K.numpy(hpsi(h, y))

    r = solve_ivp(
        fun, (0, 3), psi0, method="DOP853", t_eval=K.numpy(tlist), rtol=1e-6, atol=1e-6
    )
    for psit in r.y.T:
        print(total_z(psit, N))
    time3 = time.time()
    print(
        "matrix exponential:",
        time1 - time0,
        "tc fast implementation",
        time2 - time1,
        "scipy ode",
        time3 - time2,
    )


if __name__ == "__main__":
    main(10)
