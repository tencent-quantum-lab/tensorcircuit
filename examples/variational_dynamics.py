"""
Variational wavefunctions based on variational circuits and its dynamics.
"""

# Variational Quantum Algorithm for Quantum Dynamics
# Ref: PRL 125, 010501 (2020)

import sys
import time

sys.path.insert(0, "../")

import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")  # can set tensorflow backend or jax backend
tc.set_dtype("complex64")
# the default precision is complex64, can change to complex128 for double precision


def variational_wfn(theta, psi0):
    theta = tc.backend.reshape(theta, [l, N, 2])
    c = tc.Circuit(N, inputs=psi0)
    for i in range(0, l):
        for j in range(N - 1):
            c.exp1(j, j + 1, theta=theta[i, j, 0], unitary=tc.gates._zz_matrix)
        for j in range(N):
            c.rx(j, theta=theta[i, j, 1])

    return c.state()


ppsioverptheta = tc.backend.jit(tc.backend.jacfwd(variational_wfn, argnums=0))
# compute \partial psi /\partial theta, i.e. jacobian of wfn


def ij(i, j):
    """
    Inner product
    """
    return tc.backend.tensordot(tc.backend.conj(i), j, 1)


@tc.backend.jit
def lhs_matrix(theta, psi0):
    vij = tc.backend.vmap(ij, vectorized_argnums=0)
    vvij = tc.backend.vmap(vij, vectorized_argnums=1)
    jacobian = ppsioverptheta(theta, psi0=psi0)
    jacobian = tc.backend.transpose(jacobian)
    fim = vvij(jacobian, jacobian)
    fim = tc.backend.real(fim)
    return fim


@tc.backend.jit
def rhs_vector(theta, psi0):
    def energy(theta, psi0):
        w = variational_wfn(theta, psi0)
        wl = tc.backend.stop_gradient(w)
        wl = tc.backend.conj(wl)
        wr = w
        wl = tc.backend.reshape(wl, [1, -1])
        wr = tc.backend.reshape(wr, [-1, 1])
        e = wl @ h @ wr
        return tc.backend.real(e)[0, 0]

    eg = tc.backend.grad(energy, argnums=0)
    rhs = eg(theta, psi0)
    rhs = tc.backend.imag(rhs)
    return rhs


@tc.backend.jit
def update(theta, lhs, rhs, tau):
    # protection
    eps = 1e-4
    lhs += eps * tc.backend.eye(l * N * 2, dtype=lhs.dtype)
    return (
        tc.backend.cast(
            tau * tc.backend.solve(lhs, rhs, assume_a="sym"), dtype=theta.dtype
        )
        + theta
    )


if __name__ == "__main__":
    N = 10
    l = 5
    tau = 0.005
    steps = 200

    g = tc.templates.graphs.Line1D(N, pbc=False)
    h = tc.quantum.heisenberg_hamiltonian(
        g, hzz=1, hyy=0, hxx=0, hz=0, hx=1, hy=0, sparse=False
    )
    # TFIM Hamiltonian defined on lattice graph g (1D OBC chain)
    h = tc.array_to_tensor(h)

    psi0 = np.zeros(2**N)
    psi0[0] = 1.0
    psi0 = tc.array_to_tensor(psi0)

    theta = np.zeros([l * N * 2])
    theta = tc.array_to_tensor(theta)

    time0 = time.time()

    for n in range(steps):
        psi = variational_wfn(theta, psi0)
        lhs = lhs_matrix(theta, psi0)
        rhs = rhs_vector(theta, psi0)
        theta = update(theta, lhs, rhs, tau)
        if n % 10 == 0:
            time1 = time.time()
            print(time1 - time0)
            time0 = time1
            psi_exact = tc.backend.expm(-1j * h * n * tau) @ tc.backend.reshape(
                psi0, [-1, 1]
            )
            psi_exact = tc.backend.reshape(psi_exact, [-1])
            print(
                "time: %.2f" % (n * tau),
                "exact:",
                tc.expectation([tc.gates.z(), [0]], ket=psi_exact),
                "variational:",
                tc.expectation([tc.gates.z(), [0]], ket=psi),
            )
