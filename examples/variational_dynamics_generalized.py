"""
Variational wavefunctions based on variational circuits and its dynamics.
"""

# https://arxiv.org/pdf/1812.08767.pdf
# Eq 13, 14, based on discussion:
# https://github.com/tencent-quantum-lab/tensorcircuit/discussions/22

import sys
import time

sys.path.insert(0, "../")

import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")  # can set tensorflow backend or jax backend
tc.set_dtype("complex64")
# the default precision is complex64, can change to complex128 for double precision


@tc.backend.jit
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


def _vdot(i, j):
    return tc.backend.tensordot(tc.backend.conj(i), j, 1)


@tc.backend.jit
def lhs_matrix(theta, psi0):
    psi = variational_wfn(theta, psi0)

    def ij(i, j):
        return _vdot(i, j) + _vdot(i, psi) * _vdot(j, psi)

    vij = tc.backend.vmap(ij, vectorized_argnums=0)
    vvij = tc.backend.vmap(vij, vectorized_argnums=1)
    jacobian = ppsioverptheta(theta, psi0=psi0)
    jacobian = tc.backend.transpose(jacobian)
    fim = vvij(jacobian, jacobian)
    lhs = tc.backend.real(fim)
    return lhs


@tc.backend.jit
def rhs_vector(theta, psi0):
    def energy1(theta, psi0):
        w = variational_wfn(theta, psi0)
        wl = tc.backend.conj(w)
        wr = tc.backend.stop_gradient(w)
        wl = tc.backend.reshape(wl, [1, -1])
        wr = tc.backend.reshape(wr, [-1, 1])
        e = wl @ h @ wr  # <\partial psi0|H| psi0>
        return tc.backend.real(e)[0, 0]

    def energy2(theta, psi0):
        w = variational_wfn(theta, psi0)
        wr0 = tc.backend.stop_gradient(w)
        wr0 = tc.backend.reshape(wr0, [-1, 1])
        wl0 = tc.backend.stop_gradient(w)
        wl0 = tc.backend.conj(wl0)
        wl0 = tc.backend.reshape(wl0, [1, -1])
        e0 = wl0 @ h @ wr0  # <psi0| H | psi0>

        wl = tc.backend.conj(w)
        wl = tc.backend.reshape(wl, [1, -1])
        w0 = wl @ wr0  # <\partial psi0| psi0>
        return tc.backend.real((w0 * e0)[0, 0])

    eg1 = tc.backend.grad(energy1, argnums=0)
    eg2 = tc.backend.grad(energy2, argnums=0)

    rhs1 = eg1(theta, psi0)
    rhs1 = tc.backend.imag(rhs1)
    rhs2 = eg2(theta, psi0)
    rhs2 = tc.backend.imag(rhs2)  # should be a imaginary number
    rhs = rhs1 - rhs2
    return rhs


@tc.backend.jit
def update(theta, lhs, rhs, tau):
    # protection
    eps = 1e-3
    lhs += eps * tc.backend.eye(l * N * 2, dtype=lhs.dtype)
    dtheta = tc.backend.cast(
        tau * tc.backend.solve(lhs, rhs, assume_a="sym"), dtype=theta.dtype
    )
    return dtheta + theta


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
