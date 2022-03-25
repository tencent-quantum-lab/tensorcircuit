"""
A simple script to benchmark the approximation power of the MPS simulator.
"""
import sys

sys.path.insert(0, "../")

import tensorcircuit as tc

tc.set_backend("tensorflow")


def tfi_energy(c, n, j=1.0, h=-1.0):
    e = 0.0
    for i in range(n):
        e += h * c.expectation((tc.gates.x(), [i]))
    for i in range(n - 1):
        e += j * c.expectation((tc.gates.z(), [i]), (tc.gates.z(), [(i + 1) % n]))
    return e


def tfi_energy_mps(c, n, j=1.0, h=-1.0):
    e = 0.0
    for i in range(n):
        e += h * c.expectation_single_gate(tc.gates.x(), i)
    for i in range(n - 1):
        e += j * c.expectation_two_gates_product(
            tc.gates.z(), tc.gates.z(), i, (i + 1) % n
        )
    return e


def energy(param, mpsd=None):
    if mpsd is None:
        c = tc.Circuit(n)
    else:
        c = tc.MPSCircuit(n)
        c.set_truncation_rule(max_singular_values=mpsd)

    for i in range(n):
        c.H(i)
    for j in range(nlayers):
        for i in range(n - 1):
            c.exp1(
                i,
                (i + 1) % n,
                theta=param[2 * j, i],
                unitary=tc.gates._zz_matrix,
            )
        for i in range(n):
            c.rx(i, theta=param[2 * j + 1, i])
    if mpsd is None:
        e = tfi_energy(c, n)
    else:
        e = tfi_energy_mps(c, n)

    e = tc.backend.real(e)
    if mpsd is not None:
        fidelity = c._fidelity
    else:
        fidelity = None
    return e, c.state(), fidelity


n, nlayers = 15, 20
print("number of qubits: ", n)
print("number of layers: ", nlayers)

param = tc.backend.implicit_randu([2 * nlayers, n])
# param = tc.backend.ones([2 * nlayers, n])
# it turns out that the mps approximation power highly depends on the
# parameters, if we use ``param = tc.backend.ones``, the apprixmation ratio decays very fast
# At least, the estimated fidelity is a very good proxy metric for real fidelity
# as long as it is larger than 50%
e0, s0, _ = energy(param)
print(
    "entanglement: ",
    tc.backend.numpy(
        tc.quantum.entropy(tc.quantum.reduced_density_matrix(s0, cut=n // 2))
    ),
)

for mpsd in [2, 5, 10, 20, 50, 100]:
    e1, s1, f1 = energy(param, mpsd=mpsd)
    print("------------------------")
    print("bond dimension: ", mpsd)
    print(
        "exact energy: ",
        tc.backend.numpy(e0),
        "mps simulator energy: ",
        tc.backend.numpy(e1),
    )
    print(
        "energy relative error(%): ",
        tc.backend.numpy(tc.backend.abs((e1 - e0) / e0)) * 100,
    )
    print("estimated fidelity:", tc.backend.numpy(f1))
    print(
        "real fidelity:",
        tc.backend.numpy(
            tc.backend.abs(tc.backend.tensordot(tc.backend.conj(s1), s0, 1))
        ),
    )
