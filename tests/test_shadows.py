import pytest
from pytest_lazyfixture import lazy_fixture as lf
import numpy as np
import tensorcircuit as tc
from tensorcircuit.shadows import (
    shadow_bound,
    shadow_snapshots,
    local_snapshot_states,
    global_shadow_state,
    entropy_shadow,
    Renyi_entropy_2,
    expection_ps_shadow,
    global_shadow_state1,
    global_shadow_state2,
    slice_sub,
)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_jit(backend):
    nq, repeat = 8, 5
    ps = [1, 0, 0, 0, 2, 0, 0, 0]
    sub = (1, 3, 6, 7)
    error = 0.1
    ns, k = shadow_bound(ps, error)
    ns //= repeat

    thetas = 2 * np.random.rand(2, nq) - 1

    c = tc.Circuit(nq)
    for i in range(nq):
        c.H(i)
    for i in range(2):
        for j in range(nq):
            c.cnot(j, (j + 1) % nq)
        for j in range(nq):
            c.rz(j, theta=thetas[i, j] * np.pi)

    psi = c.state()
    pauli_strings = tc.backend.convert_to_tensor(np.random.randint(1, 4, size=(ns, nq)))
    status = tc.backend.convert_to_tensor(np.random.rand(ns, repeat))

    def classical_shadow(psi, pauli_strings, status):
        lss_states = shadow_snapshots(psi, pauli_strings, status)
        expc = expection_ps_shadow(lss_states, ps=ps, k=k)
        ent = entropy_shadow(lss_states, sub=sub, alpha=2)
        return expc, ent

    csjit = tc.backend.jit(classical_shadow)

    exact_expc = c.expectation_ps(ps=ps)
    exact_rdm = tc.quantum.reduced_density_matrix(psi, cut=[0, 2, 4, 5])
    exact_ent = tc.quantum.renyi_entropy(exact_rdm, k=2)
    expc, ent = csjit(psi, pauli_strings, status)
    expc = np.median(expc)

    assert np.abs(expc - exact_expc) < error
    assert np.abs(ent - exact_ent) < 5 * error


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_state(backend):
    nq, ns = 2, 10000

    c = tc.Circuit(nq)
    c.H(0)
    c.cnot(0, 1)

    psi = c.state()
    bell_state = psi[:, None] @ psi[None, :]

    pauli_strings = tc.backend.convert_to_tensor(np.random.randint(1, 4, size=(ns, nq)))
    status = tc.backend.convert_to_tensor(np.random.rand(ns, 2))
    lss_states = shadow_snapshots(c.state(), pauli_strings, status)
    sdw_state = global_shadow_state(lss_states)

    np.allclose(sdw_state, bell_state, atol=0.01)


# @pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
# def test_expc(backend):
#     import pennylane as qml
#
#     nq, ns, repeat = 6, 2000, 1000
#
#     thetas = 2 * np.random.rand(2, nq) - 1
#
#     c = tc.Circuit(nq)
#     for i in range(nq):
#         c.H(i)
#     for i in range(2):
#         for j in range(nq):
#             c.cnot(j, (j + 1) % nq)
#         for j in range(nq):
#             c.rz(j, theta=thetas[i, j] * np.pi)
#
#     ps = [1, 0, 0, 0, 0, 3]
#     sub = [1, 4]
#     psi = c.state()
#
#     pauli_strings = tc.backend.convert_to_tensor(np.random.randint(1, 4, size=(ns, nq)))
#     status = tc.backend.convert_to_tensor(np.random.rand(ns, repeat))
#     snapshots = shadow_snapshots(psi, pauli_strings, status, measurement_only=True)
#
#     exact_expc = c.expectation_ps(ps=ps)
#     exact_rdm = tc.quantum.reduced_density_matrix(
#         psi, cut=[i for i in range(nq) if i not in sub]
#     )
#     exact_ent = tc.quantum.renyi_entropy(exact_rdm, k=2)
#     print(exact_expc, exact_ent)
#
#     expc = np.median(expection_ps_shadow(snapshots, pauli_strings, ps=ps, k=9))
#     ent = entropy_shadow(snapshots, pauli_strings, sub, alpha=2)
#     ent2 = Renyi_entropy_2(snapshots, sub)
#     print(expc, ent, ent2)
#
#     pl_snapshots = np.asarray(snapshots).reshape(ns * repeat, nq)
#     pl_ps = np.tile(np.asarray(pauli_strings - 1)[:, None, :], (1, repeat, 1)).reshape(
#         ns * repeat, nq
#     )
#     shadow = qml.ClassicalShadow(pl_snapshots, pl_ps)
#     H = qml.PauliX(0) @ qml.PauliZ(5)
#     pl_expc = shadow.expval(H, k=9)
#     pl_ent = shadow.entropy(sub, alpha=2)
#     print(pl_expc, pl_ent)
#
#     assert np.isclose(expc, pl_expc)
#     assert np.isclose(ent, pl_ent)
