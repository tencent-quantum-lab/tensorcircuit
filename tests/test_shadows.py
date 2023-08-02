import pytest
from pytest_lazyfixture import lazy_fixture as lf
import numpy as np
import tensorcircuit as tc
from tensorcircuit.shadows import (
    shadow_bound,
    shadow_snapshots,
    local_snapshot_states,
    global_shadow_state,
    global_shadow_state1,
    global_shadow_state2,
    entropy_shadow,
    expection_ps_shadow,
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
    nq, ns = 2, 6000

    c = tc.Circuit(nq)
    c.H(0)
    c.cnot(0, 1)

    psi = c.state()
    bell_state = psi[:, None] @ psi[None, :]

    pauli_strings = tc.backend.convert_to_tensor(np.random.randint(1, 4, size=(ns, nq)))
    status = tc.backend.convert_to_tensor(np.random.rand(ns, 2))
    lss_states = shadow_snapshots(c.state(), pauli_strings, status)
    sdw_state = global_shadow_state(lss_states)

    R = np.array(sdw_state - bell_state)
    error = np.sqrt(np.trace(R.conj().T @ R))
    assert error < 0.1


# @pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
# def test_expc(backend):
#     import pennylane as qml
#     nq, ns = 8, 100000
#
#     c = tc.Circuit(nq)
#     for i in range(nq):
#         c.H(i)
#     for j in range(nq):
#         c.cnot(j, (j + 1) % nq)
#     for j in range(nq):
#         c.rx(j, theta=0.1 * np.pi)
#
#     ps = [1, 0, 2, 2, 0, 3, 1, 3]
#     exact = c.expectation_ps(ps=ps)
#
#     pauli_strings = tc.backend.convert_to_tensor(np.random.randint(1, 4, size=(ns, nq)))
#     status = tc.backend.convert_to_tensor(np.random.rand(ns, 1))
#     snapshots = shadow_snapshots(
#         c.state(), pauli_strings, status, measurement_only=True
#     )
#
#     expc = np.median(expection_ps_shadow(snapshots, pauli_strings, ps=ps, k=9))
#     ent = entropy_shadow(snapshots, pauli_strings, range(4), alpha=2)
#
#     shadow = qml.ClassicalShadow(np.asarray(snapshots[:, 0]), np.asarray(pauli_strings - 1))   # repeat == 1
#     H = qml.PauliX(0) @ qml.PauliX(6) @ qml.PauliY(2)@ qml.PauliY(3) @ qml.PauliZ(5) @ qml.PauliZ(7)
#     pl_expc = shadow.expval(H, k=9)
#     pl_ent = shadow.entropy(range(4), alpha=2)
#
#     print(np.isclose(expc, pl_expc), np.isclose(ent, pl_ent))
#
#     assert np.abs(expc - exact) < 0.1
