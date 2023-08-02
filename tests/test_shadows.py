import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf
import tensorcircuit as tc
from tensorcircuit.shadows import shadow_snapshots, local_snapshot_states, global_shadow_state, global_shadow_state1, \
    global_shadow_state2, entropy_shadow, expection_ps_shadow, shadow_bound


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_jit(backend):
    nq, ns = 8, 10000

    c = tc.Circuit(nq)
    for i in range(nq):
        c.H(i)
    for i in range(0, 2):
        for j in range(nq):
            c.cnot(j, (j + 1) % nq)
        for j in range(nq):
            c.rx(j, theta=0.5 * np.pi)

    def classical_shadow(psi, pauli_strings, status, sub, x, y, z):
        snapshots = shadow_snapshots(psi, pauli_strings, status)
        lss_states = local_snapshot_states(snapshots, pauli_strings)
        sdw_state = global_shadow_state(lss_states, sub=sub)
        expc = expection_ps_shadow(lss_states, x=x, y=y, z=z, k=20)
        ent = entropy_shadow(sdw_state, alpha=2)
        return expc, ent, lss_states, sdw_state

    csjit = tc.backend.jit(classical_shadow, static_argnums=(4, 5, 6))

    pauli_strings = tc.backend.convert_to_tensor(np.random.randint(0, 3, size=(ns, nq)))
    status = tc.backend.convert_to_tensor(np.random.rand(ns, 6))
    x, y, z = (0, 6), (2, 3), (5, 7)
    sub = [1, 3, 6, 7]

    exact = c.expectation_ps(x=x, y=y, z=z)
    expc, ent, lss_states, sdw_state = csjit(c.state(), pauli_strings, status, sub, x, y, z)
    expc = np.median(expc)

    assert np.abs(expc - exact) < 0.1


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_state(backend):
    nq, ns = 2, 10000

    c = tc.Circuit(nq)
    c.H(0)
    c.cnot(0, 1)

    psi = c.state()
    bell_state = psi[:, None] @ psi[None, :]

    pauli_strings = tc.backend.convert_to_tensor(np.random.randint(0, 3, size=(ns, nq)))
    status = tc.backend.convert_to_tensor(np.random.rand(ns, 1))
    snapshots = shadow_snapshots(c.state(), pauli_strings, status)

    lss_states = local_snapshot_states(snapshots, pauli_strings)
    sdw_state = global_shadow_state(lss_states)

    R = np.array(sdw_state - bell_state)
    assert np.sqrt(np.trace(R.conj().T @ R)) < 0.1


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_expc(backend):
    nq, ns = 6, 10000

    c = tc.Circuit(nq)
    for i in range(nq):
        c.H(i)
    for i in range(0, 2):
        for j in range(nq):
            c.cnot(j, (j + 1) % nq)
        for j in range(nq):
            c.rx(j, theta=0.1 * np.pi)

    ps = [1, 1, 0, 2, 3, 0]
    exact = c.expectation_ps(ps=ps)

    pauli_strings = tc.backend.convert_to_tensor(np.random.randint(0, 3, size=(ns, nq)))
    status = tc.backend.convert_to_tensor(np.random.rand(ns, 2))
    snapshots = shadow_snapshots(c.state(), pauli_strings, status)

    expc = np.median(expection_ps_shadow(snapshots, pauli_strings, ps=ps, k=9))
    # ent = entropy_shadow(snapshots, pauli_strings, range(4), alpha=2)
    # import pennylane as qml
    # shadow = qml.ClassicalShadow(np.asarray(snapshots[:, 0]), np.asarray(pauli_strings))   # repeat == 1
    # H = qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(3) @ qml.PauliZ(4)
    # pl_expc = shadow.expval(H, k=9)
    # pl_ent = shadow.entropy(range(4), alpha=2)
    # print(np.isclose(expc, pl_expc), np.isclose(ent, pl_ent))

    assert np.abs(expc - exact) < 0.1
