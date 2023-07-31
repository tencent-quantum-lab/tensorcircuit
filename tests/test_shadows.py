import numpy as np
import time
import pytest
from tensorcircuit.circuit import Circuit
from tensorcircuit import set_backend
from tensorcircuit.shadows import shadow_snapshots, shadow_state, local_snapshot_states, global_snapshot_states, \
    entropy_shadow, expection_ps_shadow


if __name__ == "__main__":
    backend = set_backend('jax')
    N = 6

    c = Circuit(N)
    for i in range(N):
        c.H(i)
    for i in range(0, 2):
        for j in range(N):
            c.cnot(j, (j + 1) % N)
        for j in range(N):
            c.rx(j, theta=0.1 * np.pi)

    ps = [1, 1, 0, 2, 2, 0]

    print("exact:", c.expectation_ps(ps=ps))

    psi0 = c.state()
    pauli_strings = np.random.randint(0, 3, size=(10000, N))
    snapshots = shadow_snapshots(psi0, backend.convert_to_tensor(pauli_strings), repeat=1)

    import pennylane as qml
    shadow = qml.ClassicalShadow(np.asarray(snapshots[:, 0]), np.asarray(pauli_strings))
    H = [qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(3) @ qml.PauliY(4)]
    pl_expc = shadow.expval(H, k=10)
    pl_ent = shadow.entropy(range(2), alpha=1)
    print("pl:", pl_expc, pl_ent)
    expc = expection_ps_shadow(snapshots, pauli_strings, ps=ps, k=10)
    ent = entropy_shadow(snapshots, pauli_strings, sub=range(2), alpha=1)
    print("here:", np.median(expc), ent)
    print(expc)

    # jit test
    def classical_shadow(psi, pauli_strings):
        snapshots = shadow_snapshots(psi, pauli_strings, repeat=1)
        # return local_snapshot_states(snapshots, pauli_strings, sub=[1, 3, 5])
        # return global_snapshot_states1(snapshots, pauli_strings, sub=[1, 3, 5])
        # return shadow_state(snapshots, pauli_strings, sub=[1, 3, 5])
        # return expection_ps_shadow(snapshots, pauli_strings, ps=ps, k=10)
        return entropy_shadow(snapshots, pauli_strings, sub=range(3), alpha=1)

    csjit = backend.jit(classical_shadow)

    pauli_strings = backend.convert_to_tensor(np.random.randint(0, 3, size=(10000, N)))
    res = csjit(psi0, pauli_strings)
    print(res)