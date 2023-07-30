"""Classical Shadows base class with processing functions"""
from typing import Optional, Sequence
from string import ascii_letters as ABC
from tensorcircuit.cons import backend
from tensorcircuit.circuit import Circuit
import numpy as np


def shadow_snapshots(psi, pauli_strings, repeat: int = 1):
    '''
        ns: number of snapshots
        nq: number of qubits
        cir: quantum circuit
        pauli_strings = (ns, nq)
        repeat: times to measure on one pauli string

        return:
        snapshots = (ns, repeat, nq)
    '''
    if pauli_strings.dtype not in ("int", "int32"):
        raise TypeError("Expected a int data type. " f"Got {pauli_strings.dtype}")

    # if backend.max(pauli_strings) > 2 or backend.min(pauli_strings) < 0:
    #     raise ValueError("The elements in pauli_strings must be between 0 and 2!")

    angles = backend.cast(backend.convert_to_tensor(
        np.array([[-np.pi / 2, np.pi / 4, 0], [np.pi / 3, np.arccos(1 / np.sqrt(3)), np.pi / 4], [0, 0, 0]])),
                          dtype=pauli_strings.dtype)

    nq = pauli_strings.shape[1]
    assert 2 ** nq == len(psi)

    def proj_measure(pauli_string):
        # pauli_rot = backend.onehot(pauli_string, num=3) @ angles
        pauli_rot = angles[pauli_string]
        c_ = Circuit(nq, inputs=psi)
        for i, (theta, alpha, phi) in enumerate(pauli_rot):
            c_.R(i, theta=theta, alpha=alpha, phi=phi)
        return c_.sample(batch=repeat, format="sample_bin")

    vpm = backend.vmap(proj_measure, vectorized_argnums=0)
    return vpm(pauli_strings)   # (ns, repeat, nq)


def snapshot_states(snapshots, pauli_strings):
    '''
        ns: number of snapshots
        nq: number of qubits
        pauli_strings = (ns, nq) or (ns, repeat, nq)
        snapshots = (ns, repeat, nq)

        return:
        snapshot_states = (ns, repeat, nq, 2, 2)
    '''
    if len(pauli_strings.shape) < len(snapshots.shape):
        pauli_strings = backend.tile(pauli_strings[:, None, :], (1, snapshots.shape[1], 1))  # (ns, repeat, nq)

    X_dm = backend.cast(np.array([[[1, 1], [1, 1]], [[1, -1], [-1, 1]]]) / 2, dtype=complex)
    Y_dm = backend.cast(np.array([[[1, -1j], [1j, 1]], [[1, 1j], [-1j, 1]]]) / 2, dtype=complex)
    Z_dm = backend.cast(np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]]), dtype=complex)
    pauli_dm = backend.convert_to_tensor(backend.stack((X_dm, Y_dm, Z_dm), axis=0))  # (3, 2, 2, 2)

    def dm(pauli, ss):
        return pauli_dm[pauli, ss]

    v = backend.vmap(dm, vectorized_argnums=(0, 1))
    vv = backend.vmap(v, vectorized_argnums=(0, 1))
    vvv = backend.vmap(vv, vectorized_argnums=(0, 1))

    return vvv(pauli_strings, snapshots)


def shadow_state(snapshots, pauli_strings, sub: Optional[Sequence[int]] = None):
    '''
        ns: number of snapshots
        nq: number of qubits
        pauli_strings = (ns, nq) or (ns, repeat, nq)
        snapshots = (ns, repeat, nq)

        return:
        shadow_state = (2 ** nq, 2 ** nq)
    '''
    return backend.mean(global_shadow_states(snapshot_states(snapshots, pauli_strings), sub), axis=(0, 1))


def expection_ps_shadow(snapshots, pauli_strings, x: Optional[Sequence[int]] = None, y: Optional[Sequence[int]] = None,
                   z: Optional[Sequence[int]] = None, ps: Optional[Sequence[int]] = None, k: int = 1):
    '''
        ns: number of snapshots
        nq: number of qubits
        pauli_strings = (ns, nq) or (ns, repeat, nq)
        snapshots = (ns, repeat, nq)

        return:
        expection = (1,)
    '''
    ss_states = snapshot_states(snapshots, pauli_strings)  # (ns, repeat, nq, 2, 2)
    ns, repeat, nq, _, _ = ss_states.shape
    ns *= repeat
    ss_states = backend.reshape(ss_states, (ns, nq, 2, 2))

    if ps is not None:
        ps = np.array(ps)  # (nq,)
    else:
        ps = np.zeros(nq, dtype=int)
        if x is not None:
            for i in x:
                ps[i] = 0
        if y is not None:
            for i in y:
                ps[i] = 1
        if z is not None:
            for i in z:
                ps[i] = 2

    paulis = backend.convert_to_tensor(
        backend.cast(np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]),
                     dtype=ss_states.dtype))

    def sqp(dm, p_idx):
        return 3 * backend.real(backend.trace(paulis[p_idx] @ dm))

    v = backend.vmap(sqp, vectorized_argnums=(0, 1))  # (nq,)

    def prod(dm, p_idx):
        tensor = v(dm, p_idx)
        return backend.shape_prod(tensor)

    vv = backend.vmap(prod, vectorized_argnums=0)  # (ns,)

    batch = ns // k
    means = []
    for i in range(0, ns, batch):
        means.append(backend.mean(vv(ss_states[i: i + batch], ps)))
    return means


def entropy_shadow(snapshots, pauli_strings, sub: Optional[Sequence[int]] = None, alpha: int = 1):
    '''
            ns: number of snapshots
            nq: number of qubits
            pauli_strings = (ns, nq) or (ns, repeat, nq)
            snapshots = (ns, repeat, nq)

            return:
            entropy = (1,)
        '''
    if alpha <= 0:
        raise ValueError("Alpha should not be less than 1!")

    sdw_rdm = shadow_state(snapshots, pauli_strings, sub)   # (2 ** nq, 2 ** nq)

    evs = backend.relu(backend.eigvalsh(sdw_rdm))
    if alpha == 1:
        return -backend.sum(evs * backend.relu(backend.log(evs + 1e-15)))
    else:
        return backend.log(backend.sum(backend.power(evs, alpha))) / (1 - alpha)


def local_shadow_states(ss_states, sub: Optional[Sequence[int]] = None):
    '''
        ns: number of snapshots
        nq: number of qubits
        ss_states = (ns, repeat, nq, 2, 2)

        return:
        local_shadow_states = (ns, repeat, nq, 2, 2)
    '''
    if sub is None:
        return 3 * ss_states - backend.eye(2)[None, None, None, :, :]
    else:
        sub = backend.convert_to_tensor(np.array(sub))
        return 3 * ss_states[:, :, sub] - backend.eye(2)[None, None, None, :, :]


def global_shadow_states(ss_states, sub: Optional[Sequence[int]] = None):
    '''
        ns: number of snapshots
        nq: number of qubits
        ss_states = (ns, repeat, nq, 2, 2)

        return:
        global_shadow_states = (ns, repeat, 2 ** nq, 2 ** nq)
    '''
    shadow_states = backend.transpose(local_shadow_states(ss_states, sub), (2, 0, 1, 3, 4))   # (nq, ns, repeat, 2, 2)
    nq, ns, repeat, _, _ = shadow_states.shape

    old_indices = [f"ab{ABC[2 + 2 * i: 4 + 2 * i]}" for i in range(nq)]
    new_indices = f"ab{ABC[2:2 * nq + 2:2]}{ABC[3:2 * nq + 2:2]}"

    return backend.reshape(
        backend.einsum(f'{",".join(old_indices)}->{new_indices}', *shadow_states, optimize=True),
        (ns, repeat, 2 ** nq, 2 ** nq),
    )


def global_shadow_states0(ss_states, sub: Optional[Sequence[int]] = None):
    '''
        ns: number of snapshots
        nq: number of qubits
        ss_states = (ns, repeat, nq, 2, 2)

        return:
        global_shadow_states = (ns, repeat, 2 ** nq, 2 ** nq)
    '''
    shadow_states = local_shadow_states(ss_states, sub)  # (ns, repeat, nq, 2, 2)
    ns, repeat, nq, _, _ = ss_states.shape

    old_indices = [f"{ABC[2 * i: 2 + 2 * i]}" for i in range(nq)]
    new_indices = f"{ABC[0:2 * nq:2]}{ABC[1:2 * nq:2]}"

    def tensor_prod(dms):
        return backend.reshape(backend.einsum(f'{",".join(old_indices)}->{new_indices}', *dms, optimize=True),
                               (2 ** nq, 2 ** nq))

    v = backend.vmap(tensor_prod, vectorized_argnums=0)
    vv = backend.vmap(v, vectorized_argnums=0)
    return vv(shadow_states)


def global_shadow_states1(ss_states, sub: Optional[Sequence[int]] = None):
    '''
        ns: number of snapshots
        nq: number of qubits
        ss_states = (ns, repeat, nq, 2, 2)

        return:
        global_shadow_states = (ns, repeat, 2 ** nq, 2 ** nq)
    '''
    shadow_states = local_shadow_states(ss_states, sub)   # (ns, repeat, nq, 2, 2)

    def tensor_prod(dms):
        res = dms[0]
        for dm in dms[1:]:
            res = backend.kron(res, dm)
        return res

    v = backend.vmap(tensor_prod, vectorized_argnums=0)
    vv = backend.vmap(v, vectorized_argnums=0)
    return vv(shadow_states)


if __name__ == "__main__":
    import time
    from tensorcircuit import set_backend
    backend = set_backend('jax')
    N = 6

    # old_indices = [f"a{ABC[1 + 2 * i: 3 + 2 * i]}" for i in range(N)]
    # new_indices = f"a{ABC[1:2 * N + 1:2]}{ABC[2:2 * N + 1:2]}"
    # print(old_indices, new_indices)
    # exit(0)

    c = Circuit(N)
    for i in range(N):
        c.H(i)
    for i in range(0, 2):
        for j in range(N):
            c.cnot(j, (j + 1) % N)
        for j in range(N):
            c.rx(j, theta=0.1)
    psi0 = c.state()
    # print(c.sample())

    def classical_shadow(psi, pauli_strings):
        snapshots = shadow_snapshots(psi, pauli_strings, repeat=1)
        # ss_states = snapshot_states(snapshots, pauli_strings)
        # return local_shadow_states(ss_states)
        # return global_shadow_states(ss_states)
        # return shadow_state(snapshots, pauli_strings)
        return expection_ps_shadow(snapshots, pauli_strings, x=[0, 1], y=[3, 4], k=6)
        # return entropy_shadow(snapshots, pauli_strings, sub=range(3), alpha=1)

    csjit = backend.jit(classical_shadow)

    pauli_strings = backend.convert_to_tensor(np.random.randint(0, 3, size=(100, N)))
    res = csjit(psi0, pauli_strings)
    print(res)




