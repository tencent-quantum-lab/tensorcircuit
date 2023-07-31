"""Classical Shadows functions"""
from typing import Optional, Sequence
from string import ascii_letters as ABC
import numpy as np

from .cons import backend
from .circuit import Circuit


def shadow_snapshots(psi, pauli_strings, repeat: int = 1):
    r"""To generate the shadow snapshots from given pauli-string observable on $|\psi\rangle$

    :param psi: shape = (2 ** nq, 2 ** nq), where nq is the number of qubits
    :type: Tensor
    :param pauli_strings: shape = (ns, nq), where ns is the number of pauli strings
    :type: Tensor
    :param repeat: times to measure on one pauli string
    :type: int

    :return snapshots: shape = (ns, repeat, nq)
    :rtype: Tensor
    """
    pauli_strings = backend.cast(pauli_strings, dtype=int)

    angles = backend.convert_to_tensor(
        np.array([[-np.pi / 2, np.pi / 4, 0], [np.pi / 3, np.arccos(1 / np.sqrt(3)), np.pi / 4], [0, 0, 0]]))

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


def local_snapshot_states(snapshots, pauli_strings, sub: Optional[Sequence[int]] = None):
    r"""To generate the local snapshots states from snapshots and pauli strings

    :param snapshots: shape = (ns, repeat, nq)
    :type: Tensor
    :param pauli_strings: shape = (ns, nq) or (ns, repeat, nq)
    :type: Tensor
    :param sub: qubit indices of subsystem
    :type: Optional[Sequence[int]]

    :return lss_states: shape = (ns, repeat, nq, 2, 2)
    :rtype: Tensor
    """
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

    lss_states = vvv(pauli_strings, snapshots)
    if sub is not None:
        sub = backend.convert_to_tensor(np.array(sub))
        lss_states = lss_states[:, :, sub]
    return 3 * lss_states - backend.eye(2)[None, None, None, :, :]


def global_snapshot_states(snapshots, pauli_strings=None, sub: Optional[Sequence[int]] = None):
    r"""To generate the global snapshots states from local snapshot states or snapshots and pauli strings

    :param snapshots: shape = (ns, repeat, nq, 2, 2) or (ns, repeat, nq)
    :type: Tensor
    :param pauli_strings: shape = None or (ns, nq) or (ns, repeat, nq)
    :type: Optional[Tensor]
    :param sub: qubit indices of subsystem
    :type: Optional[Sequence[int]]

    :return gss_states: shape = (ns, repeat, 2 ** nq, 2 ** nq)
    :rtype: Tensor
    """
    if pauli_strings is not None:
        assert len(snapshots.shape) == 3
        lss_states = local_snapshot_states(snapshots, pauli_strings, sub)  # (ns, repeat, nq_sub, 2, 2)
    else:
        if sub is not None:
            sub = backend.convert_to_tensor(np.array(sub))
            lss_states = snapshots[:, :, sub]  # (ns, repeat, nq_sub, 2, 2)
        else:
            lss_states = snapshots  # (ns, repeat, nq, 2, 2)

    def tensor_prod(dms):
        res = dms[0]
        for dm in dms[1:]:
            res = backend.kron(res, dm)
        return res

    v = backend.vmap(tensor_prod, vectorized_argnums=0)
    vv = backend.vmap(v, vectorized_argnums=0)
    return vv(lss_states)


def shadow_state(snapshots, pauli_strings=None, sub: Optional[Sequence[int]] = None):
    r"""To generate the shadow states from global snapshot states or local snapshot states or snapshots and pauli strings

    :param snapshots: shape = (ns, repeat, 2 ** nq, 2 ** nq) or (ns, repeat, nq, 2, 2) or (ns, repeat, nq)
    :type: Tensor
    :param pauli_strings: shape = None or (ns, nq) or (ns, repeat, nq)
    :type: Optional[Tensor]
    :param sub: qubit indices of subsystem
    :type: Optional[Sequence[int]]

    :return shadow_state: shape = (2 ** nq, 2 ** nq)
    :rtype: Tensor
    """
    if len(snapshots.shape) == 4:
        assert sub is None
        gss_states = snapshots  # (ns, repeat, 2 ** nq, 2 ** nq)
    else:
        gss_states = global_snapshot_states(snapshots, pauli_strings, sub)  # (ns, repeat, 2 ** nq_sub, 2 ** nq_sub)
    return backend.mean(gss_states, axis=(0, 1))


def expection_ps_shadow(snapshots, pauli_strings=None, x: Optional[Sequence[int]] = None,
                        y: Optional[Sequence[int]] = None, z: Optional[Sequence[int]] = None,
                        ps: Optional[Sequence[int]] = None, k: int = 1):
    r"""To calculate the expectation value of an observable on shadow snapshot states

    :param snapshots: shape = (ns, repeat, nq, 2, 2) or (ns, repeat, nq)
    :type: Tensor
    :param pauli_strings: shape = None or (ns, nq) or (ns, repeat, nq)
    :type: Optional[Tensor]
    :param sub: qubit indices of subsystem
    :type: Optional[Sequence[int]]
    :param x: sites to apply X gate, defaults to None
    :type: Optional[Sequence[int]]
    :param y: sites to apply Y gate, defaults to None
    :type: Optional[Sequence[int]]
    :param z: sites to apply Z gate, defaults to None
    :type: Optional[Sequence[int]]
    :param ps: or one can apply a ps structures instead of x, y, z, e.g. [0, 1, 3, 0, 2, 2] for X_1Z_2Y_4Y_5 defaults to None, ps can overwrite x, y and z
    :type: Optional[Sequence[int]]
    :param k: Number of equal parts to split the shadow snapshot states to compute the median of means. k=1 (default) corresponds to simply taking the mean over all shadow snapshot states.
    :type: int

    :return expectation values: shape = (k,) or (k + 1,)
    :rtype: List[Tensor]
    """
    if pauli_strings is not None:
        assert len(snapshots.shape) == 3
        lss_states = local_snapshot_states(snapshots, pauli_strings)  # (ns, repeat, nq, 2, 2)
    else:
        lss_states = snapshots  # (ns, repeat, nq, 2, 2)
    ns, repeat, nq, _, _ = lss_states.shape
    ns *= repeat
    ss_states = backend.reshape(lss_states, (ns, nq, 2, 2))

    if ps is not None:
        ps = np.array(ps)  # (nq,)
    else:
        ps = np.zeros(nq, dtype=int)
        if x is not None:
            for i in x:
                ps[i] = 1
        if y is not None:
            for i in y:
                ps[i] = 2
        if z is not None:
            for i in z:
                ps[i] = 3

    paulis = backend.convert_to_tensor(
        backend.cast(np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]),
                     dtype=ss_states.dtype))    # (4, 2, 2)

    def sqp(dm, p_idx):
        return backend.real(backend.trace(paulis[p_idx] @ dm))

    v = backend.vmap(sqp, vectorized_argnums=(0, 1))  # (nq,)

    def prod(dm):
        tensor = v(dm, ps)
        return backend.shape_prod(tensor)

    vv = backend.vmap(prod, vectorized_argnums=0)  # (ns,)

    batch = ns // k
    means = []
    for i in range(0, ns, batch):
        ans = vv(ss_states[i: i + batch])
        means.append(backend.mean(ans))
    return means


def entropy_shadow(ss_or_sd, pauli_strings=None, sub: Optional[Sequence[int]] = None, alpha: int = 1):
    r"""To calculate the Renyi entropy of a subsystem from shadow state or shadow snapshot states

    :param ss_or_sd: shadow state (shape = (2 ** nq, 2 ** nq)) or snapshot states (shape = (ns, repeat, 2 ** nq, 2 ** nq) or (ns, repeat, nq, 2, 2) or (ns, repeat, nq))
    :type: Tensor
    :param pauli_strings: shape = None or (ns, nq) or (ns, repeat, nq)
    :type: Optional[Tensor]
    :param sub: qubit indices of subsystem
    :type: Optional[Sequence[int]]
    :param alpha: order of the Renyi entropy, alpha=1 corresponds to the von Neumann entropy/
    :type: int

    :return Renyi entropy: shape = (1,)
    :rtype: Tensor
    """
    if alpha <= 0:
        raise ValueError("Alpha should not be less than 1!")

    if len(ss_or_sd.shape) == 2 and ss_or_sd.shape[0] == ss_or_sd.shape[1]:
        assert sub is None
        sdw_rdm = ss_or_sd
    else:
        sdw_rdm = shadow_state(ss_or_sd, pauli_strings, sub)   # (2 ** nq, 2 ** nq)

    evs = backend.relu(backend.eigvalsh(sdw_rdm))
    evs /= backend.sum(evs)
    if alpha == 1:
        return -backend.sum(evs * backend.log(evs + 1e-15))
    else:
        return backend.log(backend.sum(backend.power(evs, alpha))) / (1 - alpha)


def global_snapshot_states1(snapshots, pauli_strings=None, sub: Optional[Sequence[int]] = None):
    r"""To generate the global snapshots states from local snapshot states or snapshots and pauli strings

    :param snapshots: shape = (ns, repeat, nq, 2, 2) or (ns, repeat, nq)
    :type: Tensor
    :param pauli_strings: shape = None or (ns, nq) or (ns, repeat, nq)
    :type: Optional[Tensor]
    :param sub: qubit indices of subsystem
    :type: Optional[Sequence[int]]

    :return gss_states: shape = (ns, repeat, 2 ** nq, 2 ** nq)
    :rtype: Tensor
    """
    if pauli_strings is not None:
        assert len(snapshots.shape) == 3
        lss_states = local_snapshot_states(snapshots, pauli_strings, sub)  # (ns, repeat, nq_sub, 2, 2)
    else:
        if sub is not None:
            sub = backend.convert_to_tensor(np.array(sub))
            lss_states = snapshots[:, :, sub]   # (ns, repeat, nq_sub, 2, 2)
        else:
            lss_states = snapshots    # (ns, repeat, nq, 2, 2)
    lss_states = backend.transpose(lss_states, (2, 0, 1, 3, 4))   # (nq, ns, repeat, 2, 2)
    nq, ns, repeat, _, _ = lss_states.shape

    old_indices = [f"ab{ABC[2 + 2 * i: 4 + 2 * i]}" for i in range(nq)]
    new_indices = f"ab{ABC[2:2 * nq + 2:2]}{ABC[3:2 * nq + 2:2]}"

    return backend.reshape(
        backend.einsum(f'{",".join(old_indices)}->{new_indices}', *lss_states, optimize=True),
        (ns, repeat, 2 ** nq, 2 ** nq),
    )


def global_snapshot_states2(snapshots, pauli_strings=None, sub: Optional[Sequence[int]] = None):
    r"""To generate the global snapshots states from local snapshot states or snapshots and pauli strings

    :param snapshots: shape = (ns, repeat, nq, 2, 2) or (ns, repeat, nq)
    :type: Tensor
    :param pauli_strings: shape = None or (ns, nq) or (ns, repeat, nq)
    :type: Optional[Tensor]
    :param sub: qubit indices of subsystem
    :type: Optional[Sequence[int]]

    :return gss_states: shape = (ns, repeat, 2 ** nq, 2 ** nq)
    :rtype: Tensor
    """
    if pauli_strings is not None:
        assert len(snapshots.shape) == 3
        lss_states = local_snapshot_states(snapshots, pauli_strings, sub)  # (ns, repeat, nq_sub, 2, 2)
    else:
        if sub is not None:
            sub = backend.convert_to_tensor(np.array(sub))
            lss_states = snapshots[:, :, sub]  # (ns, repeat, nq_sub, 2, 2)
        else:
            lss_states = snapshots  # (ns, repeat, nq, 2, 2)
    ns, repeat, nq, _, _ = lss_states.shape

    old_indices = [f"{ABC[2 * i: 2 + 2 * i]}" for i in range(nq)]
    new_indices = f"{ABC[0:2 * nq:2]}{ABC[1:2 * nq:2]}"

    def tensor_prod(dms):
        return backend.reshape(backend.einsum(f'{",".join(old_indices)}->{new_indices}', *dms, optimize=True),
                               (2 ** nq, 2 ** nq))

    v = backend.vmap(tensor_prod, vectorized_argnums=0)
    vv = backend.vmap(v, vectorized_argnums=0)
    return vv(lss_states)








