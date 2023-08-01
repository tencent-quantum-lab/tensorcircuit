"""
Classical Shadows functions
"""
from typing import Optional, Sequence
from string import ascii_letters as ABC
import numpy as np

from .cons import backend, dtypestr, rdtypestr
from .circuit import Circuit


def shadow_snapshots(psi, pauli_strings, repeat: int = 1):
    r"""To generate the shadow snapshots from given pauli string observables on $|\psi\rangle$

    :param psi: shape = (2 ** nq, 2 ** nq), where nq is the number of qubits
    :type: Tensor
    :param pauli_strings: shape = (ns, nq), where ns is the number of pauli strings
    :type: Tensor
    :param repeat: times to measure on one pauli string
    :type: int

    :return snapshots: shape = (ns, repeat, nq)
    :rtype: Tensor
    """
    pauli_strings = backend.cast(pauli_strings, dtype="int32")

    angles = backend.cast(backend.convert_to_tensor(
        [[-np.pi / 2, np.pi / 4, 0], [np.pi / 3, np.arccos(1 / np.sqrt(3)), np.pi / 4], [0, 0, 0]]),
                          dtype=rdtypestr)  # (3, 3)

    nq = pauli_strings.shape[1]
    assert 2 ** nq == len(psi)

    def proj_measure(pauli_string):
        c_ = Circuit(nq, inputs=psi)
        for i in range(nq):
            c_.R(i, theta=backend.gather1d(backend.gather1d(angles, backend.gather1d(pauli_string, i)), 0),
                 alpha=backend.gather1d(backend.gather1d(angles, backend.gather1d(pauli_string, i)), 1),
                 phi=backend.gather1d(backend.gather1d(angles, backend.gather1d(pauli_string, i)), 1))
        return c_.sample(batch=repeat, format="sample_bin", allow_state=True)

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
    pauli_strings = backend.cast(pauli_strings, dtype="int32")
    if len(pauli_strings.shape) < len(snapshots.shape):
        pauli_strings = backend.tile(pauli_strings[:, None, :], (1, snapshots.shape[1], 1))  # (ns, repeat, nq)

    X_dm = backend.cast(backend.convert_to_tensor([[[1, 1], [1, 1]], [[1, -1], [-1, 1]]]) / 2, dtype=dtypestr)
    Y_dm = backend.cast(backend.convert_to_tensor(np.array([[[1, -1j], [1j, 1]], [[1, 1j], [-1j, 1]]]) / 2), dtype=dtypestr)
    Z_dm = backend.cast(backend.convert_to_tensor([[[1, 0], [0, 0]], [[0, 0], [0, 1]]]), dtype=dtypestr)
    pauli_dm = backend.stack((X_dm, Y_dm, Z_dm), axis=0)  # (3, 2, 2, 2)

    v = backend.vmap(lambda p, s: backend.gather1d(backend.gather1d(pauli_dm, p), s), vectorized_argnums=(0, 1))
    vv = backend.vmap(v, vectorized_argnums=(0, 1))
    vvv = backend.vmap(vv, vectorized_argnums=(0, 1))

    lss_states = vvv(pauli_strings, snapshots)
    if sub is not None:
        lss_states = backend.transpose(lss_states, (2, 0, 1, 3, 4))
        v0 = backend.vmap(lambda idx: backend.gather1d(lss_states, idx), vectorized_argnums=0)
        lss_states = backend.transpose(v0(backend.convert_to_tensor(sub)), (1, 2, 0, 3, 4))

    return 3 * lss_states - backend.eye(2)[None, None, None, :, :]


def global_shadow_state(snapshots, pauli_strings=None, sub: Optional[Sequence[int]] = None):
    r"""To generate the global shadow state from local snapshot states or snapshots and pauli strings

    :param snapshots: shape = (ns, repeat, nq, 2, 2) or (ns, repeat, nq)
    :type: Tensor
    :param pauli_strings: shape = None or (ns, nq) or (ns, repeat, nq)
    :type: Optional[Tensor]
    :param sub: qubit indices of subsystem
    :type: Optional[Sequence[int]]

    :return gsdw_state: shape = (2 ** nq, 2 ** nq)
    :rtype: Tensor
    """
    if pauli_strings is not None:
        if len(snapshots.shape) != 3:
            raise RuntimeError(
                f"snapshots should be 3-d if pauli_strings is not None, got {len(snapshots.shape)}-d instead.")
        pauli_strings = backend.cast(pauli_strings, dtype="int32")
        lss_states = local_snapshot_states(snapshots, pauli_strings, sub)  # (ns, repeat, nq_sub, 2, 2)
    else:
        if sub is not None:
            lss_states = backend.transpose(snapshots, (2, 0, 1, 3, 4))
            v0 = backend.vmap(lambda idx: backend.gather1d(lss_states, idx), vectorized_argnums=0)
            lss_states = backend.transpose(v0(backend.convert_to_tensor(sub)), (1, 2, 0, 3, 4))
        else:
            lss_states = snapshots  # (ns, repeat, nq, 2, 2)

    nq = lss_states.shape[2]

    def tensor_prod(dms):
        res = backend.gather1d(dms, 0)
        for i in range(1, nq):
            res = backend.kron(res, backend.gather1d(dms, i))
        return res

    v = backend.vmap(tensor_prod, vectorized_argnums=0)
    vv = backend.vmap(v, vectorized_argnums=0)
    gss_states = vv(lss_states)
    return backend.mean(gss_states, axis=(0, 1))


def expection_ps_shadow(snapshots, pauli_strings=None, x: Optional[Sequence[int]] = None,
                        y: Optional[Sequence[int]] = None, z: Optional[Sequence[int]] = None,
                        ps: Optional[Sequence[int]] = None, k: int = 1):
    r"""To calculate the expectation value of an observable on shadow snapshot states

    :param snapshots: shape = (ns, repeat, nq, 2, 2) or (ns, repeat, nq)
    :type: Tensor
    :param pauli_strings: shape = None or (ns, nq) or (ns, repeat, nq)
    :type: Optional[Tensor]
    :param x: sites to apply X gate, defaults to None
    :type: Optional[Sequence[int]]
    :param y: sites to apply Y gate, defaults to None
    :type: Optional[Sequence[int]]
    :param z: sites to apply Z gate, defaults to None
    :type: Optional[Sequence[int]]
    :param ps: or one can apply a ps structures instead of x, y, z, e.g. [1, 1, 0, 2, 3, 0] for X_0X_1Y_3Z_4 defaults to None, ps can overwrite x, y and z
    :type: Optional[Sequence[int]]
    :param k: Number of equal parts to split the shadow snapshot states to compute the median of means. k=1 (default) corresponds to simply taking the mean over all shadow snapshot states.
    :type: int

    :return expectation values: shape = (k,)
    :rtype: List[Tensor]
    """
    if pauli_strings is not None:
        if len(snapshots.shape) != 3:
            raise RuntimeError(
                f"snapshots should be 3-d if pauli_strings is not None, got {len(snapshots.shape)}-d instead.")
        pauli_strings = backend.cast(pauli_strings, dtype="int32")
        lss_states = local_snapshot_states(snapshots, pauli_strings)  # (ns, repeat, nq, 2, 2)
    else:
        lss_states = snapshots  # (ns, repeat, nq, 2, 2)
    ns, repeat, nq, _, _ = lss_states.shape
    ns *= repeat
    ss_states = backend.reshape(lss_states, (ns, nq, 2, 2))

    if ps is not None:
        if len(ps) != nq:
            raise RuntimeError(
                f"The number of qubits of the shadow state is {nq}, but got a {len(ps)}-qubit pauli string observable.")
        ps = np.array(ps)  # (nq,)
    else:
        ps = np.zeros(nq, dtype="int32")
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
                     dtype=dtypestr))    # (4, 2, 2)

    v = backend.vmap(lambda dm, idx: backend.real(backend.trace(backend.gather1d(paulis, idx) @ dm)),
                     vectorized_argnums=(0, 1))  # (nq,)
    vv = backend.vmap(lambda dm: backend.shape_prod(v(dm, ps)), vectorized_argnums=0)  # (ns,)

    batch = int(np.ceil(ns / k))
    return [backend.mean(vv(ss_states[i: i + batch])) for i in range(0, ns, batch)]


def entropy_shadow(ss_or_sd, pauli_strings=None, sub: Optional[Sequence[int]] = None, alpha: int = 1):
    r"""To calculate the Renyi entropy of a subsystem from shadow state or shadow snapshot states

    :param ss_or_sd: shadow state (shape = (2 ** nq, 2 ** nq)) or snapshot states (shape = (ns, repeat, nq, 2, 2) or (ns, repeat, nq))
    :type: Tensor
    :param pauli_strings: shape = None or (ns, nq) or (ns, repeat, nq)
    :type: Optional[Tensor]
    :param sub: qubit indices of subsystem
    :type: Optional[Sequence[int]]
    :param alpha: order of the Renyi entropy, alpha=1 corresponds to the von Neumann entropy
    :type: int

    :return Renyi entropy: shape = ()
    :rtype: Tensor
    """
    if alpha <= 0:
        raise ValueError("Alpha should not be less than 1!")

    if len(ss_or_sd.shape) == 2 and ss_or_sd.shape[0] == ss_or_sd.shape[1]:
        if sub is not None:
            raise ValueError("sub should be None if the input is the global shadow state.")
        sdw_rdm = ss_or_sd
    else:
        sdw_rdm = global_shadow_state(ss_or_sd, pauli_strings, sub)   # (2 ** nq, 2 ** nq)

    evs = backend.relu(backend.real(backend.eigvalsh(sdw_rdm)))
    evs /= backend.sum(evs)
    if alpha == 1:
        return -backend.sum(evs * backend.log(evs + 1e-12))
    else:
        return backend.log(backend.sum(backend.power(evs, alpha))) / (1 - alpha)


def global_shadow_state1(snapshots, pauli_strings=None, sub: Optional[Sequence[int]] = None):
    r"""To generate the global snapshots states from local snapshot states or snapshots and pauli strings

    :param snapshots: shape = (ns, repeat, nq, 2, 2) or (ns, repeat, nq)
    :type: Tensor
    :param pauli_strings: shape = None or (ns, nq) or (ns, repeat, nq)
    :type: Optional[Tensor]
    :param sub: qubit indices of subsystem
    :type: Optional[Sequence[int]]

    :return gsdw_state: shape = (2 ** nq, 2 ** nq)
    :rtype: Tensor
    """
    if pauli_strings is not None:
        if len(snapshots.shape) != 3:
            raise RuntimeError(
                f"snapshots should be 3-d if pauli_strings is not None, got {len(snapshots.shape)}-d instead.")
        pauli_strings = backend.cast(pauli_strings, dtype="int32")
        lss_states = local_snapshot_states(snapshots, pauli_strings, sub)  # (ns, repeat, nq_sub, 2, 2)
    else:
        if sub is not None:
            lss_states = backend.transpose(snapshots, (2, 0, 1, 3, 4))
            v0 = backend.vmap(lambda idx: backend.gather1d(lss_states, idx), vectorized_argnums=0)
            lss_states = backend.transpose(v0(backend.convert_to_tensor(sub)), (1, 2, 0, 3, 4))
        else:
            lss_states = snapshots    # (ns, repeat, nq, 2, 2)
    lss_states = backend.transpose(lss_states, (2, 0, 1, 3, 4))   # (nq, ns, repeat, 2, 2)
    nq, ns, repeat, _, _ = lss_states.shape

    old_indices = [f"ab{ABC[2 + 2 * i: 4 + 2 * i]}" for i in range(nq)]
    new_indices = f"ab{ABC[2:2 * nq + 2:2]}{ABC[3:2 * nq + 2:2]}"

    gss_states = backend.reshape(
        backend.einsum(f'{",".join(old_indices)}->{new_indices}', *lss_states, optimize=True),
        (ns, repeat, 2 ** nq, 2 ** nq),
    )
    return backend.mean(gss_states, axis=(0, 1))


def global_shadow_state2(snapshots, pauli_strings=None, sub: Optional[Sequence[int]] = None):
    r"""To generate the global snapshots states from local snapshot states or snapshots and pauli strings

    :param snapshots: shape = (ns, repeat, nq, 2, 2) or (ns, repeat, nq)
    :type: Tensor
    :param pauli_strings: shape = None or (ns, nq) or (ns, repeat, nq)
    :type: Optional[Tensor]
    :param sub: qubit indices of subsystem
    :type: Optional[Sequence[int]]

    :return gsdw_state: shape = (2 ** nq, 2 ** nq)
    :rtype: Tensor
    """
    if pauli_strings is not None:
        if len(snapshots.shape) != 3:
            raise RuntimeError(
                f"snapshots should be 3-d if pauli_strings is not None, got {len(snapshots.shape)}-d instead.")
        pauli_strings = backend.cast(pauli_strings, dtype="int32")
        lss_states = local_snapshot_states(snapshots, pauli_strings, sub)  # (ns, repeat, nq_sub, 2, 2)
    else:
        if sub is not None:
            lss_states = backend.transpose(snapshots, (2, 0, 1, 3, 4))
            v0 = backend.vmap(lambda idx: backend.gather1d(lss_states, idx), vectorized_argnums=0)
            lss_states = backend.transpose(v0(backend.convert_to_tensor(sub)), (1, 2, 0, 3, 4))
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
    gss_states = vv(lss_states)
    return backend.mean(gss_states, axis=(0, 1))








