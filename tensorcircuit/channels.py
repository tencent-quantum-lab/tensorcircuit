"""
Some common noise quantum channels.
"""

import sys
from typing import Any, Sequence, Optional, Union
from operator import and_
from functools import reduce

import numpy as np

from . import cons
from .cons import backend
from . import gates

thismodule = sys.modules[__name__]

Gate = gates.Gate
Tensor = Any


def _sqrt(a: Tensor) -> Tensor:
    r"""Return the square root of Tensor with default global dtype

    .. math::
        \sqrt{a}

    :param a: Tensor
    :type a: Tensor
    :return: Square root of Tensor
    :rtype: Tensor
    """
    return backend.cast(backend.sqrt(a), dtype=cons.dtypestr)


def depolarizingchannel(px: float, py: float, pz: float) -> Sequence[Gate]:
    r"""Return a Depolarizing Channel

    .. math::
        \sqrt{1-p_x-p_y-p_z}
        \begin{bmatrix}
            1 & 0\\
            0 & 1\\
        \end{bmatrix}\qquad
        \sqrt{p_x}
        \begin{bmatrix}
            0 & 1\\
            1 & 0\\
        \end{bmatrix}\qquad
        \sqrt{p_y}
        \begin{bmatrix}
            0 & -1j\\
            1j & 0\\
        \end{bmatrix}\qquad
        \sqrt{p_z}
        \begin{bmatrix}
            1 & 0\\
            0 & -1\\
        \end{bmatrix}

    :Example:

    >>> cs = depolarizingchannel(0.1, 0.15, 0.2)
    >>> tc.channels.single_qubit_kraus_identity_check(cs)

    :param px: :math:`p_x`
    :type px: float
    :param py: :math:`p_y`
    :type py: float
    :param pz: :math:`p_z`
    :type pz: float
    :return: Sequences of Gates
    :rtype: Sequence[Gate]
    """
    # assert px + py + pz <= 1
    i = Gate(_sqrt(1 - px - py - pz) * gates.i().tensor)  # type: ignore
    x = Gate(_sqrt(px) * gates.x().tensor)  # type: ignore
    y = Gate(_sqrt(py) * gates.y().tensor)  # type: ignore
    z = Gate(_sqrt(pz) * gates.z().tensor)  # type: ignore
    return [i, x, y, z]


def generaldepolarizingchannel(
    p: Union[float, Sequence[Any]], num_qubits: int = 1
) -> Sequence[Gate]:
    """Return a Depolarizing Channel

    .. math::
        \sqrt{1-p_x-p_y-p_z}
        \begin{bmatrix}
            1 & 0\\
            0 & 1\\
        \end{bmatrix}\qquad
        \sqrt{p_x}
        \begin{bmatrix}
            0 & 1\\
            1 & 0\\
        \end{bmatrix}\qquad
        \sqrt{p_y}
        \begin{bmatrix}
            0 & -1j\\
            1j & 0\\
        \end{bmatrix}\qquad
        \sqrt{p_z}
        \begin{bmatrix}
            1 & 0\\
            0 & -1\\
        \end{bmatrix}

    :Example:

    >>> cs=tc.channels.generaldepolarizingchannel([0.1,0.1,0.1],1)
    >>> tc.channels.kraus_identity_check(cs)
    >>> cs = tc.channels.generaldepolarizingchannel(0.02,2)
    >>> tc.channels.kraus_identity_check(cs)


    :param p: parameter for each Pauli channel 
    :type p: Union[float, Sequence]
    :param num_qubits: number of qubits, defaults to 1
    :type num_qubits: int, optional
    :return: Sequences of Gates
    :rtype: Sequence[Gate]
    """

    if num_qubits == 1:

        if isinstance(p, float):

            assert p > 0 and p < 1 / 3, "p should be >0 and <1/3"
            probs = [1 - 3 * p] + 3 * [p]

        elif isinstance(p, list):

            assert reduce(
                and_, [pi > 0 and pi < 1 for pi in p]
            ), "p should be >0 and <1"
            probs = [1 - sum(p)] + p  # type: ignore

        elif isinstance(p, tuple):

            p = list[p]  # type: ignore
            assert reduce(
                and_, [pi > 0 and pi < 1 for pi in p]
            ), "p should be >0 and <1"
            probs = [1 - sum(p)] + p  # type: ignore

        else:
            raise ValueError("p should be float or list")

    elif num_qubits == 2:

        if isinstance(p, float):

            assert p > 0 and p < 1, "p should be >0 and <1/15"
            probs = [1 - 15 * p] + 15 * [p]

        elif isinstance(p, list):

            assert reduce(
                and_, [pi > 0 and pi < 1 for pi in p]
            ), "p should be >0 and <1"
            probs = [1 - sum(p)] + p  # type: ignore

        elif isinstance(p, tuple):

            p = list[p]  # type: ignore
            assert reduce(
                and_, [pi > 0 and pi < 1 for pi in p]
            ), "p should be >0 and <1"
            probs = [1 - sum(p)] + p  # type: ignore

        else:
            raise ValueError("p should be float or list")

    if num_qubits == 1:
        tup = [gates.i().tensor, gates.x().tensor, gates.y().tensor, gates.z().tensor]  # type: ignore
    if num_qubits == 2:
        tup = [
            gates.ii().tensor,  # type: ignore
            gates.ix().tensor,  # type: ignore
            gates.iy().tensor,  # type: ignore
            gates.iz().tensor,  # type: ignore
            gates.xi().tensor,  # type: ignore
            gates.xx().tensor,  # type: ignore
            gates.xy().tensor,  # type: ignore
            gates.xz().tensor,  # type: ignore
            gates.yi().tensor,  # type: ignore
            gates.yx().tensor,  # type: ignore
            gates.yy().tensor,  # type: ignore
            gates.yz().tensor,  # type: ignore
            gates.zi().tensor,  # type: ignore
            gates.zx().tensor,  # type: ignore
            gates.zy().tensor,  # type: ignore
            gates.zz().tensor,  # type: ignore
        ]

    Gkarus = []
    for pro, paugate in zip(probs, tup):
        Gkarus.append(Gate(_sqrt(pro) * paugate))

    return Gkarus


def amplitudedampingchannel(gamma: float, p: float) -> Sequence[Gate]:
    r"""
    Return an amplitude damping channel.
    Notice: Amplitude damping corrspondings to p = 1.

    .. math::
        \sqrt{p}
        \begin{bmatrix}
            1 & 0\\
            0 & \sqrt{1-\gamma}\\
        \end{bmatrix}\qquad
        \sqrt{p}
        \begin{bmatrix}
            0 & \sqrt{\gamma}\\
            0 & 0\\
        \end{bmatrix}\qquad
        \sqrt{1-p}
        \begin{bmatrix}
            \sqrt{1-\gamma} & 0\\
            0 & 1\\
        \end{bmatrix}\qquad
        \sqrt{1-p}
        \begin{bmatrix}
            0 & 0\\
            \sqrt{\gamma} & 0\\
        \end{bmatrix}

    :Example:

    >>> cs = amplitudedampingchannel(0.25, 0.3)
    >>> tc.channels.single_qubit_kraus_identity_check(cs)

    :param gamma: the damping parameter of amplitude (:math:`\gamma`)
    :type gamma: float
    :param p: :math:`p`
    :type p: float
    :return: An amplitude damping channel with given :math:`\gamma` and :math:`p`
    :rtype: Sequence[Gate]
    """
    # https://cirq.readthedocs.io/en/stable/docs/noise.html
    # https://github.com/quantumlib/Cirq/blob/master/cirq/ops/common_channels.py

    g00 = Gate(np.array([[1, 0], [0, 0]], dtype=cons.npdtype))
    g01 = Gate(np.array([[0, 1], [0, 0]], dtype=cons.npdtype))
    g10 = Gate(np.array([[0, 0], [1, 0]], dtype=cons.npdtype))
    g11 = Gate(np.array([[0, 0], [0, 1]], dtype=cons.npdtype))
    m0 = _sqrt(p) * (g00 + _sqrt(1 - gamma) * g11)
    m1 = _sqrt(p) * (_sqrt(gamma) * g01)
    m2 = _sqrt(1 - p) * (_sqrt(1 - gamma) * g00 + g11)
    m3 = _sqrt(1 - p) * (_sqrt(gamma) * g10)
    return [m0, m1, m2, m3]


def resetchannel() -> Sequence[Gate]:
    r"""Reset channel

    .. math::
        \begin{bmatrix}
            1 & 0\\
            0 & 0\\
        \end{bmatrix}\qquad
        \begin{bmatrix}
            0 & 1\\
            0 & 0\\
        \end{bmatrix}

    :Example:

    >>> cs = resetchannel()
    >>> tc.channels.single_qubit_kraus_identity_check(cs)

    :return: Reset channel
    :rtype: Sequence[Gate]
    """
    m0 = Gate(np.array([[1, 0], [0, 0]], dtype=cons.npdtype))
    m1 = Gate(np.array([[0, 1], [0, 0]], dtype=cons.npdtype))
    return [m0, m1]


def phasedampingchannel(gamma: float) -> Sequence[Gate]:
    r"""Return a phase damping channel with given :math:`\gamma`

    .. math::
        \begin{bmatrix}
            1 & 0\\
            0 & \sqrt{1-\gamma}\\
        \end{bmatrix}\qquad
        \begin{bmatrix}
            0 & 0\\
            0 & \sqrt{\gamma}\\
        \end{bmatrix}

    :Example:

    >>> cs = phasedampingchannel(0.6)
    >>> tc.channels.single_qubit_kraus_identity_check(cs)

    :param gamma: The damping parameter of phase (:math:`\gamma`)
    :type gamma: float
    :return: A phase damping channel with given :math:`\gamma`
    :rtype: Sequence[Gate]
    """
    g00 = Gate(np.array([[1, 0], [0, 0]], dtype=cons.npdtype))
    g11 = Gate(np.array([[0, 0], [0, 1]], dtype=cons.npdtype))
    m0 = 1.0 * (g00 + _sqrt(1 - gamma) * g11)  # 1* ensure gate
    m1 = _sqrt(gamma) * g11
    return [m0, m1]


def kraus_identity_check(kraus: Sequence[Gate]) -> None:
    r"""Check identity of a single qubit Kraus operators.

    :Examples:

    >>> cs = resetchannel()
    >>> tc.channels.kraus_identity_check(cs)

    .. math::
        \sum_{k}^{} K_k^{\dagger} K_k = I

    :param kraus: List of Kraus operators.
    :type kraus: Sequence[Gate]
    """

    dim = backend.shape_tuple(kraus[0].tensor)
    shape = (len(dim), len(dim))
    placeholder = backend.zeros(shape)
    placeholder = backend.cast(placeholder, dtype=cons.dtypestr)
    for k in kraus:
        k = Gate(backend.reshape(k.tensor, [len(dim), len(dim)]))
        placeholder += backend.conj(backend.transpose(k.tensor, [1, 0])) @ k.tensor
    np.testing.assert_allclose(placeholder, np.eye(shape[0]), atol=1e-5)


single_qubit_kraus_identity_check = kraus_identity_check  # backward compatibility


def kraus_to_super_gate(kraus_list: Sequence[Gate]) -> Tensor:
    r"""Convert Kraus operators to one Tensor (as one Super Gate).

    .. math::
        \sum_{k}^{} K_k \otimes K_k^{\dagger}

    :param kraus_list: A sequence of Gate
    :type kraus_list: Sequence[Gate]
    :return: The corresponding Tensor of the list of Kraus operators
    :rtype: Tensor
    """
    kraus_tensor_list = [k.tensor for k in kraus_list]
    k = kraus_tensor_list[0]
    u = backend.kron(k, backend.conj(k))
    for k in kraus_tensor_list[1:]:
        u += backend.kron(k, backend.conj(k))
    return u


def _collect_channels() -> Sequence[str]:
    r"""Return channels names in this module.

    :Example:

    >>> tc.channels._collect_channels()
    ['amplitudedamping', 'depolarizing', 'phasedamping', 'reset']

    :return: A list of channel names
    :rtype: Sequence[str]
    """
    cs = []
    for name in dir(thismodule):
        if name.endswith("channel"):
            n = name[:-7]
            cs.append(n)
    return cs


channels = _collect_channels()
# channels = ["depolarizing", "amplitudedamping", "reset", "phasedamping"]
