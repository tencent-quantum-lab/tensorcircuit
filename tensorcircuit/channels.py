"""
some common noise quantum channels
"""
import sys
import numpy as np
from typing import Tuple, List, Callable, Union, Optional, Sequence, Any

from . import gates
from . import cons
from .backends import backend  # type: ignore

thismodule = sys.modules[__name__]

Gate = gates.Gate
Tensor = Any


def _sqrt(a: Tensor) -> Tensor:
    return backend.cast(backend.sqrt(a), dtype=cons.dtypestr)


def depolarizingchannel(px: float, py: float, pz: float) -> Sequence[Gate]:
    # assert px + py + pz <= 1
    i = Gate(_sqrt(1 - px - py - pz) * gates.i().tensor)  # type: ignore
    x = Gate(_sqrt(px) * gates.x().tensor)  # type: ignore
    y = Gate(_sqrt(py) * gates.y().tensor)  # type: ignore
    z = Gate(_sqrt(pz) * gates.z().tensor)  # type: ignore
    return [i, x, y, z]


def amplitudedampingchannel(gamma: float, p: float) -> Sequence[Gate]:
    # https://cirq.readthedocs.io/en/stable/docs/noise.html
    # https://github.com/quantumlib/Cirq/blob/master/cirq/ops/common_channels.py
    # amplitude damping corrspondings to p=1
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
    m0 = Gate(np.array([[1, 0], [0, 0]], dtype=cons.npdtype))
    m1 = Gate(np.array([[0, 1], [0, 0]], dtype=cons.npdtype))
    return [m0, m1]


def phasedampingchannel(gamma: float) -> Sequence[Gate]:
    g00 = Gate(np.array([[1, 0], [0, 0]], dtype=cons.npdtype))
    g11 = Gate(np.array([[0, 0], [0, 1]], dtype=cons.npdtype))
    m0 = g00 + _sqrt(1 - gamma) * g11
    m1 = _sqrt(gamma) * g11
    return [m0, m1]


def single_qubit_kraus_identity_check(kraus: Sequence[Gate]) -> None:
    placeholder = backend.zeros([2, 2])
    placeholder = backend.cast(placeholder, dtype=cons.dtypestr)
    for k in kraus:
        placeholder += backend.conj(backend.transpose(k.tensor, [1, 0])) @ k.tensor
    np.testing.assert_allclose(placeholder, np.eye(2), atol=1e-5)


def kraus_to_super_gate(kraus_list: Sequence[Gate]) -> Tensor:
    kraus_tensor_list = [k.tensor for k in kraus_list]
    k = kraus_tensor_list[0]
    u = backend.kron(k, backend.conj(k))
    for k in kraus_tensor_list[1:]:
        u += backend.kron(k, backend.conj(k))
    return u


def _collect_channels() -> Sequence[str]:
    cs = []
    for name in dir(thismodule):
        if name.endswith("channel"):
            n = name[:-7]
            cs.append(n)
    return cs


channels = _collect_channels()
# channels = ["depolarizing", "amplitudedamping", "reset", "phasedamping"]
