"""
some common noise quantum channels
"""

import numpy as np
from typing import Tuple, List, Callable, Union, Optional, Sequence, Any

from . import gates
from . import cons
from .backends import backend  # type: ignore

Gate = gates.Gate


def depolarizingchannel(px: float, py: float, pz: float) -> Sequence[Gate]:
    i = np.sqrt(1 - px - py - pz) * gates.i()  # type: ignore
    x = np.sqrt(px) * gates.x()  # type: ignore
    y = np.sqrt(py) * gates.y()  # type: ignore
    z = np.sqrt(pz) * gates.z()  # type: ignore
    return [i, x, y, z]


def single_qubit_kraus_identity_check(kraus: Sequence[Gate]) -> None:
    placeholder = backend.zeros([2, 2])
    placeholder = backend.cast(placeholder, dtype=cons.dtypestr)
    for k in kraus:
        placeholder += backend.conj(backend.transpose(k.tensor, [1, 0])) @ k.tensor
    np.testing.assert_allclose(placeholder, np.eye(2), atol=1e-5)
