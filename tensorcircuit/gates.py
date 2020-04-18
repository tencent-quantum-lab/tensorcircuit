"""
Declarations of single qubit and two-qubit gates.
"""

import sys
from copy import deepcopy
from typing import Optional, Any
from functools import partial

import numpy as np
from scipy.stats import unitary_group
import tensornetwork as tn

from .cons import npdtype, backend

thismodule = sys.modules[__name__]

Tensor = Any

# Common single qubit states as np.ndarray objects
zero_state = np.array([1.0, 0.0], dtype=npdtype)
one_state = np.array([0.0, 1.0], dtype=npdtype)
plus_state = 1.0 / np.sqrt(2) * (zero_state + one_state)
minus_state = 1.0 / np.sqrt(2) * (zero_state - one_state)

# Common single qubit gates as np.ndarray objects
_h_matrix = 1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]])
_i_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
_x_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
_y_matrix = np.array([[0.0, -1j], [1j, 0.0]])
_z_matrix = np.array([[1.0, 0.0], [0.0, -1.0]])

_cnot_matrix = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)
_cz_matrix = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
    ]
)
_swap_matrix = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


_toffoli_matrix = np.array(
    [
        [1.0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1.0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1.0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1.0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1.0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1.0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1.0],
        [0, 0, 0, 0, 0, 0, 1.0, 0],
    ]
)


class Gate(tn.Node):  # type: ignore
    """
    Wrapper of tn.Node, quantum gate
    """

    pass


def num_to_tensor(*num: float) -> Any:
    l = [backend.convert_to_tensor(np.array(n).astype(npdtype)) for n in num]
    if len(l) == 1:
        return l[0]
    return l


def array_to_tensor(*num: np.array) -> Any:
    l = [backend.convert_to_tensor(n.astype(npdtype)) for n in num]
    if len(l) == 1:
        return l[0]
    return l


def gate_wrapper(m: np.array, n: Optional[str] = None) -> Gate:
    if not n:
        n = "unknowngate"
    return Gate(deepcopy(m), name=n)


def meta_gate() -> None:
    for name in dir(thismodule):
        if name.endswith("_matrix") and name.startswith("_"):
            n = name[1:-7]
            m = getattr(thismodule, name)
            if m.shape[0] == 4:
                m = np.reshape(m, newshape=(2, 2, 2, 2))
            if m.shape[0] == 8:
                m = np.reshape(m, newshape=(2, 2, 2, 2, 2, 2))
            m = m.astype(npdtype)
            temp = partial(gate_wrapper, m, n)
            setattr(thismodule, n + "gate", temp)
            setattr(thismodule, n, temp)


meta_gate()


def rgate(theta: float = 0, alpha: float = 0, phi: float = 0) -> Gate:
    theta, phi, alpha = num_to_tensor(theta, phi, alpha)
    i, x, y, z = array_to_tensor(_i_matrix, _x_matrix, _y_matrix, _z_matrix)
    unitary = (
        backend.cos(theta) * i
        - backend.i() * backend.cos(phi) * backend.sin(alpha) * backend.sin(theta) * x
        - backend.i() * backend.sin(phi) * backend.sin(alpha) * backend.sin(theta) * y
        - backend.i() * backend.sin(theta) * backend.cos(alpha) * z
    )
    return Gate(unitary)


r = rgate


def rgate_theoretical(theta: float = 0, alpha: float = 0, phi: float = 0) -> Gate:
    theta, phi, alpha = num_to_tensor(theta, phi, alpha)
    mx = backend.sin(alpha) * backend.cos(phi)
    my = backend.sin(alpha) * backend.sin(phi)
    mz = backend.cos(alpha)
    x, y, z = array_to_tensor(_x_matrix, _y_matrix, _z_matrix)

    unitary = backend.expm(-backend.i() * theta * (mx * x + my * y + mz * z))
    return Gate(unitary)


def random_single_qubit_gate() -> Gate:
    """
    Returns the random single qubit gate described in https://arxiv.org/abs/2002.07730.
    """

    # Get the random parameters
    theta, alpha, phi = np.random.rand(3) * 2 * np.pi

    return rgate(theta, alpha, phi)


def random_two_qubit_gate() -> Gate:
    """
    Returns a random two-qubit gate.
    """
    unitary = unitary_group.rvs(dim=4)
    unitary = np.reshape(unitary, newshape=(2, 2, 2, 2))
    return Gate(deepcopy(unitary), name="R2Q")
