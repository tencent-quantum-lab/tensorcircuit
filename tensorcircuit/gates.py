"""Declarations of single qubit and two-qubit gates."""

import sys
from copy import deepcopy
from typing import Optional
from functools import partial

import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group
import tensornetwork as tn

from .cons import npdtype

thismodule = sys.modules[__name__]

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
    mx = np.sin(alpha) * np.cos(phi)
    my = np.sin(alpha) * np.sin(phi)
    mz = np.cos(alpha)
    unitary = expm(-1j * theta * (mx * _x_matrix + my * _y_matrix * mz * _z_matrix))

    return Gate(unitary)


r = rgate


def random_single_qubit_gate(
    seed: Optional[int] = None, angle_scale: float = 1.0
) -> Gate:
    """
    Returns the random single qubit gate described in https://arxiv.org/abs/2002.07730.

    :param seed: Seed for random number generator.
    :param  angle_scale: Floating point value to scale angles by. Default 1.

    """
    if seed:
        np.random.seed(seed)

    # Get the random parameters
    theta, alpha, phi = np.random.rand(3) * 2 * np.pi
    mx = np.sin(alpha) * np.cos(phi)
    my = np.sin(alpha) * np.sin(phi)
    mz = np.cos(alpha)

    theta *= angle_scale

    # Get the unitary
    unitary = expm(-1j * theta * (mx * _x_matrix + my * _y_matrix * mz * _z_matrix))

    return Gate(unitary)


def random_two_qubit_gate() -> Gate:
    """
    Returns a random two-qubit gate.

    :seed: Seed for random number generator.
    """
    unitary = unitary_group.rvs(dim=4)
    unitary = np.reshape(unitary, newshape=(2, 2, 2, 2))
    return Gate(deepcopy(unitary), name="R2Q")
