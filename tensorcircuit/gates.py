"""
declarations of single-qubit and two-qubit gates and their corresponding matrix
"""

import sys
from copy import deepcopy
from functools import reduce
from typing import Any, Callable, Optional, Union
from operator import mul

import numpy as np
import tensornetwork as tn
from scipy.stats import unitary_group

from .cons import backend, dtypestr, npdtype

thismodule = sys.modules[__name__]

Tensor = Any
Array = Any

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
_s_matrix = np.array([[1.0, 0.0], [0.0, 1j]])
_t_matrix = np.array([[1.0, 0.0], [0.0, np.exp(np.pi / 4 * 1j)]])
_wroot_matrix = (
    1
    / np.sqrt(2)
    * np.array([[1, -1 / np.sqrt(2) * (1 + 1.0j)], [1 / np.sqrt(2) * (1 - 1.0j), 1]])
)


_ii_matrix = np.kron(_i_matrix, _i_matrix)
_xx_matrix = np.kron(_x_matrix, _x_matrix)
_yy_matrix = np.kron(_y_matrix, _y_matrix)
_zz_matrix = np.kron(_z_matrix, _z_matrix)
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

_cy_matrix = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0j],
        [0.0, 0.0, 1.0j, 0.0],
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


def __rmul__(self: tn.Node, lvalue: Union[float, complex]) -> "Gate":
    newg = Gate(lvalue * self.tensor)
    return newg


tn.Node.__rmul__ = __rmul__


class Gate(tn.Node):  # type: ignore
    """
    Wrapper of tn.Node, quantum gate
    """

    pass


def num_to_tensor(*num: Union[float, Tensor], dtype: Optional[str] = None) -> Any:
    r"""
    Convert the inputs to Tensor with specified dtype

    :param *num: inputs
    :param dtype: dtype of the output Tensors
    :returns: List of Tensors
    :rtype: List[Tensor]
    
    Example::
        # TODO(@PeterYu): Add examples
    """
    
    l = []
    if not dtype:
        dtype = dtypestr
    for n in num:
        if not backend.is_tensor(n):
            l.append(backend.cast(backend.convert_to_tensor(n), dtype=dtype))
        else:
            l.append(backend.cast(n, dtype=dtype))
    if len(l) == 1:
        return l[0]
    return l


array_to_tensor = num_to_tensor


def gate_wrapper(m: Tensor, n: Optional[str] = None) -> Gate:
    if not n:
        n = "unknowngate"
    m = m.astype(npdtype)
    return Gate(deepcopy(m), name=n)


class GateF:
    def __init__(self, m: Tensor, n: Optional[str] = None):
        if not n:
            n = "unknowngate"
        self.m = m
        self.n = n

    def __call__(self, *args: Any, **kws: Any) -> Gate:
        m = self.m.astype(npdtype)
        return Gate(deepcopy(m), name=self.n)

    def adjoint(self, *args: Any, **kws: Any) -> Gate:
        m = self.__call__(*args, **kws)
        m.tensor = backend.adjoint(m.tensor)
        return m

    # TODO(@refraction-ray): controlled

    def __str__(self) -> str:
        return self.n

    __repr__ = __str__


class GateVF(GateF):
    def __init__(self, f: Callable[..., Gate], n: Optional[str] = None):
        if not n:
            n = "unknowngate"
        self.f = f
        self.n = n

    def __call__(self, *args: Any, **kws: Any) -> Gate:
        return self.f(*args, **kws)


def meta_gate() -> None:
    """
    Inner helper function to generate gate functions, such as ``z()`` from ``_z_matrix``
    """
    for name in dir(thismodule):
        if name.endswith("_matrix") and name.startswith("_"):
            n = name[1:-7]
            m = getattr(thismodule, name)
            if m.shape[0] == 4:
                m = np.reshape(m, newshape=(2, 2, 2, 2))
            if m.shape[0] == 8:
                m = np.reshape(m, newshape=(2, 2, 2, 2, 2, 2))
            m = m.astype(npdtype)
            # not enough for new mechanism: register method on class instead of instance
            # temp = partial(gate_wrapper, m, n)
            # temp.__name__ = n
            temp = GateF(m, n)
            setattr(thismodule, n + "gate", temp)
            setattr(thismodule, n, temp)


meta_gate()

pauli_gates = [i(), x(), y(), z()]  # type: ignore


def matrix_for_gate(gate: Gate) -> Tensor:
    """
    Convert Gate to Tensor
    
    :param gate: input Gate
    :type gate: Gate
    :return: corresponding Tensor
    :rtype: Tensor
    """
    
    t = gate.tensor
    l = int(np.sqrt(t.size))
    t = t.reshape([l, l])
    return t


def bmatrix(a: Array) -> str:
    """
    Returns a LaTeX bmatrix

    :param a: 2D numpy array
    :type a: np.array
    :raises ValueError: [description]
    :return: latex str for bmatrix of array a
    :rtype: str
    """
    #   Adopted from https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix/17131750

    if len(a.shape) > 2:
        raise ValueError("bmatrix can at most display two dimensions")
    lines = str(a).replace("[", "").replace("]", "").splitlines()
    rv = [r"\begin{bmatrix}"]
    rv += ["    " + " & ".join(l.split()) + r"\\" for l in lines]
    rv[-1] = rv[-1][:-2]
    rv += [r" \end{bmatrix}"]
    return "".join(rv)


def r_gate(theta: float = 0, alpha: float = 0, phi: float = 0) -> Gate:
    r"""
    .. math::
        R(\theta, \phi, \alpha) = i\cos(\theta)
    .. math::
        - i \cos(\phi) \sin(\alpha) \sin(\theta) X
    .. math::
        - i \sin(\phi) \sin(\alpha) \sin(\theta) Y
    .. math::
        - i \sin(\theta) \cos(\alpha) Z
        
    :param theta:  angle in radians
    :type theta: float, optional
    :param alpha: angle in radians
    :type alpha: float, optional
    :param phi: angle in radians
    :type phi: float, optional
    
    :return: R Gate
    :rtype: Gate
    """
    theta, phi, alpha = num_to_tensor(theta, phi, alpha)
    i, x, y, z = array_to_tensor(_i_matrix, _x_matrix, _y_matrix, _z_matrix)
    unitary = (
        backend.cos(theta) * i
        - backend.i() * backend.cos(phi) * backend.sin(alpha) * backend.sin(theta) * x
        - backend.i() * backend.sin(phi) * backend.sin(alpha) * backend.sin(theta) * y
        - backend.i() * backend.sin(theta) * backend.cos(alpha) * z
    )
    return Gate(unitary)


# r = r_gate


def rx_gate(theta: float = 0) -> Gate:
    r"""
    .. math::
        RX(\theta) = e^{-i\frac{\theta}{2}X}
        
    :param theta: angle in radians
    :type theta: float, optional
    :return: RX gate
    :rtype: Gate
    """
    i, x = array_to_tensor(_i_matrix, _x_matrix)
    theta = num_to_tensor(theta)
    unitary = backend.cos(theta / 2.0) * i - backend.i() * backend.sin(theta / 2.0) * x
    return Gate(unitary)


# rx = rx_gate


def ry_gate(theta: float = 0) -> Gate:
    r"""
    .. math::
        RY(\theta) = e^{-i\frac{\theta}{2}Y}
        
    :param theta: angle in radians
    :type theta: float, optional
    :return: RY gate
    :rtype: Gate
    """
    i, y = array_to_tensor(_i_matrix, _y_matrix)
    theta = num_to_tensor(theta)
    unitary = backend.cos(theta / 2.0) * i - backend.i() * backend.sin(theta / 2.0) * y
    return Gate(unitary)


# ry = ry_gate


def rz_gate(theta: float = 0) -> Gate:
    r"""
    .. math::
        RX(\theta) = e^{-i\frac{\theta}{2}Z}
        
    :param theta: angle in radians
    :type theta: float, optional
    :return: RZ Gate
    :rtype: Gate
    """
    i, z = array_to_tensor(_i_matrix, _z_matrix)
    theta = num_to_tensor(theta)
    unitary = backend.cos(theta / 2.0) * i - backend.i() * backend.sin(theta / 2.0) * z
    return Gate(unitary)


# rz = rz_gate


def rgate_theoretical(theta: float = 0, alpha: float = 0, phi: float = 0) -> Gate:
    r"""
    .. math::
        mx = \sin(\alpha) \cos(\phi) X
    .. math::
        my = \sin(\alpha) \sin(\phi) Y
    .. math::
        mz = \cos(\alpha) Z
    .. math::
        R(\theta, \alpha, \phi) = e^{-i\theta (mx+my+mz)}
    :param theta: angle in radians
    :type theta: float, optional
    :param alpha: angle in radians
    :type alpha: float, optional
    :param phi: angle in radians
    :type phi: float, optional
    :return: Rotation Gate
    :rtype: Gate
    """
    theta, phi, alpha = num_to_tensor(theta, phi, alpha)
    mx = backend.sin(alpha) * backend.cos(phi)
    my = backend.sin(alpha) * backend.sin(phi)
    mz = backend.cos(alpha)
    x, y, z = array_to_tensor(_x_matrix, _y_matrix, _z_matrix)

    unitary = backend.expm(-backend.i() * theta * (mx * x + my * y + mz * z))
    return Gate(unitary)


def random_single_qubit_gate() -> Gate:
    """
    Random single qubit gate described in https://arxiv.org/abs/2002.07730.
    
    :return: A random single qubit gate
    :rtype: Gate
    """
    # Get the random parameters
    theta, alpha, phi = np.random.rand(3) * 2 * np.pi  # type: ignore

    return r_gate(theta, alpha, phi)  # type: ignore


# rs = random_single_qubit_gate  # deprecated


def iswap_gate(theta: float = 1.0) -> Gate:
    r"""
    
    .. math::
        d_1 =
        \begin{pmatrix}
            1 & 0 & 0 & 1\\
            0 & 0 & 0 & 0\\
            0 & 0 & 0 & 0\\
            0 & 0 & 0 & 0\\
        \end{pmatrix}
        d_2 =
        \begin{pmatrix}
            0 & 0 & 0 & 0\\
            0 & 1 & 0 & 0\\
            0 & 0 & 1 & 0\\
            0 & 0 & 0 & 0\\
        \end{pmatrix}
        od =
        \begin{pmatrix}
            0 & 0 & 0 & 0\\
            0 & 0 & 1 & 0\\
            0 & 1 & 0 & 0\\
            0 & 0 & 0 & 0\\
        \end{pmatrix}
          
    .. math::
        iSwap(\theta) = d_1 + \cos(\frac{\pi}{2} \theta ) d_2 + j \sin(\frac{\pi}{2} \theta ) od
    
    # TODO(@YHPeter): need review
    
    :param theta: angle in radians
    :type theta: float
    :return: iSwap Gate
    :rtype: Gate
    """
    d1 = np.array([[1.0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1.0]])
    d2 = np.array([[0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 0]])
    od = np.array([[0, 0, 0, 0], [0, 0, 1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 0]])
    d1, d2, od = array_to_tensor(d1, d2, od)
    theta = num_to_tensor(theta)
    unitary = (
        d1
        + backend.cos(theta * np.pi / 2) * d2
        + 1.0j * backend.sin(theta * np.pi / 2) * od
    )
    unitary = backend.reshape(unitary, [2, 2, 2, 2])
    return Gate(unitary)


# iswap = iswap_gate


def cr_gate(theta: float = 0, alpha: float = 0, phi: float = 0) -> Gate:
    r"""  
    .. math::
        CR(\theta, \phi, \alpha) = j + i\cos(\theta)
    .. math::
        - i \cos(\phi) \sin(\alpha) sin(\theta) X
    .. math::
        - i \sin(\phi) \sin(\alpha) sin(\theta) Y
    .. math::
        - i \sin(\theta) \cos(\alpha) Z
        
    :param theta:  angle in radians
    :type theta: float, optional
    :param alpha: angle in radians
    :type alpha: float, optional
    :param phi: angle in radians
    :type phi: float, optional
    
    :return: CR Gate
    :rtype: Gate
    """
    theta, phi, alpha = num_to_tensor(theta, phi, alpha)
    u = np.array([[1.0, 0.0], [0.0, 0.0]])
    d = np.array([[0.0, 0.0], [0.0, 1.0]])
    j = np.kron(u, _i_matrix)
    i = np.kron(d, _i_matrix)
    x = np.kron(d, _x_matrix)
    y = np.kron(d, _y_matrix)
    z = np.kron(d, _z_matrix)

    j, i, x, y, z = array_to_tensor(j, i, x, y, z)
    unitary = (
        j
        + backend.cos(theta) * i
        - backend.i() * backend.cos(phi) * backend.sin(alpha) * backend.sin(theta) * x
        - backend.i() * backend.sin(phi) * backend.sin(alpha) * backend.sin(theta) * y
        - backend.i() * backend.sin(theta) * backend.cos(alpha) * z
    )
    unitary = backend.reshape(unitary, [2, 2, 2, 2])
    return Gate(unitary)


# cr = cr_gate


def random_two_qubit_gate() -> Gate:
    """
    Returns a random two-qubit gate.
    :return: a random two-qubit gate
    :rtype: Gate
    """
    unitary = unitary_group.rvs(dim=4).astype(
        npdtype
    )  # the default is np.complex128 without astype
    unitary = np.reshape(unitary, newshape=(2, 2, 2, 2))
    return Gate(deepcopy(unitary), name="R2Q")


def any_gate(unitary: Tensor, name: str = "any") -> Gate:
    """
    Note one should provide the gate with properly reshaped

    :param unitary: corresponding gate
    """
    # deepcopy roadblocks tf.function, pls take care of the unitary outside
    unitary = backend.reshape2(unitary)
    # nleg = int(np.log2(backend.sizen(unitary)))
    # unitary = backend.reshape(unitary, [2 for _ in range(nleg)])
    return Gate(unitary, name=name)


# any = any_gate


def exponential_gate(unitary: Tensor, theta: float, name: str = "none") -> Gate:
    r"""
    .. math::
        exp(U) = e^{-i \theta U}
    # TODO(@YHPeter): need review
    :param unitary: input unitary
    :type unitary: Tensor
    :param theta: angle in radians
    :type theta: float
    :param name: suffix of Gate name
    """
    theta = num_to_tensor(theta)
    mat = backend.expm(-backend.i() * theta * unitary)
    dimension = reduce(mul, mat.shape)
    nolegs = int(np.log(dimension) / np.log(2))
    mat = backend.reshape(mat, shape=[2 for _ in range(nolegs)])
    return Gate(mat, name="exp-" + name)


exp_gate = exponential_gate
# exp = exponential_gate


def exponential_gate_unity(unitary: Tensor, theta: float, name: str = "none") -> Gate:
    theta, unitary = num_to_tensor(theta, unitary)
    # n = len(unitary.shape)
    size = int(reduce(mul, unitary.shape))
    n = int(np.log2(size))
    i = np.eye(2 ** (int(n / 2)))
    i = i.reshape([2 for _ in range(n)])
    unitary = backend.reshape(unitary, [2 for _ in range(n)])
    it = array_to_tensor(i)
    mat = backend.cos(theta) * it - 1.0j * backend.sin(theta) * unitary
    return Gate(mat, name="exp1-" + name)


exp1_gate = exponential_gate_unity
# exp1 = exponential_gate_unity


def meta_vgate() -> None:
    for f in ["r", "rx", "ry", "rz", "iswap", "any", "exp", "exp1", "cr"]:
        setattr(thismodule, f, GateVF(getattr(thismodule, f + "_gate"), f))


meta_vgate()
