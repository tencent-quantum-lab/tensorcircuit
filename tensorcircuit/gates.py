"""
Declarations of single-qubit and two-qubit gates and their corresponding matrix.
"""

import sys
from copy import deepcopy
from functools import reduce, partial
from typing import Any, Callable, Optional, Sequence, List, Union, Tuple
from operator import mul

import numpy as np
import tensornetwork as tn
from scipy.stats import unitary_group

from .cons import backend, dtypestr, npdtype
from .backends import get_backend  # type: ignore
from .utils import arg_alias

thismodule = sys.modules[__name__]

Tensor = Any
Array = Any
Operator = Any  # QuOperator

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

_ix_matrix = np.kron(_i_matrix, _x_matrix)
_iy_matrix = np.kron(_i_matrix, _y_matrix)
_iz_matrix = np.kron(_i_matrix, _z_matrix)
_xi_matrix = np.kron(_x_matrix, _i_matrix)
_yi_matrix = np.kron(_y_matrix, _i_matrix)
_zi_matrix = np.kron(_z_matrix, _i_matrix)

_xy_matrix = np.kron(_x_matrix, _y_matrix)
_xz_matrix = np.kron(_x_matrix, _z_matrix)
_yx_matrix = np.kron(_y_matrix, _x_matrix)
_yz_matrix = np.kron(_y_matrix, _z_matrix)
_zx_matrix = np.kron(_z_matrix, _x_matrix)
_zy_matrix = np.kron(_z_matrix, _y_matrix)


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

_fredkin_matrix = np.array(
    [
        [1.0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1.0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1.0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1.0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1.0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1.0, 0],
        [0, 0, 0, 0, 0, 1.0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1.0],
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

    def __repr__(self) -> str:
        """Formatted output of Gate

        :Example:

        >>> tc.gates.ry(0.5)
        >>> # OR
        >>> print(repr(tc.gates.ry(0.5)))
        Gate(
            name: '__unnamed_node__',
            tensor:
                <tf.Tensor: shape=(2, 2), dtype=complex64, numpy=
                array([[ 0.9689124 +0.j, -0.24740396+0.j],
                    [ 0.24740396+0.j,  0.9689124 +0.j]], dtype=complex64)>,
            edges: [
                Edge(Dangling Edge)[0],
                Edge(Dangling Edge)[1]
            ])
        """
        sp = " " * 4
        edges = self.get_all_edges()
        edges_text = [edge.__repr__().replace("\n", "").strip() for edge in edges]
        edges_out = f"[" + f"\n{sp*2}" + f",\n{sp*2}".join(edges_text) + f"\n{sp}]"
        tensor_out = f"\n{sp*2}" + self.tensor.__repr__().replace("\n", f"\n{sp*2}")
        return (
            f"{self.__class__.__name__}(\n"
            f"{sp}name: {self.name!r},\n"
            f"{sp}tensor:{tensor_out},\n"
            f"{sp}edges: {edges_out})"
        )

    def copy(self, conjugate: bool = False) -> "Gate":
        result = super().copy(conjugate=conjugate)
        result.__class__ = Gate
        return result  # type: ignore


def num_to_tensor(*num: Union[float, Tensor], dtype: Optional[str] = None) -> Any:
    r"""
    Convert the inputs to Tensor with specified dtype.

    :Example:

    >>> from tensorcircuit.gates import num_to_tensor
    >>> # OR
    >>> from tensorcircuit.gates import array_to_tensor
    >>>
    >>> x, y, z = 0, 0.1, np.array([1])
    >>>
    >>> tc.set_backend('numpy')
    numpy_backend
    >>> num_to_tensor(x, y, z)
    [array(0.+0.j, dtype=complex64), array(0.1+0.j, dtype=complex64), array([1.+0.j], dtype=complex64)]
    >>>
    >>> tc.set_backend('tensorflow')
    tensorflow_backend
    >>> num_to_tensor(x, y, z)
    [<tf.Tensor: shape=(), dtype=complex64, numpy=0j>,
     <tf.Tensor: shape=(), dtype=complex64, numpy=(0.1+0j)>,
     <tf.Tensor: shape=(1,), dtype=complex64, numpy=array([1.+0.j], dtype=complex64)>]
    >>>
    >>> tc.set_backend('pytorch')
    pytorch_backend
    >>> num_to_tensor(x, y, z)
    [tensor(0.+0.j), tensor(0.1000+0.j), tensor([1.+0.j])]
    >>>
    >>> tc.set_backend('jax')
    jax_backend
    >>> num_to_tensor(x, y, z)
    [DeviceArray(0.+0.j, dtype=complex64),
     DeviceArray(0.1+0.j, dtype=complex64),
     DeviceArray([1.+0.j], dtype=complex64)]

    :param num: inputs
    :type num: Union[float, Tensor]
    :param dtype: dtype of the output Tensors
    :type dtype: str, optional
    :return: List of Tensors
    :rtype: List[Tensor]
    """
    # TODO(@YHPeter): fix __doc__ for same function with different names

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
    def __init__(
        self, m: Tensor, n: Optional[str] = None, ctrl: Optional[List[int]] = None
    ):
        if not n:
            n = "unknowngate"
        self.m = m
        self.n = n
        self.ctrl = ctrl

    def __call__(self, *args: Any, **kws: Any) -> Gate:
        # m = array_to_tensor(self.m)
        # m = backend.cast(m, dtypestr)
        m = self.m.astype(npdtype)
        return Gate(deepcopy(m), name=self.n)

    def adjoint(self) -> "GateF":
        m = self.__call__()
        npb = get_backend("numpy")
        shape0 = npb.shape_tuple(m.tensor)
        m0 = npb.reshapem(m.tensor)
        ma = npb.adjoint(m0)
        if np.allclose(m0, ma, atol=1e-5):
            name = self.n
        else:
            name = self.n + "d"
        ma = npb.reshape(ma, shape0)
        return GateF(ma, name, self.ctrl)

    # TODO(@refraction-ray): adjoint gate convention finally determined
    def ided(self, before: bool = True) -> "GateF":
        def f(*args: Any, **kws: Any) -> Any:
            m = self.__call__(*args, **kws)
            u = m.tensor
            u = backend.reshapem(u)
            if before:
                iu = backend.kron(backend.eye(2), u)
                iu = backend.reshape2(iu)
                return Gate(iu, name="ip" + self.n)
            else:  # before=False
                iu = backend.kron(u, backend.eye(2))
                iu = backend.reshape2(iu)
                return Gate(iu, name="ia" + self.n)

        if before:
            name_prefix = "ip"
        else:
            name_prefix = "ia"
        return GateVF(f, name_prefix + self.n)

    def controlled(self) -> "GateF":
        def f(*args: Any, **kws: Any) -> Any:
            m = self.__call__(*args, **kws)
            u = m.tensor
            u = backend.reshapem(u)
            s = int(u.shape[-1])
            upper = backend.concat([backend.eye(s), backend.zeros([s, s])])
            lower = backend.concat([backend.zeros([s, s]), u])
            cu = backend.concat([upper, lower], axis=-1)
            cu = backend.reshape2(cu)

            return Gate(cu, name="c" + self.n)

        if not self.ctrl:
            ctrl = [1]
        else:
            ctrl = [1] + self.ctrl
        return GateVF(f, "c" + self.n, ctrl)

    def ocontrolled(self) -> "GateF":
        def f(*args: Any, **kws: Any) -> Any:
            m = self.__call__(*args, **kws)
            u = m.tensor
            u = backend.reshapem(u)
            s = int(u.shape[-1])
            lower = backend.concat([backend.zeros([s, s]), backend.eye(s)])
            upper = backend.concat([u, backend.zeros([s, s])])
            ocu = backend.concat([upper, lower], axis=-1)
            ocu = backend.reshape2(ocu)

            # TODO(@refraction-ray): ctrl convention to be finally determined
            return Gate(ocu, name="o" + self.n)

        if not self.ctrl:
            ctrl = [0]
        else:
            ctrl = [0] + self.ctrl
        return GateVF(f, "o" + self.n, ctrl)

    def __str__(self) -> str:
        return self.n

    __repr__ = __str__


class GateVF(GateF):
    def __init__(
        self,
        f: Callable[..., Gate],
        n: Optional[str] = None,
        ctrl: Optional[List[int]] = None,
    ):
        if not n:
            n = "unknowngate"
        self.f = f
        self.n = n
        self.ctrl = ctrl

    def __call__(self, *args: Any, **kws: Any) -> Gate:
        return self.f(*args, **kws)

    def adjoint(self) -> "GateVF":
        def f(*args: Any, **kws: Any) -> Gate:
            m = self.__call__(*args, **kws)
            npb = get_backend("numpy")
            shape0 = npb.shape_tuple(m.tensor)
            m0 = npb.reshapem(m.tensor)
            ma = npb.adjoint(m0)
            if np.allclose(m0, ma, atol=1e-5):
                name = self.n
            else:
                name = self.n + "d"
            ma = npb.reshape(ma, shape0)
            return Gate(ma, name)

        return GateVF(f, self.n + "d", self.ctrl)


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
            # m = m.astype(npdtype)
            # not enough for new mechanism: register method on class instead of instance
            # temp = partial(gate_wrapper, m, n)
            # temp.__name__ = n
            temp = GateF(m, n)
            setattr(thismodule, n + "gate", temp)
            setattr(thismodule, n + "_gate", temp)
            setattr(thismodule, n, temp)


meta_gate()

pauli_gates = [i(), x(), y(), z()]  # type: ignore


def matrix_for_gate(gate: Gate, tol: float = 1e-6) -> Tensor:
    r"""
    Convert Gate to numpy array.

    :Example:

    >>> gate = tc.gates.r_gate()
    >>> tc.gates.matrix_for_gate(gate)
        array([[1.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j]], dtype=complex64)

    :param gate: input Gate
    :type gate: Gate
    :return: Corresponding Tensor
    :rtype: Tensor
    """

    t = gate.tensor
    t = backend.reshapem(t)
    t = backend.numpy(t)
    t.real[abs(t.real) < tol] = 0.0
    t.imag[abs(t.imag) < tol] = 0.0
    return t


def bmatrix(a: Array) -> str:
    r"""
    Returns a :math:`\LaTeX` bmatrix.

    :Example:

    >>> gate = tc.gates.r_gate()
    >>> array = tc.gates.matrix_for_gate(gate)
    >>> array
    array([[1.+0.j, 0.+0.j],
        [0.+0.j, 1.+0.j]], dtype=complex64)
    >>> print(tc.gates.bmatrix(array))
    \begin{bmatrix}    1.+0.j & 0.+0.j\\    0.+0.j & 1.+0.j \end{bmatrix}

    Formatted Display:

    .. math::
        \begin{bmatrix}    1.+0.j & 0.+0.j\\    0.+0.j & 1.+0.j \end{bmatrix}

    :param a: 2D numpy array
    :type a: np.array
    :raises ValueError: ValueError("bmatrix can at most display two dimensions")
    :return: :math:`\LaTeX`-formatted string for bmatrix of the array a
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


def phase_gate(theta: float = 0) -> Gate:
    r"""
    The phase gate

    .. math::
        \textrm{phase}(\theta) =
        \begin{pmatrix}
            1 & 0 \\
            0 & e^{i\theta} \\
        \end{pmatrix}

    :param theta: angle in radians, defaults to 0
    :type theta: float, optional
    :return: phase gate
    :rtype: Gate
    """
    i00, i11 = array_to_tensor(np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]))
    unitary = i00 + backend.exp(1.0j * theta) * i11
    return Gate(unitary)


def get_u_parameter(m: Tensor) -> Tuple[float, float, float]:
    """
    From the single qubit unitary to infer three angles of IBMUgate,

    :param m: numpy array, no backend agnostic version for now
    :type m: Tensor
    :return: theta, phi, lbd
    :rtype: Tuple[Tensor, Tensor, Tensor]
    """
    # ref:
    # https://github.com/Qiskit/qiskit-terra/blob/6125f5cbbf322268f53328f23c0be348c4fe0771/qiskit/quantum_info/synthesis/two_qubit_decompose.py#L44
    phase = np.linalg.det(m) ** (-1 / 2)
    U = phase * m  # U in SU(2)

    theta = 2 * np.arccos(np.abs(U[1, 1]))

    # Find phi and lambda
    lbdpphi = 2 * np.angle(U[1, 1])
    lbdmphi = -2 * np.angle(U[1, 0])
    lbd = (lbdpphi + lbdmphi) / 2
    phi = (lbdpphi - lbdmphi) / 2
    return theta, phi, lbd


def u_gate(theta: float = 0, phi: float = 0, lbd: float = 0) -> Gate:
    r"""
    IBMQ U gate following the converntion of OpenQASM3.0.
    See `OpenQASM doc <https://openqasm.com/language/gates.html#built-in-gates>`_

    .. math::

        \begin{split}U(\theta,\phi,\lambda) := \left(\begin{array}{cc}
        \cos(\theta/2) & -e^{i\lambda}\sin(\theta/2) \\
        e^{i\phi}\sin(\theta/2) & e^{i(\phi+\lambda)}\cos(\theta/2) \end{array}\right).\end{split}

    :param theta: _description_, defaults to 0
    :type theta: float, optional
    :param phi: _description_, defaults to 0
    :type phi: float, optional
    :param lbd: _description_, defaults to 0
    :type lbd: float, optional
    :return: _description_
    :rtype: Gate
    """
    i00, i01, i10, i11 = array_to_tensor(
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, 0]]),
        np.array([[0, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
    )
    unitary = (
        backend.cos(theta / 2) * i00
        - backend.exp(1.0j * lbd) * backend.sin(theta / 2) * i01
        + backend.exp(1.0j * phi) * backend.sin(theta / 2) * i10
        + backend.exp(1.0j * (phi + lbd)) * backend.cos(theta / 2) * i11
    )
    return Gate(unitary)


def r_gate(theta: float = 0, alpha: float = 0, phi: float = 0) -> Gate:
    r"""
    General single qubit rotation gate

    .. math::
        R(\theta, \alpha, \phi) = j \cos(\theta) I
        - j \cos(\phi) \sin(\alpha) \sin(\theta) X
        - j \sin(\phi) \sin(\alpha) \sin(\theta) Y
        - j \sin(\theta) \cos(\alpha) Z

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
    Rotation gate along :math:`x` axis.

    .. math::
        RX(\theta) = e^{-j\frac{\theta}{2}X}

    :param theta: angle in radians
    :type theta: float, optional
    :return: RX Gate
    :rtype: Gate
    """
    i, x = array_to_tensor(_i_matrix, _x_matrix)
    theta = num_to_tensor(theta)
    unitary = backend.cos(theta / 2.0) * i - backend.i() * backend.sin(theta / 2.0) * x
    return Gate(unitary)


# rx = rx_gate


def ry_gate(theta: float = 0) -> Gate:
    r"""
    Rotation gate along :math:`y` axis.

    .. math::
        RY(\theta) = e^{-j\frac{\theta}{2}Y}

    :param theta: angle in radians
    :type theta: float, optional
    :return: RY Gate
    :rtype: Gate
    """
    i, y = array_to_tensor(_i_matrix, _y_matrix)
    theta = num_to_tensor(theta)
    unitary = backend.cos(theta / 2.0) * i - backend.i() * backend.sin(theta / 2.0) * y
    return Gate(unitary)


# ry = ry_gate


def rz_gate(theta: float = 0) -> Gate:
    r"""
    Rotation gate along :math:`z` axis.

    .. math::
        RZ(\theta) = e^{-j\frac{\theta}{2}Z}

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
    Rotation gate implemented by matrix exponential. The output is the same as `rgate`.

    .. math::
        R(\theta, \alpha, \phi) = e^{-j \theta \left[\sin(\alpha) \cos(\phi) X
                                                   + \sin(\alpha) \sin(\phi) Y
                                                   + \cos(\alpha) Z\right]}

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

    :return: A random single-qubit gate
    :rtype: Gate
    """
    # Get the random parameters
    theta, alpha, phi = np.random.rand(3) * 2 * np.pi  # type: ignore

    return r_gate(theta, alpha, phi)  # type: ignore


# rs = random_single_qubit_gate  # deprecated


def iswap_gate(theta: float = 1.0) -> Gate:
    r"""
    iSwap gate.

    .. math::
        \textrm{iSwap}(\theta) =
        \begin{pmatrix}
            1 & 0 & 0 & 0\\
            0 & \cos(\frac{\pi}{2} \theta ) & j \sin(\frac{\pi}{2} \theta ) & 0\\
            0 & j \sin(\frac{\pi}{2} \theta ) & \cos(\frac{\pi}{2} \theta ) & 0\\
            0 & 0 & 0 & 1\\
        \end{pmatrix}

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
    Controlled rotation gate. When the control qubit is 1, `rgate` is applied to the target qubit.

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

    :return: A random two-qubit gate
    :rtype: Gate
    """
    unitary = unitary_group.rvs(dim=4).astype(
        npdtype
    )  # the default is np.complex128 without astype
    unitary = np.reshape(unitary, newshape=(2, 2, 2, 2))
    return Gate(deepcopy(unitary), name="R2Q")


def any_gate(unitary: Tensor, name: str = "any") -> Gate:
    """
    Note one should provide the gate with properly reshaped.

    :param unitary: corresponding gate
    :type unitary: Tensor
    :param name: The name of the gate.
    :type name: str
    :return: the resulted gate
    :rtype: Gate
    """
    # deepcopy roadblocks tf.function, pls take care of the unitary outside
    if isinstance(unitary, Gate):
        unitary.tensor = backend.cast(unitary.tensor, dtypestr)
        return unitary
    unitary = backend.cast(unitary, dtypestr)
    unitary = backend.reshape2(unitary)
    # nleg = int(np.log2(backend.sizen(unitary)))
    # unitary = backend.reshape(unitary, [2 for _ in range(nleg)])
    return Gate(unitary, name=name)


# any = any_gate


@partial(arg_alias, alias_dict={"unitary": ["hermitian", "hamiltonian"]})
def exponential_gate(unitary: Tensor, theta: float, name: str = "none") -> Gate:
    r"""
    Exponential gate.

    .. math::
        \textrm{exp}(U) = e^{-j \theta U}

    :param unitary: input unitary :math:`U`
    :type unitary: Tensor
    :param theta: angle in radians
    :type theta: float
    :param name: suffix of Gate name
    :return: Exponential Gate
    :rtype: Gate
    """
    theta = num_to_tensor(theta)
    mat = backend.expm(-backend.i() * theta * unitary)
    dimension = reduce(mul, mat.shape)
    nolegs = int(np.log(dimension) / np.log(2))
    mat = backend.reshape(mat, shape=[2 for _ in range(nolegs)])
    return Gate(mat, name="exp-" + name)


exp_gate = exponential_gate
# exp = exponential_gate


@partial(arg_alias, alias_dict={"unitary": ["hermitian", "hamiltonian"]})
def exponential_gate_unity(
    unitary: Tensor, theta: float, half: bool = False, name: str = "none"
) -> Gate:
    r"""
    Faster exponential gate directly implemented based on RHS. Only works when :math:`U^2 = I` is an identity matrix.

    .. math::
        \textrm{exp}(U) &= e^{-j \theta U} \\
                &= \cos(\theta) I - j \sin(\theta) U \\

    :param unitary: input unitary :math:`U`
    :type unitary: Tensor
    :param theta: angle in radians
    :type theta: float
    :param half: if True, the angel theta is mutiplied by 1/2,
        defaults to False
    :type half: bool
    :param name: suffix of Gate name
    :type name: str, optional
    :return: Exponential Gate
    :rtype: Gate
    """
    theta, unitary = num_to_tensor(theta, unitary)
    size = int(reduce(mul, unitary.shape))
    n = int(np.log2(size))
    i = np.eye(2 ** (int(n / 2)))
    i = i.reshape([2 for _ in range(n)])
    unitary = backend.reshape(unitary, [2 for _ in range(n)])
    it = array_to_tensor(i)
    if half is True:
        theta = theta / 2.0
    mat = backend.cos(theta) * it - 1.0j * backend.sin(theta) * unitary
    return Gate(mat, name="exp1-" + name)


exp1_gate = exponential_gate_unity
# exp1 = exponential_gate_unity
rzz_gate = partial(exp1_gate, unitary=_zz_matrix, half=True)
rxx_gate = partial(exp1_gate, unitary=_xx_matrix, half=True)
ryy_gate = partial(exp1_gate, unitary=_yy_matrix, half=True)


def multicontrol_gate(unitary: Tensor, ctrl: Union[int, Sequence[int]] = 1) -> Operator:
    r"""
    Multicontrol gate. If the control qubits equal to ``ctrl``, :math:`U` is applied to the target qubits.

    E.g., ``multicontrol_gate(tc.gates._zz_matrix, [1, 0, 1])`` returns a gate of 5 qubits,
        where the last 2 qubits are applied :math:`ZZ` gate,
        if the first 3 qubits are :math:`\ket{101}`.

    :param unitary: input unitary :math:`U`
    :type unitary: Tensor
    :param ctrl: control bit sequence
    :type ctrl: Union[int, Sequence[int]]
    :return: Multicontrol Gate
    :rtype: Operator
    """
    if isinstance(unitary, tn.Node):
        unitary = unitary.tensor
    unitary = backend.reshapem(unitary)
    rend = backend.stack(
        [backend.cast(unitary, dtypestr), backend.eye(backend.shape_tuple(unitary)[-1])]
    )
    rend = backend.reshape2(rend)
    rn = tn.Node(rend)
    nodes = []
    eps = 1e-5
    if isinstance(ctrl, int):
        ctrl = [ctrl]
    if int(ctrl[0] + eps) == 1:
        leftend = np.zeros([2, 2, 2])
        leftend[1, 1, 0] = 1
        leftend[0, 0, 1] = 1
    else:
        leftend = np.zeros([2, 2, 2])
        leftend[0, 0, 0] = 1
        leftend[1, 1, 1] = 1
    nodes.append(tn.Node(array_to_tensor(leftend)))
    for c in ctrl[1:]:
        mid = np.zeros([2, 2, 2, 2])
        if c == 1:
            mid[0, 1, 1, 0] = 1
            mid[1, 0, 0, 1] = 1
            mid[1, 1, 1, 1] = 1
            mid[0, 0, 0, 1] = 1
        else:  # c= 0
            mid[0, 0, 0, 0] = 1
            mid[1, 1, 1, 1] = 1
            mid[1, 0, 0, 1] = 1
            mid[0, 1, 1, 1] = 1
        nodes.append(tn.Node(array_to_tensor(mid)))

    nodes.append(rn)

    if len(nodes) == 2:
        nodes[0][2] ^ nodes[1][0]
    else:
        nodes[0][2] ^ nodes[1][0]
        for i in range(1, len(nodes) - 1):
            nodes[i][3] ^ nodes[i + 1][0]

    from .quantum import QuOperator

    l = int((len(nodes[-1].edges) - 1) / 2)
    gate = QuOperator(
        [nodes[0][0]]
        + [n[1] for n in nodes[1:-1]]
        + [nodes[-1][i] for i in range(1, 1 + l)],
        [nodes[0][1]]
        + [n[2] for n in nodes[1:-1]]
        + [nodes[-1][i] for i in range(1 + l, 1 + 2 * l)],
    )

    return gate


def mpo_gate(mpo: Operator, name: str = "mpo") -> Operator:
    return mpo


def meta_vgate() -> None:
    for f in [
        "r",
        "u",
        "rx",
        "ry",
        "rz",
        "phase",
        "iswap",
        "any",
        "exp",
        "exp1",
        "cr",
        "rzz",
        "rxx",
        "ryy",
    ]:
        for funcname in [f, f + "gate"]:
            setattr(thismodule, funcname, GateVF(getattr(thismodule, f + "_gate"), f))
    for f in ["cu", "crx", "cry", "crz", "cphase"]:
        for funcname in [f, f + "gate"]:
            setattr(thismodule, funcname, getattr(thismodule, f[1:]).controlled())
    for f in ["ox", "oy", "oz", "orx", "ory", "orz"]:
        for funcname in [f, f + "gate"]:
            setattr(thismodule, funcname, getattr(thismodule, f[1:]).ocontrolled())
    for f in ["sd", "td"]:
        for funcname in [f, f + "gate"]:
            setattr(thismodule, funcname, getattr(thismodule, f[:-1]).adjoint())
    for f in ["multicontrol", "mpo"]:  # mpo type gate
        for funcname in [f, f + "gate"]:
            setattr(thismodule, funcname, GateVF(getattr(thismodule, f + "_gate"), f))


meta_vgate()
