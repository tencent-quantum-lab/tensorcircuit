"""
Shortcuts for measurement patterns on circuit
"""
# circuit in, circuit out
# pylint: disable=invalid-name

from functools import wraps
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from .graphs import Grid2DCoord
from .. import gates as G
from ..circuit import Circuit as Circ
from ..cons import backend

Circuit = Any  # we don't use the real circuit class as too many mypy complains emerge
Tensor = Any
Graph = Any


def state_centric(f: Callable[..., Circuit]) -> Callable[..., Tensor]:
    """
    Function decorator wraps the function with the first input and output in the format of circuit,
    the wrapped function has the first input and the output as the state tensor.

    :param f: Function with the fist input and the output as ``Circuit`` object.
    :type f: Callable[..., Circuit]
    :return: Wrapped function with the first input and the output as the state tensor correspondingly.
    :rtype: Callable[..., Tensor]
    """

    @wraps(f)
    def wrapper(s: Tensor, *args: Any, **kws: Any) -> Tensor:
        n = backend.sizen(s)
        n = int(np.log2(n))
        c = Circ(n, inputs=s)
        c = f(c, *args, **kws)
        s = c.state()
        return s

    return wrapper


def Bell_pair_block(
    c: Circuit, links: Optional[Sequence[Tuple[int, int]]] = None
) -> Circuit:
    """
    For each pair in links, the input product state |00> is transformed as (01>-|10>)

    :param c: Circuit in
    :type c: Circuit
    :param links: pairs indices for Bell pairs, defaults to None, corresponds to neighbor links
    :type links: Optional[Sequence[Tuple[int, int]]], optional
    :return: Circuit out
    :rtype: Circuit
    """
    # from |00> return |01>-|10>
    n = c._nqubits
    if links is None:
        links = [(i, i + 1) for i in range(0, n - 1, 2)]
    for a, b in links:
        c.X(a)
        c.H(a)
        c.cnot(a, b)
        c.X(b)
    return c


def Grid2D_entangling(
    c: Circuit, coord: Grid2DCoord, unitary: Tensor, params: Tensor, **kws: Any
) -> Circuit:
    i = 0
    for a, b in coord.all_rows():
        c.exp1(a, b, unitary=unitary, theta=params[i], **kws)
        i += 1
    for a, b in coord.all_cols():
        c.exp1(a, b, unitary=unitary, theta=params[i], **kws)
        i += 1
    return c


def QAOA_block(
    c: Circuit, g: Graph, paramzz: Tensor, paramx: Tensor, **kws: Any
) -> Circuit:
    if backend.sizen(paramzz) == 1:
        for e1, e2 in g.edges:
            c.exp1(
                e1,
                e2,
                unitary=G._zz_matrix,
                theta=paramzz * g[e1][e2].get("weight", 1.0),
                **kws
            )
    else:
        i = 0
        for e1, e2 in g.edges:
            c.exp1(e1, e2, unitary=G._zz_matrix, theta=paramzz[i], **kws)
            i += 1

    if backend.sizen(paramx) == 1:
        for n in g.nodes:
            c.rx(n, theta=paramx)
    else:
        i = 0
        for n in g.nodes:
            c.rx(n, theta=paramx[i])
            i += 1
    return c


def example_block(
    c: Circuit, param: Tensor, nlayers: int = 2, is_split: bool = False
) -> Circuit:
    r"""
    The circuit ansatz is firstly one layer of Hadamard gates and
    then we have ``nlayers`` blocks of :math:`e^{i\theta Z_iZ_{i+1}}` two-qubit gate in ladder layout,
    following rx gate.

    :param c: The circuit
    :type c: Circuit
    :param param: paramter tensor with 2*nlayer*n elements
    :type param: Tensor
    :param nlayers: number of ZZ+RX blocks, defaults to 2
    :type nlayers: int, optional
    :param is_split: whether use SVD split to reduce ZZ gate bond dimension,
        defaults to False
    :type is_split: bool, optional
    :return: The circuit with example ansatz attached
    :rtype: Circuit
    """
    # used for test and demonstrations
    if is_split:
        split_conf = {
            "max_singular_values": 2,
            "fixed_choice": 1,
        }
    else:
        split_conf = None  # type: ignore
    n = c._nqubits
    param = backend.reshape(param, [2 * nlayers, n])
    for i in range(n):
        c.H(i)
    for j in range(nlayers):
        for i in range(n - 1):
            c.exp1(
                i, i + 1, unitary=G._zz_matrix, theta=param[2 * j, i], split=split_conf
            )
        for i in range(n):
            c.rx(i, theta=param[2 * j + 1, i])
    return c
