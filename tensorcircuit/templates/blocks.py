"""
Shortcuts for measurement patterns on circuit
"""
# circuit in, circuit out
# pylint: disable=invalid-name

from functools import wraps
from typing import Any, Callable, Optional, Sequence, Tuple, List

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
        split_conf = None
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


def qft(
    c: Circuit,
    *index: int,
    do_swaps: bool = True,
    inverse: bool = False,
    insert_barriers: bool = False
) -> Circuit:
    """
    This function applies quantum fourier transformation (QFT) to the selected circuit lines

    :param c: Circuit in
    :type c: Circuit
    :param *index: the indices of the circuit lines to apply QFT
    :type *index: List[int]
    :param do_swaps: Whether to include the final swaps in the QFT
    :type do_swaps: bool
    :param inverse: If True, the inverse Fourier transform is constructed
    :type inverse: bool
    :param insert_barriers: If True, barriers are inserted as visualization improvement
    :type insert_barriers: bool
    :return: Circuit c
    :rtype: Circuit
    """
    assert len(set(index)) == len(index), "There should not be any repetitive qubits"
    if inverse:
        if do_swaps:
            for i in range(len(index) // 2):
                c.swap(index[i], index[len(index) - 1 - i])
        for i in range(len(index) - 1, -1, -1):
            rotation = -np.pi / 2
            for j in range(len(index) - 1, i, -1):
                c.cphase(index[j], index[i], theta=rotation)
                rotation /= 2
            c.H(index[i])
            if insert_barriers:
                c.barrier_instruction(range(min(index), max(index) + 1))
    else:
        for i in range(len(index)):
            c.H(index[i])
            rotation = np.pi / 2
            for j in range(i + 1, len(index)):
                c.cphase(index[j], index[i], theta=rotation)
                rotation /= 2
            if insert_barriers:
                c.barrier_instruction(range(min(index), max(index) + 1))
        if do_swaps:
            for i in range(len(index) // 2):
                c.swap(index[i], index[len(index) - 1 - i])
    return c


def QAOA_ansatz_for_Ising(
    params: list,
    nlayers: int,
    pauli_terms: Tensor,
    weights: list,
    mixer: str = "X",
    gap: int = 5,
) -> Circuit:
    """
    Construct the QAOA ansatz for the Ising Model.
    The number of qubits is determined by `pauli_terms`.

    :param params: A list of parameter values used in the QAOA ansatz.
    :param nlayers: The number of layers in the QAOA ansatz.
    :pauli_terms: A list of Pauli terms, where each term is represented as a list of 0/1 series.
    :param weights: A list of weights corresponding to each Pauli term.
    :param mixer: mixer type. The options are "X", "XY_ring", "XY_par_ring", "XY_full", and "QAMPA".
    """
    nqubits = len(pauli_terms[0])
    c = Circ(nqubits)
    for i in range(nqubits):
        c.h(i)  # Apply Hadamard gate to each qubit

    for j in range(nlayers):
        # cost terms
        for k in range(len(pauli_terms)):
            term = pauli_terms[k]
            index_of_ones = []
            for l in range(len(term)):
                if term[l] == 1:
                    index_of_ones.append(l)
            if len(index_of_ones) == 1:
                c.rz(index_of_ones[0], theta=2 * weights[k] * params[2 * j])
                # Apply Rz gate with angle determined by weight and current parameter value
            elif len(index_of_ones) == 2:
                c.exp1(
                    index_of_ones[0],
                    index_of_ones[1],
                    unitary=G._zz_matrix,
                    theta=weights[k] * params[2 * j],
                )
                # Apply exp1 gate with a custom unitary (zz_matrix) and angle determined by weight and current parameter value
            else:
                raise ValueError("Invalid number of Z terms")

        # standard mixer
        if mixer == "X":
            for i in range(nqubits):
                c.rx(
                    i, theta=params[2 * j + 1]
                )  # Apply Rx gate with angle determined by current parameter value
        # Parity ring XY mixer
        elif mixer == "XY":
            pairs = []
            for index in [0, 1]:
                while index + 2 <= nqubits:
                    pairs.append([index, index + 1])
                    index += 2
            for pair in pairs:
                c.exp1(pair[0], pair[1], unitary=G._xx_matrix, theta=params[2 * j + 1])
                c.exp1(pair[0], pair[1], unitary=G._yy_matrix, theta=params[2 * j + 1])
            c.exp1(nqubits - 1, 0, unitary=G._xx_matrix, theta=params[2 * j + 1])
            c.exp1(nqubits - 1, 0, unitary=G._yy_matrix, theta=params[2 * j + 1])

        # XY mixer with full couplings
        elif mixer == "XY_full":
            for q0 in range(nqubits - 1):
                for q1 in range(q0 + 1, nqubits):
                    c.exp1(q0, q1, unitary=G._xx_matrix, theta=params[2 * j + 1])
                    c.exp1(q0, q1, unitary=G._yy_matrix, theta=params[2 * j + 1])

        # Parity ring ZZ mixer
        elif mixer == "ZZ":
            pairs = []
            for index in [0, 1]:
                while index + 2 <= nqubits:
                    pairs.append([index, index + 1])
                    index += 2
            for pair in pairs:
                c.exp1(pair[0], pair[1], unitary=G._zz_matrix, theta=params[2 * j + 1])
            c.exp1(nqubits - 1, 0, unitary=G._zz_matrix, theta=params[2 * j + 1])

        # ZZ mixer with full couplings
        elif mixer == "ZZ_full":
            for q0 in range(nqubits - 1):
                for q1 in range(q0, nqubits):
                    c.exp1(q0, q1, unitary=G._zz_matrix, theta=params[2 * j + 1])

        else:
            raise ValueError("Invalid mixer type.")

    return c
