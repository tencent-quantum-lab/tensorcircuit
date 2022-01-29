"""
shortcuts for measurement patterns on circuit
"""
# circuit in, scalar out

from typing import Any

from ..circuit import Circuit
from ..cons import backend, dtypestr
from .. import gates as G

Tensor = Any
Graph = Any  # nx.graph


def any_measurements(c: Circuit, structures: Tensor, onehot: bool = False) -> Tensor:
    """
    This measurements pattern is specifically suitable for vmap. Parameterize the Pauli string
    to be measured.

    :param c: [description]
    :type c: Circuit
    :param structures: parameter tensors determines what Pauli string to be measured,
        shape is [nwires, 4] if onehot is False.
    :type structures: Tensor
    :param onehot: [description], defaults to False
    :type onehot: bool, optional
    :return: [description]
    :rtype: Tensor
    """
    if onehot is True:
        structuresc = backend.cast(structures, dtype="int32")
        structuresc = backend.onehot(structuresc, num=4)
        structuresc = backend.cast(structuresc, dtype=dtypestr)
    else:
        structuresc = structures
    nwires = c._nqubits
    obs = []
    for i in range(nwires):
        obs.append(
            [
                G.Gate(
                    sum(
                        [
                            structuresc[i, k] * g.tensor
                            for k, g in enumerate(G.pauli_gates)
                        ]
                    )
                ),
                (i,),
            ]
        )
    loss = c.expectation(*obs, reuse=False)  # type: ignore
    # TODO(@refraction-ray): is reuse=True in this setup has user case?
    return backend.real(loss)


def sparse_expectation(c: Circuit, hamiltonian: Tensor) -> Tensor:
    """
    [summary]

    :param c: [description]
    :type c: Circuit
    :param hamiltonian: COO_sparse_matrix
    :type hamiltonian: Tensor
    :return: a real and scalar tensor of shape []
    :rtype: Tensor
    """
    state = c.wavefunction(form="ket")
    tmp = backend.sparse_dense_matmul(hamiltonian, state)
    expt = backend.adjoint(state) @ tmp
    return backend.real(expt)[0, 0]


def heisenberg_measurements(
    c: Circuit,
    g: Graph,
    hzz: float = 1.0,
    hxx: float = 1.0,
    hyy: float = 1.0,
    hz: float = 0.0,
    hx: float = 0.0,
    hy: float = 0.0,
    reuse: bool = True,
) -> Tensor:
    loss = 0.0
    for e in g.edges:
        loss += (
            g[e[0]][e[1]]["weight"]
            * hzz
            * c.expectation((G.z(), [e[0]]), (G.z(), [e[1]]), reuse=reuse)  # type: ignore
        )
        loss += (
            g[e[0]][e[1]]["weight"]
            * hyy
            * c.expectation((G.y(), [e[0]]), (G.y(), [e[1]]), reuse=reuse)  # type: ignore
        )
        loss += (
            g[e[0]][e[1]]["weight"]
            * hxx
            * c.expectation((G.x(), [e[0]]), (G.x(), [e[1]]), reuse=reuse)  # type: ignore
        )
    if hx != 0:
        for i in range(len(g.nodes)):
            loss += hx * c.expectation((G.x(), [i]), reuse=reuse)  # type: ignore
    if hy != 0:
        for i in range(len(g.nodes)):
            loss += hy * c.expectation((G.y(), [i]), reuse=reuse)  # type: ignore
    if hz != 0:
        for i in range(len(g.nodes)):
            loss += hz * c.expectation((G.z(), [i]), reuse=reuse)  # type: ignore
    return loss


def spin_glass_measurements(c: Circuit, g: Graph, reuse: bool = True) -> Tensor:
    loss = 0
    for e1, e2 in g.edges:
        loss += g[e1][e2].get("weight", 1.0) * c.expectation(
            (G.z(), [e1]), (G.z(), [e2]), reuse=reuse  # type: ignore
        )
    for n in g.nodes:
        loss += g.nodes[n].get("weight", 0.0) * c.expectation((G.z(), [n]), reuse=reuse)  # type: ignore
    return backend.real(loss)
