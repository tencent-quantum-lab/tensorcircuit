"""
Shortcuts for measurement patterns on circuit
"""
# circuit in, scalar out

from typing import Any

from ..circuit import Circuit
from ..cons import backend, dtypestr
from ..quantum import QuOperator
from .. import gates as G

Tensor = Any
Graph = Any  # nx.graph


def any_measurements(
    c: Circuit, structures: Tensor, onehot: bool = False, reuse: bool = False
) -> Tensor:
    """
    This measurements pattern is specifically suitable for vmap. Parameterize the Pauli string
    to be measured.

    :example:

    .. code-block:: python

        c = tc.Circuit(3)
        c.rx(0, theta=1.0)
        c.cnot(0, 1)
        c.cnot(1, 2)
        c.ry(2, theta=-1.0)

        z0x2 = c.expectation([tc.gates.z(), [0]], [tc.gates.x(), [2]])
        z0x2p1 = tc.templates.measurements.parameterized_measurements(
            c, tc.array_to_tensor(np.array([3, 0, 1])), onehot=True
        )
        z0x2p2 = tc.templates.measurements.parameterized_measurements(
            c,
            tc.array_to_tensor(np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])),
            onehot=False,
        )
        np.testing.assert_allclose(z0x2, z0x2p1)
        np.testing.assert_allclose(z0x2, z0x2p2)

    :param c: The circuit to be measured
    :type c: Circuit
    :param structures: parameter tensors determines what Pauli string to be measured,
        shape is [nwires, 4] if ``onehot`` is False and [nwires] if ``onehot`` is True.
    :type structures: Tensor
    :param onehot: defaults to False. If set to be True,
        structures will first go through onehot procedure.
    :type onehot: bool, optional
    :param reuse: reuse the wavefunction when computing the expectations, defaults to be False
    :type reuse: bool, optional
    :return: The expectation value of given Pauli string by the tensor ``structures``.
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
    loss = c.expectation(*obs, reuse=reuse)  # type: ignore
    return backend.real(loss)


parameterized_measurements = any_measurements


def any_local_measurements(
    c: Circuit, structures: Tensor, onehot: bool = False, reuse: bool = True
) -> Tensor:
    """
    This measurements pattern is specifically suitable for vmap. Parameterize the local
    Pauli string to be measured.

    :example:

    .. code-block:: python

        c = tc.Circuit(3)
        c.X(0)
        c.cnot(0, 1)
        c.H(-1)
        basis = tc.backend.convert_to_tensor(np.array([3, 3, 1]))
        z0, z1, x2 = tc.templates.measurements.parameterized_local_measurements(
            c, structures=basis, onehot=True
        )
        # -1, -1, 1


    :param c: The circuit to be measured
    :type c: Circuit
    :param structures: parameter tensors determines what Pauli string to be measured,
        shape is [nwires, 4] if ``onehot`` is False and [nwires] if ``onehot`` is True.
    :type structures: Tensor
    :param onehot: defaults to False. If set to be True,
        structures will first go through onehot procedure.
    :type onehot: bool, optional
    :param reuse: reuse the wavefunction when computing the expectations, defaults to be True
    :type reuse: bool, optional
    :return: The expectation value of given Pauli string by the tensor ``structures``.
    :rtype: Tensor
    """
    if onehot is True:
        structuresc = backend.cast(structures, dtype="int32")
        structuresc = backend.onehot(structuresc, num=4)
        structuresc = backend.cast(structuresc, dtype=dtypestr)
    else:
        structuresc = structures
    nwires = c._nqubits
    loss = []
    for i in range(nwires):
        loss.append(
            c.expectation(
                (
                    G.Gate(
                        sum(
                            [
                                structuresc[i, k] * g.tensor
                                for k, g in enumerate(G.pauli_gates)
                            ]
                        )
                    ),
                    [i],
                ),
                reuse=reuse,
            )
        )

    return backend.real(backend.stack(loss))


parameterized_local_measurements = any_local_measurements


def operator_expectation(c: Circuit, hamiltonian: Any) -> Tensor:
    """
    Evaluate Hamiltonian expectation where ``hamiltonian`` can be dense matrix, sparse matrix or MPO.

    :param c: The circuit whose output state is used to evaluate the expectation
    :type c: Circuit
    :param hamiltonian: Hamiltonian matrix in COO_sparse_matrix form
    :type hamiltonian: Tensor
    :return: a real and scalar tensor of shape [] as the expectation value
    :rtype: Tensor
    """
    if isinstance(hamiltonian, QuOperator):
        return mpo_expectation(c, hamiltonian)
    elif backend.is_sparse(hamiltonian):
        return sparse_expectation(c, hamiltonian)
    else:
        w = c.state(form="ket")
        e = (backend.adjoint(w) @ hamiltonian @ w)[0, 0]
        return backend.real(e)


def sparse_expectation(c: Circuit, hamiltonian: Tensor) -> Tensor:
    """
    Evaluate Hamiltonian expectation where ``hamiltonian`` is kept in sparse matrix form to save space

    :param c: The circuit whose output state is used to evaluate the expectation
    :type c: Circuit
    :param hamiltonian: Hamiltonian matrix in COO_sparse_matrix form
    :type hamiltonian: Tensor
    :return: a real and scalar tensor of shape [] as the expectation value
    :rtype: Tensor
    """
    state = c.wavefunction(form="ket")
    tmp = backend.sparse_dense_matmul(hamiltonian, state)
    expt = backend.adjoint(state) @ tmp
    return backend.real(expt)[0, 0]


def mpo_expectation(c: Circuit, mpo: QuOperator) -> Tensor:
    """
    Evaluate expectation of operator ``mpo`` defined in ``QuOperator`` MPO format
    with the output quantum state from circuit ``c``.

    :param c: The circuit for the output state
    :type c: Circuit
    :param mpo: MPO operator
    :type mpo: QuOperator
    :return: a real and scalar tensor of shape [] as the expectation value
    :rtype: Tensor
    """
    mps = c.get_quvector()
    e = (mps.adjoint() @ mpo @ mps).eval_matrix()
    return backend.real(e)[0, 0]


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
    r"""
    Evaluate Heisenberg energy expectation, whose Hamiltonian is defined on the lattice graph ``g`` as follows:
    (e are edges in graph ``g`` where e1 and e2 are two nodes for edge e and v are nodes in graph ``g``)

    .. math::

        H = \sum_{e\in g} w_e (h_{xx} X_{e1}X_{e2} + h_{yy} Y_{e1}Y_{e2} + h_{zz} Z_{e1}Z_{e2})
         + \sum_{v\in g} (h_x X_v + h_y Y_v + h_z Z_v)

    :example:

    .. code-block:: python

        g = tc.templates.graphs.Line1D(n=5)
        c = tc.Circuit(5)
        c.X(0)
        energy = tc.templates.measurements.heisenberg_measurements(c, g) # 1

    :param c: Circuit to be measured
    :type c: Circuit
    :param g: Lattice graph defining Heisenberg Hamiltonian
    :type g: Graph
    :param hzz: [description], defaults to 1.0
    :type hzz: float, optional
    :param hxx: [description], defaults to 1.0
    :type hxx: float, optional
    :param hyy: [description], defaults to 1.0
    :type hyy: float, optional
    :param hz: [description], defaults to 0.0
    :type hz: float, optional
    :param hx: [description], defaults to 0.0
    :type hx: float, optional
    :param hy: [description], defaults to 0.0
    :type hy: float, optional
    :param reuse: [description], defaults to True
    :type reuse: bool, optional
    :return: Value of Heisenberg energy
    :rtype: Tensor
    """
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
    return backend.real(loss)


def spin_glass_measurements(c: Circuit, g: Graph, reuse: bool = True) -> Tensor:
    r"""
    Compute spin glass energy defined on graph ``g`` expectation for output state of the circuit ``c``.
    The Hamiltonian to be evaluated is defined as (first term is determined by node weights while
    the second term is determined by edge weights of the graph):

    .. math::

        H = \sum_{v\in g} w_v Z_v + \sum_{e\in g} w_e Z_{e1} Z_{e2}

    :example:

    .. code-block:: python

        import networkx as nx

        # building the lattice graph for spin glass Hamiltonian
        g = nx.Graph()
        g.add_node(0, weight=1)
        g.add_node(1, weight=-1)
        g.add_node(2, weight=1)
        g.add_edge(0, 1, weight=-1)
        g.add_edge(1, 2, weight=-1)

        c = tc.Circuit(3)
        c.X(1)
        energy = tc.templates.measurements.spin_glass_measurements(c, g)
        print(energy) # 5.0

    :param c: The quantum circuit
    :type c: Circuit
    :param g: The graph for spin glass Hamiltonian definition
    :type g: Graph
    :param reuse: Whether measure the circuit with reusing the wavefunction, defaults to True
    :type reuse: bool, optional
    :return: The spin glass energy expectation value
    :rtype: Tensor
    """
    loss = 0
    for e1, e2 in g.edges:
        loss += g[e1][e2].get("weight", 1.0) * c.expectation(
            (G.z(), [e1]), (G.z(), [e2]), reuse=reuse  # type: ignore
        )
    for n in g.nodes:
        loss += g.nodes[n].get("weight", 0.0) * c.expectation((G.z(), [n]), reuse=reuse)  # type: ignore
    return backend.real(loss)
