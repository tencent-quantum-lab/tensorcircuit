"""
quantum circuit class but with density matrix simulator
"""

from functools import partial, reduce
from operator import add
from typing import Tuple, List, Callable, Union, Optional, Sequence, Any

import numpy as np
import tensornetwork as tn
import graphviz

from . import gates
from .cons import dtypestr, npdtype, backend, contractor
from .circuit import Circuit

Gate = gates.Gate
Tensor = Any

# TODO: Monte Carlo State Circuit Simulator
# note not all channels but only deploarizing channel can be simulated in that Monte Carlo way with pure state simulators
class DMCircuit:
    def __init__(self, nqubits: int, empty: bool = False) -> None:
        if not empty:
            _prefix = "qb-"
            if nqubits < 2:
                raise ValueError(
                    f"Number of qubits must be greater than 2 but is {nqubits}."
                )

            # Get nodes on the interior
            nodes = [
                tn.Node(
                    np.array(
                        [
                            [[1.0]],
                            [[0.0]],
                        ],
                        dtype=npdtype,
                    ),
                    name=_prefix + str(x + 1),
                )
                for x in range(nqubits - 2)
            ]

            # Get nodes on the end
            nodes.insert(
                0,
                tn.Node(
                    np.array(
                        [
                            [1.0],
                            [0.0],
                        ],
                        dtype=npdtype,
                    ),
                    name=_prefix + str(0),
                ),
            )
            nodes.append(
                tn.Node(
                    np.array(
                        [
                            [1.0],
                            [0.0],
                        ],
                        dtype=npdtype,
                    ),
                    name=_prefix + str(nqubits - 1),
                )
            )

            # Connect edges between middle nodes
            for i in range(1, nqubits - 2):
                tn.connect(nodes[i].get_edge(2), nodes[i + 1].get_edge(1))

            # Connect end nodes to the adjacent middle nodes
            if nqubits < 3:
                tn.connect(nodes[0].get_edge(1), nodes[1].get_edge(1))
            else:
                tn.connect(
                    nodes[0].get_edge(1), nodes[1].get_edge(1)
                )  # something wrong here?
                tn.connect(nodes[-1].get_edge(1), nodes[-2].get_edge(2))

            self._nqubits = nqubits
            self._rfront = [n.get_edge(0) for n in nodes]
            lnodes, self._lfront = self._copy(nodes, self._rfront, conj=True)
            lnodes.extend(nodes)
            self._nodes = lnodes
            self._meta_apply()

    def _meta_apply(self) -> None:
        for g in Circuit.sgates:
            setattr(
                self,
                g,
                partial(self.apply_general_gate_delayed, getattr(gates, g), name=g),
            )
            setattr(
                self,
                g.upper(),
                partial(self.apply_general_gate_delayed, getattr(gates, g), name=g),
            )

        for g in Circuit.vgates:
            setattr(
                self,
                g,
                partial(
                    self.apply_general_variable_gate_delayed, getattr(gates, g), name=g
                ),
            )
            setattr(
                self,
                g.upper(),
                partial(
                    self.apply_general_variable_gate_delayed, getattr(gates, g), name=g
                ),
            )

    def _copy(
        self,
        nodes: Sequence[tn.Node],
        dangling: Optional[Sequence[tn.Edge]] = None,
        conj: Optional[bool] = False,
    ) -> Tuple[List[tn.Node], List[tn.Edge]]:
        """
        copy all nodes and dangling edges correspondingly

        :return:
        """
        ndict, edict = tn.copy(nodes, conjugate=conj)
        newnodes = []
        for n in nodes:
            newnodes.append(ndict[n])
        newfront = []
        if not dangling:
            dangling = []
            for n in nodes:
                dangling.extend([e for e in n])
        for e in dangling:
            newfront.append(edict[e])
        return newnodes, newfront

    def _copy_DMCircuit(self) -> "DMCircuit":
        newnodes, newfront = self._copy(self._nodes, self._lfront + self._rfront)
        newDMCircuit = DMCircuit(self._nqubits, empty=True)
        newDMCircuit._nqubits = self._nqubits
        newDMCircuit._lfront = newfront[: self._nqubits]
        newDMCircuit._rfront = newfront[self._nqubits :]
        newDMCircuit._nodes = newnodes
        return newDMCircuit

    def _contract(self) -> None:
        t = contractor(self._nodes, output_edge_order=self._lfront + self._rfront)
        self._nodes = [t]

    def apply_general_gate(
        self, gate: Gate, *index: int, name: Optional[str] = None
    ) -> None:
        assert len(index) == len(set(index))
        noe = len(index)
        rgated, _ = self._copy([gate], conj=True)
        rgate = rgated[0]
        for i, ind in enumerate(index):
            gate.get_edge(i + noe) ^ self._lfront[ind]
            self._lfront[ind] = gate.get_edge(i)
            rgate.get_edge(i + noe) ^ self._rfront[ind]
            self._rfront[ind] = rgate.get_edge(i)
        self._nodes.append(gate)
        self._nodes.append(rgate)

    def apply_general_gate_delayed(
        self, gatef: Callable[[], Gate], *index: int, name: Optional[str] = None
    ) -> None:
        gate = gatef()
        self.apply_general_gate(gate, *index, name=name)

    def apply_general_variable_gate_delayed(
        self,
        gatef: Callable[..., Gate],
        *index: int,
        name: Optional[str] = None,
        **vars: float,
    ) -> None:
        gate = gatef(**vars)
        self.apply_general_gate(gate, *index, name=name)

    @staticmethod
    def check_kraus(kraus: Sequence[Gate]) -> bool:  # TODO
        return True

    def apply_general_kraus(
        self, kraus: Sequence[Gate], index: Sequence[Tuple[int]]
    ) -> None:
        # TODO: quick way to apply layers of kraus: seems no simply way to do that?
        self.check_kraus(kraus)
        assert len(kraus) == len(index) or len(index) == 1
        if len(index) == 1:
            index = [index[0] for _ in range(len(kraus))]
        self._contract()
        circuits = []
        for k, i in zip(kraus, index):
            dmc = self._copy_DMCircuit()
            dmc.apply_general_gate(k, *i)
            dd = dmc.densitymatrix()
            circuits.append(dd)
        tensor = reduce(add, circuits)
        tensor = backend.reshape(tensor, [2 for _ in range(2 * self._nqubits)])
        self._nodes = [Gate(tensor)]
        dangling = [e for e in self._nodes[0]]
        self._lfront = dangling[: self._nqubits]
        self._rfront = dangling[self._nqubits :]

    def densitymatrix(self, check: bool = False) -> tn.Node.tensor:
        if len(self._nodes) > 1:
            self._contract()
        nodes, d_edges = self._copy(self._nodes, self._lfront + self._rfront)
        t = contractor(nodes, output_edge_order=d_edges)
        dm = backend.reshape(t.tensor, shape=[2 ** self._nqubits, 2 ** self._nqubits])
        if check:
            self.check_density_matrix(dm)
        return dm

    def expectation(self, *ops: Tuple[tn.Node, List[int]]) -> tn.Node.tensor:
        if len(self._nodes) > 1:
            self._contract()
        newdm, newdang = self._copy(self._nodes, self._lfront + self._rfront)
        occupied = set()
        nodes = newdm
        for op, index in ops:
            noe = len(index)
            for j, e in enumerate(index):
                if e in occupied:
                    raise ValueError("Cannot measure two operators in one index")
                newdang[e + self._nqubits] ^ op.get_edge(j)
                newdang[e] ^ op.get_edge(j + noe)
                occupied.add(e)
            nodes.append(op)
        for j in range(self._nqubits):
            if j not in occupied:  # edge1[j].is_dangling invalid here!
                newdang[j] ^ newdang[j + self._nqubits]
        return contractor(nodes).tensor

    @staticmethod
    def check_density_matrix(dm: Tensor) -> None:
        assert np.allclose(backend.trace(dm), 1.0, atol=1e-5)

    def to_graphviz(
        self,
        graph: graphviz.Graph = None,
        include_all_names: bool = False,
        engine: str = "neato",
    ) -> graphviz.Graph:
        """
        Not an ideal visualization for quantum circuit, but reserve here as a general approch to show tensornetwork

        :param graph:
        :param include_all_names:
        :param engine:
        :return:
        """
        # Modified from tensornetwork codebase
        nodes = self._nodes
        if graph is None:
            # pylint: disable=no-member
            graph = graphviz.Graph("G", engine=engine)
        for node in nodes:
            if not node.name.startswith("__") or include_all_names:
                label = node.name
            else:
                label = ""
            graph.node(str(id(node)), label=label)
        seen_edges = set()
        for node in nodes:
            for i, edge in enumerate(node.edges):
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                if not edge.name.startswith("__") or include_all_names:
                    edge_label = edge.name + ": " + str(edge.dimension)
                else:
                    edge_label = ""
                if edge.is_dangling():
                    # We need to create an invisible node for the dangling edge
                    # to connect to.
                    graph.node(
                        "{}_{}".format(str(id(node)), i),
                        label="",
                        _attributes={"style": "invis"},
                    )
                    graph.edge(
                        "{}_{}".format(str(id(node)), i),
                        str(id(node)),
                        label=edge_label,
                    )
                else:
                    graph.edge(
                        str(id(edge.node1)),
                        str(id(edge.node2)),
                        label=edge_label,
                    )
        return graph
