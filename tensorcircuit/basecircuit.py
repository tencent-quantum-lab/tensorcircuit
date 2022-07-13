"""
Quantum circuit: common methods for all circuit classes as MixIn
"""
# pylint: disable=invalid-name

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensornetwork as tn

from . import gates
from .cons import npdtype, backend, dtypestr, contractor
from .simplify import _split_two_qubit_gate

Gate = gates.Gate
Tensor = Any

sgates = (
    ["i", "x", "y", "z", "h", "t", "s", "td", "sd", "wroot"]
    + ["cnot", "cz", "swap", "cy", "iswap", "ox", "oy", "oz"]
    + ["toffoli", "fredkin"]
)
vgates = [
    "r",
    "cr",
    "rx",
    "ry",
    "rz",
    "rxx",
    "ryy",
    "rzz",
    "crx",
    "cry",
    "crz",
    "orx",
    "ory",
    "orz",
    "any",
    "exp",
    "exp1",
]
mpogates = ["multicontrol", "mpo"]
gate_aliases = [
    ["cnot", "cx"],
    ["fredkin", "cswap"],
    ["toffoli", "ccnot"],
    ["any", "unitary"],
]


class BaseCircuit:
    _nodes: List[tn.Node]
    _front: List[tn.Edge]
    _nqubits: int
    is_dm: bool
    _qir: List[Dict[str, Any]]
    split: Optional[Dict[str, Any]]

    @staticmethod
    def all_zero_nodes(n: int, d: int = 2, prefix: str = "qb-") -> Sequence[tn.Node]:
        l = [0.0 for _ in range(d)]
        l[0] = 1.0
        nodes = [
            tn.Node(
                np.array(
                    l,
                    dtype=npdtype,
                ),
                name=prefix + str(x),
            )
            for x in range(n)
        ]
        return nodes

    @staticmethod
    def front_from_nodes(nodes: List[tn.Node]) -> List[tn.Edge]:
        return [n.get_edge(0) for n in nodes]

    @staticmethod
    def copy(
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
            newn = ndict[n]
            newn.is_dagger = conj
            newn.flag = getattr(n, "flag", "") + "copy"
            newn.id = getattr(n, "id", id(n))
            newnodes.append(newn)
        newfront = []
        if not dangling:
            dangling = []
            for n in nodes:
                dangling.extend([e for e in n])
        for e in dangling:
            newfront.append(edict[e])
        return newnodes, newfront

    def _copy(
        self, conj: Optional[bool] = False
    ) -> Tuple[List[tn.Node], List[tn.Edge]]:
        return self.copy(self._nodes, self._front, conj)

    def apply_general_gate(
        self,
        gate: Gate,
        *index: int,
        name: Optional[str] = None,
        split: Optional[Dict[str, Any]] = None,
        mpo: bool = False,
        ir_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        if name is None:
            name = ""
        gate_dict = {
            "gate": gate,
            "index": index,
            "name": name,
            "split": split,
            "mpo": mpo,
        }
        if ir_dict is not None:
            ir_dict.update(gate_dict)
        else:
            ir_dict = gate_dict
        self._qir.append(ir_dict)
        assert len(index) == len(set(index))
        noe = len(index)
        nq = self._nqubits
        applied = False
        split_conf = None
        if split is not None:
            split_conf = split
        elif self.split is not None:
            split_conf = self.split

        if not mpo:
            if (split_conf is not None) and noe == 2:
                results = _split_two_qubit_gate(gate, **split_conf)
                # max_err cannot be jax jitted
                if results is not None:
                    n1, n2, is_swap = results
                    n1.flag = "gate"
                    n1.is_dagger = False
                    n1.name = name
                    n1.id = id(n1)
                    n2.flag = "gate"
                    n2.is_dagger = False
                    n2.id = id(n2)
                    n2.name = name
                    if is_swap is False:
                        n1[1] ^ self._front[index[0]]
                        n2[2] ^ self._front[index[1]]
                        self._nodes.append(n1)
                        self._nodes.append(n2)
                        self._front[index[0]] = n1[0]
                        self._front[index[1]] = n2[1]
                        if self.is_dm:
                            [n1l, n2l], _ = self.copy([n1, n2], conj=True)
                            n1l[1] ^ self._front[index[0] + nq]
                            n2l[2] ^ self._front[index[1] + nq]
                            self._nodes.append(n1l)
                            self._nodes.append(n2l)
                            self._front[index[0] + nq] = n1l[0]
                            self._front[index[1] + nq] = n2l[1]
                    else:
                        n2[2] ^ self._front[index[0]]
                        n1[1] ^ self._front[index[1]]
                        self._nodes.append(n1)
                        self._nodes.append(n2)
                        self._front[index[0]] = n1[0]
                        self._front[index[1]] = n2[1]
                        if self.is_dm:
                            [n1l, n2l], _ = self.copy([n1, n2], conj=True)
                            n2l[1] ^ self._front[index[0] + nq]
                            n1l[2] ^ self._front[index[1] + nq]
                            self._nodes.append(n1l)
                            self._nodes.append(n2l)
                            self._front[index[0] + nq] = n1l[0]
                            self._front[index[1] + nq] = n2l[1]
                    applied = True

            if applied is False:
                gate.name = name
                gate.flag = "gate"
                gate.is_dagger = False
                gate.id = id(gate)
                self._nodes.append(gate)
                if self.is_dm:
                    lgates, _ = self.copy([gate], conj=True)
                    lgate = lgates[0]
                    self._nodes.append(lgate)
                for i, ind in enumerate(index):
                    gate.get_edge(i + noe) ^ self._front[ind]
                    self._front[ind] = gate.get_edge(i)
                    if self.is_dm:
                        lgate.get_edge(i + noe) ^ self._front[ind + nq]
                        self._front[ind + nq] = lgate.get_edge(i)

        else:  # gate in MPO format
            gatec = gate.copy()
            for n in gatec.nodes:
                n.flag = "gate"
                n.is_dagger = False
                n.id = id(gate)
                n.name = name
            self._nodes += gatec.nodes
            if self.is_dm:
                gateconj = gate.adjoint()
                for n0, n in zip(gatec.nodes, gateconj.nodes):
                    n.flag = "gate"
                    n.is_dagger = True
                    n.id = id(n0)
                    n.name = name
                self._nodes += gateconj.nodes

            for i, ind in enumerate(index):
                gatec.in_edges[i] ^ self._front[ind]
                self._front[ind] = gatec.out_edges[i]
                if self.is_dm:
                    gateconj.out_edges[i] ^ self._front[ind + nq]
                    self._front[ind + nq] = gateconj.in_edges[i]

        self.state_tensor = None  # refresh the state cache

    apply = apply_general_gate

    @staticmethod
    def apply_general_variable_gate_delayed(
        gatef: Callable[..., Gate],
        name: Optional[str] = None,
        mpo: bool = False,
    ) -> Callable[..., None]:
        if name is None:
            name = getattr(gatef, "n")

        def apply(self: "BaseCircuit", *index: int, **vars: Any) -> None:
            split = None
            localname = name
            if "name" in vars:
                localname = vars["name"]
                del vars["name"]
            if "split" in vars:
                split = vars["split"]
                del vars["split"]
            gate_dict = {
                "gatef": gatef,
                "index": index,
                "name": localname,
                "split": split,
                "mpo": mpo,
                "parameters": vars,
            }
            # self._qir.append(gate_dict)
            gate = gatef(**vars)
            self.apply_general_gate(
                gate,
                *index,
                name=localname,
                split=split,
                mpo=mpo,
                ir_dict=gate_dict,
            )  # type: ignore

        return apply

    @staticmethod
    def apply_general_gate_delayed(
        gatef: Callable[[], Gate],
        name: Optional[str] = None,
        mpo: bool = False,
    ) -> Callable[..., None]:
        # it is more like a register instead of apply
        # nested function must be utilized, functools.partial doesn't work for method register on class
        # see https://re-ra.xyz/Python-中实例方法动态绑定的几组最小对立/
        if name is None:
            name = getattr(gatef, "n")
        defaultname = name

        def apply(
            self: "BaseCircuit",
            *index: int,
            split: Optional[Dict[str, Any]] = None,
            name: Optional[str] = None,
        ) -> None:
            if name is not None:
                localname = name
            else:
                localname = defaultname  # type: ignore

            # split = None
            gate = gatef()
            gate_dict = {"gatef": gatef}

            self.apply_general_gate(
                gate,
                *index,
                name=localname,
                split=split,
                mpo=mpo,
                ir_dict=gate_dict,
            )

        return apply

    def _copy_state_tensor(
        self, conj: bool = False, reuse: bool = True
    ) -> Tuple[List[tn.Node], List[tn.Edge]]:
        if reuse:
            t = getattr(self, "state_tensor", None)
            if t is None:
                nodes, d_edges = self._copy()
                t = contractor(nodes, output_edge_order=d_edges)
                setattr(self, "state_tensor", t)
            ndict, edict = tn.copy([t], conjugate=conj)
            newnodes = []
            newnodes.append(ndict[t])
            newfront = []
            for e in t.edges:
                newfront.append(edict[e])
            return newnodes, newfront
        return self._copy(conj)  # type: ignore

    def expectation_before(
        self,
        *ops: Tuple[tn.Node, List[int]],
        reuse: bool = True,
        **kws: Any,
    ) -> List[tn.Node]:
        nq = self._nqubits
        if self.is_dm is True:
            nodes, newdang = self._copy_state_tensor(reuse=reuse)
        else:
            nodes1, edge1 = self._copy_state_tensor(reuse=reuse)
            nodes2, edge2 = self._copy_state_tensor(conj=True, reuse=reuse)
            nodes = nodes1 + nodes2
            newdang = edge1 + edge2
        occupied = set()
        for op, index in ops:
            if not isinstance(op, tn.Node):
                # op is only a matrix
                op = backend.reshape2(op)
                op = backend.cast(op, dtype=dtypestr)
                op = gates.Gate(op)
            else:
                op.tensor = backend.cast(op.tensor, dtype=dtypestr)
            if isinstance(index, int):
                index = [index]
            noe = len(index)
            for j, e in enumerate(index):
                if e in occupied:
                    raise ValueError("Cannot measure two operators in one index")
                newdang[e + nq] ^ op.get_edge(j)
                newdang[e] ^ op.get_edge(j + noe)
                occupied.add(e)
            op.flag = "operator"
            op.is_dagger = False
            op.id = id(op)
            nodes.append(op)
        for j in range(nq):
            if j not in occupied:  # edge1[j].is_dangling invalid here!
                newdang[j] ^ newdang[j + nq]
        return nodes  # type: ignore

    def expectation_ps(
        self,
        x: Optional[Sequence[int]] = None,
        y: Optional[Sequence[int]] = None,
        z: Optional[Sequence[int]] = None,
        reuse: bool = True,
        **kws: Any,
    ) -> Tensor:
        """
        Shortcut for Pauli string expectation.
        x, y, z list are for X, Y, Z positions

        :Example:

        >>> c = tc.Circuit(2)
        >>> c.X(0)
        >>> c.H(1)
        >>> c.expectation_ps(x=[1], z=[0])
        array(-0.99999994+0.j, dtype=complex64)

        :param x: _description_, defaults to None
        :type x: Optional[Sequence[int]], optional
        :param y: _description_, defaults to None
        :type y: Optional[Sequence[int]], optional
        :param z: _description_, defaults to None
        :type z: Optional[Sequence[int]], optional
        :param reuse: whether to cache and reuse the wavefunction, defaults to True
        :type reuse: bool, optional
        :return: Expectation value
        :rtype: Tensor
        """
        obs = []
        if x is not None:
            for i in x:
                obs.append([gates.x(), [i]])  # type: ignore
        if y is not None:
            for i in y:
                obs.append([gates.y(), [i]])  # type: ignore
        if z is not None:
            for i in z:
                obs.append([gates.z(), [i]])  # type: ignore
        return self.expectation(*obs, reuse=reuse, **kws)  # type: ignore
