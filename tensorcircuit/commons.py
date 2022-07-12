"""
Quantum circuit: common methods for all circuit classes
"""
# pylint: disable=invalid-name

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensornetwork as tn

from . import gates
from .cons import npdtype
from .simplify import _split_two_qubit_gate

Gate = gates.Gate
Tensor = Any
BaseCircuit = Any


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


def front_from_nodes(nodes: List[tn.Node]) -> List[tn.Edge]:
    return [n.get_edge(0) for n in nodes]


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


def copy_circuit(
    circuit: BaseCircuit, conj: Optional[bool] = False
) -> Tuple[List[tn.Node], List[tn.Edge]]:
    return copy(circuit._nodes, circuit._front, conj)


def apply_general_gate(
    circuit: BaseCircuit,
    gate: Gate,
    *index: int,
    name: Optional[str] = None,
    split: Optional[Dict[str, Any]] = None,
    mpo: bool = False,
    ir_dict: Optional[Dict[str, Any]] = None,
    is_dm: bool = False,
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
    circuit._qir.append(ir_dict)
    assert len(index) == len(set(index))
    noe = len(index)
    applied = False
    split_conf = None
    if split is not None:
        split_conf = split
    elif circuit.split is not None:
        split_conf = circuit.split

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
                    n1[1] ^ circuit._front[index[0]]
                    n2[2] ^ circuit._front[index[1]]
                    circuit._nodes.append(n1)
                    circuit._nodes.append(n2)
                    circuit._front[index[0]] = n1[0]
                    circuit._front[index[1]] = n2[1]
                    if is_dm:
                        [n1l, n2l], _ = copy([n1, n2], conj=True)
                        n1l[1] ^ circuit._lfront[index[0]]
                        n2l[2] ^ circuit._lfront[index[1]]
                        circuit._nodes.append(n1l)
                        circuit._nodes.append(n2l)
                        circuit._lfront[index[0]] = n1l[0]
                        circuit._lfront[index[1]] = n2l[1]
                else:
                    n2[2] ^ circuit._front[index[0]]
                    n1[1] ^ circuit._front[index[1]]
                    circuit._nodes.append(n1)
                    circuit._nodes.append(n2)
                    circuit._front[index[0]] = n1[0]
                    circuit._front[index[1]] = n2[1]
                    if is_dm:
                        [n1l, n2l], _ = copy([n1, n2], conj=True)
                        n2l[1] ^ circuit._lfront[index[0]]
                        n1l[2] ^ circuit._lfront[index[1]]
                        circuit._nodes.append(n1l)
                        circuit._nodes.append(n2l)
                        circuit._lfront[index[0]] = n1l[0]
                        circuit._lfront[index[1]] = n2l[1]
                applied = True

        if applied is False:
            gate.name = name
            gate.flag = "gate"
            gate.is_dagger = False
            gate.id = id(gate)
            circuit._nodes.append(gate)
            if is_dm:
                lgates, _ = copy([gate], conj=True)
                lgate = lgates[0]
                circuit._nodes.append(lgate)
            for i, ind in enumerate(index):
                gate.get_edge(i + noe) ^ circuit._front[ind]
                circuit._front[ind] = gate.get_edge(i)
                if is_dm:
                    lgate.get_edge(i + noe) ^ circuit._lfront[ind]
                    circuit._lfront[ind] = lgate.get_edge(i)

    else:  # gate in MPO format
        gatec = gate.copy()
        for n in gatec.nodes:
            n.flag = "gate"
            n.is_dagger = False
            n.id = id(gate)
            n.name = name
        circuit._nodes += gatec.nodes
        if is_dm:
            gateconj = gate.adjoint()
            for n0, n in zip(gatec.nodes, gateconj.nodes):
                n.flag = "gate"
                n.is_dagger = True
                n.id = id(n0)
                n.name = name
            circuit._nodes += gateconj.nodes

        for i, ind in enumerate(index):
            gatec.in_edges[i] ^ circuit._front[ind]
            circuit._front[ind] = gatec.out_edges[i]
            if is_dm:
                gateconj.out_edges[i] ^ circuit._lfront[ind]
                circuit._lfront[ind] = gateconj.in_edges[i]

    circuit.state_tensor = None  # refresh the state cache


def apply_general_variable_gate_delayed(
    gatef: Callable[..., Gate],
    name: Optional[str] = None,
    mpo: bool = False,
    is_dm: bool = False,
) -> Callable[..., None]:
    if name is None:
        name = getattr(gatef, "n")

    def apply(self: BaseCircuit, *index: int, **vars: Any) -> None:
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
        apply_general_gate(
            self,
            gate,
            *index,
            name=localname,
            split=split,
            mpo=mpo,
            ir_dict=gate_dict,
            is_dm=is_dm,
        )  # type: ignore

    return apply


def apply_general_gate_delayed(
    gatef: Callable[[], Gate],
    name: Optional[str] = None,
    mpo: bool = False,
    is_dm: bool = False,
) -> Callable[..., None]:
    # it is more like a register instead of apply
    # nested function must be utilized, functools.partial doesn't work for method register on class
    # see https://re-ra.xyz/Python-中实例方法动态绑定的几组最小对立/
    if name is None:
        name = getattr(gatef, "n")
    defaultname = name

    def apply(
        self: BaseCircuit,
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

        apply_general_gate(
            self,
            gate,
            *index,
            name=localname,
            split=split,
            mpo=mpo,
            ir_dict=gate_dict,
            is_dm=is_dm,
        )

    return apply


def expectation_ps(
    c: BaseCircuit,
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
    return c.expectation(*obs, reuse=reuse, **kws)  # type: ignore
