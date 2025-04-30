"""
Module for functions adding layers of circuits
"""

import itertools
import logging
import sys
from typing import Sequence, Union, Any, Optional, Tuple, List

import networkx as nx
import numpy as np
import tensorflow as tf

from ..circuit import Circuit
from ..densitymatrix import DMCircuit
from ..gates import num_to_tensor, array_to_tensor, _swap_matrix
from ..channels import depolarizingchannel
from ..abstractcircuit import sgates

logger = logging.getLogger(__name__)

try:
    import cirq
except ImportError as e:
    logger.warning(e)
    logger.warning("Therefore some functionality in %s may not work" % __name__)


thismodule = sys.modules[__name__]

Tensor = Any  # tf.Tensor
Graph = Any  # nx.Graph
Symbol = Any  # sympy.Symbol


def _resolve(symbol: Union[Symbol, Tensor], i: int = 0) -> Tensor:
    """
    Make sure the layer is compatible with both multi-param and single param requirements。

    What could be the input: list/tuple of sympy.symbol, tf.tensor with 1D or 0D shape
    """
    if isinstance(symbol, list) or isinstance(symbol, tuple):
        return symbol[i]
    elif tf.is_tensor(symbol):  # tf.tensor of 1D or 2D
        if len(symbol.shape) == 1:
            return symbol[i]
        else:  # len(shape) == 0
            return symbol
    else:  # sympy.symbol
        return symbol


def generate_double_gate(gates: str) -> None:
    d1, d2 = gates[0], gates[1]

    def f(
        circuit: Circuit, qubit1: int, qubit2: int, symbol: Union[Tensor, float]
    ) -> Circuit:
        if d1 == "x":
            circuit.H(qubit1)  # type: ignore
        elif d1 == "y":
            circuit.rx(qubit1, theta=num_to_tensor(-np.pi / 2))  # type: ignore
        if d2 == "x":
            circuit.H(qubit2)  # type: ignore
        elif d2 == "y":
            circuit.rx(qubit2, theta=num_to_tensor(-np.pi / 2))  # type: ignore
        circuit.CNOT(qubit1, qubit2)  # type: ignore
        circuit.rz(qubit2, theta=symbol)  # type: ignore
        circuit.CNOT(qubit1, qubit2)  # type: ignore
        if d1 == "x":
            circuit.H(qubit1)  # type: ignore
        elif d1 == "y":
            circuit.rx(qubit1, theta=num_to_tensor(np.pi / 2))  # type: ignore
        if d2 == "x":
            circuit.H(qubit2)  # type: ignore
        elif d2 == "y":
            circuit.rx(qubit2, theta=num_to_tensor(np.pi / 2))  # type: ignore
        return circuit

    f.__doc__ = """%sgate""" % gates
    setattr(thismodule, gates + "gate", f)


def generate_gate_layer(gate: str) -> None:
    r"""
    $$e^{-i\theta \sigma}$$

    :param gate:
    :type gate: str
    :return:
    """

    def f(
        circuit: Circuit, symbol: Union[Tensor, float] = None, g: Graph = None
    ) -> Circuit:  # compatible with graph mode
        symbol0 = _resolve(symbol)
        if gate.lower() in sgates:
            for n in range(circuit._nqubits):
                getattr(circuit, gate)(n)
        else:
            for n in range(circuit._nqubits):
                getattr(circuit, gate)(n, theta=2 * symbol0)
        return circuit

    f.__doc__ = """%slayer""" % gate
    f.__repr__ = """%slayer""" % gate  # type: ignore
    f.__trainable__ = False if gate in sgates else True  # type: ignore
    setattr(thismodule, gate + "layer", f)


def generate_any_gate_layer(gate: str) -> None:
    r"""
    $$e^{-i\theta_i \sigma}$$

    :param gate:
    :type gate: str
    :return:
    """

    def f(
        circuit: Circuit, symbol: Union[Tensor, float] = None, g: Graph = None
    ) -> Circuit:  # compatible with graph mode
        if gate.lower() in sgates:
            for n in range(circuit._nqubits):
                getattr(circuit, gate)(n)
        else:
            for n in range(circuit._nqubits):
                getattr(circuit, gate)(n, theta=2 * symbol[n])  # type: ignore
        return circuit

    f.__doc__ = """any%slayer""" % gate
    f.__repr__ = """any%slayer""" % gate  # type: ignore
    f.__trainable__ = False if gate in sgates else True  # type: ignore
    setattr(thismodule, "any" + gate + "layer", f)


def generate_any_double_gate_layer(gates: str) -> None:
    def f(circuit: Circuit, symbol: Union[Tensor, float], g: Graph = None) -> Circuit:
        if g is None:
            g = nx.complete_graph(circuit._nqubits)
        for i, e in enumerate(g.edges):
            qubit1, qubit2 = e
            getattr(thismodule, gates + "gate")(
                circuit,
                qubit1,
                qubit2,
                -symbol[i] * g[e[0]][e[1]].get("weight", 1.0) * 2,  # type: ignore
            )
            ## should be better as * 2 # e^{-i\theta H}, H=-ZZ
        return circuit

    f.__doc__ = """any%slayer""" % gates
    f.__repr__ = """any%slayer""" % gates  # type: ignore
    f.__trainable__ = True  # type: ignore
    setattr(thismodule, "any" + gates + "layer", f)


def generate_double_gate_layer(gates: str) -> None:
    def f(circuit: Circuit, symbol: Union[Tensor, float], g: Graph = None) -> Circuit:
        symbol0 = _resolve(symbol)
        if g is None:
            g = nx.complete_graph(circuit._nqubits)
        for e in g.edges:
            qubit1, qubit2 = e
            getattr(thismodule, gates + "gate")(
                circuit, qubit1, qubit2, -symbol0 * g[e[0]][e[1]].get("weight", 1.0) * 2
            )  ## should be better as * 2 # e^{-i\theta H}, H=-ZZ
        return circuit

    f.__doc__ = """%slayer""" % gates
    f.__repr__ = """%slayer""" % gates  # type: ignore
    f.__trainable__ = True  # type: ignore
    setattr(thismodule, gates + "layer", f)


def generate_double_gate_layer_bitflip(gates: str) -> None:
    # deprecated, as API are consistent now for DMCircuit and Circuit
    def f(
        circuit: DMCircuit, symbol: Union[Tensor, float], g: Graph, *params: float
    ) -> DMCircuit:
        symbol0 = _resolve(symbol)
        for e in g.edges:
            qubit1, qubit2 = e
            getattr(thismodule, gates + "gate")(
                circuit,
                qubit1,
                qubit2,
                -symbol0 * g[e[0]][e[1]].get("weight", 1.0) * 2,
            )  ## should be better as * 2 # e^{-i\theta H}, H=-ZZ
            circuit.apply_general_kraus(
                depolarizingchannel(params[0], params[1], params[2]), [(e[0],)]
            )
            circuit.apply_general_kraus(
                depolarizingchannel(params[0], params[1], params[2]), [(e[1],)]
            )
        return circuit

    f.__doc__ = """%slayer_bitflip""" % gates
    f.__repr__ = """%slayer_bitflip""" % gates  # type: ignore
    f.__trainable__ = True  # type: ignore
    setattr(thismodule, gates + "layer_bitflip", f)


def generate_double_gate_layer_bitflip_mc(gates: str) -> None:
    def f(
        circuit: Circuit, symbol: Union[Tensor, float], g: Graph, *params: float
    ) -> Circuit:
        symbol0 = _resolve(symbol)
        for e in g.edges:
            qubit1, qubit2 = e
            getattr(thismodule, gates + "gate")(
                circuit,
                qubit1,
                qubit2,
                -symbol0 * g[e[0]][e[1]].get("weight", 1.0) * 2,
            )  ## should be better as * 2 # e^{-i\theta H}, H=-ZZ
            circuit.depolarizing(e[0], px=params[0], py=params[1], pz=params[2])  # type: ignore
            circuit.depolarizing(e[1], px=params[0], py=params[1], pz=params[2])  # type: ignore
        return circuit

    f.__doc__ = """%slayer_bitflip_mc""" % gates
    f.__repr__ = """%slayer_bitflip_mc""" % gates  # type: ignore
    f.__trainable__ = True  # type: ignore
    setattr(thismodule, gates + "layer_bitflip_mc", f)


def generate_any_double_gate_layer_bitflip_mc(gates: str) -> None:
    def f(
        circuit: Circuit, symbol: Union[Tensor, float], g: Graph = None, *params: float
    ) -> Circuit:
        if g is None:
            g = nx.complete_graph(circuit._nqubits)
        for i, e in enumerate(g.edges):
            qubit1, qubit2 = e
            getattr(thismodule, gates + "gate")(
                circuit,
                qubit1,
                qubit2,
                -symbol[i] * g[e[0]][e[1]].get("weight", 1.0) * 2,  # type: ignore
            )
            ## should be better as * 2 # e^{-i\theta H}, H=-ZZ
            circuit.depolarizing(e[0], px=params[0], py=params[1], pz=params[2])  # type: ignore
            circuit.depolarizing(e[1], px=params[0], py=params[1], pz=params[2])  # type: ignore
        return circuit

    f.__doc__ = """any%slayer_bitflip_mc""" % gates
    f.__repr__ = """any%slayer_bitflip_mc""" % gates  # type: ignore
    f.__trainable__ = True  # type: ignore
    setattr(thismodule, "any" + gates + "layer_bitflip_mc", f)


def generate_double_layer_block(gates: Tuple[str]) -> None:
    d1, d2 = gates[0], gates[1]  # type: ignore

    def f(circuit: Circuit, symbol: Tensor, g: Graph = None) -> Circuit:
        if g is None:
            g = nx.complete_graph(circuit._nqubits)
        getattr(thismodule, d1 + "layer")(circuit, symbol[0], g)
        getattr(thismodule, d2 + "layer")(circuit, symbol[1], g)
        return circuit

    f.__doc__ = """%s_%s_block""" % (d1, d2)
    f.__repr__ = """%s_%s_block""" % (d1, d2)  # type: ignore
    f.__trainable__ = False if (d1 in sgates) and (d2 in sgates) else True  # type: ignore
    setattr(thismodule, "%s_%s_block" % (d1, d2), f)


def anyswaplayer(circuit: Circuit, symbol: Tensor, g: Graph) -> Circuit:
    for i, e in enumerate(g.edges):
        qubit1, qubit2 = e
        circuit.exp1(  # type: ignore
            qubit1,
            qubit2,
            unitary=array_to_tensor(_swap_matrix),
            theta=symbol[i] * g[e[0]][e[1]].get("weight", 1.0),
        )

    return circuit


def anyswaplayer_bitflip_mc(
    circuit: Circuit, symbol: Tensor, g: Graph, px: float, py: float, pz: float
) -> Circuit:
    for i, e in enumerate(g.edges):
        qubit1, qubit2 = e
        circuit.exp1(  # type: ignore
            qubit1,
            qubit2,
            unitary=array_to_tensor(_swap_matrix),
            theta=symbol[i] * g[e[0]][e[1]].get("weight", 1.0),
        )
        circuit.depolarizing(e[0], px=px, py=py, pz=pz)  # type: ignore
        circuit.depolarizing(e[1], px=px, py=py, pz=pz)  # type: ignore
    return circuit


for gate in ["rx", "ry", "rz", "H", "I"]:
    generate_gate_layer(gate)
    generate_any_gate_layer(gate)

for gates in itertools.product(*[["x", "y", "z"] for _ in range(2)]):
    gates = gates[0] + gates[1]
    generate_double_gate(gates)  # type: ignore
    generate_double_gate_layer(gates)  # type: ignore
    generate_any_double_gate_layer(gates)  # type: ignore
    generate_double_gate_layer_bitflip(gates)  # type: ignore
    generate_double_gate_layer_bitflip_mc(gates)  # type: ignore
    generate_any_double_gate_layer_bitflip_mc(gates)  # type: ignore


for gates in itertools.product(
    *[["rx", "ry", "rz", "xx", "yy", "zz"] for _ in range(2)]
):
    generate_double_layer_block(gates)  # type: ignore


def bitfliplayer(ci: DMCircuit, g: Graph, px: float, py: float, pz: float) -> None:
    n = len(g.nodes)
    for i in range(n):
        ci.apply_general_kraus(depolarizingchannel(px, py, pz), [(i,)])
    bitfliplayer.__repr__ = """bitfliplayer"""  # type: ignore
    bitfliplayer.__trainable__ = True  # type: ignore


def bitfliplayer_mc(ci: Circuit, g: Graph, px: float, py: float, pz: float) -> None:
    n = len(g.nodes)
    for i in range(n):
        ci.depolarizing(i, px=px, py=py, pz=pz)  # type: ignore
    bitfliplayer.__repr__ = """bitfliplayer_mc"""  # type: ignore
    bitfliplayer.__trainable__ = True  # type: ignore


# TODO(@refraction-ray): should move and refactor the above functions to templates/layers.

## below is similar layer but in cirq API instead of tensrocircuit native API
## special notes to the API, the arguments order are different due to historical reason compared to tc layers API
## and we have no attention to further maintain the cirq codebase below, availability is not guaranteend


def generate_qubits(g: Graph) -> List[Any]:
    return sorted([v for _, v in g.nodes.data("qubit")])


try:
    basis_rotation = {
        "x": (cirq.H, cirq.H),
        "y": (cirq.rx(-np.pi / 2), cirq.rx(np.pi / 2)),
        "z": None,
    }

    def generate_cirq_double_gate(gates: str) -> None:
        d1, d2 = gates[0], gates[1]
        r1, r2 = basis_rotation[d1], basis_rotation[d2]

        def f(
            circuit: cirq.Circuit,
            qubit1: cirq.GridQubit,
            qubit2: cirq.GridQubit,
            symbol: Symbol,
        ) -> cirq.Circuit:
            if r1 is not None:
                circuit.append(r1[0](qubit1))
            if r2 is not None:
                circuit.append(r2[0](qubit2))
            circuit.append(cirq.CNOT(qubit1, qubit2))
            circuit.append(cirq.rz(symbol)(qubit2))
            circuit.append(cirq.CNOT(qubit1, qubit2))
            if r1 is not None:
                circuit.append(r1[1](qubit1))
            if r2 is not None:
                circuit.append(r2[1](qubit2))
            return circuit

        f.__doc__ = """%sgate""" % gates
        setattr(thismodule, "cirq" + gates + "gate", f)

    def cirqswapgate(
        circuit: cirq.Circuit,
        qubit1: cirq.GridQubit,
        qubit2: cirq.GridQubit,
        symbol: Symbol,
    ) -> cirq.Circuit:
        circuit.append(cirq.SwapPowGate(exponent=symbol)(qubit1, qubit2))
        return circuit

    def cirqcnotgate(
        circuit: cirq.Circuit,
        qubit1: cirq.GridQubit,
        qubit2: cirq.GridQubit,
        symbol: Symbol,
    ) -> cirq.Circuit:
        circuit.append(cirq.CNOT(qubit1, qubit2))
        return circuit

    def generate_cirq_gate_layer(gate: str) -> None:
        r"""
        $$e^{-i\theta \sigma}$$

        :param gate:
        :type gate: str
        :return:
        """

        def f(
            circuit: cirq.Circuit,
            g: Graph,
            symbol: Symbol,
            qubits: Optional[Sequence[Any]] = None,
        ) -> cirq.Circuit:
            symbol0 = _resolve(symbol[0])
            if not qubits:
                qubits = generate_qubits(g)
            rotation = getattr(cirq, gate)
            if isinstance(rotation, cirq.Gate):
                circuit.append(rotation.on_each(qubits))
            else:  # function
                circuit.append(rotation(2.0 * symbol0).on_each(qubits))
            return circuit

        f.__doc__ = """%slayer""" % gate
        f.__repr__ = """%slayer""" % gate  # type: ignore
        f.__trainable__ = False if isinstance(getattr(cirq, gate), cirq.Gate) else True  # type: ignore
        setattr(thismodule, "cirq" + gate + "layer", f)

    def generate_cirq_any_gate_layer(gate: str) -> None:
        r"""
        $$e^{-i\theta \sigma}$$

        :param gate:
        :type gate: str
        :return:
        """

        def f(
            circuit: cirq.Circuit,
            g: Graph,
            symbol: Symbol,
            qubits: Optional[Sequence[Any]] = None,
        ) -> cirq.Circuit:
            if not qubits:
                qubits = generate_qubits(g)
            rotation = getattr(cirq, gate)
            for i, q in enumerate(qubits):
                circuit.append(rotation(2.0 * symbol[i])(q))
            return circuit

        f.__doc__ = """any%slayer""" % gate
        f.__repr__ = """any%slayer""" % gate  # type: ignore
        f.__trainable__ = True  # type: ignore
        setattr(thismodule, "cirqany" + gate + "layer", f)

    def generate_cirq_double_gate_layer(gates: str) -> None:
        def f(
            circuit: cirq.Circuit,
            g: Graph,
            symbol: Symbol,
            qubits: Optional[Sequence[Any]] = None,
        ) -> cirq.Circuit:
            symbol0 = _resolve(symbol)
            for e in g.edges:
                qubit1 = g.nodes[e[0]]["qubit"]
                qubit2 = g.nodes[e[1]]["qubit"]
                getattr(thismodule, "cirq" + gates + "gate")(
                    circuit, qubit1, qubit2, -symbol0 * g[e[0]][e[1]]["weight"] * 2
                )  ## should be better as * 2 # e^{-i\theta H}, H=-ZZ
            return circuit

        f.__doc__ = """%slayer""" % gates
        f.__repr__ = """%slayer""" % gates  # type: ignore
        f.__trainable__ = True  # type: ignore
        setattr(thismodule, "cirq" + gates + "layer", f)

    def generate_cirq_any_double_gate_layer(gates: str) -> None:
        """
        The following function should be used to generate layers with special case.
        As its soundness depends on the nature of the task or problem, it doesn't always make sense.

        :param gates:
        :type gates: str
        :return:
        """

        def f(
            circuit: cirq.Circuit,
            g: Graph,
            symbol: Symbol,
            qubits: Optional[Sequence[Any]] = None,
        ) -> cirq.Circuit:
            for i, e in enumerate(g.edges):
                qubit1 = g.nodes[e[0]]["qubit"]
                qubit2 = g.nodes[e[1]]["qubit"]
                getattr(thismodule, "cirq" + gates + "gate")(
                    circuit, qubit1, qubit2, -symbol[i] * g[e[0]][e[1]]["weight"] * 2
                )  ## should be better as * 2 # e^{-i\theta H}, H=-ZZ
            return circuit

        f.__doc__ = """any%slayer""" % gates
        f.__repr__ = """any%slayer""" % gates  # type: ignore
        f.__trainable__ = True  # type: ignore
        setattr(thismodule, "cirqany" + gates + "layer", f)

    for gate in ["rx", "ry", "rz", "H"]:
        generate_cirq_gate_layer(gate)
        if gate != "H":
            generate_cirq_any_gate_layer(gate)

    for gates in itertools.product(*[["x", "y", "z"] for _ in range(2)]):
        gates = gates[0] + gates[1]
        generate_cirq_double_gate(gates)  # type: ignore
        generate_cirq_double_gate_layer(gates)  # type: ignore
        generate_cirq_any_double_gate_layer(gates)  # type: ignore

    generate_cirq_double_gate_layer("swap")
    generate_cirq_any_double_gate_layer("swap")
    generate_cirq_double_gate_layer("cnot")

except NameError as e:
    logger.warning(e)
    logger.warning("cirq layer generation disabled")
