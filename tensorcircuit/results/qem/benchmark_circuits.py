"""
circuits for quantum chip benchmark
"""

from typing import Any, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

import networkx as nx

try:
    from mitiq.benchmarks import (
        generate_ghz_circuit,
        generate_mirror_circuit,
        generate_rb_circuits,
        generate_w_circuit,
    )
    from mitiq.interface import convert_from_mitiq
except ModuleNotFoundError:
    logger.warning("mitiq is not installed, please ``pip install mitiq`` first")


from ... import Circuit


def ghz_circuit(num_qubits: int) -> Tuple[Any, Dict[str, float]]:
    cirq = generate_ghz_circuit(num_qubits)
    ideal = {"0" * num_qubits: 0.5, "1" * num_qubits: 0.5}
    qisc = convert_from_mitiq(cirq, "qiskit")
    circuit = Circuit.from_qiskit(qisc, qisc.num_qubits)
    return circuit, ideal


def w_circuit(num_qubits: int) -> Tuple[Any, Dict[str, float]]:
    # Efficient quantum algorithms for GHZ and W states https://arxiv.org/abs/1807.05572
    # Werner-state with linear complexity  {'1000': 0.25, '0100': 0.25, '0010': 0.25, '0001': 0.25}
    cirq = generate_w_circuit(num_qubits)

    ideal = {}
    for i in range(num_qubits):
        bitstring = "0" * i + "1" + "0" * (num_qubits - i - 1)
        ideal[bitstring] = 1 / num_qubits

    qisc = convert_from_mitiq(cirq, "qiskit")
    circuit = Circuit.from_qiskit(qisc, qisc.num_qubits)
    return circuit, ideal


def rb_circuit(num_qubits: int, depth: int) -> Tuple[Any, Dict[str, float]]:
    # num_qubits limited to 1 or 2
    cirq = generate_rb_circuits(num_qubits, depth)[0]
    ideal = {"0" * num_qubits: 1.0}
    qisc = convert_from_mitiq(cirq, "qiskit")
    circuit = Circuit.from_qiskit(qisc, qisc.num_qubits)
    return circuit, ideal


def mirror_circuit(
    depth: int,
    two_qubit_gate_prob: float,
    connectivity_graph: nx.Graph,
    seed: int,
    two_qubit_gate_name: str = "CNOT",
) -> Tuple[Any, Dict[str, float]]:
    # Measuring the Capabilities of Quantum Computers https://arxiv.org/pdf/2008.11294.pdf
    cirq, bitstring_list = generate_mirror_circuit(
        nlayers=depth,
        two_qubit_gate_prob=two_qubit_gate_prob,
        connectivity_graph=connectivity_graph,
        two_qubit_gate_name=two_qubit_gate_name,
        seed=seed,
    )
    ideal_bitstring = "".join(map(str, bitstring_list))
    ideal = {ideal_bitstring: 1.0}
    qisc = convert_from_mitiq(cirq, "qiskit")
    circuit = Circuit.from_qiskit(qisc, qisc.num_qubits)
    return circuit, ideal


def QAOA_circuit(
    graph: List[Tuple[int]], weight: List[float], params: List[float]
) -> Any:  # internal API don't use
    nlayers = len(params)

    qlist = []
    for i in range(len(graph)):
        for j in range(2):
            qlist.append(graph[i][j])
    qlist = list(set(qlist))

    nqubit = max(qlist) + 1

    c = Circuit(nqubit)
    for i in qlist:
        c.h(i)  # type: ignore

    for i in range(nlayers):
        for e in range(len(graph)):
            c.cnot(graph[e][0], graph[e][1])  # type: ignore
            c.rz(graph[e][1], theta=params[i, 0] * weight[e])  # type: ignore
            c.cnot(graph[e][0], graph[e][1])  # type: ignore

        for k in qlist:
            c.rx(k, theta=params[i, 1] * 2)  # type: ignore

    return c
