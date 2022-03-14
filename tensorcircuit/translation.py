"""
Circuit object translation in different packages
"""

from typing import Any, Dict, List, Optional

import numpy as np
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi

from . import gates
from .circuit import Circuit

Tensor = Any


def perm_matrix(n: int) -> Tensor:
    p_mat = np.zeros([2 ** n, 2 ** n])
    for i in range(2 ** n):
        bit = i
        revs_i = 0
        for j in range(n):
            if bit & 0b1:
                revs_i += 1 << (n - j - 1)
            bit = bit >> 1
        p_mat[i, revs_i] = 1
    return p_mat


def qir2qiskit(qir: List[Dict[str, Any]], n: int) -> Any:
    qiskit_circ = QuantumCircuit(n)
    for gate_info in qir:
        index = gate_info["index"]
        gate_name = str(gate_info["gatef"])
        qis_name = gate_name
        if gate_name in ["exp", "exp1", "any", "multicontrol"]:
            if gate_info["name"] != gate_name:
                qis_name += "_" + gate_info["name"]
        parameters = {}
        if "parameters" in gate_info:
            parameters = gate_info["parameters"]
        if gate_name in [
            "h",
            "x",
            "y",
            "z",
            "i",
            "s",
            "t",
            "swap",
            "iswap",
            "cnot",
            "toffoli",
            "fredkin",
            "cnot",
            "cx",
            "cy",
            "cz",
        ]:
            getattr(qiskit_circ, gate_name)(*index)
        elif gate_name in ["sd", "td"]:
            getattr(qiskit_circ, gate_name + "g")(*index)
        elif gate_name in ["ox", "oy", "oz"]:
            getattr(qiskit_circ, "c" + gate_name[1:])(*index, ctrl_state=0)
        elif gate_name == "wroot":
            wroot_op = qi.Operator(
                np.reshape(
                    gate_info["gatef"](**parameters).tensor,
                    [2 ** len(index), 2 ** len(index)],
                )
            )
            qiskit_circ.unitary(wroot_op, index, label=qis_name)
        elif gate_name in ["rx", "ry", "rz", "crx", "cry", "crz"]:
            getattr(qiskit_circ, gate_name)(parameters["theta"], *index)
        elif gate_name in ["orx", "ory", "orz"]:
            getattr(qiskit_circ, "c" + gate_name[1:])(
                parameters["theta"], *index, ctrl_state=0
            )
        elif gate_name in ["exp", "exp1"]:
            unitary = parameters["unitary"]
            theta = parameters["theta"]
            exp_op = qi.Operator(unitary)
            index_reversed = [x for x in index[::-1]]
            qiskit_circ.hamiltonian(
                exp_op, time=theta, qubits=index_reversed, label=qis_name
            )
        elif gate_name in ["mpo", "multicontrol"]:
            qop = qi.Operator(
                np.reshape(
                    gate_info["gatef"](**parameters).eval_matrix(),
                    [2 ** len(index), 2 ** len(index)],
                )
            )
            qiskit_circ.unitary(qop, index[::-1], label=qis_name)
        else:  # r cr any gate
            qop = qi.Operator(
                np.reshape(
                    gate_info["gatef"](**parameters).tensor,
                    [2 ** len(index), 2 ** len(index)],
                )
            )
            qiskit_circ.unitary(qop, index[::-1], label=qis_name)
    return qiskit_circ


def ctrl_str2ctrl_state(ctrl_str: str, nctrl: int) -> List[int]:
    ctrl_state_bin = int(ctrl_str)
    return [0x1 & (ctrl_state_bin >> (i)) for i in range(nctrl)]


def qiskit2tc(
    qcdata: List[List[Any]], n: int, inputs: Optional[List[float]] = None
) -> Any:
    if inputs is None:
        tc_circuit: Any = Circuit(n)
    else:
        tc_circuit = Circuit(n, inputs=inputs)
    for gate_info in qcdata:
        idx = [qb.index for qb in gate_info[1]]
        gate_name = gate_info[0].name
        parameters = gate_info[0].params
        if gate_name in [
            "h",
            "x",
            "y",
            "z",
            "i",
            "s",
            "t",
            "swap",
            "iswap",
            "cnot",
            "toffoli",
            "fredkin",
            "cx",
            "cy",
            "cz",
        ]:
            getattr(tc_circuit, gate_name)(*idx)
        elif gate_name in ["sdg", "tdg"]:
            getattr(tc_circuit, gate_name[:-1])(*idx)
        elif gate_name in ["cx_o0", "cy_o0", "cz_o0"]:
            getattr(tc_circuit, "o" + gate_name[1])(*idx)
        elif gate_name in ["rx", "ry", "rz", "crx", "cry", "crz"]:
            getattr(tc_circuit, gate_name)(*idx, theta=parameters)
        elif gate_name in ["crx_o0", "cry_o0", "crz_o0"]:
            getattr(tc_circuit, "o" + gate_name[1:-3])(*idx, theta=parameters)
        elif gate_name == "hamiltonian":
            tc_circuit.exp(*idx, theta=parameters[-1], unitary=parameters[0])
        elif gate_name == "cswap":
            tc_circuit.fredkin(*idx)
        elif gate_name[:3] == "ccx":
            if gate_name[3:] == "":
                tc_circuit.toffoli(*idx)
            else:
                ctrl_state = ctrl_str2ctrl_state(gate_name[5:], len(idx) - 1)
                tc_circuit.multicontrol(
                    *idx, ctrl=ctrl_state, unitary=gates._x_matrix, name="x"
                )
        elif gate_name[:3] == "mcx":
            if gate_name[3:] == "":
                tc_circuit.multicontrol(
                    *idx, ctrl=[1] * (len(idx) - 1), unitary=gates._x_matrix, name="x"
                )
            else:
                ctrl_state = ctrl_str2ctrl_state(gate_name[5:], len(idx) - 1)
                tc_circuit.multicontrol(
                    *idx, ctrl=ctrl_state, unitary=gates._x_matrix, name="x"
                )
        elif gate_name[0] == "c":
            for i in range(1, len(gate_name)):
                if (gate_name[-i] == "o") & (gate_name[-i - 1] == "_"):
                    break
            if i == len(gate_name) - 1:
                base_gate = gate_info[0].base_gate
                ctrl_state = [1] * base_gate.num_qubits
                idx = idx[: -base_gate.num_qubits] + idx[-base_gate.num_qubits :][::-1]
                print(idx)
                tc_circuit.multicontrol(
                    *idx, ctrl=ctrl_state, unitary=base_gate.to_matrix()
                )
            else:
                base_gate = gate_info[0].base_gate
                ctrl_state = ctrl_str2ctrl_state(
                    gate_name[-i + 1 :], len(idx) - base_gate.num_qubits
                )
                idx = idx[: -base_gate.num_qubits] + idx[-base_gate.num_qubits :][::-1]
                tc_circuit.multicontrol(
                    *idx, ctrl=ctrl_state, unitary=base_gate.to_matrix()
                )
        else:  # unitary gate
            idx_inverse = (x for x in idx[::-1])
            tc_circuit.any(*idx_inverse, unitary=gate_info[0].to_matrix())
    return tc_circuit
