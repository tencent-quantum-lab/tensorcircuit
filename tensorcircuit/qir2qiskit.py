from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
import numpy as np


def qir2qiskit(qir, n):
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
        elif gate_name == "exp":
            unitary = parameters["unitary"]
            theta = parameters["theta"]
            exp_op = qi.Operator(unitary)
            index_reversed = [x for x in index[::-1]]
            qiskit_circ.hamiltonian(
                exp_op, time=theta, qubits=index_reversed, label=qis_name
            )
        elif gate_name == "exp1":
            unitary = parameters["unitary"]
            theta = parameters["theta"]
            exp_op = qi.Operator(
                np.cos(theta) * np.eye(2 ** len(index)) - 1.0j * np.sin(theta) * unitary
            )
            qiskit_circ.unitary(exp_op, index[::-1], label=qis_name)
        elif gate_name == "multicontrol":
            qop = qi.Operator(
                np.reshape(
                    gate_info["gatef"](**parameters).eval_matrix(),
                    [2 ** len(index), 2 ** len(index)],
                )
            )
            qiskit_circ.unitary(qop, index[::-1], label=qis_name)
        elif gate_name == "mpo":
            print("mpo gate not support now")   # gate_info["gatef"](**parameters).eval_matrix() fails
        else:  # r cr any gate
            qop = qi.Operator(
                np.reshape(
                    gate_info["gatef"](**parameters).tensor,
                    [2 ** len(index), 2 ** len(index)],
                )
            )
            qiskit_circ.unitary(qop, index[::-1], label=qis_name)
    return qiskit_circ
