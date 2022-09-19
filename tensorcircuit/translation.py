"""
Circuit object translation in different packages
"""

from typing import Any, Dict, List, Optional
from copy import deepcopy
import logging
import numpy as np

logger = logging.getLogger(__name__)


try:
    from qiskit import QuantumCircuit
    import qiskit.quantum_info as qi
    from qiskit.extensions.exceptions import ExtensionError
except ImportError:
    logger.warning(
        "Please first ``pip install qiskit`` to enable related functionality"
    )

from . import gates
from .circuit import Circuit
from .densitymatrix import DMCircuit2
from .cons import backend
from .interfaces.tensortrans import tensor_to_numpy


Tensor = Any


def perm_matrix(n: int) -> Tensor:
    r"""
    Generate a permutation matrix P.
    Due to the different convention or qubits' order in qiskit and tensorcircuit,
    the unitary represented by the same circuit is different. They are related
    by this permutation matrix P:
    P @ U_qiskit @ P = U_tc

    :param n: # of qubits
    :type n: int
    :return: The permutation matrix P
    :rtype: Tensor
    """
    p_mat = np.zeros([2**n, 2**n])
    for i in range(2**n):
        bit = i
        revs_i = 0
        for j in range(n):
            if bit & 0b1:
                revs_i += 1 << (n - j - 1)
            bit = bit >> 1
        p_mat[i, revs_i] = 1
    return p_mat


def qir2qiskit(qir: List[Dict[str, Any]], n: int) -> Any:
    r"""
    Generate a qiskit quantum circuit using the quantum intermediate
    representation (qir) in tensorcircuit.

    :Example:

    >>> c = tc.Circuit(2)
    >>> c.H(1)
    >>> c.X(1)
    >>> qisc = tc.translation.qir2qiskit(c.to_qir(), 2)
    >>> qisc.data
    [(Instruction(name='h', num_qubits=1, num_clbits=0, params=[]), [Qubit(QuantumRegister(2, 'q'), 1)], []),
     (Instruction(name='x', num_qubits=1, num_clbits=0, params=[]), [Qubit(QuantumRegister(2, 'q'), 1)], [])]

    :param qir: The quantum intermediate representation of a circuit.
    :type qir: List[Dict[str, Any]]
    :param n: # of qubits
    :type n: int
    :return: qiskit QuantumCircuit object
    :rtype: Any
    """
    qiskit_circ = QuantumCircuit(n)
    for gate_info in qir:
        index = gate_info["index"]
        gate_name = str(gate_info["gatef"])
        qis_name = gate_name
        if gate_name in ["exp", "exp1", "any", "multicontrol"]:
            qis_name = gate_info["name"]
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
                    backend.numpy(gate_info["gatef"](**parameters).tensor),
                    [2 ** len(index), 2 ** len(index)],
                )
            )
            qiskit_circ.unitary(wroot_op, index, label=qis_name)
        elif gate_name in ["rx", "ry", "rz", "crx", "cry", "crz"]:
            getattr(qiskit_circ, gate_name)(
                np.real(
                    backend.numpy(gates.array_to_tensor(parameters["theta"]))
                ).item(),
                *index
            )
        elif gate_name in ["orx", "ory", "orz"]:
            getattr(qiskit_circ, "c" + gate_name[1:])(
                np.real(
                    backend.numpy(gates.array_to_tensor(parameters["theta"]))
                ).item(),
                *index,
                ctrl_state=0
            )
        elif gate_name in ["exp", "exp1"]:
            unitary = backend.numpy(backend.convert_to_tensor(parameters["unitary"]))
            theta = backend.numpy(backend.convert_to_tensor(parameters["theta"]))
            theta = np.array(np.real(theta)).astype(np.float64)
            # cast theta to real, since qiskit only support unitary.
            # Error can be presented if theta is actually complex in this procedure.
            exp_op = qi.Operator(unitary)
            index_reversed = [x for x in index[::-1]]
            qiskit_circ.hamiltonian(
                exp_op, time=theta, qubits=index_reversed, label=qis_name
            )
        elif gate_name in ["mpo", "multicontrol"]:
            qop = qi.Operator(
                np.reshape(
                    backend.numpy(gate_info["gatef"](**parameters).eval_matrix()),
                    [2 ** len(index), 2 ** len(index)],
                )
            )
            qiskit_circ.unitary(qop, index[::-1], label=qis_name)
        # TODO(@refraction-ray): support for phase gate and U, cU gate for the circuit translation
        else:  # r cr any gate
            qop = qi.Operator(
                np.reshape(
                    backend.numpy(gate_info["gatef"](**parameters).tensor),
                    [2 ** len(index), 2 ** len(index)],
                )
            )
            try:
                qiskit_circ.unitary(qop, index[::-1], label=qis_name)
            except ExtensionError:
                logger.warning(
                    "omit non unitary gate in tensorcircuit when transforming to qiskit"
                )
                qiskit_circ.unitary(
                    np.eye(2 ** len(index)), index[::-1], label=qis_name
                )

    return qiskit_circ


def ctrl_str2ctrl_state(ctrl_str: str, nctrl: int) -> List[int]:
    ctrl_state_bin = int(ctrl_str)
    return [0x1 & (ctrl_state_bin >> (i)) for i in range(nctrl)]


def qiskit2tc(
    qcdata: List[List[Any]],
    n: int,
    inputs: Optional[List[float]] = None,
    is_dm: bool = False,
) -> Any:
    r"""
    Generate a tensorcircuit circuit using the quantum circuit data in qiskit.

    :Example:

    >>> qisc = QuantumCircuit(2)
    >>> qisc.h(0)
    >>> qisc.x(1)
    >>> qc = tc.translation.qiskit2tc(qisc.data, 2)
    >>> qc.to_qir()[0]['gatef']
    h

    :param qcdata: Quantum circuit data from qiskit.
    :type qcdata: List[List[Any]]
    :param n: # of qubits
    :type n: int
    :param inputs: Input state of the circuit. Default is None.
    :type inputs: Optional[List[float]]
    :return: A quantum circuit in tensorcircuit
    :rtype: Any
    """
    if is_dm:
        Circ = DMCircuit2
    else:
        Circ = Circuit  # type: ignore
    if inputs is None:
        tc_circuit: Any = Circ(n)
    else:
        tc_circuit = Circ(n, inputs=inputs)
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
                # print(idx)
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


def tensor_to_json(a: Any) -> Any:
    if (
        isinstance(a, float)
        or isinstance(a, int)
        or isinstance(a, complex)
        or isinstance(a, list)
        or isinstance(a, tuple)
    ):
        a = np.array(a)
    a = tensor_to_numpy(a)

    ar = np.real(a)
    ai = np.imag(a)
    return [ar.tolist(), ai.tolist()]


def json_to_tensor(a: Any) -> Any:
    ar = np.array(a[0])
    ai = np.array(a[1])
    return ar + 1.0j * ai


def qir2json(
    qir: List[Dict[str, Any]], simplified: bool = False
) -> List[Dict[str, Any]]:
    """
    transform qir to json compatible list of dict where array is replaced by real and imaginary list

    :param qir: _description_
    :type qir: List[Dict[str, Any]]
    :param simplified: If False, keep all info for each gate, defaults to be False.
        If True, suitable for IO since less information is required
    :type simplified: bool
    :return: _description_
    :rtype: List[Dict[str, Any]]
    """
    logger.warning(
        "experimental feature subject to fast protocol and implementation change, try on your own risk"
    )
    qir = deepcopy(qir)
    tcqasm = []
    for r in qir:
        if r["mpo"] is True:
            nm = backend.reshapem(r["gate"].eval())
        else:
            nm = backend.reshapem(r["gate"].tensor)
        nmr, nmi = tensor_to_json(nm)
        if backend.shape_tuple(nm)[0] == backend.shape_tuple(nm)[1] == 2:
            uparams = list(gates.get_u_parameter(backend.numpy(nm)))
        else:
            uparams = []
        params = r.get("parameters", {})
        for k, v in params.items():
            params[k] = tensor_to_json(v)

        ditem = {
            "name": r["gatef"].n,
            "qubits": list(r["index"]),
        }
        unsupport_list = ["any", "unitary", "mpo", "exp", "exp1", "r", "cr"]
        if not simplified:
            ditem.update(
                {
                    "matrix": [nmr, nmi],
                    "uparams": uparams,
                    "parameters": params,
                    "mpo": r["mpo"],
                }
            )
        else:  # update only when necessary: simplified=True
            if ditem["name"] in unsupport_list and uparams:
                ditem.update({"uparams": uparams})
            elif ditem["name"] in unsupport_list:
                ditem.update({"matrix": [nmr, nmi]})
            if params:
                ditem.update({"parameters": params})
            if r["mpo"]:
                ditem.update({"mpo": r["mpo"]})
        tcqasm.append(ditem)
    return tcqasm


def json2qir(tcqasm: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.warning(
        "experimental feature subject to fast protocol and implementation change, try on your own risk"
    )
    tcqasm = deepcopy(tcqasm)
    qir = []
    for d in tcqasm:
        param = d.get("parameters", {})
        for k, v in param.items():
            # tree_map doesn't work here since the end leaves are list
            param[k] = json_to_tensor(v)
        if "matrix" in d:
            gatem = json_to_tensor(d["matrix"])
        else:
            gatem = getattr(gates, d["name"])(**param)
        qir.append(
            {
                "gate": gatem,
                "index": tuple(d["qubits"]),
                "mpo": d.get("mpo", False),
                "split": {},
                "parameters": param,
                "gatef": getattr(gates, d["name"]),
                "name": d["name"],
            }
        )
    return qir
