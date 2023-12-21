"""
Circuit object translation in different packages
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
from copy import deepcopy
import logging
import numpy as np

logger = logging.getLogger(__name__)


try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate
    from qiskit.extensions import UnitaryGate
    import qiskit.quantum_info as qi
    from qiskit.extensions.exceptions import ExtensionError
    from qiskit.circuit.quantumcircuitdata import CircuitInstruction
    from qiskit.circuit.parametervector import ParameterVectorElement
    from qiskit.circuit import Parameter, ParameterExpression
except ImportError:
    logger.warning(
        "Please first ``pip install -U qiskit`` to enable related functionality in translation module"
    )
    CircuitInstruction = Any

try:
    import cirq
except ImportError:
    logger.warning(
        "Please first ``pip install -U cirq`` to enable related functionality in translation module"
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


def _get_float(parameters: Any, key: str, default: int = 0) -> float:
    return np.real(backend.numpy(gates.array_to_tensor(parameters.get(key, default)))).item()  # type: ignore


def _merge_extra_qir(
    qir: List[Dict[str, Any]], extra_qir: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    nqir = []
    # caution on the same pos!
    inds: Dict[int, List[Dict[str, Any]]] = {}
    for d in extra_qir:
        if d["pos"] not in inds:
            inds[d["pos"]] = []
        inds[d["pos"]].append(d)
    for i, q in enumerate(qir):
        if i in inds:
            nqir += inds[i]
        nqir.append(q)
    for k in inds:
        if k >= len(qir):
            nqir += inds[k]
    return nqir


def qir2cirq(
    qir: List[Dict[str, Any]], n: int, extra_qir: Optional[List[Dict[str, Any]]] = None
) -> Any:
    r"""
    Generate a cirq circuit using the quantum intermediate
    representation (qir) in tensorcircuit.

    :Example:

    >>> c = tc.Circuit(2)
    >>> c.H(1)
    >>> c.X(1)
    >>> cisc = tc.translation.qir2cirq(c.to_qir(), 2)
    >>> print(cisc)
    1: ───H───X───

    :param qir: The quantum intermediate representation of a circuit.
    :type qir: List[Dict[str, Any]]
    :param n: # of qubits
    :type n: int
    :param extra_qir: The extra quantum IR of tc circuit including measure and reset on hardware,
        defaults to None
    :type extra_qir: Optional[List[Dict[str, Any]]]
    :return: qiskit cirq object
    :rtype: Any

    #TODO(@erertertet):
    add default theta to iswap gate
    add more cirq built-in gate instead of customized
    add unitary test with tolerance
    add support of cirq built-in ControlledGate for multiplecontroll
    support more element in qir, e.g. barrier, measure...
    """

    class CustomizedCirqGate(cirq.Gate):  # type: ignore
        def __init__(self, uMatrix: Any, name: str, nqubit: int):
            super(CustomizedCirqGate, self)
            self.uMatrix = uMatrix
            self.name = name
            self.nqubit = nqubit

        def _num_qubits_(self) -> int:
            return self.nqubit

        def _unitary_(self) -> Any:
            return self.uMatrix

        def _circuit_diagram_info_(
            self, *args: Optional[cirq.CircuitDiagramInfoArgs]
        ) -> List[str]:
            return [self.name] * self.nqubit

    if extra_qir is not None and len(extra_qir) > 0:
        qir = _merge_extra_qir(qir, extra_qir)
    qbits = cirq.LineQubit.range(n)
    cmd = []
    for gate_info in qir:
        index = [qbits[i] for i in gate_info["index"]]
        gate_name = str(gate_info["gatef"])
        if "parameters" in gate_info:
            parameters = gate_info["parameters"]
        if gate_name in [
            "h",
            "i",
            "x",
            "y",
            "z",
            "s",
            "t",
            "fredkin",
            "toffoli",
            "cnot",
            "swap",
        ]:
            cmd.append(getattr(cirq, gate_name.upper())(*index))
        elif gate_name in ["rx", "ry", "rz"]:
            cmd.append(
                getattr(cirq, gate_name)(_get_float(parameters, "theta")).on(*index)
            )
        elif gate_name == "iswap":
            cmd.append(
                cirq.ISwapPowGate(
                    exponent=_get_float(parameters, "theta", default=1)
                ).on(*index)
            )
        elif gate_name in ["mpo", "multicontrol"]:
            gatem = np.reshape(
                backend.numpy(gate_info["gatef"](**parameters).eval_matrix()),
                [2 ** len(index), 2 ** len(index)],
            )
            ci_name = gate_info["name"]
            cgate = CustomizedCirqGate(gatem, ci_name, len(index))
            cmd.append(cgate.on(*index))
        else:
            # Add Customized Gate if there is no match
            gatem = np.reshape(
                gate_info["gate"].tensor,
                [2 ** len(index), 2 ** len(index)],
            )
            # Note: unitary test is not working for some of the generated matrix,
            # probably add tolerance unitary test later
            cgate = CustomizedCirqGate(gatem, gate_name, len(index))
            cmd.append(cgate.on(*index))
    cirq_circuit = cirq.Circuit(*cmd)
    return cirq_circuit


def qir2qiskit(
    qir: List[Dict[str, Any]],
    n: int,
    extra_qir: Optional[List[Dict[str, Any]]] = None,
    initialization: Optional[Tensor] = None,
) -> Any:
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
    :param extra_qir: The extra quantum IR of tc circuit including measure and reset on hardware,
        defaults to None
    :type extra_qir: Optional[List[Dict[str, Any]]]
    :param initialization: Circuit initial state in qiskit format
    :type initialization: Optional[Tensor]
    :return: qiskit QuantumCircuit object
    :rtype: Any
    """
    if extra_qir is not None and len(extra_qir) > 0:
        qir = _merge_extra_qir(qir, extra_qir)
        qiskit_circ = QuantumCircuit(n, n)
    else:
        qiskit_circ = QuantumCircuit(n)
    if initialization is not None:
        qiskit_circ.initialize(initialization)
    measure_cbit = 0
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
        elif gate_name == "u":
            getattr(qiskit_circ, "u")(
                _get_float(parameters, "theta"),
                _get_float(parameters, "phi"),
                _get_float(parameters, "lbd"),
                *index,
            )
        elif gate_name == "cu":
            getattr(qiskit_circ, "cu")(
                _get_float(parameters, "theta"),
                _get_float(parameters, "phi"),
                _get_float(parameters, "lbd"),
                0,  # gamma
                *index,
            )
        elif gate_name == "iswap":
            qiskit_circ.append(
                XXPlusYYGate(np.pi * _get_float(parameters, "theta", 1), np.pi),
                index,
            )
        elif gate_name == "wroot":
            getattr(qiskit_circ, "u")(np.pi / 2, -np.pi / 4, np.pi / 4, *index)
        elif gate_name in ["rx", "ry", "rz", "crx", "cry", "crz", "rxx", "ryy", "rzz"]:
            getattr(qiskit_circ, gate_name)(_get_float(parameters, "theta"), *index)
        elif gate_name in ["phase", "cphase"]:
            getattr(qiskit_circ, gate_name[:-4])(
                _get_float(parameters, "theta"), *index
            )
        elif gate_name in ["orx", "ory", "orz"]:
            getattr(qiskit_circ, "c" + gate_name[1:])(
                _get_float(parameters, "theta"), *index, ctrl_state=0
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
        elif gate_name == "multicontrol":
            unitary = backend.numpy(backend.convert_to_tensor(parameters["unitary"]))
            ctrl_str = "".join(map(str, parameters["ctrl"]))[::-1]
            gate = UnitaryGate(unitary, label=qis_name).control(
                len(ctrl_str), ctrl_state=ctrl_str
            )
            qiskit_circ.append(gate, gate_info["index"])
        elif gate_name == "mpo":
            qop = qi.Operator(
                np.reshape(
                    backend.numpy(gate_info["gatef"](**parameters).eval_matrix()),
                    [2 ** len(index), 2 ** len(index)],
                )
            )
            qiskit_circ.unitary(qop, index[::-1], label=qis_name)
        elif gate_name == "measure":
            qiskit_circ.measure(index[0], measure_cbit)
            measure_cbit += 1
        elif gate_name == "reset":
            qiskit_circ.reset(index[0])
        elif gate_name == "barrier":
            qiskit_circ.barrier(*index)
        else:  # r cr any gate
            gatem = np.reshape(
                backend.numpy(gate_info["gatef"](**parameters).tensor),
                [2 ** len(index), 2 ** len(index)],
            )
            qop = qi.Operator(gatem)
            try:
                qiskit_circ.unitary(qop, index[::-1], label=qis_name)
            except (ExtensionError, ValueError) as _:
                logger.warning(
                    "omit non unitary gate in tensorcircuit when transforming to qiskit: %s"
                    % gate_name
                )
                qiskit_circ.unitary(
                    np.eye(2 ** len(index)), index[::-1], label=qis_name
                )

    return qiskit_circ


def _translate_qiskit_params(
    gate_info: CircuitInstruction, binding_params: Any
) -> List[float]:
    parameters = []
    for p in gate_info[0].params:
        if isinstance(p, ParameterVectorElement):
            parameters.append(binding_params[p.index])
        elif isinstance(p, Parameter):
            parameters.append(binding_params[p])
        elif isinstance(p, ParameterExpression):
            if len(p.parameters) == 0:
                parameters.append(float(p))
                continue
            if len(p.parameters) != 1:
                raise ValueError(
                    f"Can't translate parameter expression with more than 1 parameters: {p}"
                )
            p_real = list(p.parameters)[0]
            if not isinstance(p_real, ParameterVectorElement):
                raise TypeError(
                    "Parameters in parameter expression should be ParameterVectorElement"
                )
            # note "sym" != "sim"
            expr = p.sympify().simplify()
            # only allow simple expressions like 1.0 * theta
            if not expr.is_Mul:
                raise ValueError(f"Unsupported parameter expression: {p}")
            arg1, arg2 = expr.args
            if arg1.is_number and arg2.is_symbol:
                coeff = arg1
            elif arg1.is_symbol and arg2.is_number:
                coeff = arg2
            else:
                raise ValueError(f"Unsupported parameter expression: {p}")
            # taking real part here because using complex type will result in a type error
            # for tf backend when the binding parameter is real
            parameters.append(float(coeff) * binding_params[p_real.index])
        else:
            # numbers, arrays, etc.
            parameters.append(p)
    return parameters


def ctrl_str2ctrl_state(ctrl_str: str, nctrl: int) -> List[int]:
    ctrl_state_bin = int(ctrl_str)
    return [0x1 & (ctrl_state_bin >> (i)) for i in range(nctrl)]


def qiskit2tc(
    qcdata: List[CircuitInstruction],
    n: int,
    inputs: Optional[List[float]] = None,
    is_dm: bool = False,
    circuit_constructor: Any = None,
    circuit_params: Optional[Dict[str, Any]] = None,
    binding_params: Optional[Union[Sequence[float], Dict[Any, float]]] = None,
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
    :type qcdata: List[CircuitInstruction]
    :param n: # of qubits
    :type n: int
    :param inputs: Input state of the circuit. Default is None.
    :type inputs: Optional[List[float]]
    :param circuit_constructor: ``Circuit``, ``DMCircuit`` or ``MPSCircuit``
    :type circuit_contructor: Any
    :param circuit_params: kwargs given in Circuit.__init__ construction function, default to None.
    :type circuit_params: Optional[Dict[str, Any]]
    :param binding_params: (variational) parameters for the circuit.
        Could be either a sequence or dictionary depending on the type of parameters in the Qiskit circuit.
        For ``ParameterVectorElement`` use sequence. For ``Parameter`` use dictionary
    :type binding_params: Optional[Union[Sequence[float], Dict[Any, float]]]
    :return: A quantum circuit in tensorcircuit
    :rtype: Any
    """
    if circuit_constructor is not None:
        Circ = circuit_constructor
    elif is_dm:
        Circ = DMCircuit2
    else:
        Circ = Circuit
    if circuit_params is None:
        circuit_params = {}
    if "nqubits" not in circuit_params:
        circuit_params["nqubits"] = n
    if (
        len(qcdata) > 0
        and qcdata[0][0].name == "initialize"
        and "inputs" not in circuit_params
    ):
        circuit_params["inputs"] = perm_matrix(n) @ np.array(qcdata[0][0].params)
    if inputs is not None:
        circuit_params["inputs"] = inputs

    tc_circuit: Any = Circ(**circuit_params)
    for gate_info in qcdata:
        idx = [qb.index for qb in gate_info[1]]
        gate_name = gate_info[0].name
        parameters = _translate_qiskit_params(gate_info, binding_params)
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
        elif gate_name in ["rx", "ry", "rz", "crx", "cry", "crz", "rxx", "ryy", "rzz"]:
            getattr(tc_circuit, gate_name)(*idx, theta=parameters)
        elif gate_name in ["crx_o0", "cry_o0", "crz_o0"]:
            getattr(tc_circuit, "o" + gate_name[1:-3])(*idx, theta=parameters)
        elif gate_name in ["r"]:
            getattr(tc_circuit, "u")(
                *idx,
                theta=parameters[0],
                phi=parameters[1] - np.pi / 2,
                lbd=-parameters[1] + np.pi / 2,
            )
        elif gate_name in ["u3", "u"]:
            getattr(tc_circuit, "u")(
                *idx, theta=parameters[0], phi=parameters[1], lbd=parameters[2]
            )
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
        elif gate_name[0] == "c" and gate_name[:7] != "circuit" and gate_name != "cu":
            # qiskit cu bug, see https://github.com/tencent-quantum-lab/tensorcircuit/issues/199
            for i in range(1, len(gate_name)):
                if (gate_name[-i] == "o") & (gate_name[-i - 1] == "_"):
                    break
            if i == len(gate_name) - 1:
                base_gate = gate_info[0].base_gate
                ctrl_state = [1] * base_gate.num_qubits
                idx = idx[: -base_gate.num_qubits] + idx[-base_gate.num_qubits :][::-1]
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
        elif gate_name == "measure":
            tc_circuit.measure_instruction(*idx)
            # logger.warning(
            #     "qiskit to tc translation currently doesn't support measure instruction, just skipping"
            # )
        elif gate_name == "reset":
            tc_circuit.reset_instruction(*idx)
            # logger.warning(
            #     "qiskit to tc translation currently doesn't support reset instruction, just skipping"
            # )
        elif gate_name == "barrier":
            tc_circuit.barrier_instruction(*idx)
        elif gate_name == "initialize":
            # already taken care of when initializing
            continue
        elif not hasattr(gate_info[0], "__array__"):
            # an instruction containing a lot of gates.
            # the condition is based on
            # https://github.com/Qiskit/qiskit-terra/blob/2f5944d60140ceb6e30d5b891e8ffec735247ce9/qiskit/circuit/gate.py#L43
            qiskit_circuit = gate_info[0].definition
            if qiskit_circuit is None:
                logger.warning(
                    f"qiskit to tc translation doesn't support {gate_name} instruction, skipping"
                )
                continue
            c = Circuit.from_qiskit(qiskit_circuit, binding_params=binding_params)
            tc_circuit.append(c, idx)
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
    if np.iscomplexobj(a):
        ai = np.imag(a)
        return [ar.tolist(), ai.tolist()]
    else:
        return [ar.tolist()]


def json_to_tensor(a: Any) -> Any:
    ar = np.array(a[0])
    if len(a) == 1:
        return ar
    else:
        assert len(a) == 2
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


def eqasm2tc(
    eqasm: str, nqubits: Optional[int] = None, headers: Tuple[int, int] = (6, 1)
) -> Circuit:
    """
    Translation qexe/eqasm instruction to tensorcircuit Circuit object

    :param eqasm: _description_
    :type eqasm: str
    :param nqubits: _description_, defaults to None
    :type nqubits: Optional[int], optional
    :param headers: lines of ignored code at the head and the tail, defaults to (6, 1)
    :type headers: Tuple[int, int], optional
    :return: _description_
    :rtype: Circuit
    """
    eqasm_list = eqasm.split("\n")
    if nqubits is None:
        nqubits = len(eqasm_list[2].split(","))
    eqasm_list = eqasm_list[headers[0] : -headers[1]]
    c = Circuit(nqubits)
    # measure z not considered
    for inst in eqasm_list:
        if inst.startswith("bs"):
            inst_list = inst.split(" ")
            if inst_list[2].startswith("RZ"):
                exponent = int(inst_list[2][3:])
                index = (int(inst_list[3][1:]),)
                c.rz(*index, theta=2 * np.pi / 2**exponent)  # type: ignore
            elif inst_list[2] == "Z/2":
                c.rz(*index, theta=-np.pi / 2)  # type: ignore
            elif inst_list[2] == "-Z/2":
                c.rz(*index, theta=np.pi / 2)  # type: ignore
            # TODO(@refraction-ray): Z/2 convention to be double checked
            else:
                gate_name = inst_list[2].lower()
                if len(inst_list) == 4:
                    index = (int(inst_list[3][1:]),)
                elif len(inst_list) == 5:
                    index = (int(inst_list[3][2:-1]), int(inst_list[4][1:-1]))  # type: ignore
                else:
                    raise ValueError("Unknown format for eqasm: %s" % str(inst_list))
                getattr(c, gate_name)(*index)

    return c


def qiskit_from_qasm_str_ordered_measure(qasm_str: str) -> Any:
    """
    qiskit ``from_qasm_str`` method cannot keep the order of measure as the qasm file,
    we provide this alternative function in case the order of measure instruction matters

    :param qasm_str: open qasm str
    :type qasm_str: str
    :return: ``qiskit.circuit.QuantumCircuit``
    :rtype: Any
    """
    measure_sequence = []
    qasm_instruction = []
    for line in qasm_str.split("\n"):
        if line.startswith("measure"):
            index = int(line.split(" ")[1][2:-1])
            cindex = int(line.split(" ")[3].strip(";").split("[")[1][:-1])
            # sometimes we have qasm as "measure q[3] -> meas[0];"

            measure_sequence.append((index, cindex))
        else:
            qasm_instruction.append(line)

    qc = QuantumCircuit.from_qasm_str("\n".join(qasm_instruction))
    for qid, cid in measure_sequence:
        qc.measure(qid, cid)
    return qc
