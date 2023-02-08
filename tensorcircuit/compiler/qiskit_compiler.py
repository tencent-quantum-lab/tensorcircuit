"""
compiler interface via qiskit
"""

from typing import Any, Dict, Optional
import re

from ..abstractcircuit import AbstractCircuit
from ..circuit import Circuit
from ..translation import qiskit_from_qasm_str_ordered_measure


def _free_pi(s: str) -> str:
    # dirty trick to get rid of pi in openqasm from qiskit
    rs = []
    pistr = "3.141592653589793"
    s = s.replace("pi", pistr)
    for r in s.split("\n"):
        inc = re.search(r"\(.*\)", r)
        if inc is None:
            rs.append(r)
        else:
            v = r[inc.start() : inc.end()]
            v = eval(v)
            r = r[: inc.start()] + "(" + str(v) + ")" + r[inc.end() :]
            rs.append(r)
    return "\n".join(rs)


def _comment_qasm(s: str) -> str:
    """
    return the qasm str in comment format

    :param s: _description_
    :type s: str
    :return: _description_
    :rtype: str
    """
    nslist = []
    nslist.append("//circuit begins")
    for line in s.split("\n"):
        nslist.append("//" + line)
    nslist.append("//circuit ends")
    return "\n".join(nslist)


def _comment_dict(d: Dict[int, int], name: str = "logical_physical_mapping") -> str:
    """
    save a dict in commented qasm

    :param d: _description_
    :type d: Dict[int, int]
    :param name: _description_, defaults to "logical_physical_mapping"
    :type name: str, optional
    :return: _description_
    :rtype: str
    """
    nslist = []
    nslist.append("//%s begins" % name)
    for k, v in d.items():
        nslist.append("// " + str(k) + " : " + str(v))
    nslist.append("//%s ends" % name)
    return "\n".join(nslist)


def _get_positional_logical_mapping_from_qiskit(qc: Any) -> Dict[int, int]:
    i = 0
    positional_logical_mapping = {}
    for inst in qc.data:
        if inst[0].name == "measure":
            positional_logical_mapping[i] = inst[1][0].index
            i += 1

    return positional_logical_mapping


def _get_logical_physical_mapping_from_qiskit(
    qc_after: Any, qc_before: Any = None
) -> Dict[int, int]:
    """
    get ``logical_physical_mapping`` from qiskit Circuit by comparing the circuit after and before compiling

    :param qc_after: qiskit ``QuantumCircuit`` after compiling
    :type qc_after: Any
    :param qc_before: qiskit ``QuantumCircuit`` before compiling,
        if None, measure(q, q) is assumed
    :type qc_before: Any
    :return: logical_physical_mapping
    :rtype: Dict[int, int]
    """
    logical_physical_mapping = {}
    for inst in qc_after.data:
        if inst[0].name == "measure":
            if qc_before is None:
                logical_q = inst[2][0].index
            else:
                for instb in qc_before.data:
                    if (
                        instb[0].name == "measure"
                        and instb[2][0].index == inst[2][0].index
                    ):
                        logical_q = instb[1][0].index
                        break
            logical_physical_mapping[logical_q] = inst[1][0].index
    return logical_physical_mapping


def _add_measure_all_if_none(qc: Any) -> Any:
    for inst in qc.data:
        if inst[0].name == "measure":
            break
    else:
        qc.measure_all()
    return qc


def qiskit_compile(
    circuit: Any,
    info: Optional[Dict[str, Any]] = None,
    output: str = "tc",
    compiled_options: Optional[Dict[str, Any]] = None,
) -> Any:
    from qiskit.compiler import transpile
    from qiskit.transpiler.passes import RemoveBarriers

    if isinstance(circuit, AbstractCircuit):
        circuit = circuit.to_qiskit(enable_instruction=True)
    elif isinstance(circuit, str):
        circuit = qiskit_from_qasm_str_ordered_measure(circuit)
    # else qiskit circuit
    circuit = _add_measure_all_if_none(circuit)
    if compiled_options is None:
        compiled_options = {
            "basis_gates": ["h", "rz", "cx"],
            "optimization_level": 2,
        }
    ncircuit = transpile(circuit, **compiled_options)
    ncircuit = RemoveBarriers()(ncircuit)

    if output.lower() in ["qasm", "openqasm"]:
        r0 = ncircuit.qasm()

    elif output.lower() in ["qiskit", "ibm"]:
        r0 = ncircuit

    elif output.lower() in ["tc", "tensorcircuit"]:
        s = _free_pi(ncircuit.qasm())
        r0 = Circuit.from_openqasm(
            s,
            keep_measure_order=True,
        )

    else:
        raise ValueError("Unknown output format: %s" % output)

    r1 = {}
    nlpm = _get_logical_physical_mapping_from_qiskit(ncircuit, circuit)
    # new_logical_physical_mapping
    if info is not None and "logical_physical_mapping" in info:
        r1["logical_physical_mapping"] = {
            k: nlpm[v] for k, v in info["logical_physical_mapping"].items()
        }
    else:
        r1["logical_physical_mapping"] = nlpm
    if info is not None and "positional_logical_mapping" in info:
        r1["positional_logical_mapping"] = info["positional_logical_mapping"]
    else:  # info is none, assume circuit is the logical circuit
        r1["positional_logical_mapping"] = _get_positional_logical_mapping_from_qiskit(
            circuit
        )
    # TODO(@refraction-ray): more info to be added into r1 dict

    return (r0, r1)
