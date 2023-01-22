"""
compiler interface via qiskit
"""

from typing import Any, Dict, Tuple, Optional
import re

from ..abstractcircuit import AbstractCircuit
from ..circuit import Circuit


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


def _get_mappings_from_qiskit(qc: Any) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    get the ``positional_logical_mapping`` and ``logical_physical_mapping`` from qiskit Circuit

    :param qc: qiskit ``QuantumCircuit``
    :type qc: Any
    :return: _description_
    :rtype: Tuple[Dict[int, int], Dict[int, int]]
    """
    logical_physical_mapping = {}
    positional_logical_mapping = {}
    i = 0
    for inst in qc.data:
        if inst[0].name == "measure":
            logical_physical_mapping[inst[2][0].index] = inst[1][0].index
            positional_logical_mapping[i] = inst[2][0].index
            i += 1
    return positional_logical_mapping, logical_physical_mapping


def _add_measure_all_if_none(qc: Any) -> Any:
    for inst in qc.data:
        if inst[0].name == "measure":
            break
    else:
        qc.measure_all()
    return qc


def qiskit_compile(
    circuit: Any,
    output: str = "tc",
    info: bool = False,
    compiled_options: Optional[Dict[str, Any]] = None,
) -> Any:
    from qiskit.compiler import transpile
    from qiskit.transpiler.passes import RemoveBarriers

    if isinstance(circuit, AbstractCircuit):
        circuit = circuit.to_qiskit(enable_instruction=True)
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
        r0 = Circuit.from_qiskit(ncircuit)

    else:
        raise ValueError("Unknown output format: %s" % output)

    if info is False:
        return r0

    r1 = {}

    (
        r1["positional_logical_mapping"],
        r1["logical_physical_mapping"],
    ) = _get_mappings_from_qiskit(ncircuit)
    # TODO(@refraction-ray): more info to be added into r1 dict

    return (r0, r1)
