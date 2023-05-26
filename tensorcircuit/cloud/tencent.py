"""
Cloud provider from Tencent
"""

from typing import Any, Dict, List, Optional, Sequence, Union
from datetime import datetime
from json import dumps
import logging
from functools import partial

from .config import tencent_base_url
from .utils import rpost_json
from .abstraction import Device, sep, Task
from ..abstractcircuit import AbstractCircuit
from ..utils import is_sequence, arg_alias
from ..circuit import Circuit
from ..translation import eqasm2tc

logger = logging.getLogger(__name__)


def tencent_headers(token: Optional[str] = None) -> Dict[str, str]:
    if token is None:
        token = "ANY;0"
    headers = {"Authorization": "Bearer " + token}
    return headers


def error_handling(r: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(r, dict):
        raise ValueError("failed to get legal response from the server")
    if "err" in r:
        raise ValueError(r["err"])
    return r


def list_devices(token: Optional[str] = None, **kws: Any) -> List[Device]:
    json: Dict[Any, Any] = kws
    r = rpost_json(
        tencent_base_url + "device/find", json=json, headers=tencent_headers(token)
    )
    r = error_handling(r)
    ds = r["devices"]
    rs = []
    for d in ds:
        rs.append(Device.from_name("tencent" + sep + d["id"]))
    return rs


def list_properties(device: Device, token: Optional[str] = None) -> Dict[str, Any]:
    json = {"id": device.name}
    r = rpost_json(
        tencent_base_url + "device/detail", json=json, headers=tencent_headers(token)
    )
    r = error_handling(r)
    if "device" in r:
        r = r["device"]
        if "links" in r:
            links_dict = {}
            for link in r["links"]:
                links_dict[(link["A"], link["B"])] = link
            r["links"] = links_dict
        if "bits" in r:
            bits_dict = {}
            for bit in r["bits"]:
                bits_dict[bit["Qubit"]] = bit
            r["bits"] = bits_dict
        r["native_gates"] = ["h", "rz", "x", "y", "z", "cz", "cx"]  # handcoded
        return r  # type: ignore
    else:
        raise ValueError("No device with the name: %s" % device)

    # list properties should at least contain the following items
    """
    {'id': '20xmon',
     'type': 'CHIP',
     'state': 'on',
     'native_gates': ["rz"]
     'links': {(0, 1): { 'CZErrRate': 0.03, 'at': 1673605888},
                ...},
     'bits': {0: { 'At': 1673605888,
                'Freqency': 4420,
                'ReadoutF0Err': 0.0415,
                'ReadoutF1Err': 0.1006,
                'SingleQubitErrRate': 0.00095,
                'SingleQubitGateLenInNs': 30,
                'T1': 35.5,
                'T2': 5.4},
                ...,},
    'langs': ['tQASM', 'eQASM']}
    """


def _replace_rz_to_st(qasm: str) -> str:
    nqasm = []
    for line in qasm.split("\n"):
        if line.startswith("rz(pi/2)") or line.startswith("rz(5*pi/2)"):
            line = " ".join(["s"] + line.split(" ")[1:])
        elif line.startswith("rz(-pi/2)") or line.startswith("rz(3*pi/2)"):
            line = " ".join(["sdg"] + line.split(" ")[1:])
        elif line.startswith("rz(pi/4)"):
            line = " ".join(["t"] + line.split(" ")[1:])
        elif line.startswith("rz(-pi/4)"):
            line = " ".join(["tdg"] + line.split(" ")[1:])

        nqasm.append(line)
    return "\n".join(nqasm)


@partial(
    arg_alias,
    alias_dict={
        "compiling": ["compiled"],
        "compiled_options": ["qiskit_compiled_options"],
    },
)
def submit_task(
    device: Device,
    token: str,
    lang: str = "OPENQASM",
    shots: Union[int, Sequence[int]] = 1024,
    version: str = "1",
    prior: int = 1,
    circuit: Optional[Union[AbstractCircuit, Sequence[AbstractCircuit]]] = None,
    source: Optional[Union[str, Sequence[str]]] = None,
    remarks: Optional[str] = None,
    group: Optional[str] = None,
    compiling: bool = False,
    compiled_options: Optional[Dict[str, Any]] = None,
    enable_qiskit_initial_mapping: bool = False,
    # measure: Optional[Sequence[int]] = None,
    enable_qos_qubit_mapping: bool = True,
    enable_qos_gate_decomposition: bool = True,
    enable_qos_initial_mapping: bool = False,
    qos_dry_run: bool = False,
    **kws: Any
) -> List[Task]:
    """
    Submit task via tencent provider, we suggest to enable one of the compiling functionality:
    either in tc: frontend or in qos: backend. If both are enabled, try on your own risk, some
    qubit mapping may fail silently. If the user directly provide ``source`` or qiskit Circuit in ``circuit``,
    the qubit mapping should be taken care of by the users.

    :param device: [description]
    :type device: Device
    :param token: [description]
    :type token: str
    :param lang: language choice for ``source``, defaults to "OPENQASM"
    :type lang: str, optional
    :param shots: number of measurement shots, defaults to 1024
    :type shots: Union[int, Sequence[int]], optional
    :param version: submit task protocol version, defaults to "1"
    :type version: str, optional
    :param prior: priority for the task queue, defaults to 1
    :type prior: int, optional
    :param circuit: tensorcircuit or qiskit circuit object, defaults to None
    :type circuit: Optional[Union[AbstractCircuit, Sequence[AbstractCircuit]]], optional
    :param source: directly given circuit representation, defaults to None
    :type source: Optional[Union[str, Sequence[str]]], optional
    :param remarks: remarks on the task, defaults to None
    :type remarks: Optional[str], optional
    :param compiling: whether compiling in tc via qiskit compiling system,
        defaults to False
    :type compiling: bool, optional
    :param compiled_options: compiling options for qiskit ``transpile`` method,
        defaults to None
    :type compiled_options: Optional[Dict[str, Any]], optional
    :param enable_qos_qubit_mapping: whether to insert swap if necessary in qos, defaults to True
    :type enable_qos_qubit_mapping: bool, optional
    :param enable_qos_gate_decomposition: whether to compile the gate in qos, defaults to True
    :type enable_qos_gate_decomposition: bool, optional
    :param enable_qos_initial_mapping: whether to run an initial qubit mapping in qos,
        defaults to False
    :type enable_qos_initial_mapping: bool, optional
    :param qos_dry_run: when dry run, only compiled circuit is returned (no real circuit execution),
        defaults to False
    :type qos_dry_run: bool, optional
    :return: Task object or List of Task for batch submission
    :rtype: List[Task]
    """
    # :param measure: [deprecated] which group of qubit to measure,
    #     defaults to None, the measure result is in the order of qubit index
    #     instead of the ``measure`` list
    # :type measure: Optional[Sequence[int]], optional
    if source is None:
        if compiled_options is None and compiling is True:
            links = device.topology()
            if (
                enable_qiskit_initial_mapping is True
                and isinstance(links, list)
                and len(links) > 1
                and isinstance(links[0], list)
            ):
                coupling_map = links
            else:
                coupling_map = None
            compiled_options = {
                "basis_gates": ["h", "x", "y", "z", "s", "t", "rz", "cx", "cz"],
                "optimization_level": 2,
                "coupling_map": coupling_map,
            }

        def c2qasm(c: Any, compiling: bool) -> str:
            from ..compiler.qiskit_compiler import qiskit_compile
            from qiskit.circuit import QuantumCircuit

            if compiling is True:
                s, _ = qiskit_compile(
                    c, output="qasm", info=None, compiled_options=compiled_options
                )
            else:
                if isinstance(c, QuantumCircuit):
                    s = c.qasm()
                    # nq = c.num_qubits
                else:
                    s = c.to_openqasm()
                    # nq = c._nqubits
            # s = _free_pi(s) # tQuk translation now supports this
            # if measure is not None:  # ad hoc partial measurement
            #     slist = s.split("\n")[:-1]
            #     if len(slist) > 3 and not slist[3].startswith("creg"):
            #         slist.insert(3, "creg c[%s];" % nq)
            #     for m in measure:
            #         slist.append("measure q[%s]->c[%s];" % (m, m))
            #     slist.append("")
            #     s = "\n".join(slist)
            s = _replace_rz_to_st(s)
            return s  # type: ignore

        if is_sequence(circuit):
            source = [c2qasm(c, compiling) for c in circuit]  # type: ignore
        else:
            source = c2qasm(circuit, compiling)
        lang = "OPENQASM"

    if len(device.name.split("?")) > 1:
        device_str = device.name
    else:
        oint = 0
        if enable_qos_qubit_mapping:
            oint += 1
        if enable_qos_gate_decomposition:
            oint += 2
        if enable_qos_initial_mapping:
            oint += 4
        device_str = device.name + "?o=" + str(oint)
    if qos_dry_run:
        device_str += "&dry"

    if is_sequence(source):
        # batched mode
        json = []
        if not is_sequence(shots):
            shots = [shots for _ in source]  # type: ignore
        for sc, sh in zip(source, shots):  # type: ignore
            json.append(
                {
                    "device": device_str,
                    "shots": sh,
                    "source": sc,
                    "version": version,
                    "lang": lang,
                    "prior": prior,
                    "remarks": remarks,
                    "group": group,
                }
            )

    else:
        json = {  # type: ignore
            "device": device_str,
            "shots": shots,
            "source": source,
            "version": version,
            "lang": lang,
            "prior": prior,
            "remarks": remarks,
            "group": group,
        }
    r = rpost_json(
        tencent_base_url + "task/submit", json=json, headers=tencent_headers(token)
    )
    r = error_handling(r)
    rtn = []
    for t in r["tasks"]:
        if "err" in t or "id" not in t:
            if "err" in t:
                logger.warning(t["err"])
            else:
                logger.warning("unsuccessful submission of the task:\n" + dumps(r))
        else:
            ti = Task(id_=t["id"], device=device)
            rtn.append(ti)
    if not is_sequence(source):
        return rtn[0]  # type: ignore
    elif len(rtn) == 0:
        raise ValueError("All tasks submitted failed")
    else:
        return rtn


def resubmit_task(task: Task, token: str) -> Task:
    # TODO(@refraction-ray): batch resubmit
    json = {"id": task.id_}
    r = rpost_json(
        tencent_base_url + "task/start", json=json, headers=tencent_headers(token)
    )
    r = error_handling(r)
    try:
        return Task(id_=r["tasks"][0]["id"])

    except KeyError:
        raise ValueError(dumps(r))


def remove_task(task: Task, token: str) -> Any:
    # TODO(@refraction-ray): batch cancel
    json = {"id": task.id_}
    r = rpost_json(
        tencent_base_url + "task/remove", json=json, headers=tencent_headers(token)
    )
    r = error_handling(r)
    return r


def list_tasks(device: Device, token: str, **filter_kws: Any) -> List[Task]:
    json = filter_kws
    if device is not None:
        json["device"] = device.name
    r = rpost_json(
        tencent_base_url + "task/find?pn=1&npp=50",
        json=json,
        headers=tencent_headers(token),
    )
    r = error_handling(r)
    try:
        rtn = []
        for t in r["tasks"]:
            rtn.append(
                Task(
                    id_=t["id"],
                    device=Device.from_name("tencent" + sep + t["device"]),
                )
            )
        return rtn
    except KeyError:
        raise ValueError(dumps(r))


def get_task_details(
    task: Task, device: Device, token: str, prettify: bool
) -> Dict[str, Any]:
    json = {"id": task.id_}
    r = rpost_json(
        tencent_base_url + "task/detail", json=json, headers=tencent_headers(token)
    )
    r = error_handling(r)
    try:
        if "result" in r["task"]:
            if "counts" in r["task"]["result"]:
                r["task"]["results"] = r["task"]["result"]["counts"]
            else:
                r["task"]["results"] = r["task"]["result"]
        if "optimization" in r["task"]:
            if (
                "pairs" in r["task"]["optimization"]
                and r["task"]["optimization"]["pairs"] is not None
            ):
                r["task"]["optimization"]["pairs"] = {
                    int(k): int(v)
                    for k, v in r["task"]["optimization"]["pairs"].items()
                }
                # r["logical_physical_mapping"] = r["task"]["optimization"]["pairs"]

        # if not "logical_physical_mapping" in r["task"]:
        # r["task"]["logical_physical_mapping"] = None

        if prettify is False:
            return r["task"]  # type: ignore
        # make the results more readable
        r = r["task"]
        if "at" in r:
            r["at"] = datetime.fromtimestamp(r["at"] / 1e6)
        if "ts" in r:
            for k in r["ts"]:
                r["ts"][k] = datetime.fromtimestamp(r["ts"][k] / 1e6)
        if "source" in r:
            r["frontend"] = Circuit.from_openqasm(r["source"])
        if "optimization" in r and r["state"] == "completed":
            try:
                r["backend"] = eqasm2tc(r["optimization"]["progs"][0]["code"])
            except KeyError:
                pass
        return r  # type: ignore
    except KeyError:
        raise ValueError(dumps(r))

    # make the return at least contain the following terms across different providers
    """
    'id': 'd947cd76-a961-4c22-b295-76287c9fdaa3',
    'state': 'completed',
    'at': 1666752095915849,
    'shots': 1024,
    'source': 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nh q[0];\nh q[1];\nh q[2];\n', 
    'device': 'simulator:aer',
    'results':  {'000': 123,
    '001': 126,
    '010': 131,
    '011': 128,
    '100': 122,
    '101': 135,
    '110': 128,
    '111': 131}
    """
