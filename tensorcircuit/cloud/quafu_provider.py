"""
Cloud provider from QuaFu: http://quafu.baqis.ac.cn/
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import logging

from quafu import User, QuantumCircuit
from quafu import Task as Task_

from .abstraction import Device, sep, Task
from ..abstractcircuit import AbstractCircuit
from ..utils import is_sequence

logger = logging.getLogger(__name__)


def list_devices(token: Optional[str] = None, **kws: Any) -> List[Device]:
    raise NotImplementedError


def list_properties(device: Device, token: Optional[str] = None) -> Dict[str, Any]:
    raise NotImplementedError


def submit_task(
    device: Device,
    token: str,
    shots: Union[int, Sequence[int]] = 1024,
    circuit: Optional[Union[AbstractCircuit, Sequence[AbstractCircuit]]] = None,
    source: Optional[Union[str, Sequence[str]]] = None,
    compile: bool = True,
    **kws: Any
) -> Task:
    if source is None:

        def c2qasm(c: Any) -> str:
            from qiskit.circuit import QuantumCircuit

            if isinstance(c, QuantumCircuit):
                s = c.qasm()
                # nq = c.num_qubits
            else:
                s = c.to_openqasm()
            return s  # type: ignore

        if not is_sequence(circuit):
            source = c2qasm(circuit)
        else:
            source = [c2qasm(c) for c in circuit]  # type: ignore
    user = User()
    user.save_apitoken(token)

    def c2task(source: str) -> Task:
        nq = int(source.split("\n")[2].split("[")[1].split("]")[0])  # type: ignore
        qc = QuantumCircuit(nq)
        qc.from_openqasm(source)
        task = Task_()
        device_name = device.name.split(sep)[-1]
        task.config(backend=device_name, shots=shots, compile=compile)
        res = task.send(qc, wait=False)
        wrapper = Task(res.taskid, device=device)
        return wrapper

    if not is_sequence(source):
        return c2task(source)  # type: ignore
    else:
        return [c2task(s) for s in source]  # type: ignore


def resubmit_task(task: Task, token: str) -> Task:
    raise NotImplementedError


def remove_task(task: Task, token: str) -> Any:
    raise NotImplementedError


def list_tasks(device: Device, token: str, **filter_kws: Any) -> List[Task]:
    raise NotImplementedError


def get_task_details(
    task: Task, device: Device, token: str, prettify: bool
) -> Dict[str, Any]:
    # id results
    r = {}
    r["id"] = task.id_
    t = Task_()
    r["results"] = dict(t.retrieve(task.id_).counts)  # type: ignore
    if r["results"]:
        r["state"] = "completed"
    else:
        r["state"] = "pending"
    return r
