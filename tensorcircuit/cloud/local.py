"""
Cloud provider from local machine
"""

from typing import Any, Dict, List, Optional, Union, Sequence
from uuid import uuid4
import time

from .abstraction import Device, sep, Task
from ..utils import is_sequence
from ..abstractcircuit import AbstractCircuit

local_devices = ["testing"]

task_list: Dict[str, Any] = {}  # memory only task cache


def list_devices(token: Optional[str] = None, **kws: Any) -> List[Device]:
    rs = []
    for d in local_devices:
        rs.append(Device.from_name("local" + sep + d))
    return rs


def get_task_details(
    task: Task, device: Device, token: str, prettify: bool
) -> Dict[str, Any]:
    if task.id_ in task_list:
        return task_list[task.id_]  # type: ignore
    raise ValueError("no task with id: %s" % task.id_)


def submit_task(
    device: Device,
    token: str,
    shots: Union[int, Sequence[int]] = 1024,
    version: str = "1",
    circuit: Optional[Union[AbstractCircuit, Sequence[AbstractCircuit]]] = None,
    **kws: Any
) -> List[Task]:
    def _circuit2result(c: AbstractCircuit) -> Dict[str, Any]:
        if device.name in ["testing", "default"]:
            count = c.sample(batch=shots, allow_state=True, format="count_dict_bin")  # type: ignore
        else:
            raise ValueError("Unsupported device from local provider: %s" % device.name)
        d = {
            "id": str(uuid4()),
            "state": "completed",
            "at": time.time() * 1e6,
            "shots": shots,
            "device": device.name,
            "results": count,
        }
        return d

    if is_sequence(circuit):
        tl = []
        for c in circuit:  # type: ignore
            d = _circuit2result(c)
            task_list[d["id"]] = d
            tl.append(Task(id_=d["id"], device=device))
        return tl
    else:
        d = _circuit2result(circuit)  # type: ignore
        task_list[d["id"]] = d

        return Task(id_=d["id"], device=device)  # type: ignore


def list_tasks(device: Device, token: str, **filter_kws: Any) -> List[Task]:
    r = []
    for t, v in task_list.items():
        if (device is not None and v["device"] == device.name) or device is None:
            r.append(Task(id_=t, device=Device.from_name("local" + sep + v["device"])))
    return r
