"""
Cloud provider from Tencent
"""

from typing import Any, Dict, List, Optional, Sequence, Union
from json import dumps

from .config import tencent_base_url
from .utils import rpost_json
from .abstraction import Device, sep, Task
from ..abstractcircuit import AbstractCircuit
from ..utils import is_sequence


def tencent_headers(token: Optional[str] = None) -> Dict[str, str]:
    if token is None:
        token = "ANY;0"
    headers = {"Authorization": "Bearer " + token}
    return headers


def list_devices(token: Optional[str] = None) -> List[Device]:
    json: Dict[Any, Any] = {}
    r = rpost_json(
        tencent_base_url + "device/find", json=json, headers=tencent_headers(token)
    )
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
    if "device" in r:
        return r["device"]  # type: ignore
    else:
        raise ValueError("No device with the name: %s" % device)


def submit_task(
    device: Device,
    token: str,
    lang: str,
    shots: Union[int, Sequence[int]] = 1024,
    version: str = "1",
    prior: int = 1,
    circuit: Optional[Union[AbstractCircuit, Sequence[AbstractCircuit]]] = None,
    source: Optional[Union[str, Sequence[str]]] = None,
    remarks: Optional[str] = None,
) -> List[Task]:
    if source is None:
        pass  # circuit to source logic
    if is_sequence(source):
        # batched mode
        json = []
        if not is_sequence(shots):
            shots = [shots for _ in source]  # type: ignore
        for sc, sh in zip(source, shots):  # type: ignore
            json.append(
                {
                    "device": device.name,
                    "shots": sh,
                    "source": sc,
                    "version": version,
                    "lang": lang,
                    "prior": prior,
                    "remarks": remarks,
                }
            )

    else:
        json = {  # type: ignore
            "device": device.name,
            "shots": shots,
            "source": source,
            "version": version,
            "lang": lang,
            "prior": prior,
            "remarks": remarks,
        }
    r = rpost_json(
        tencent_base_url + "task/submit", json=json, headers=tencent_headers(token)
    )
    try:
        return [Task(id_=t["id"], device=device) for t in r["tasks"]]
    except KeyError:
        raise ValueError(dumps(r))
