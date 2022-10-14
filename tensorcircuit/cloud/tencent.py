"""
Cloud provider from Tencent
"""

from typing import Any, Dict, List, Optional

from .config import tencent_base_url
from .utils import rpost_json
from .abstraction import Device, sep


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
