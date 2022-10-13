"""
Cloud provider from Tencent
"""

from typing import Any, Dict, Optional

from .config import tencent_base_url
from .utils import rpost_json


def tencent_headers(token: Optional[str] = None) -> Dict[str, str]:
    if token is None:
        token = "ANY;0"
    headers = {"Authorization": "Bearer " + token}
    return headers


def list_devices(token: Optional[str] = None) -> Any:
    json: Dict[Any, Any] = {}
    r = rpost_json(
        tencent_base_url + "device/find", json=json, headers=tencent_headers(token)
    )
    return r["devices"]
