"""
main entrypoints of cloud module
"""

from typing import Any, List, Optional, Dict, Union
from base64 import b64decode, b64encode
from functools import partial
import json
import os
import sys

from .abstraction import Provider, Device, Task, sep
from . import tencent
from ..cons import backend

package_name = "tensorcircuit"
thismodule = sys.modules[__name__]


default_provider = Provider.from_name("tencent")
avail_providers = ["tencent"]


def list_providers() -> List[Provider]:
    return [get_provider(s) for s in avail_providers]


def set_provider(
    provider: Optional[Union[str, Provider]] = None, set_global: bool = True
) -> Provider:
    if provider is None:
        provider = default_provider
    provider = Provider.from_name(provider)
    if set_global:
        for module in sys.modules:
            if module.startswith(package_name):
                setattr(sys.modules[module], "default_provider", provider)
    return provider


set_provider()
get_provider = partial(set_provider, set_global=False)

default_device = Device.from_name("tencent::hello")


def set_device(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
    set_global: bool = True,
) -> Device:
    if device is None:
        device = default_device
    device = Device.from_name(device, provider)
    if provider is None:
        provider = device.provider
    if set_global:
        for module in sys.modules:
            if module.startswith(package_name):
                setattr(sys.modules[module], "default_device", device)
    return device


set_device()
get_device = partial(set_device, set_global=False)


def b64encode_s(s: str) -> str:
    return b64encode(s.encode("utf-8")).decode("utf-8")


def b64decode_s(s: str) -> str:
    return b64decode(s.encode("utf-8")).decode("utf-8")


saved_token: Dict[str, Any] = {}


def set_token(
    token: Optional[str] = None,
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
    cached: bool = True,
) -> Dict[str, Any]:
    global saved_token
    tcdir = os.path.dirname(os.path.abspath(__file__))
    authpath = os.path.join(tcdir, "auth.json")
    if provider is None:
        provider = default_provider
    provider = Provider.from_name(provider)
    if device is not None:
        device = Device.from_name(device, provider)
    # if device is None:
    #     device = default_device

    if token is None:
        if cached and os.path.exists(authpath):
            with open(authpath, "r") as f:
                file_token = json.load(f)
                file_token = backend.tree_map(b64decode_s, file_token)
        else:
            file_token = {}
        file_token.update(saved_token)
        saved_token = file_token
    else:  # with token
        if device is None:
            if provider is None:
                provider = Provider.from_name("tencent")
            added_token = {provider.name + sep: token}
        else:
            added_token = {provider.name + sep + device.name: token}
        saved_token.update(added_token)

    if cached:
        file_token = backend.tree_map(b64encode_s, saved_token)
        with open(authpath, "w") as f:
            json.dump(file_token, f)

    return saved_token


set_token()


def get_token(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
) -> Optional[str]:
    if provider is None:
        provider = default_provider
    provider = Provider.from_name(provider)
    if device is not None:
        device = Device.from_name(device, provider)
    target = provider.name + sep
    if device is not None:
        target = target + device.name
    for k, v in saved_token.items():
        if k == target:
            return v  # type: ignore
    return None


# token json structure
# {"tencent::": token1, "tencent::20xmon":  token2}


def list_devices(
    provider: Optional[Union[str, Provider]] = None, token: Optional[str] = None
) -> Any:
    if provider is None:
        provider = default_provider
    provider = Provider.from_name(provider)
    if token is None:
        token = provider.get_token()
    if provider.name == "tencent":
        return tencent.list_devices(token)
    else:
        raise ValueError("Unsupported provider: %s" % provider.name)


def list_properties(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    if device is None:
        device = default_device
    device = Device.from_name(device, provider)
    if provider is None:
        provider = device.provider

    if token is None:
        token = device.get_token()  # type: ignore
    if provider.name == "tencent":  # type: ignore
        return tencent.list_properties(device, token)
    else:
        raise ValueError("Unsupported provider: %s" % provider.name)  # type: ignore


def submit_task(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
    token: Optional[str] = None,
    **task_kws: Any,
) -> List[Task]:
    if device is None:
        device = default_device
    device = Device.from_name(device, provider)
    if provider is None:
        provider = device.provider

    if token is None:
        token = device.get_token()  # type: ignore

    if provider.name == "tencent":  # type: ignore
        return tencent.submit_task(device, token, **task_kws)  # type: ignore
    else:
        raise ValueError("Unsupported provider: %s" % provider.name)  # type: ignore
