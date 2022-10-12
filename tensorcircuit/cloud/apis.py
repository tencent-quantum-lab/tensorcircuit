"""
main entrypoints of cloud module
"""

from typing import Any, Callable, Optional, Dict, Sequence, Union
from base64 import b64decode, b64encode
from functools import partial, wraps
import json
import os
import sys

from ..cons import backend

package_name = "tensorcircuit"
thismodule = sys.modules[__name__]


class Provider:
    activated_provider: Dict[str, "Provider"] = {}

    def __init__(self, name: str, lower: bool = True):
        if lower is True:
            name = name.lower()
        self.name = name

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__

    @classmethod
    def from_name(cls, provider: Union[str, "Provider"] = "tencent") -> "Provider":
        if provider is None:
            provider = "tencent"
        if isinstance(provider, cls):
            p = provider
        elif isinstance(provider, str):
            if provider in cls.activated_provider:
                return cls.activated_provider[provider]
            else:
                p = cls(provider)
                cls.activated_provider[provider] = p
        else:
            raise ValueError(
                "Unsupported format for `provider` argument: %s" % provider
            )
        return p


class Device:
    def __init__(self, name: str, lower: bool = False):
        if lower is True:
            name = name.lower()
        self.name = name

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__

    @classmethod
    def from_name(cls, device: Optional[Union[str, "Device"]] = None) -> "Device":
        if device is None:
            raise ValueError("Must specify on device instead of default ``None``")
        if isinstance(device, cls):
            d = device
        elif isinstance(device, str):
            d = cls(device)
        else:
            raise ValueError("Unsupported format for `provider` argument: %s" % device)
        return d


def convert_provider_device(
    f: Callable[..., Any],
    provider_argnums: Optional[Sequence[int]] = None,
    device_argnums: Optional[Sequence[int]] = None,
    provider_kws: Optional[Sequence[str]] = None,
    device_kws: Optional[Sequence[str]] = None,
) -> Callable[..., Any]:
    if provider_argnums is None:
        provider_argnums = []
    if isinstance(provider_argnums, int):
        provider_argnums = [provider_argnums]
    if device_argnums is None:
        device_argnums = []
    if isinstance(device_argnums, int):
        device_argnums = [device_argnums]
    if provider_kws is None:
        provider_kws = []
    if device_kws is None:
        device_kws = []

    @wraps(f)
    def wrapper(*args: Any, **kws: Any) -> Any:
        nargs = list(args)
        nkws = kws
        for i, arg in enumerate(args):
            if i in provider_argnums:  # type: ignore
                nargs[i] = Provider.from_name(arg)
            elif i in device_argnums:  # type: ignore
                nargs[i] = Device.from_name(arg)
        for k, v in kws.items():
            if k in provider_kws:  # type: ignore
                nkws[k] = Provider.from_name(v)
            if k in device_kws:  # type: ignore
                nkws[k] = Device.from_name(v)
        return f(*nargs, **nkws)

    return wrapper


@partial(convert_provider_device, provider_argnums=0, provider_kws="provider")
def set_provider(provider: Optional[str] = None, set_global: bool = True) -> Provider:
    if set_global:
        for module in sys.modules:
            if module.startswith(package_name):
                setattr(sys.modules[module], "provider", provider)
    return provider  # type: ignore


get_provider = partial(set_provider, set_global=False)


def b64encode_s(s: str) -> str:
    return b64encode(s.encode("utf-8")).decode("utf-8")


def b64decode_s(s: str) -> str:
    return b64decode(s.encode("utf-8")).decode("utf-8")


saved_token: Dict[str, Any] = {}


@partial(
    convert_provider_device,
    provider_argnums=1,
    provider_kws="provider",
    device_argnums=2,
    device_kws="device",
)
def set_token(
    token: Optional[str] = None,
    provider: Optional[str] = None,
    device: Optional[str] = None,
    cached: bool = True,
) -> Dict[str, Any]:
    global saved_token
    tcdir = os.path.dirname(os.path.abspath(__file__))
    authpath = os.path.join(tcdir, "auth.json")

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
                provider = Provider.from_name("tencent")  # type: ignore
            added_token = {provider.name + "~": token}  # type: ignore
        else:
            added_token = {provider.name + "~" + device.name: token}  # type: ignore
        saved_token.update(added_token)

    if cached:
        file_token = backend.tree_map(b64encode_s, saved_token)
        with open(authpath, "w") as f:
            json.dump(file_token, f)

    return saved_token


set_token()
# token json structure
# {"tencent~": token1, "tencent~20xmon":  token2}
