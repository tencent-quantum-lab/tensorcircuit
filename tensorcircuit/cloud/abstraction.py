"""
Abstraction for Provider, Device and Task
"""

from typing import Any, Dict, List, Optional, Union
import time


class Provider:
    activated_providers: Dict[str, "Provider"] = {}

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
            if provider in cls.activated_providers:
                return cls.activated_providers[provider]
            else:
                p = cls(provider)
                cls.activated_providers[provider] = p
        else:
            raise ValueError(
                "Unsupported format for `provider` argument: %s" % provider
            )
        return p

    def set_token(self, token: str, cached: bool = True) -> Any:
        from .apis import set_token

        return set_token(token, self, cached=cached)

    def get_token(self) -> str:
        from .apis import get_token

        return get_token(self)  # type: ignore

    def list_devices(self) -> Any:
        from .apis import list_devices

        return list_devices(self)

    def get_device(self, device: Optional[Union[str, "Device"]]) -> "Device":
        from .apis import get_device

        return get_device(self, device)

    def list_tasks(self, **filter_kws: Any) -> List["Task"]:
        from .apis import list_tasks

        return list_tasks(self, **filter_kws)


sep = "::"


class Device:
    activated_devices: Dict[str, "Device"] = {}

    def __init__(
        self,
        name: str,
        provider: Optional[Union[str, Provider]] = None,
        lower: bool = True,
    ):
        if lower is True:
            name = name.lower()
        if provider is not None:
            self.provider = Provider.from_name(provider)
            if len(name.split(sep)) > 1:
                self.name = name.split(sep)[1]
            else:
                self.name = name
        else:  # no explicit provider
            if len(name.split(sep)) == 1:
                name = "tencent" + sep + name  # default provider
            self.name = name.split(sep)[1]
            self.provider = Provider.from_name(name.split(sep)[0])

    def __str__(self) -> str:
        return self.provider.name + sep + self.name

    __repr__ = __str__

    @classmethod
    def from_name(
        cls,
        device: Optional[Union[str, "Device"]] = None,
        provider: Optional[Union[str, Provider]] = None,
    ) -> "Device":
        if device is None:
            raise ValueError("Must specify on device instead of default ``None``")
        if isinstance(device, cls):
            d = device
        elif isinstance(device, str):
            if device in cls.activated_devices:
                return cls.activated_devices[device]
            else:
                d = cls(device, provider)
                cls.activated_devices[device] = d
        else:
            raise ValueError("Unsupported format for `provider` argument: %s" % device)
        return d

    def set_token(self, token: str, cached: bool = True) -> Any:
        from .apis import set_token

        return set_token(token, provider=self.provider, device=self, cached=cached)

    def get_token(self) -> Optional[str]:
        from .apis import get_token

        s = get_token(provider=self.provider, device=self)
        if s is not None:
            return s
        # fallback to provider default
        return get_token(provider=self.provider)

    def list_properties(self) -> Dict[str, Any]:
        from .apis import list_properties

        return list_properties(self.provider, self)

    def submit_task(self, **task_kws: Any) -> List["Task"]:
        from .apis import submit_task

        return submit_task(provider=self.provider, device=self, **task_kws)

    def get_task(self, taskid: str) -> "Task":
        from .apis import get_task

        return get_task(taskid, device=self)

    def list_tasks(self, **filter_kws: Any) -> List["Task"]:
        from .apis import list_tasks

        return list_tasks(self.provider, self, **filter_kws)


class Task:
    def __init__(self, id_: str, device: Optional[Device] = None):
        self.id_ = id_
        self.device = device

    def __repr__(self) -> str:
        return self.device.__repr__() + "~~" + self.id_

    __str__ = __repr__

    def get_device(self) -> Device:
        if self.device is None:
            return Device.from_name(self.details()["device"])
        else:
            return Device.from_name(self.device)

    def details(self) -> Dict[str, Any]:
        from .apis import get_task_details

        return get_task_details(self)

    def state(self) -> str:
        r = self.details()
        return r["state"]  # type: ignore

    status = state

    def resubmit(self) -> "Task":
        from .apis import resubmit_task

        return resubmit_task(self)

    def results(self, format: Optional[str] = None, blocked: bool = False) -> Any:
        # TODO(@refraction-ray): support different formats compatible with tc,
        # also support format_ alias
        if not blocked:
            if self.state() != "completed":
                raise ValueError("Task %s is not completed yet" % self.id_)
            r = self.details()["results"]
            r = {k: v for k, v in sorted(r.items(), key=lambda item: -item[1])}  # type: ignore
            return r
        else:
            s = self.state()
            while s != "completed":
                if s in ["failed"]:
                    raise ValueError("Task %s is in %s state" % (self.id_, s))
                time.sleep(1.0)
                s = self.state()
            return self.results(format=format, blocked=False)
