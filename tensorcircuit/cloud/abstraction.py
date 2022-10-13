"""
Abstraction for Provider, Device and Task
"""

from typing import Any, Dict, Optional, Union


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
            return s  # type: ignore
        # fallback to provider default
        return get_token(provider=self.provider)  # type: ignore
