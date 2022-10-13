"""
Abstraction for Provider, Device and Task
"""

from typing import Any, Dict, Optional, Union


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

    def set_token(self, token: str, cached: bool = True) -> Any:
        from .apis import set_token

        return set_token(token, self, cached=cached)

    def get_token(self) -> str:
        from .apis import get_token

        return get_token(self)  # type: ignore


class Device:
    def __init__(self, name: str, lower: bool = True):
        if lower is True:
            name = name.lower()
        if len(name.split("~")) == 1:
            name = "tencent" + name  # default provider
        self.name = name.split("~")[1]
        self.provider = Provider.from_name(name.split("~")[0])

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

    def set_token(self, token: str, cached: bool = True) -> Any:
        from .apis import set_token

        return set_token(token, provider=self.provider, device=self, cached=cached)

    def get_token(self) -> str:
        from .apis import get_token

        s = get_token(provider=self.provider, device=self)
        if s is not None:
            return s  # type: ignore
        # fallback to provider default
        s = get_token(provider=self.provider)
        if s is not None:
            return s  # type: ignore
        raise LookupError("no token for device: %s" % self.name)
