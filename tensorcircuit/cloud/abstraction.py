"""
Abstraction for Provider, Device and Task
"""

from typing import Any, Dict, List, Optional, Union
from functools import partial
import time

from ..results import readout_mitigation as rem
from ..results import counts
from ..utils import arg_alias


class TCException(BaseException):
    pass


class TaskException(TCException):
    pass


class TaskUnfinished(TaskException):
    def __init__(self, taskid: str, state: str):
        self.taskid = taskid
        self.state = state
        super().__init__(
            "Task %s is not completed yet, now in %s state" % (self.taskid, self.state)
        )


class TaskFailed(TaskException):
    def __init__(self, taskid: str, state: str, message: str):
        self.taskid = taskid
        self.state = state
        self.message = message
        super().__init__(
            "Task %s is in %s state with err message %s"
            % (self.taskid, self.state, self.message)
        )


class Provider:
    """
    Provider abstraction for cloud connection, eg. "tencent", "local"
    """

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
    """
    Device abstraction for cloud connection, eg. quantum chips
    """

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
            if len(name.split(sep)) > 1:
                self.name = name.split(sep)[1]
                self.provider = Provider.from_name(name.split(sep)[0])
            else:
                from .apis import get_provider

                self.name = name
                self.provider = get_provider()

    def __str__(self) -> str:
        return self.provider.name + sep + self.name

    __repr__ = __str__

    @classmethod
    def from_name(
        cls,
        device: Union[str, "Device"],
        provider: Optional[Union[str, Provider]] = None,
    ) -> "Device":
        # if device is None:
        # raise ValueError("Must specify on device instead of default ``None``")
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

    def get_task(self, taskid: str) -> "Task":
        from .apis import get_task

        return get_task(taskid, device=self)

    def submit_task(self, **task_kws: Any) -> List["Task"]:
        from .apis import submit_task

        return submit_task(provider=self.provider, device=self, **task_kws)

    def list_tasks(self, **filter_kws: Any) -> List["Task"]:
        from .apis import list_tasks

        return list_tasks(self.provider, self, **filter_kws)


sep2 = "~~"


class Task:
    """
    Task abstraction for quantum jobs on the cloud
    """

    def __init__(self, id_: str, device: Optional[Device] = None):
        self.id_ = id_
        self.device = device

    def __repr__(self) -> str:
        return self.device.__repr__() + sep2 + self.id_

    __str__ = __repr__

    def get_device(self) -> Device:
        """
        Query which device the task is run on

        :return: _description_
        :rtype: Device
        """
        if self.device is None:
            return Device.from_name(self.details()["device"])
        else:
            return Device.from_name(self.device)

    def details(self, **kws: Any) -> Dict[str, Any]:
        """
        Get the current task details

        :return: _description_
        :rtype: Dict[str, Any]
        """
        from .apis import get_task_details

        return get_task_details(self, **kws)

    def state(self) -> str:
        """
        Query the current task status

        :return: _description_
        :rtype: str
        """
        r = self.details()
        return r["state"]  # type: ignore

    status = state

    def resubmit(self) -> "Task":
        """
        resubmit the task

        :return: the resubmitted task
        :rtype: Task
        """
        from .apis import resubmit_task

        return resubmit_task(self)

    @partial(arg_alias, alias_dict={"format": ["format_"]})
    def results(
        self,
        format: Optional[str] = None,
        blocked: bool = False,
        mitigated: bool = False,
        calibriation_options: Optional[Dict[str, Any]] = None,
        readout_mit: Optional[rem.ReadoutMit] = None,
        mitigation_options: Optional[Dict[str, Any]] = None,
    ) -> counts.ct:
        """
        get task results of the qjob

        :param format: unsupported now, defaults to None, which is "count_dict_bin"
        :type format: Optional[str], optional
        :param blocked: whether blocked to wait until the result is returned, defaults to False,
            which raise error when the task is unfinished
        :type blocked: bool, optional
        :param mitigated: whether enable readout error mitigation, defaults to False
        :type mitigated: bool, optional
        :param calibriation_options: option dict for ``ReadoutMit.cals_from_system``,
            defaults to None
        :type calibriation_options: Optional[Dict[str, Any]], optional
        :param readout_mit: if given, directly use the calibriation info on ``readout_mit``,
            defaults to None
        :type readout_mit: Optional[rem.ReadoutMit], optional
        :param mitigation_options: option dict for ``ReadoutMit.apply_correction``, defaults to None
        :type mitigation_options: Optional[Dict[str, Any]], optional
        :return: count dict results
        :rtype: Any
        """
        if not blocked:
            s = self.state()
            if s != "completed":
                raise TaskUnfinished(self.id_, s)
            r = self.details()["results"]
            r = counts.sort_count(r)  # type: ignore
        else:
            s = self.state()
            tries = 0
            while s != "completed":
                if s in ["failed"]:
                    err = self.details().get("err", "")
                    raise TaskFailed(self.id_, s, err)
                time.sleep(0.5 + tries / 10)
                tries += 1
                s = self.state()
            r = self.results(format=format, blocked=False, mitigated=False)
        if mitigated is False:
            return r  # type: ignore
        nqubit = len(list(r.keys())[0])

        # mitigated is True:
        if readout_mit is None and getattr(self, "readout_mit", None) is None:

            def run(cs: Any, shots: Any) -> Any:
                """
                current workaround for batch
                """
                from .apis import submit_task

                # ts = []
                # for c in cs:
                #     ts.append(
                #         submit_task(circuit=c, shots=shots, device=self.get_device())
                #     )
                #     time.sleep(0.3)
                ts = submit_task(circuit=cs, shots=shots, device=self.get_device())
                return [t.results(blocked=True) for t in ts]  # type: ignore

            shots = self.details()["shots"]
            readout_mit = rem.ReadoutMit(run)
            if calibriation_options is None:
                calibriation_options = {}
            readout_mit.cals_from_system(
                list(range(nqubit)), shots, **calibriation_options
            )
            self.readout_mit = readout_mit
        elif readout_mit is None:
            readout_mit = self.readout_mit

        if mitigation_options is None:
            mitigation_options = {}
        miti_count = readout_mit.apply_correction(
            r, list(range(nqubit)), **mitigation_options
        )
        return counts.sort_count(miti_count)
