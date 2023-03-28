"""
Abstraction for Provider, Device and Task
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from functools import partial
import time

import networkx as nx

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

    def list_devices(self, **kws: Any) -> Any:
        from .apis import list_devices

        return list_devices(self, **kws)

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
        lower: bool = False,
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
        self.readout_mit: Any = None

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
            if len(device.split(sep)) > 1:
                provider = device.split(sep)[0]
                device = device.split(sep)[1]
            if provider is None:
                pn = ""
            elif isinstance(provider, str):
                pn = provider
            else:
                pn = provider.name
            if pn + sep + device in cls.activated_devices:
                return cls.activated_devices[pn + sep + device]
            else:
                d = cls(device, provider)

                cls.activated_devices[pn + sep + device] = d
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
        """
        List all device properties in as dict

        :return: [description]
        :rtype: Dict[str, Any]
        """
        from .apis import list_properties

        return list_properties(self.provider, self)

    def native_gates(self) -> List[str]:
        """
        List native gates supported for the device, str conforms qiskit convention

        :return: _description_
        :rtype: List[str]
        """
        properties = self.list_properties()

        if "native_gates" in properties:
            return properties["native_gates"]  # type: ignore
        return []

    def topology(self) -> List[Tuple[int, int]]:
        """
        Get the bidirectional topology link list of the device

        :return: [description]
        :rtype: List[Tuple[int, int]]
        """
        properties = self.list_properties()
        if "links" not in properties:
            return  # type: ignore
        links = []
        for link in properties["links"]:
            links.append((link[0], link[1]))
            links.append((link[1], link[0]))
        links = list(set(links))
        links = [list(link) for link in links]  # type: ignore
        # compatible with coupling_map in qiskit
        return links

    def topology_graph(self, visualize: bool = False) -> nx.Graph:
        """
        Get the qubit topology in ``nx.Graph`` or directly visualize it

        :param visualize: [description], defaults to False
        :type visualize: bool, optional
        :return: [description]
        :rtype: nx.Graph
        """
        pro = self.list_properties()
        if not ("links" in pro and "bits" in pro):
            return  # type: ignore
        g = nx.Graph()
        node_color = []
        edge_color = []
        for i in pro["bits"]:
            g.add_node(i)
            node_color.append(pro["bits"][i]["T1"])
        for e1, e2 in pro["links"]:
            g.add_edge(e1, e2)
            edge_color.append(pro["links"][(e1, e2)]["CZErrRate"])
        if visualize is False:
            return g
        from matplotlib import colormaps

        # pos1 = nx.planar_layout(g)
        # pos2 = nx.spring_layout(g, pos=pos1, k=2)
        pos = nx.kamada_kawai_layout(g)
        return nx.draw(
            g,
            pos=pos,
            with_labels=True,
            node_size=600,
            node_color=node_color,
            cmap=colormaps["Wistia"],
            vmin=max(min(node_color) - 5, 0),
            width=2.5,
            edge_color=edge_color,
            edge_cmap=colormaps["gray"],
            edge_vmin=0,
            edge_vmax=max(edge_color) * 1.2,
        )

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
        self.more_details: Dict[str, Any] = {}

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

    @partial(arg_alias, alias_dict={"blocked": ["wait"]})
    def details(self, blocked: bool = False, **kws: Any) -> Dict[str, Any]:
        """
        Get the current task details


        :param blocked: whether return until task is finished, defaults to False
        :type blocked: bool
        :return: _description_
        :rtype: Dict[str, Any]
        """
        from .apis import get_task_details

        if blocked is False:
            dt = get_task_details(self, **kws)
            dt.update(self.more_details)
            return dt
        s = self.state()
        tries = 0
        while s == "pending":
            time.sleep(0.5 + tries / 10)
            tries += 1
            s = self.state()
        return self.details(**kws)  # type: ignore

    def get_logical_physical_mapping(self) -> Optional[Dict[int, int]]:
        d = self.details()
        try:
            mp = d["optimization"]["pairs"]
        except KeyError:
            if "qubits" in d and isinstance(d["qubits"], int):
                mp = {i: i for i in range(d["qubits"])}
            else:
                mp = None
        return mp  # type: ignore

    def add_details(self, **kws: Any) -> None:
        self.more_details.update(kws)

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

    @partial(arg_alias, alias_dict={"format": ["format_"], "blocked": ["wait"]})
    def results(
        self,
        format: Optional[str] = None,
        blocked: bool = True,
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
        device = self.get_device()
        if device.provider.name != "tencent":
            raise ValueError("Only tencent provider supports auto readout mitigation")
        if readout_mit is None and getattr(device, "readout_mit", None) is None:

            def run(cs: Any, shots: Any) -> Any:
                """
                current workaround for batch
                """
                from .apis import submit_task

                ts = submit_task(circuit=cs, shots=shots, device=device.name + "?o=0")
                return [t.results(blocked=True) for t in ts]  # type: ignore

            shots = self.details()["shots"]
            readout_mit = rem.ReadoutMit(run)
            if calibriation_options is None:
                calibriation_options = {}
            readout_mit.cals_from_system(
                list(range(nqubit)), shots, **calibriation_options
            )
            device.readout_mit = readout_mit
        elif readout_mit is None:
            readout_mit = device.readout_mit

        if mitigation_options is None:
            try:
                mitigation_options = {
                    "logical_physical_mapping": self.details()["optimization"]["pairs"]
                }
            except KeyError:
                mitigation_options = {}
        miti_count = readout_mit.apply_correction(
            r, list(range(nqubit)), **mitigation_options
        )
        return counts.sort_count(miti_count)
