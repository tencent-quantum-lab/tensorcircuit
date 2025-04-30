"""
main entrypoints of cloud module
"""

from typing import Any, List, Optional, Dict, Union, Tuple
from base64 import b64decode, b64encode
from functools import partial
import json
import os
import sys
import logging

from .abstraction import Provider, Device, Task, sep, sep2

logger = logging.getLogger(__name__)


try:
    from . import tencent  # type: ignore
except (ImportError, ModuleNotFoundError):
    logger.warning("fail to load cloud provider module: tencent")

try:
    from . import local
except (ImportError, ModuleNotFoundError):
    logger.warning("fail to load cloud provider module: local")

try:
    from . import quafu_provider
except (ImportError, ModuleNotFoundError):
    pass
    # logger.warning("fail to load cloud provider module: quafu")

package_name = "tensorcircuit"
thismodule = sys.modules[__name__]


default_provider = Provider.from_name("tencent")
avail_providers = ["tencent", "local"]


def list_providers() -> List[Provider]:
    """
    list all cloud providers that tensorcircuit supports

    :return: _description_
    :rtype: List[Provider]
    """
    return [get_provider(s) for s in avail_providers]


def set_provider(
    provider: Optional[Union[str, Provider]] = None, set_global: bool = True
) -> Provider:
    """
    set default provider for the program

    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param set_global: whether set, defaults to True,
        if False, equivalent to ``get_provider``
    :type set_global: bool, optional
    :return: _description_
    :rtype: Provider
    """
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

default_device = Device.from_name("tencent::simulator:tc")


def set_device(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
    set_global: bool = True,
) -> Device:
    """
    set the default device

    :param provider: provider of the device, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: the device, defaults to None
    :type device: Optional[Union[str, Device]], optional
    :param set_global: whether set, defaults to True,
        if False, equivalent to ``get_device``, defaults to True
    :type set_global: bool, optional
    :return: _description_
    :rtype: Device
    """
    if provider is not None and device is None:
        provider, device = None, provider
    if device is None and provider is not None:
        raise ValueError("Please specify the device apart from the provider")
    if device is None:
        device = default_device

    if isinstance(device, str):
        if len(device.split(sep)) > 1:
            provider, device = device.split(sep)
            provider = Provider.from_name(provider)
            device = Device.from_name(device, provider)
        else:
            if provider is None:
                provider = get_provider()
            provider = Provider.from_name(provider)
            device = Device.from_name(device, provider)
    else:
        if provider is None:
            provider = get_provider()
        provider = Provider.from_name(provider)
        device = Device.from_name(device, provider)

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


def _preprocess(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
) -> Tuple[Provider, Device]:
    """
    Smartly determine the provider and device based on the input

    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: _description_, defaults to None
    :type device: Optional[Union[str, Device]], optional
    :return: a pair of provider and device after preprocessing
    :rtype: Tuple[Provider, Device]
    """
    if provider is not None and device is None:
        provider, device = None, provider
    if device is None:
        device = get_device()
    if isinstance(device, str):
        if len(device.split(sep)) > 1:
            device = Device.from_name(device, provider)
        else:
            if provider is None:
                provider = get_provider()
            device = Device.from_name(device, provider)
    if provider is None:
        provider = device.provider
    if isinstance(provider, str):
        provider = Provider.from_name(provider)
    return provider, device  # type: ignore


def set_token(
    token: Optional[str] = None,
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
    cached: bool = True,
    clear: bool = False,
) -> Dict[str, Any]:
    """
    Set API token for given provider or specifically to given device

    :param token: the API token, defaults to None
    :type token: Optional[str], optional
    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: _description_, defaults to None
    :type device: Optional[Union[str, Device]], optional
    :param cached: whether save on the disk, defaults to True
    :type cached: bool, optional
    :param clear: if True, clear all token saved, defaults to False
    :type clear: bool, optional
    :return: _description_
    :rtype: Dict[str, Any]
    """
    global saved_token
    homedir = os.path.expanduser("~")
    authpath = os.path.join(homedir, ".tc.auth.json")
    # provider, device = _preprocess(provider, device)
    if clear is True:
        saved_token = {}
    if token is None:
        if cached and os.path.exists(authpath):
            try:
                with open(authpath, "r") as f:
                    file_token = json.load(f)
                    file_token = {k: b64decode_s(v) for k, v in file_token.items()}
                    # file_token = backend.tree_map(b64decode_s, file_token)
            except json.JSONDecodeError:
                logger.warning("token file loading failure, set empty token instead")
                # TODO(@refraction-ray): better conflict solve with multiprocessing
                file_token = {}
        else:
            file_token = {}
        file_token.update(saved_token)
        saved_token = file_token
    else:  # with token
        if isinstance(provider, str):
            provider = Provider.from_name(provider)
        if device is None:
            if provider is None:
                provider = default_provider
            added_token = {provider.name + sep: token}
        else:
            device = Device.from_name(device)
            if provider is None:
                provider = device.provider  # type: ignore
            if provider is None:
                provider = default_provider
            added_token = {provider.name + sep + device.name: token}  # type: ignore
        saved_token.update(added_token)

    if cached:
        # file_token = backend.tree_map(b64encode_s, saved_token)
        file_token = {k: b64encode_s(v) for k, v in saved_token.items()}
        if file_token:
            with open(authpath, "w") as f:
                json.dump(file_token, f)

    return saved_token


set_token()


def get_token(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
) -> Optional[str]:
    """
    Get API token setted for given provider or device,
    if no device token saved, the corresponding provider tken is returned

    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: _description_, defaults to None
    :type device: Optional[Union[str, Device]], optional
    :return: _description_
    :rtype: Optional[str]
    """
    if provider is None:
        provider = get_provider()
    provider = Provider.from_name(provider)
    target = provider.name + sep
    if device is not None:
        device = Device.from_name(device, provider)
        target = target + device.name
    for k, v in saved_token.items():
        if k == target:
            return v  # type: ignore
    return None


# token json structure
# {"tencent::": token1, "tencent::20xmon":  token2}


def list_devices(
    provider: Optional[Union[str, Provider]] = None,
    token: Optional[str] = None,
    **kws: Any,
) -> List[Device]:
    """
    List all devices under a provider

    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param token: _description_, defaults to None
    :type token: Optional[str], optional
    :return: _description_
    :rtype: Any
    """
    if provider is None:
        provider = default_provider
    provider = Provider.from_name(provider)
    if token is None:
        token = provider.get_token()
    if provider.name == "tencent":
        return tencent.list_devices(token, **kws)  # type: ignore
    elif provider.name == "local":
        return local.list_devices(token, **kws)
    else:
        raise ValueError("Unsupported provider: %s" % provider.name)


def list_properties(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List properties of a given device

    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: _description_, defaults to None
    :type device: Optional[Union[str, Device]], optional
    :param token: _description_, defaults to None
    :type token: Optional[str], optional
    :return: Propeties dict
    :rtype: Dict[str, Any]
    """
    # if provider is not None and device is None:
    #     provider, device = None, provider
    # if device is None:
    #     device = default_device
    # device = Device.from_name(device, provider)
    # if provider is None:
    #     provider = device.provider
    provider, device = _preprocess(provider, device)

    if token is None:
        token = device.get_token()  # type: ignore
    if provider.name == "tencent":  # type: ignore
        return tencent.list_properties(device, token)  # type: ignore
    elif provider.name == "local":
        raise ValueError("Unsupported method for local backend")
    else:
        raise ValueError("Unsupported provider: %s" % provider.name)  # type: ignore


def get_task(
    taskid: str,
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
) -> Task:
    """
    Get ``Task`` object from task string, the binding device can also be provided

    :param taskid: _description_
    :type taskid: str
    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: _description_, defaults to None
    :type device: Optional[Union[str, Device]], optional
    :return: _description_
    :rtype: Task
    """
    if provider is not None and device is None:
        provider, device = None, provider
    if device is not None:  # device can be None for identify tasks
        device = Device.from_name(device, provider)
    elif len(taskid.split(sep2)) > 1:
        device = Device(taskid.split(sep2)[0])
        taskid = taskid.split(sep2)[1]
    return Task(taskid, device=device)


def get_task_details(
    taskid: Union[str, Task], token: Optional[str] = None, prettify: bool = False
) -> Dict[str, Any]:
    """
    Get task details dict given task id

    :param taskid: _description_
    :type taskid: Union[str, Task]
    :param token: _description_, defaults to None
    :type token: Optional[str], optional
    :param prettify: whether make the returned dict more readable and more phythonic,
        defaults to False
    :type prettify: bool
    :return: _description_
    :rtype: Dict[str, Any]
    """
    if isinstance(taskid, str):
        task = Task(taskid)
    else:
        task = taskid
    if task.device is not None:
        device = task.device
    else:
        device = default_device
    if token is None:
        token = device.get_token()
    provider = device.provider

    if provider.name == "tencent":
        return tencent.get_task_details(task, device, token, prettify)  # type: ignore
    elif provider.name == "local":
        return local.get_task_details(task, device, token, prettify)  # type: ignore
    elif provider.name == "quafu":
        return quafu_provider.get_task_details(task, device, token, prettify)  # type: ignore

    else:
        raise ValueError("Unsupported provider: %s" % provider.name)  # type: ignore


def submit_task(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
    token: Optional[str] = None,
    **task_kws: Any,
) -> List[Task]:
    """
    submit task to the cloud platform, batch submission default enabled

    .. seealso::

        :py:meth:`tensorcircuit.cloud.tencent.submit_task`

    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: _description_, defaults to None
    :type device: Optional[Union[str, Device]], optional
    :param token: _description_, defaults to None
    :type token: Optional[str], optional
    :param task_kws: all necessary keywords arguments for task submission,
        see detailed API in each provider backend:
        1. tencent - :py:meth:`tensorcircuit.cloud.tencent.submit_task`
    :type task_kws: Any
    :return: The task object
    :rtype: List[Task]
    """
    # if device is None:
    #     device = get_device()
    # if isinstance(device, str):
    #     if len(device.split(sep)) > 1:
    #         device = Device(device, provider)
    #     else:
    #         if provider is None:
    #             provider = get_provider()
    #         device = Device(device, provider)
    # if provider is None:
    #     provider = device.provider
    provider, device = _preprocess(provider, device)

    if token is None:
        token = device.get_token()  # type: ignore
    if provider.name == "tencent":  # type: ignore
        return tencent.submit_task(device, token, **task_kws)  # type: ignore
    elif provider.name == "local":  # type: ignore
        return local.submit_task(device, token, **task_kws)  # type: ignore
    elif provider.name == "quafu":  # type: ignore
        return quafu_provider.submit_task(device, token, **task_kws)  # type: ignore
    else:
        raise ValueError("Unsupported provider: %s" % provider.name)  # type: ignore


def resubmit_task(
    task: Optional[Union[str, Task]],
    token: Optional[str] = None,
) -> Task:
    """
    Rerun the given task

    :param task: _description_
    :type task: Optional[Union[str, Task]]
    :param token: _description_, defaults to None
    :type token: Optional[str], optional
    :return: _description_
    :rtype: Task
    """
    if isinstance(task, str):
        task = Task(task)
    device = task.get_device()  # type: ignore
    if token is None:
        token = device.get_token()
    provider = device.provider

    if provider.name == "tencent":  # type: ignore
        return tencent.resubmit_task(task, token)  # type: ignore
    elif provider.name == "local":
        raise ValueError("Unsupported method for local backend")
    else:
        raise ValueError("Unsupported provider: %s" % provider.name)  # type: ignore


def remove_task(
    task: Optional[Union[str, Task]],
    token: Optional[str] = None,
) -> Task:
    if isinstance(task, str):
        task = Task(task)
    device = task.get_device()  # type: ignore
    if token is None:
        token = device.get_token()
    provider = device.provider

    if provider.name == "tencent":  # type: ignore
        return tencent.remove_task(task, token)  # type: ignore
    elif provider.name == "local":
        raise ValueError("Unsupported method for local backend")
    else:
        raise ValueError("Unsupported provider: %s" % provider.name)  # type: ignore


def list_tasks(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
    token: Optional[str] = None,
    **filter_kws: Any,
) -> List[Task]:
    """
    List tasks based on given filters

    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: _description_, defaults to None
    :type device: Optional[Union[str, Device]], optional
    :param token: _description_, defaults to None
    :type token: Optional[str], optional
    :return: list of task object that satisfy these filter criteria
    :rtype: List[Task]
    """
    if provider is None:
        provider = default_provider
    provider = Provider.from_name(provider)
    if token is None:
        token = provider.get_token()  # type: ignore
    if device is not None:
        device = Device.from_name(device)
    if provider.name == "tencent":  # type: ignore
        return tencent.list_tasks(device, token, **filter_kws)  # type: ignore
    elif provider.name == "local":  # type: ignore
        return local.list_tasks(device, token, **filter_kws)  # type: ignore
    else:
        raise ValueError("Unsupported provider: %s" % provider.name)  # type: ignore
