import sys
import os
import pytest

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc
from tensorcircuit.cloud import apis
from tensorcircuit.cloud import config


def test_get_token():
    apis.set_token(config.tencent_token)
    assert apis.get_token(provider="Tencent") == config.tencent_token
    p = apis.get_provider("tencent")
    assert p.get_token() == config.tencent_token


def test_list_devices():
    print(apis.list_devices())
    p = apis.get_provider()
    print(p.list_devices())


def test_get_device():
    d1 = apis.get_device(device="tencent::hello")
    assert d1.name == "hello"
    assert d1.provider.name == "tencent"
    d2 = apis.get_device(device="hello")
    assert d2.name == "hello"
    assert d2.provider.name == "tencent"
    p = apis.get_provider()
    d3 = p.get_device("tencent::hello")
    assert d3.name == "hello"
    assert d3.provider.name == "tencent"
    d4 = p.get_device("hello")
    assert d4.name == "hello"
    assert d4.provider.name == "tencent"


def test_list_properties():
    d = apis.get_device(device="simulator:aer")
    print(d.list_properties())
    print(apis.list_properties(device="simulator:aer"))
    with pytest.raises(ValueError):
        apis.list_properties(device="hell")


def test_submit_task():
    c = tc.Circuit(3)
    c.H(0)
    c.H(1)
    c.H(2)
    t = apis.submit_task(device="simulator:aer", circuit=c)
    r = t.details()
    assert r["task"]["state"] in ["pending", "completed"]
