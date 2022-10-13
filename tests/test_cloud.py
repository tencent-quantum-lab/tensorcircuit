import sys
import os

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
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
    d1 = apis.get_device(device="tecent::hello")
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
