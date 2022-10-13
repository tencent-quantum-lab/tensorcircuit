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
