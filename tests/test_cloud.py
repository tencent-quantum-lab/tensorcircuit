import sys
import os
import time
import pytest
import numpy as np

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc
from tensorcircuit.cloud import apis, wrapper
from tensorcircuit.results import counts

if "TC_CLOUD_TEST" not in os.environ:
    pytest.skip(allow_module_level=True)
    # skip on CI due to no token


def test_get_token():
    print(apis.get_token(provider="Tencent"))
    p = apis.get_provider("tencent")
    print(p.get_token())
    print(p.get_device("simulator:tc").get_token())


def test_list_devices():
    print(apis.list_devices())
    p = apis.get_provider()
    print(p.list_devices())
    print(p.list_devices(state="on"))


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


def test_get_device_cache():
    d1 = apis.get_device("local::testing")
    d2 = apis.get_device(provider="local", device="testing")
    apis.set_provider("local")
    d3 = apis.get_device("testing")
    assert id(d1) == id(d2)
    assert id(d3) == id(d1)
    apis.set_provider("tencent")
    d4 = apis.get_device("testing")
    assert d4.provider.name == "tencent"
    assert id(d4) != id(d1)


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
    t = apis.submit_task(device="simulator:tc", circuit=c)
    r = t.details()
    assert r["state"] in ["pending", "completed"]
    print(t.results(blocked=True))
    assert t.get_logical_physical_mapping() == {0: 0, 1: 1, 2: 2}


def test_resubmit_task():
    c = tc.Circuit(3)
    c.H(0)
    c.H(1)
    t = apis.submit_task(device="simulator:aer", circuit=c)
    time.sleep(15)
    t1 = apis.resubmit_task(t)
    print(t.details())
    print(t1.details(wait=True))


def test_get_task():
    apis.set_device("simulator:tcn1")
    c = tc.Circuit(2)
    c.cx(0, 1)
    t = apis.submit_task(circuit=c)
    t1 = apis.get_task(t.id_)
    assert t1.id_ == t.id_
    t2 = apis.get_device("tencent::simulator:tcn1").get_task(t.id_)
    assert t2.id_ == t.id_

    apis.set_device()


def test_list_tasks():
    d = apis.get_device(device="simulator:aer")
    print(d.list_tasks())
    print(apis.list_tasks(device="simulator:tc"))


def test_local_list_device():
    dd = apis.list_devices(provider="local")
    assert dd[0].name == "testing"


def test_local_submit_task():
    c = tc.Circuit(2)
    c.h(0)
    c.cx(0, 1)

    t = apis.submit_task(device="local::testing", circuit=c, shots=2048)
    r = t.results(blocked=True)
    assert counts.kl_divergence({"00": 0.5, "11": 0.5}, r) < 0.1
    print(t.details())
    print(t.get_device())


def test_local_list_tasks():
    print(apis.list_tasks(provider="local"))


def test_local_batch_submit():
    apis.set_provider("local")
    c = tc.Circuit(2)
    c.h(1)
    c.ry(1, theta=0.8)

    ts = apis.submit_task(device="testing", circuit=[c, c])
    print(ts[0].results())

    apis.set_device("testing")
    ts = apis.submit_task(circuit=[c, c])
    print(ts[1].results())
    print(ts[1].details())
    apis.set_provider("tencent")


def test_batch_exp_ps():
    pss = [[1, 0], [0, 3]]
    c = tc.Circuit(2)
    c.h(0)
    c.x(1)
    np.testing.assert_allclose(wrapper.batch_expectation_ps(c, pss), [1, -1], atol=1e-5)
    np.testing.assert_allclose(
        wrapper.batch_expectation_ps(c, pss, ws=[1, -0.5]), 1.5, atol=1e-5
    )
    np.testing.assert_allclose(
        wrapper.batch_expectation_ps(c, pss, device="simulator:tcn1"),
        [1, -1],
        atol=1e-1,
    )
    np.testing.assert_allclose(
        wrapper.batch_expectation_ps(c, pss, device="local::default", with_rem=False),
        [1, -1],
        atol=1e-1,
    )


def test_batch_submit_template():
    run = tc.cloud.wrapper.batch_submit_template(
        device="simulator:tc", batch_limit=2, prior=10
    )
    cs = []
    for i in range(4):
        c = tc.Circuit(4)
        c.h(i)
        cs.append(c)

    rs = run(cs[:3], prior=1)
    assert len(rs) == 3
    rs = run(cs[:4])
    assert len(rs) == 4


def test_allz_batch():
    n = 5

    def makec(inputs, params):
        c = tc.Circuit(n)
        for i in range(n):
            c.rx(i, theta=inputs[i])
        for i in range(n):
            c.rz(i, theta=params[0, i])
        for i in range(n - 1):
            c.cx(i, i + 1)
        for i in range(n):
            c.rx(i, theta=params[1, i])
        return c

    pss = []
    for i in range(n):
        ps = [0 for _ in range(n)]
        ps[i] = 3
        pss.append(ps)

    def exp_val(c, device=None):
        rs = tc.cloud.wrapper.batch_expectation_ps(c, pss, device)
        return tc.backend.stack(rs)

    def qmlf(inputs, params, device=None):
        c = makec(inputs, params)
        return exp_val(c, device)

    inputs = tc.array_to_tensor(np.array([0, 1, 0, 1, 0]))
    params = np.ones([2, n])
    print(qmlf(inputs, params, device="9gmon"))
    print(qmlf(inputs, params))
