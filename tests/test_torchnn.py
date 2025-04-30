import os
import sys
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)

import tensorcircuit as tc

try:
    import torch
except ImportError:
    pytest.skip("torch is not installed", allow_module_level=True)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
def test_quantumnet(backend):
    n = 6
    nlayers = 2

    def qpred(x, weights):
        c = tc.Circuit(n)
        for i in range(n):
            c.rx(i, theta=x[i])
        for j in range(nlayers):
            for i in range(n - 1):
                c.cnot(i, i + 1)
            for i in range(n):
                c.rx(i, theta=weights[2 * j, i])
                c.ry(i, theta=weights[2 * j + 1, i])
        ypred = tc.backend.stack([c.expectation_ps(x=[i]) for i in range(n)])
        return tc.backend.real(ypred)

    if tc.backend.name == "pytorch":
        use_interface = False
    else:
        use_interface = True

    ql = tc.TorchLayer(
        qpred, weights_shape=[2 * nlayers, n], use_interface=use_interface
    )

    yp = ql(torch.ones([3, n]))
    print(yp)

    np.testing.assert_allclose(yp.shape, np.array([3, n]))


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_inputs_multiple(backend):
    n = 3
    p = 0.1
    K = tc.backend
    torchb = tc.get_backend("pytorch")

    def f(state, noise, weights):
        c = tc.Circuit(n, inputs=state)
        for i in range(n):
            c.rz(i, theta=weights[i])
        for i in range(n):
            c.depolarizing(i, px=p, py=p, pz=p, status=noise[i])
        return K.real(c.expectation_ps(x=[0]))

    layer = tc.TorchLayer(f, [n], use_vmap=True, vectorized_argnums=[0, 1])
    state = torchb.ones([2, 2**n]) / 2 ** (n / 2)
    noise = 0.2 * torchb.ones([2, n], dtype="float32")
    l = layer(state, noise)
    lsum = torchb.sum(l)
    print(l)
    lsum.backward()
    for p in layer.parameters():
        print(p.grad)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
def test_torchnn_hardware(backend):
    n = 2

    def qf(inputs, param):
        inputs = tc.backend.convert_to_tensor(tc.get_backend("pytorch").numpy(inputs))
        param = tc.backend.convert_to_tensor(tc.get_backend("pytorch").numpy(param))

        c = tc.Circuit(n)
        c.rx(0, theta=inputs[0])
        c.rx(1, theta=inputs[1])
        c.h(1)
        c.rzz(0, 1, theta=param[0])
        r = tc.backend.stack([c.expectation_ps(z=[i]) for i in range(n)])

        r = tc.get_backend("pytorch").convert_to_tensor(tc.backend.numpy(r))
        return torch.real(r)

    ql = tc.torchnn.HardwareNet(qf, [1])
    qnet = torch.nn.Sequential(ql, torch.nn.Linear(2, 1))
    print(qnet(torch.ones([5, 2])))
