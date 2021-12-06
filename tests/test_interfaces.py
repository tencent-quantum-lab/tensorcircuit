import os
import sys
import pytest
from pytest_lazyfixture import lazy_fixture as lf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)

try:
    import torch
except ImportError:
    pytest.skip("torch not available", allow_module_level=True)

import numpy as np
import tensorcircuit as tc
from tensorcircuit import interfaces


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_torch_interface(backend):
    n = 4

    def f(param):
        c = tc.Circuit(n)
        c = tc.templates.blocks.example_block(c, param)
        loss = c.expectation(
            [
                tc.gates.x(),
                [
                    1,
                ],
            ]
        )
        return tc.backend.real(loss)

    f_jit = tc.backend.jit(f)

    f_jit_torch = interfaces.torch_interface(f_jit)

    param = torch.ones([4, n], requires_grad=True)
    l = f_jit_torch(param)
    l = l ** 2
    l.backward()

    pg = param.grad
    np.testing.assert_allclose(pg.shape, [4, n])
    np.testing.assert_allclose(pg[0, 1], -2.146e-3, atol=1e-5)

    def f2(paramzz, paramx):
        c = tc.Circuit(n)
        for i in range(n):
            c.H(i)
        for j in range(2):
            for i in range(n - 1):
                c.exp1(i, i + 1, unitary=tc.gates._zz_matrix, theta=paramzz[j, i])
            for i in range(n):
                c.rx(i, theta=paramx[j, i])
        loss1 = c.expectation(
            [
                tc.gates.x(),
                [
                    1,
                ],
            ]
        )
        loss2 = c.expectation(
            [
                tc.gates.x(),
                [
                    2,
                ],
            ]
        )
        return tc.backend.real(loss1), tc.backend.real(loss2)

    f2_torch = interfaces.torch_interface(f2, jit=True)

    paramzz = torch.ones([2, n], requires_grad=True)
    paramx = torch.ones([2, n], requires_grad=True)

    l1, l2 = f2_torch(paramzz, paramx)
    l = l1 - l2
    l.backward()

    pg = paramzz.grad
    np.testing.assert_allclose(pg.shape, [2, n])
    np.testing.assert_allclose(pg[0, 0], -0.41609, atol=1e-5)

    def f3(x):
        return tc.backend.real(x ** 2)

    f3_torch = interfaces.torch_interface(f3)
    param3 = torch.ones([2], dtype=torch.complex64, requires_grad=True)
    l3 = f3_torch(param3)
    l3 = torch.sum(l3)
    l3.backward()
    pg = param3.grad
    np.testing.assert_allclose(pg, 2 * np.ones([2]).astype(np.complex64), atol=1e-5)
