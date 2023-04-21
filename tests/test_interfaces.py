# pylint: disable=invalid-name

import os
import sys
from functools import partial
import pytest
from pytest_lazyfixture import lazy_fixture as lf
from scipy import optimize
import tensorflow as tf
import jax

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)

try:
    import torch

    is_torch = True
except ImportError:
    is_torch = False

import numpy as np
import tensorcircuit as tc


@pytest.mark.skipif(is_torch is False, reason="torch not installed")
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

    f_jit_torch = tc.interfaces.torch_interface(f_jit, enable_dlpack=True)

    param = torch.ones([4, n], requires_grad=True)
    l = f_jit_torch(param)
    l = l**2
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

    f2_torch = tc.interfaces.torch_interface(f2, jit=True, enable_dlpack=True)

    paramzz = torch.ones([2, n], requires_grad=True)
    paramx = torch.ones([2, n], requires_grad=True)

    l1, l2 = f2_torch(paramzz, paramx)
    l = l1 - l2
    l.backward()

    pg = paramzz.grad
    np.testing.assert_allclose(pg.shape, [2, n])
    np.testing.assert_allclose(pg[0, 0], -0.41609, atol=1e-5)

    def f3(x):
        return tc.backend.real(x**2)

    f3_torch = tc.interfaces.torch_interface(f3)
    param3 = torch.ones([2], dtype=torch.complex64, requires_grad=True)
    l3 = f3_torch(param3)
    l3 = torch.sum(l3)
    l3.backward()
    pg = param3.grad
    np.testing.assert_allclose(pg, 2 * np.ones([2]).astype(np.complex64), atol=1e-5)


@pytest.mark.skipif(is_torch is False, reason="torch not installed")
@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_torch_interface_kws(backend):
    def f(param, n):
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

    f_jit_torch = tc.interfaces.torch_interface_kws(f, jit=True, enable_dlpack=True)

    param = torch.ones([4, 4], requires_grad=True)
    l = f_jit_torch(param, n=4)
    l = l**2
    l.backward()

    pg = param.grad
    np.testing.assert_allclose(pg.shape, [4, 4])
    np.testing.assert_allclose(pg[0, 1], -2.146e-3, atol=1e-5)

    def f2(paramzz, paramx, n, nlayer):
        c = tc.Circuit(n)
        for i in range(n):
            c.H(i)
        for j in range(nlayer):  # 2
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

    f2_torch = tc.interfaces.torch_interface_kws(f2, jit=True, enable_dlpack=True)

    paramzz = torch.ones([2, 4], requires_grad=True)
    paramx = torch.ones([2, 4], requires_grad=True)

    l1, l2 = f2_torch(paramzz, paramx, n=4, nlayer=2)
    l = l1 - l2
    l.backward()

    pg = paramzz.grad
    np.testing.assert_allclose(pg.shape, [2, 4])
    np.testing.assert_allclose(pg[0, 0], -0.41609, atol=1e-5)


@pytest.mark.skipif(is_torch is False, reason="torch not installed")
@pytest.mark.xfail(
    (int(tf.__version__.split(".")[1]) < 9)
    or (int("".join(jax.__version__.split(".")[1:])) < 314),
    reason="version too low for tf or jax",
)
@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_torch_interface_dlpack_complex(backend):
    def f3(x):
        return tc.backend.real(x**2)

    f3_torch = tc.interfaces.torch_interface(f3, enable_dlpack=True)
    param3 = torch.ones([2], dtype=torch.complex64, requires_grad=True)
    l3 = f3_torch(param3)
    l3 = torch.sum(l3)
    l3.backward()
    pg = param3.grad
    np.testing.assert_allclose(pg, 2 * np.ones([2]).astype(np.complex64), atol=1e-5)


@pytest.mark.skipif(is_torch is False, reason="torch not installed")
@pytest.mark.xfail(reason="see comment link below")
@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_torch_interface_pytree(backend):
    # pytree cannot support in pytorch autograd function...
    # https://github.com/pytorch/pytorch/issues/55509
    def f4(x):
        return tc.backend.sum(x["a"] ** 2), tc.backend.sum(x["b"] ** 3)

    f4_torch = tc.interfaces.torch_interface(f4, jit=False)
    param4 = {
        "a": torch.ones([2], requires_grad=True),
        "b": torch.ones([2], requires_grad=True),
    }

    def f4_post(x):
        r1, r2 = f4_torch(param4)
        l4 = r1 + r2
        return l4

    pg = tc.get_backend("pytorch").grad(f4_post)(param4)
    np.testing.assert_allclose(
        pg["a"], 2 * np.ones([2]).astype(np.complex64), atol=1e-5
    )


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_tf_interface(backend):
    def f0(params):
        c = tc.Circuit(1)
        c.rx(0, theta=params[0])
        c.ry(0, theta=params[1])
        return tc.backend.real(c.expectation([tc.gates.z(), [0]]))

    f = tc.interfaces.tf_interface(f0, ydtype=tf.float32, jit=True, enable_dlpack=True)

    tfb = tc.get_backend("tensorflow")
    grads = tfb.jit(tfb.grad(f))(tfb.ones([2], dtype="float32"))
    np.testing.assert_allclose(
        tfb.real(grads), np.array([-0.45464867, -0.45464873]), atol=1e-5
    )

    f = tc.interfaces.tf_interface(f0, ydtype="float32", jit=False)

    grads = tfb.grad(f)(tf.ones([2]))
    np.testing.assert_allclose(grads, np.array([-0.45464867, -0.45464873]), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_tf_interface_2(backend):
    def f1(a, b):
        sa, sb = tc.backend.sum(a), tc.backend.sum(b)
        return sa + sb, sa - sb

    f = tc.interfaces.tf_interface(f1, ydtype=["float32", "float32"], jit=True)

    def f_post(a, b):
        p, m = f(a, b)
        return p + m

    tfb = tc.get_backend("tensorflow")

    grads = tfb.jit(tfb.grad(f_post))(
        tf.ones([2], dtype=tf.float32), tf.ones([2], dtype=tf.float32)
    )

    np.testing.assert_allclose(grads, 2 * np.ones([2]), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_tf_interface_3(backend, highp):
    def f1(a, b):
        sa, sb = tc.backend.sum(a), tc.backend.sum(b)
        return sa + sb

    f = tc.interfaces.tf_interface(f1, ydtype="float64", jit=True)

    tfb = tc.get_backend("tensorflow")

    grads = tfb.jit(tfb.grad(f))(
        tf.ones([2], dtype=tf.float64), tf.ones([2], dtype=tf.float64)
    )
    np.testing.assert_allclose(grads, np.ones([2]), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_scipy_interface(backend):
    n = 3

    def f(param):
        c = tc.Circuit(n)
        for i in range(n):
            c.rx(i, theta=param[0, i])
            c.rz(i, theta=param[1, i])
        loss = c.expectation(
            [
                tc.gates.y(),
                [
                    0,
                ],
            ]
        )
        return tc.backend.real(loss)

    if tc.backend.name != "numpy":
        f_scipy = tc.interfaces.scipy_optimize_interface(f, shape=[2, n])
        r = optimize.minimize(f_scipy, np.zeros([2 * n]), method="L-BFGS-B", jac=True)
        # L-BFGS-B may has issue with float32
        # see: https://github.com/scipy/scipy/issues/5832
        np.testing.assert_allclose(r["fun"], -1.0, atol=1e-5)

    f_scipy = tc.interfaces.scipy_optimize_interface(f, shape=[2, n], gradient=False)
    r = optimize.minimize(f_scipy, np.zeros([2 * n]), method="COBYLA")
    np.testing.assert_allclose(r["fun"], -1.0, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("torchb"), lf("tfb"), lf("jaxb")])
def test_numpy_interface(backend):
    def f(params, n):
        c = tc.Circuit(n)
        for i in range(n):
            c.rx(i, theta=params[i])
        for i in range(n - 1):
            c.cnot(i, i + 1)
        r = tc.backend.real(c.expectation_ps(z=[n - 1]))
        return r

    n = 3
    f_np = tc.interfaces.numpy_interface(f, jit=False)
    r = f_np(np.ones([n]), n)
    np.testing.assert_allclose(r, 0.1577285, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb"), lf("torchb")])
def test_args_transformation(backend):
    ans = tc.interfaces.general_args_to_numpy(
        (
            tc.backend.ones([2]),
            {
                "a": tc.get_backend("tensorflow").ones([]),
                "b": [tc.get_backend("numpy").zeros([2, 1])],
            },
        )
    )
    print(ans)
    np.testing.assert_allclose(ans[1]["b"][0], np.zeros([2, 1], dtype=np.complex64))
    ans1 = tc.interfaces.numpy_args_to_backend(
        ans, target_backend="jax", dtype="float32"
    )
    print(ans1[1]["a"].dtype)
    ans1 = tc.interfaces.numpy_args_to_backend(
        ans,
        target_backend="jax",
        dtype=("complex64", {"a": "float32", "b": ["complex64"]}),
    )
    print(ans1[1]["a"].dtype)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
def test_dlpack_transformation(backend):
    blist = ["tensorflow", "jax"]
    if is_torch is True:
        blist.append("pytorch")
    for b in blist:
        ans = tc.interfaces.general_args_to_backend(
            args=tc.backend.ones([2], dtype="float32"),
            target_backend=b,
            enable_dlpack=True,
        )
        ans = tc.interfaces.which_backend(ans).device_move(ans, "cpu")
        np.testing.assert_allclose(ans, np.ones([2]))


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_args_to_tensor(backend):
    @partial(
        tc.interfaces.args_to_tensor,
        argnums=[0, 1, 2],
        gate_to_tensor=True,
        qop_to_tensor=True,
    )
    def f(a, b, c, d):
        return a, b, c, d

    r = f(np.ones([2]), tc.backend.ones([1, 2]), {"a": [tf.zeros([3])]}, np.ones([2]))
    a = r[0]
    b = r[1]
    c = r[2]["a"][0]
    d = r[3]
    assert tc.interfaces.which_backend(a, return_backend=False) == tc.backend.name
    assert tc.interfaces.which_backend(b, return_backend=False) == tc.backend.name
    assert tc.interfaces.which_backend(c, return_backend=False) == tc.backend.name
    assert tc.interfaces.which_backend(d, return_backend=False) == "numpy"
    # print(f([np.ones([2]), np.ones([1])], {"a": np.ones([3])}))
    # print(f([tc.Gate(np.ones([2, 2])), tc.Gate(np.ones([2, 2, 2, 2]))], np.ones([2])))

    a, b, c, d = f(
        [tc.Gate(np.ones([2, 2])), tc.Gate(np.ones([2, 2, 2, 2]))],
        tc.QuOperator.from_tensor(np.ones([2, 2, 2, 2, 2, 2])),
        np.ones([2, 2, 2, 2]),
        tf.zeros([1, 2]),
    )
    assert tc.interfaces.which_backend(a[0], return_backend=False) == tc.backend.name
    assert tc.backend.shape_tuple(a[1]) == (4, 4)
    assert tc.interfaces.which_backend(b, return_backend=False) == tc.backend.name
    assert tc.interfaces.which_backend(d, return_backend=False) == "tensorflow"
    assert tc.backend.shape_tuple(b) == (8, 8)
    assert tc.backend.shape_tuple(c) == (2, 2, 2, 2)

    @partial(
        tc.interfaces.args_to_tensor,
        argnums=[0, 1, 2],
        tensor_as_matrix=False,
        gate_to_tensor=True,
        gate_as_matrix=False,
        qop_to_tensor=True,
        qop_as_matrix=False,
    )
    def g(a, b, c):
        return a, b, c

    a, b, c = g(
        [tc.Gate(np.ones([2, 2])), tc.Gate(np.ones([2, 2, 2, 2]))],
        tc.QuOperator.from_tensor(np.ones([2, 2, 2, 2, 2, 2])),
        np.ones([2, 2, 2, 2]),
    )

    assert tc.interfaces.which_backend(a[0], return_backend=False) == tc.backend.name
    assert tc.backend.shape_tuple(a[1]) == (2, 2, 2, 2)
    assert tc.backend.shape_tuple(b.eval()) == (2, 2, 2, 2, 2, 2)
    assert tc.backend.shape_tuple(c) == (2, 2, 2, 2)
