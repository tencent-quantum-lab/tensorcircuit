import sys
import os
from functools import partial

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf
import tensorflow as tf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc

dtype = np.complex64
ii = np.eye(4, dtype=dtype)
iir = ii.reshape([2, 2, 2, 2])
ym = np.array([[0, -1.0j], [1.0j, 0]], dtype=dtype)
zm = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)
yz = np.kron(ym, zm)
yzr = yz.reshape([2, 2, 2, 2])


def universal_vmap():
    def sum_real(x, y):
        return tc.backend.real(x + y)

    vop = tc.backend.vmap(sum_real, vectorized_argnums=(0, 1))
    t = tc.gates.array_to_tensor(np.ones([20, 1]))
    return vop(t, 2.0 * t)


def test_vmap_np():
    r = universal_vmap()
    assert r.shape == (20, 1)


def test_vmap_jax(jaxb):
    r = universal_vmap()
    assert r.shape == (20, 1)


def test_vmap_tf(tfb):
    r = universal_vmap()
    assert r.numpy()[0, 0] == 3.0


@pytest.mark.skip(
    reason="pytorch backend to be fixed with newly added complex dtype support"
)
def test_vmap_torch(torchb):
    r = universal_vmap()
    assert r.numpy()[0, 0] == 3.0


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_backend_scatter(backend):
    assert np.allclose(
        tc.backend.scatter(
            tc.array_to_tensor(np.arange(8), dtype="int32"),
            tc.array_to_tensor(np.array([[1], [4]]), dtype="int32"),
            tc.array_to_tensor(np.array([0, 0]), dtype="int32"),
        ),
        np.array([0, 0, 2, 3, 0, 5, 6, 7]),
        atol=1e-4,
    )
    assert np.allclose(
        tc.backend.scatter(
            tc.array_to_tensor(np.arange(8).reshape([2, 4]), dtype="int32"),
            tc.array_to_tensor(np.array([[0, 2], [1, 2], [1, 3]]), dtype="int32"),
            tc.array_to_tensor(np.array([0, 99, 0]), dtype="int32"),
        ),
        np.array([[0, 1, 0, 3], [4, 5, 99, 0]]),
        atol=1e-4,
    )
    answer = np.arange(8).reshape([2, 2, 2])
    answer[0, 1, 0] = 99
    assert np.allclose(
        tc.backend.scatter(
            tc.array_to_tensor(np.arange(8).reshape([2, 2, 2]), dtype="int32"),
            tc.array_to_tensor(np.array([[0, 1, 0]]), dtype="int32"),
            tc.array_to_tensor(np.array([99]), dtype="int32"),
        ),
        answer,
        atol=1e-4,
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_backend_methods(backend):
    # TODO(@refraction-ray): add more methods
    assert np.allclose(
        tc.backend.softmax(tc.array_to_tensor(np.ones([3, 2]), dtype="float32")),
        np.ones([3, 2]) / 6.0,
        atol=1e-4,
    )

    arr = np.random.normal(size=(6, 6))
    assert np.allclose(
        tc.backend.relu(tc.array_to_tensor(arr, dtype="float32")),
        np.maximum(arr, 0),
        atol=1e-4,
    )

    assert np.allclose(
        tc.backend.adjoint(tc.array_to_tensor(arr + 1.0j * arr)),
        arr.T - 1.0j * arr.T,
        atol=1e-4,
    )

    ans = np.array([[1, 0.5j], [-0.5j, 1]])
    ans2 = ans @ ans
    ansp = tc.backend.sqrtmh(tc.array_to_tensor(ans2))
    print(ansp @ ansp, ans @ ans)
    assert np.allclose(ansp @ ansp, ans @ ans, atol=1e-4)

    assert np.allclose(tc.backend.sum(tc.array_to_tensor(np.arange(4))), 6, atol=1e-4)

    indices = np.array([[1, 2], [0, 1]])
    ans = np.array([[[0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0]]])
    assert np.allclose(tc.backend.one_hot(indices, 3), ans, atol=1e-4)

    a = tc.array_to_tensor(np.array([1, 1, 3, 2, 2, 1]), dtype="int32")
    assert np.allclose(tc.backend.unique_with_counts(a)[0].shape[0], 3)

    assert np.allclose(
        tc.backend.cumsum(tc.array_to_tensor(np.array([[0.2, 0.2], [0.2, 0.4]]))),
        np.array([0.2, 0.4, 0.6, 1.0]),
        atol=1e-4,
    )

    assert np.allclose(
        tc.backend.max(tc.backend.ones([2, 2], "float32")), 1.0, atol=1e-4
    )
    assert np.allclose(
        tc.backend.min(
            tc.backend.cast(
                tc.backend.convert_to_tensor(np.array([[1.0, 2.0], [2.0, 3.0]])),
                "float64",
            ),
            axis=1,
        ),
        np.array([1.0, 2.0]),
        atol=1e-4,
    )  # by default no keepdim


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_tree_map(backend):
    def f(a, b):
        return a + b

    r = tc.backend.tree_map(
        f, {"a": tc.backend.ones([2])}, {"a": 2 * tc.backend.ones([2])}
    )
    assert np.allclose(r["a"], 3 * np.ones([2]), atol=1e-4)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_backend_randoms(backend):
    @partial(tc.backend.jit, static_argnums=0)
    def random_matrixn(key):
        tc.backend.set_random_state(key)
        r1 = tc.backend.implicit_randn(shape=[2, 2], mean=0.5)
        r2 = tc.backend.implicit_randn(shape=[2, 2], mean=0.5)
        return r1, r2

    key = 42
    if tc.backend.name == "tensorflow":
        key = tf.random.Generator.from_seed(42)
    r11, r12 = random_matrixn(key)
    if tc.backend.name == "tensorflow":
        key = tf.random.Generator.from_seed(42)
    r21, r22 = random_matrixn(key)
    assert np.allclose(r11, r21, atol=1e-4)
    assert np.allclose(r12, r22, atol=1e-4)
    assert not np.allclose(r11, r12, atol=1e-4)

    def random_matrixu(key):
        tc.backend.set_random_state(key)
        r1 = tc.backend.implicit_randu(shape=[2, 2], high=2)
        r2 = tc.backend.implicit_randu(shape=[2, 2], high=1)
        return r1, r2

    key = 42
    r31, r32 = random_matrixu(key)
    assert np.allclose(r31.shape, [2, 2])
    assert np.any(r32 > 0)
    assert not np.allclose(r31, r32, atol=1e-4)

    def random_matrixc(key):
        tc.backend.set_random_state(key)
        r1 = tc.backend.implicit_randc(a=[1, 2, 3], shape=(2, 2))
        r2 = tc.backend.implicit_randc(a=[1, 2, 3], shape=(2, 2), p=[0.1, 0.4, 0.5])
        return r1, r2

    r41, r42 = random_matrixc(key)
    assert np.allclose(r41.shape, [2, 2])
    assert np.any((r42 > 0) & (r42 < 4))


def vqe_energy(inputs, param, n, nlayers):
    c = tc.Circuit(n, inputs=inputs)
    paramc = tc.backend.cast(param, "complex64")

    for i in range(n):
        c.H(i)
    for j in range(nlayers):
        for i in range(n - 1):
            c.any(
                i,
                i + 1,
                unitary=tc.backend.cos(paramc[2 * j, i]) * iir
                + tc.backend.sin(paramc[2 * j, i]) * 1.0j * yzr,
            )
        for i in range(n):
            c.rx(i, theta=paramc[2 * j + 1, i])
    e = 0.0
    for i in range(n):
        e += c.expectation((tc.gates.x(), [i]))
    for i in range(n - 1):  # OBC
        e += c.expectation((tc.gates.z(), [i]), (tc.gates.z(), [(i + 1) % n]))
    e = tc.backend.real(e)
    return e


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_vvag(backend):
    n = 4
    nlayers = 3
    inp = tc.backend.ones([2 ** n]) / 2 ** (n / 2)
    param = tc.backend.ones([2 * nlayers, n])
    inp = tc.backend.cast(inp, "complex64")
    param = tc.backend.cast(param, "complex64")

    vqe_energy_p = partial(vqe_energy, n=n, nlayers=nlayers)

    vag = tc.backend.value_and_grad(vqe_energy_p, argnums=(0, 1))
    v0, (g00, g01) = vag(inp, param)

    batch = 8
    inps = tc.backend.ones([batch, 2 ** n]) / 2 ** (n / 2)
    inps = tc.backend.cast(inps, "complex64")

    pvag = tc.backend.vvag(vqe_energy_p, argnums=(0, 1))
    v1, (g10, g11) = pvag(inps, param)
    assert np.allclose(v1[0], v0, atol=1e-4)
    assert np.allclose(g10[0], g00, atol=1e-4)
    assert np.allclose(g11 / batch, g01, atol=1e-4)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_vvag_dict(backend):
    def dict_plus(x, y):
        a = x["a"]
        return tc.backend.real((a + y)[0])

    dp_vvag = tc.backend.vvag(dict_plus, vectorized_argnums=1, argnums=0)
    x = {"a": tc.backend.ones([1])}
    y = tc.backend.ones([20, 1])
    v, g = dp_vvag(x, y)
    assert np.allclose(v.shape, [20])
    assert np.allclose(g["a"], 20.0, atol=1e-4)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
def test_vjp(backend):
    def f(x):
        return x ** 2

    inputs = tc.backend.ones([2, 2])
    v, g = tc.backend.vjp(f, inputs, inputs)
    np.testing.assert_allclose(v, inputs, atol=1e-5)
    np.testing.assert_allclose(g, 2 * inputs, atol=1e-5)

    def f2(x, y):
        return x + y, x - y

    inputs = [tc.backend.ones([2]), tc.backend.ones([2])]
    v = [2.0 * t for t in inputs]
    v, g = tc.backend.vjp(f2, inputs, v)
    np.testing.assert_allclose(v[1], np.zeros([2]), atol=1e-5)
    np.testing.assert_allclose(g[0], 4 * np.ones([2]), atol=1e-5)


def test_jax_svd(jaxb, highp):
    def l(A):
        u, _, v, _ = tc.backend.svd(A)
        return tc.backend.real(u[0, 0] * v[0, 0])

    def numericald(A):
        eps = 1e-6
        DA = np.zeros_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                dA = np.zeros_like(A)
                dA[i, j] = 1
                DA[i, j] = (l(A + eps * dA) - l(A)) / eps - 1.0j * (
                    l(A + eps * 1.0j * dA) - l(A)
                ) / eps
        return DA

    def analyticald(A):
        A = tc.backend.convert_to_tensor(A)
        g = tc.backend.grad(l)
        return g(A)

    for shape in [(2, 2), (3, 3), (2, 3), (4, 2)]:
        m = np.random.normal(size=shape).astype(
            np.complex128
        ) + 1.0j * np.random.normal(size=shape).astype(np.complex128)
        print(m)
        np.testing.assert_allclose(numericald(m), analyticald(m), atol=1e-3)
