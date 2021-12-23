# pylint: disable=invalid-name

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


def test_grad_torch(torchb):
    a = tc.backend.ones([2], dtype="float32")

    @tc.backend.grad
    def f(x):
        return tc.backend.sum(x)

    np.testing.assert_allclose(f(a), np.ones([2]), atol=1e-5)


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


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
def test_vjp_complex(backend):
    def f(x):
        return tc.backend.conj(x)

    inputs = tc.backend.ones([1]) + 1.0j * tc.backend.ones([1])
    v = tc.backend.ones([1], dtype="complex64")
    v, g = tc.backend.vjp(f, inputs, v)
    np.testing.assert_allclose(tc.backend.numpy(g), np.ones([1]), atol=1e-5)

    def f2(x):
        return x ** 2

    inputs = tc.backend.ones([1]) + 1.0j * tc.backend.ones([1])
    v = tc.backend.ones([1], dtype="complex64")  # + 1.0j * tc.backend.ones([1])
    v, g = tc.backend.vjp(f2, inputs, v)
    # note how vjp definition on complex function is different in jax backend
    if tc.backend.name == "jax":
        np.testing.assert_allclose(tc.backend.numpy(g), 2 + 2j, atol=1e-5)
    else:
        np.testing.assert_allclose(tc.backend.numpy(g), 2 - 2j, atol=1e-5)


# TODO(@refraction-ray): consistent and unified pytree utils for pytorch backend?


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_vjp_pytree(backend):
    def f3(d):
        return d["a"] + d["b"], d["a"]

    inputs = {"a": tc.backend.ones([2]), "b": tc.backend.ones([1])}
    v = (tc.backend.ones([2]), tc.backend.zeros([2]))
    v, g = tc.backend.vjp(f3, inputs, v)
    np.testing.assert_allclose(v[0], 2 * np.ones([2]), atol=1e-5)
    np.testing.assert_allclose(g["a"], np.ones([2]), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
def test_jvp(backend):
    def f(x):
        return x ** 2

    inputs = tc.backend.ones([2, 2])
    v, g = tc.backend.jvp(f, inputs, inputs)
    np.testing.assert_allclose(v, inputs, atol=1e-5)
    np.testing.assert_allclose(g, 2 * inputs, atol=1e-5)

    def f2(x, y):
        return x + y, x - y

    inputs = [tc.backend.ones([2]), tc.backend.ones([2])]
    v = [2.0 * t for t in inputs]
    v, g = tc.backend.jvp(f2, inputs, v)
    np.testing.assert_allclose(v[1], np.zeros([2]), atol=1e-5)
    np.testing.assert_allclose(g[0], 4 * np.ones([2]), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
def test_jvp_complex(backend):
    def f(x):
        return tc.backend.conj(x)

    inputs = tc.backend.ones([1]) + 1.0j * tc.backend.ones([1])
    v = tc.backend.ones([1], dtype="complex64")
    v, g = tc.backend.jvp(f, inputs, v)
    # numpy auto numpy doesn't work for torch conjugate tensor
    np.testing.assert_allclose(tc.backend.numpy(g), np.ones([1]), atol=1e-5)

    def f2(x):
        return x ** 2

    inputs = tc.backend.ones([1]) + 1.0j * tc.backend.ones([1])
    v = tc.backend.ones([1]) + 1.0j * tc.backend.ones([1])
    v, g = tc.backend.jvp(f2, inputs, v)
    np.testing.assert_allclose(tc.backend.numpy(g), 4.0j, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_jvp_pytree(backend):
    def f3(d):
        return d["a"] + d["b"], d["a"]

    inputs = {"a": tc.backend.ones([2]), "b": tc.backend.ones([1])}
    v = (tc.backend.ones([2]), tc.backend.zeros([2]))
    v, g = tc.backend.vjp(f3, inputs, v)
    np.testing.assert_allclose(v[0], 2 * np.ones([2]), atol=1e-5)
    np.testing.assert_allclose(g["a"], np.ones([2]), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
@pytest.mark.parametrize("mode", ["jacfwd", "jacrev"])
def test_jac(backend, mode):
    # make no sense for torch backend when you have no real vmap interface
    backend_jac = getattr(tc.backend, mode)

    def f(x):
        return x ** 2

    x = tc.backend.ones([3])
    jacf = backend_jac(f)
    np.testing.assert_allclose(jacf(x), 2 * np.eye(3), atol=1e-5)

    def f2(x):
        return x ** 2, x

    jacf2 = backend_jac(f2)
    np.testing.assert_allclose(jacf2(x)[1], np.eye(3), atol=1e-5)
    np.testing.assert_allclose(jacf2(x)[0], 2 * np.eye(3), atol=1e-5)

    def f3(x, y):
        return x + y ** 2

    jacf3 = backend_jac(f3, argnums=(0, 1))
    np.testing.assert_allclose(jacf3(x, x)[1], 2 * np.eye(3), atol=1e-5)

    def f4(x, y):
        return x ** 2, y

    # note the subtle difference of two tuples order in jacrev and jacfwd for current API
    # the value happen to be the same here, though
    jacf4 = backend_jac(f4, argnums=(0, 1))
    np.testing.assert_allclose(jacf4(x, x)[1][1], np.eye(3), atol=1e-5)
    np.testing.assert_allclose(jacf4(x, x)[0][1], np.zeros([3, 3]), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
@pytest.mark.parametrize("mode", ["jacfwd", "jacrev"])
def test_jac_md_input(backend, mode):
    backend_jac = getattr(tc.backend, mode)

    def f(x):
        return x ** 2

    x = tc.backend.ones([2, 3])
    jacf = backend_jac(f)
    np.testing.assert_allclose(jacf(x).shape, [2, 3, 2, 3], atol=1e-5)

    def f2(x):
        return tc.backend.sum(x, axis=0)

    x = tc.backend.ones([2, 3])
    jacf2 = backend_jac(f2)
    np.testing.assert_allclose(jacf2(x).shape, [3, 2, 3], atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
@pytest.mark.parametrize("mode", ["jacfwd", "jacrev"])
def test_jac_tall(backend, mode):
    backend_jac = getattr(tc.backend, mode)

    h = tc.backend.ones([5, 3])

    def f(x):
        x = tc.backend.reshape(x, [-1, 1])
        return tc.backend.reshape(h @ x, [-1])

    x = tc.backend.ones([3])
    jacf = backend_jac(f)
    np.testing.assert_allclose(jacf(x), np.ones([5, 3]), atol=1e-5)


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


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_sparse_methods(backend):
    values = tc.backend.convert_to_tensor(np.array([1.0, 2.0]))
    values = tc.backend.cast(values, "complex64")
    indices = tc.backend.convert_to_tensor(np.array([[0, 0], [1, 1]]))
    indices = tc.backend.cast(indices, "int64")
    spa = tc.backend.coo_sparse_matrix(indices, values, shape=[4, 4])
    vec = tc.backend.ones([4, 1])
    assert tc.backend.is_sparse(spa) is True
    assert tc.backend.is_sparse(vec) is False
    np.testing.assert_allclose(
        tc.backend.to_dense(spa),
        np.array(
            [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.complex64
        ),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        tc.backend.sparse_dense_matmul(spa, vec),
        np.array([[1], [2], [0], [0]], dtype=np.complex64),
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_backend_randoms_v2(backend):
    g = tc.backend.get_random_state(42)
    for t in tc.backend.stateful_randc(g, 3, [3]):
        assert t >= 0
        assert t < 3
    key = tc.backend.get_random_state(42)
    r = []
    for _ in range(2):
        key, subkey = tc.backend.random_split(key)
        r.append(tc.backend.stateful_randc(subkey, 3, [5]))
    assert tuple(r[0]) != tuple(r[1])


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_backend_randoms_v3(backend):
    tc.backend.set_random_state(42)
    for _ in range(2):
        r1 = tc.backend.implicit_randu()
    key = tc.backend.get_random_state(42)
    for _ in range(2):
        key, subkey = tc.backend.random_split(key)
        r2 = tc.backend.stateful_randu(subkey)
    np.testing.assert_allclose(r1, r2, atol=1e-5)

    @tc.backend.jit
    def f(key):
        tc.backend.set_random_state(key)
        r = []
        for _ in range(3):
            r.append(tc.backend.implicit_randu()[0])
        return r

    @tc.backend.jit
    def f2(key):
        r = []
        for _ in range(3):
            key, subkey = tc.backend.random_split(key)
            r.append(tc.backend.stateful_randu(subkey)[0])
        return r

    key = tc.backend.get_random_state(43)
    r = f(key)
    key = tc.backend.get_random_state(43)
    r1 = f2(key)
    np.testing.assert_allclose(r[-1], r1[-1], atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_function_level_set(backend):
    def f(x):
        return tc.backend.ones([x])

    f_jax_128 = tc.set_function_backend("jax")(tc.set_function_dtype("complex128")(f))
    # note the order to enable complex 128 in jax backend

    assert f_jax_128(3).dtype.__str__() == "complex128"


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_function_level_set_contractor(backend):
    @tc.set_function_contractor("branch")
    def f():
        return tc.contractor

    print(f())
    print(tc.contractor)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_with_level_set(backend):
    with tc.runtime_backend("jax"):
        with tc.runtime_dtype("complex128"):
            with tc.runtime_contractor("branch"):
                assert tc.backend.ones([2]).dtype.__str__() == "complex128"
                print(tc.contractor)
    print(tc.contractor)
    print(tc.backend.name)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
def test_grad_has_aux(backend):
    def f(x):
        return tc.backend.real(x ** 2), x ** 3

    vag = tc.backend.value_and_grad(f, has_aux=True)

    np.testing.assert_allclose(
        vag(tc.backend.ones([]))[1], 2 * tc.backend.ones([]), atol=1e-5
    )

    def f2(x):
        return tc.backend.real(x ** 2), (x ** 3, tc.backend.ones([3]))

    gs = tc.backend.grad(f2, has_aux=True)
    np.testing.assert_allclose(gs(tc.backend.ones([]))[0], 2.0, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb"), lf("tfb")])
def test_solve(backend):
    A = np.array([[2, 1, 0], [1, 2, 0], [0, 0, 1]], dtype=np.float32)
    A = tc.backend.convert_to_tensor(A)
    x = np.ones([3, 1], dtype=np.float32)
    x = tc.backend.convert_to_tensor(x)
    b = (A @ x)[:, 0]
    print(A.shape, b.shape)
    xp = tc.backend.solve(A, b, assume_a="her")
    np.testing.assert_allclose(xp, x[:, 0], atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_optimizers(backend):
    if tc.backend.name == "jax":
        try:
            import optax
        except ImportError:
            pytest.skip("optax is not installed")

    def f(params, n):
        c = tc.Circuit(n)
        c = tc.templates.blocks.example_block(c, params["a"])
        c = tc.templates.blocks.example_block(c, params["b"])
        return tc.backend.real(c.expectation([tc.gates.x(), [n // 2]]))

    vags = tc.backend.jit(tc.backend.value_and_grad(f, argnums=0), static_argnums=1)

    def get_opt():
        if tc.backend.name == "tensorflow":
            optimizer1 = tf.keras.optimizers.Adam(5e-2)
            opt = tc.backend.optimizer(optimizer1)
        elif tc.backend.name == "jax":
            optimizer2 = optax.adam(5e-2)
            opt = tc.backend.optimizer(optimizer2)
        else:
            raise ValueError("%s doesn't support optimizer interface" % tc.backend.name)
        return opt

    n = 3
    opt = get_opt()

    params = {
        "a": tc.backend.implicit_randn([4, n]),
        "b": tc.backend.implicit_randn([4, n]),
    }

    for _ in range(20):
        loss, grads = vags(params, n)
        params = opt.update(grads, params)
        print(loss)

    assert loss < -0.8

    def f2(params, n):
        c = tc.Circuit(n)
        c = tc.templates.blocks.example_block(c, params)
        return tc.backend.real(c.expectation([tc.gates.x(), [n // 2]]))

    vags2 = tc.backend.jit(tc.backend.value_and_grad(f2, argnums=0), static_argnums=1)

    params = tc.backend.implicit_randn([4, n])
    opt = get_opt()

    for _ in range(20):
        loss, grads = vags2(params, n)
        print(grads, params)
        params = opt.update(grads, params)
        print(loss)

    assert loss < -0.8
