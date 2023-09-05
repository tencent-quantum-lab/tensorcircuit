# pylint: disable=invalid-name

import sys
import os
from functools import partial

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf
import tensorflow as tf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc

dtype = np.complex64


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


# @pytest.mark.skip(
#     reason="pytorch backend to be fixed with newly added complex dtype support"
# )
def test_vmap_torch(torchb):
    r = universal_vmap()
    assert r.numpy()[0, 0] == 3.0


def test_grad_torch(torchb):
    a = tc.backend.ones([2], dtype="float32")

    # @partial(tc.backend.jit, jit_compile=True)
    @tc.backend.grad
    def f(x):
        return tc.backend.sum(x)

    np.testing.assert_allclose(f(a), np.ones([2]), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_backend_scatter(backend):
    np.testing.assert_allclose(
        tc.backend.scatter(
            tc.array_to_tensor(np.arange(8), dtype="int32"),
            tc.array_to_tensor(np.array([[1], [4]]), dtype="int32"),
            tc.array_to_tensor(np.array([0, 0]), dtype="int32"),
        ),
        np.array([0, 0, 2, 3, 0, 5, 6, 7]),
        atol=1e-4,
    )
    np.testing.assert_allclose(
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
    np.testing.assert_allclose(
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
    np.testing.assert_allclose(
        tc.backend.softmax(tc.array_to_tensor(np.ones([3, 2]), dtype="float32")),
        np.ones([3, 2]) / 6.0,
        atol=1e-4,
    )

    arr = np.random.normal(size=(6, 6))

    np.testing.assert_allclose(
        tc.backend.adjoint(tc.array_to_tensor(arr + 1.0j * arr)),
        arr.T - 1.0j * arr.T,
        atol=1e-4,
    )

    arr = tc.backend.zeros([5], dtype="float32")
    np.testing.assert_allclose(
        tc.backend.sigmoid(arr),
        tc.backend.ones([5]) * 0.5,
        atol=1e-4,
    )
    ans = np.array([[1, 0.5j], [-0.5j, 1]])
    ans2 = ans @ ans
    ansp = tc.backend.sqrtmh(tc.array_to_tensor(ans2))
    print(ansp @ ansp, ans @ ans)
    np.testing.assert_allclose(ansp @ ansp, ans @ ans, atol=1e-4)

    np.testing.assert_allclose(
        tc.backend.sum(tc.array_to_tensor(np.arange(4))), 6, atol=1e-4
    )

    indices = np.array([[1, 2], [0, 1]])
    ans = np.array([[[0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0]]])
    np.testing.assert_allclose(tc.backend.one_hot(indices, 3), ans, atol=1e-4)

    a = tc.array_to_tensor(np.array([1, 1, 3, 2, 2, 1]), dtype="int32")
    np.testing.assert_allclose(tc.backend.unique_with_counts(a)[0].shape[0], 3)

    np.testing.assert_allclose(
        tc.backend.cumsum(tc.array_to_tensor(np.array([[0.2, 0.2], [0.2, 0.4]]))),
        np.array([0.2, 0.4, 0.6, 1.0]),
        atol=1e-4,
    )

    np.testing.assert_allclose(
        tc.backend.max(tc.backend.ones([2, 2], "float32")), 1.0, atol=1e-4
    )
    np.testing.assert_allclose(
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

    np.testing.assert_allclose(
        tc.backend.concat([tc.backend.ones([2, 2]), tc.backend.ones([1, 2])]),
        tc.backend.ones([3, 2]),
        atol=1e-5,
    )

    np.testing.assert_allclose(
        tc.backend.gather1d(
            tc.array_to_tensor(np.array([0, 1, 2])),
            tc.array_to_tensor(np.array([2, 1, 0]), dtype="int32"),
        ),
        np.array([2, 1, 0]),
        atol=1e-5,
    )

    def sum_(carry, x):
        return carry + x

    r = tc.backend.scan(sum_, tc.backend.ones([10, 2]), tc.backend.zeros([2]))
    np.testing.assert_allclose(r, 10 * np.ones([2]), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb"), lf("torchb")])
def test_backend_methods_2(backend):
    np.testing.assert_allclose(tc.backend.mean(tc.backend.ones([10])), 1.0, atol=1e-5)
    # acos acosh asin asinh atan atan2 atanh cosh (cos) tan tanh sinh (sin)
    np.testing.assert_allclose(
        tc.backend.acos(tc.backend.ones([2], dtype="float32")),
        np.arccos(tc.backend.ones([2])),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        tc.backend.acosh(tc.backend.ones([2], dtype="float32")),
        np.arccosh(tc.backend.ones([2])),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        tc.backend.asin(tc.backend.ones([2], dtype="float32")),
        np.arcsin(tc.backend.ones([2])),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        tc.backend.asinh(tc.backend.ones([2], dtype="float32")),
        np.arcsinh(tc.backend.ones([2])),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        tc.backend.atan(0.5 * tc.backend.ones([2], dtype="float32")),
        np.arctan(0.5 * tc.backend.ones([2])),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        tc.backend.atan2(
            tc.backend.ones([1], dtype="float32"), tc.backend.ones([1], dtype="float32")
        ),
        np.arctan2(
            tc.backend.ones([1], dtype="float32"), tc.backend.ones([1], dtype="float32")
        ),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        tc.backend.atanh(0.5 * tc.backend.ones([2], dtype="float32")),
        np.arctanh(0.5 * tc.backend.ones([2])),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        tc.backend.cosh(tc.backend.ones([2], dtype="float32")),
        np.cosh(tc.backend.ones([2])),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        tc.backend.tan(tc.backend.ones([2], dtype="float32")),
        np.tan(tc.backend.ones([2])),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        tc.backend.tanh(tc.backend.ones([2], dtype="float32")),
        np.tanh(tc.backend.ones([2])),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        tc.backend.sinh(0.5 * tc.backend.ones([2], dtype="float32")),
        np.sinh(0.5 * tc.backend.ones([2])),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        tc.backend.eigvalsh(tc.backend.ones([2, 2])), np.array([0, 2]), atol=1e-5
    )
    np.testing.assert_allclose(
        tc.backend.left_shift(
            tc.backend.convert_to_tensor(np.array([4, 3])),
            tc.backend.convert_to_tensor(np.array([1, 1])),
        ),
        np.array([8, 6]),
    )
    np.testing.assert_allclose(
        tc.backend.right_shift(
            tc.backend.convert_to_tensor(np.array([4, 3])),
            tc.backend.convert_to_tensor(np.array([1, 1])),
        ),
        np.array([2, 1]),
    )
    np.testing.assert_allclose(
        tc.backend.mod(
            tc.backend.convert_to_tensor(np.array([4, 3])),
            tc.backend.convert_to_tensor(np.array([2, 2])),
        ),
        np.array([0, 1]),
    )
    np.testing.assert_allclose(
        tc.backend.arange(3),
        np.array([0, 1, 2]),
    )
    np.testing.assert_allclose(
        tc.backend.arange(1, 5, 2),
        np.array([1, 3]),
    )
    assert tc.backend.dtype(tc.backend.ones([])) == "complex64"
    edges = [-1, 3.3, 9.1, 10.0]
    values = tc.backend.convert_to_tensor(np.array([0.0, 4.1, 12.0], dtype=np.float32))
    r = tc.backend.numpy(tc.backend.searchsorted(edges, values))
    np.testing.assert_allclose(r, np.array([1, 2, 4]))
    p = tc.backend.convert_to_tensor(
        np.array(
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.4], dtype=np.float32
        )
    )
    r = tc.backend.probability_sample(10000, p, status=np.random.uniform(size=[10000]))
    _, r = np.unique(r, return_counts=True)
    np.testing.assert_allclose(
        r - tc.backend.numpy(p) * 10000.0, np.zeros([10]), atol=200, rtol=1
    )
    np.testing.assert_allclose(
        tc.backend.std(tc.backend.cast(tc.backend.arange(1, 4), "float32")),
        0.81649658,
        atol=1e-5,
    )
    arr = np.random.normal(size=(6, 6))
    np.testing.assert_allclose(
        tc.backend.relu(tc.array_to_tensor(arr, dtype="float32")),
        np.maximum(arr, 0),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        tc.backend.det(tc.backend.convert_to_tensor(np.eye(3) * 2)), 8, atol=1e-5
    )


# @pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb"), lf("torchb")])
# def test_backend_array(backend):
#     a = tc.backend.array([[0, 1], [1, 0]])
#     assert tc.interfaces.which_backend(a).name == tc.backend.name
#     a = tc.backend.array([[0, 1], [1, 0]], dtype=tc.rdtypestr)
#     assert tc.dtype(a) == "float32"


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb"), lf("torchb")])
def test_device_cpu_only(backend):
    a = tc.backend.ones([])
    dev_str = tc.backend.device(a)
    assert dev_str in ["cpu", "gpu:0"]
    tc.backend.device_move(a, dev_str)


@pytest.mark.skipif(
    len(tf.config.list_physical_devices()) == 1, reason="no GPU detected"
)
@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
def test_device_cpu_gpu(backend):
    a = tc.backend.ones([])
    a1 = tc.backend.device_move(a, "gpu:0")
    dev_str = tc.backend.device(a1)
    assert dev_str == "gpu:0"
    a2 = tc.backend.device_move(a1, "cpu")
    assert tc.backend.device(a2) == "cpu"


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
def test_dlpack(backend):
    a = tc.backend.ones([2, 2], dtype="float64")
    cap = tc.backend.to_dlpack(a)
    a1 = tc.backend.from_dlpack(cap)
    np.testing.assert_allclose(a, a1, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb"), lf("torchb")])
def test_arg_cmp(backend):
    np.testing.assert_allclose(tc.backend.argmax(tc.backend.ones([3], "float64")), 0)
    np.testing.assert_allclose(
        tc.backend.argmax(
            tc.array_to_tensor(np.array([[1, 2], [3, 4]]), dtype="float64")
        ),
        np.array([1, 1]),
    )
    np.testing.assert_allclose(
        tc.backend.argmin(
            tc.array_to_tensor(np.array([[1, 2], [3, 4]]), dtype="float64"), axis=-1
        ),
        np.array([0, 0]),
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb"), lf("torchb")])
def test_tree_map(backend):
    def f(a, b):
        return a + b

    r = tc.backend.tree_map(
        f, {"a": tc.backend.ones([2])}, {"a": 2 * tc.backend.ones([2])}
    )
    np.testing.assert_allclose(r["a"], 3 * np.ones([2]), atol=1e-4)

    def _add(a, b):
        return a + b

    ans = tc.backend.tree_map(
        _add,
        {"a": tc.backend.ones([2]), "b": tc.backend.ones([3])},
        {"a": tc.backend.ones([2]), "b": tc.backend.ones([3])},
    )
    np.testing.assert_allclose(ans["a"], 2 * np.ones([2]))


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
    np.testing.assert_allclose(r11, r21, atol=1e-4)
    np.testing.assert_allclose(r12, r22, atol=1e-4)
    assert not np.allclose(r11, r12, atol=1e-4)

    def random_matrixu(key):
        tc.backend.set_random_state(key)
        r1 = tc.backend.implicit_randu(shape=[2, 2], high=2)
        r2 = tc.backend.implicit_randu(shape=[2, 2], high=1)
        return r1, r2

    key = 42
    r31, r32 = random_matrixu(key)
    np.testing.assert_allclose(r31.shape, [2, 2])
    assert np.any(r32 > 0)
    assert not np.allclose(r31, r32, atol=1e-4)

    def random_matrixc(key):
        tc.backend.set_random_state(key)
        r1 = tc.backend.implicit_randc(a=[1, 2, 3], shape=(2, 2))
        r2 = tc.backend.implicit_randc(a=[1, 2, 3], shape=(2, 2), p=[0.1, 0.4, 0.5])
        return r1, r2

    r41, r42 = random_matrixc(key)
    np.testing.assert_allclose(r41.shape, [2, 2])
    assert np.any((r42 > 0) & (r42 < 4))


def vqe_energy(inputs, param, n, nlayers):
    c = tc.Circuit(n, inputs=inputs)
    paramc = tc.backend.cast(param, "complex64")

    for i in range(n):
        c.H(i)
    for j in range(nlayers):
        for i in range(n - 1):
            c.ryy(i, i + 1, theta=paramc[2 * j, i])
            # c.any(
            #     i,
            #     i + 1,
            #     unitary=tc.backend.cos(paramc[2 * j, i]) * iir
            #     + tc.backend.sin(paramc[2 * j, i]) * 1.0j * yzr,
            # )
        for i in range(n):
            c.rx(i, theta=paramc[2 * j + 1, i])
    e = 0.0
    for i in range(n):
        e += c.expectation((tc.gates.x(), [i]))
    for i in range(n - 1):  # OBC
        e += c.expectation((tc.gates.z(), [i]), (tc.gates.z(), [(i + 1) % n]))
    e = tc.backend.real(e)
    return e


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
def test_vvag(backend):
    n = 4
    nlayers = 3
    inp = tc.backend.ones([2**n]) / 2 ** (n / 2)
    param = tc.backend.ones([2 * nlayers, n])
    # inp = tc.backend.cast(inp, "complex64")
    # param = tc.backend.cast(param, "complex64")

    vqe_energy_p = partial(vqe_energy, n=n, nlayers=nlayers)

    vg = tc.backend.value_and_grad(vqe_energy_p, argnums=(0, 1))
    v0, (g00, g01) = vg(inp, param)

    batch = 8
    inps = tc.backend.ones([batch, 2**n]) / 2 ** (n / 2)
    inps = tc.backend.cast(inps, "complex64")

    pvag = tc.backend.vvag(vqe_energy_p, argnums=(0, 1))
    v1, (g10, g11) = pvag(inps, param)
    print(v1.shape, g10.shape, g11.shape)
    np.testing.assert_allclose(v1[0], v0, atol=1e-4)
    np.testing.assert_allclose(g10[0], g00, atol=1e-4)
    np.testing.assert_allclose(g11 / batch, g01, atol=1e-4)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_vvag_dict(backend):
    def dict_plus(x, y):
        a = x["a"]
        return tc.backend.real((a + y)[0])

    dp_vvag = tc.backend.vvag(dict_plus, vectorized_argnums=1, argnums=0)
    x = {"a": tc.backend.ones([1])}
    y = tc.backend.ones([20, 1])
    v, g = dp_vvag(x, y)
    np.testing.assert_allclose(v.shape, [20])
    np.testing.assert_allclose(g["a"], 20.0, atol=1e-4)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
def test_vjp(backend):
    def f(x):
        return x**2

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
        return x**2

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
        return x**2

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
        return x**2

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
        return x**2

    x = tc.backend.ones([3])
    jacf = backend_jac(f)
    np.testing.assert_allclose(jacf(x), 2 * np.eye(3), atol=1e-5)

    def f2(x):
        return x**2, x

    jacf2 = backend_jac(f2)
    np.testing.assert_allclose(jacf2(x)[1], np.eye(3), atol=1e-5)
    np.testing.assert_allclose(jacf2(x)[0], 2 * np.eye(3), atol=1e-5)

    def f3(x, y):
        return x + y**2

    jacf3 = backend_jac(f3, argnums=(0, 1))
    jacf3jit = tc.backend.jit(backend_jac(f3, argnums=(0, 1)))
    np.testing.assert_allclose(jacf3jit(x, x)[1], 2 * np.eye(3), atol=1e-5)
    np.testing.assert_allclose(jacf3(x, x)[1], 2 * np.eye(3), atol=1e-5)

    def f4(x, y):
        return x**2, y

    # note the subtle difference of two tuples order in jacrev and jacfwd for current API
    # the value happen to be the same here, though
    jacf4 = backend_jac(f4, argnums=(0, 1))
    jacf4jit = tc.backend.jit(backend_jac(f4, argnums=(0, 1)))
    np.testing.assert_allclose(jacf4jit(x, x)[1][1], np.eye(3), atol=1e-5)
    np.testing.assert_allclose(jacf4jit(x, x)[0][1], np.zeros([3, 3]), atol=1e-5)
    np.testing.assert_allclose(jacf4(x, x)[1][1], np.eye(3), atol=1e-5)
    np.testing.assert_allclose(jacf4(x, x)[0][1], np.zeros([3, 3]), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
@pytest.mark.parametrize("mode", ["jacfwd", "jacrev"])
def test_jac_md_input(backend, mode):
    backend_jac = getattr(tc.backend, mode)

    def f(x):
        return x**2

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


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb")])
def test_vvag_has_aux(backend):
    def f(x):
        y = tc.backend.sum(x)
        return tc.backend.real(y**2), y

    fvvag = tc.backend.vvag(f, has_aux=True)
    (_, v1), _ = fvvag(tc.backend.ones([10, 2]))
    np.testing.assert_allclose(v1, 2 * tc.backend.ones([10]))


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


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb"), lf("torchb")])
def test_qr(backend, highp):
    def get_random_complex(shape):
        result = np.random.random(shape) + np.random.random(shape) * 1j
        return tc.backend.convert_to_tensor(result.astype(dtype))

    np.random.seed(0)
    A1 = get_random_complex((2, 2))
    A2 = tc.backend.convert_to_tensor(np.array([[1.0, 0.0], [0.0, 0.0]]).astype(dtype))
    X = get_random_complex((2, 2))

    def func(A, x):
        x = tc.backend.cast(x, "complex64")
        Q, R = tc.backend.qr(A + X * x)
        return tc.backend.real(tc.backend.sum(tc.backend.matmul(Q, R)))

    def grad(A, x):
        return tc.backend.grad(func, argnums=1)(A, x)

    for A in [A1, A2]:
        epsilon = tc.backend.convert_to_tensor(1e-3)
        n_grad = (func(A, epsilon) - func(A, -epsilon)) / (2 * epsilon)
        a_grad = grad(A, tc.backend.convert_to_tensor(0.0))
        np.testing.assert_allclose(n_grad, a_grad, atol=1e-3)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_sparse_methods(backend):
    values = tc.backend.convert_to_tensor(np.array([1.0, 2.0]))
    values = tc.backend.cast(values, "complex64")
    indices = tc.backend.convert_to_tensor(np.array([[0, 0], [1, 1]]))
    indices = tc.backend.cast(indices, "int64")
    spa = tc.backend.coo_sparse_matrix(indices, values, shape=[4, 4])
    vec = tc.backend.ones([4, 1])
    da = np.array(
        [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.complex64
    )
    assert tc.backend.is_sparse(spa) is True
    assert tc.backend.is_sparse(vec) is False
    np.testing.assert_allclose(
        tc.backend.to_dense(spa),
        da,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        tc.backend.sparse_dense_matmul(spa, vec),
        np.array([[1], [2], [0], [0]], dtype=np.complex64),
        atol=1e-5,
    )
    spa_np = tc.backend.numpy(spa)
    np.testing.assert_allclose(spa_np.todense(), da, atol=1e-6)
    np.testing.assert_allclose(
        tc.backend.to_dense(tc.backend.coo_sparse_matrix_from_numpy(spa_np)),
        da,
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


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_with_level_set_return(backend):
    with tc.runtime_backend("jax") as K:
        assert K.name == "jax"


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
def test_grad_has_aux(backend):
    def f(x):
        return tc.backend.real(x**2), x**3

    vg = tc.backend.value_and_grad(f, has_aux=True)

    np.testing.assert_allclose(
        vg(tc.backend.ones([]))[1], 2 * tc.backend.ones([]), atol=1e-5
    )

    def f2(x):
        return tc.backend.real(x**2), (x**3, tc.backend.ones([3]))

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


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb"), lf("torchb")])
def test_treeutils(backend):
    d0 = {"a": np.ones([2]), "b": [tc.backend.zeros([]), tc.backend.ones([1, 1])]}
    leaves, treedef = tc.backend.tree_flatten(d0)
    d1 = tc.backend.tree_unflatten(treedef, leaves)
    d2 = tc.backend.tree_map(lambda x: 2 * x, d1)
    np.testing.assert_allclose(2 * np.ones([1, 1]), d2["b"][1])


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
def test_optimizers(backend):
    if tc.backend.name == "jax":
        try:
            import optax
        except ImportError:
            pytest.skip("optax is not installed")

    if tc.backend.name == "pytorch":
        try:
            import torch
        except ImportError:
            pytest.skip("torch is not installed")

    def f(params, n):
        c = tc.Circuit(n)
        c = tc.templates.blocks.example_block(c, params["a"])
        c = tc.templates.blocks.example_block(c, params["b"])
        return tc.backend.real(c.expectation([tc.gates.x(), [n // 2]]))

    vgs = tc.backend.jit(tc.backend.value_and_grad(f, argnums=0), static_argnums=1)

    def get_opt():
        if tc.backend.name == "tensorflow":
            optimizer1 = tf.keras.optimizers.Adam(5e-2)
            opt = tc.backend.optimizer(optimizer1)
        elif tc.backend.name == "jax":
            optimizer2 = optax.adam(5e-2)
            opt = tc.backend.optimizer(optimizer2)
        elif tc.backend.name == "pytorch":
            optimizer3 = partial(torch.optim.Adam, lr=5e-2)
            opt = tc.backend.optimizer(optimizer3)
        else:
            raise ValueError("%s doesn't support optimizer interface" % tc.backend.name)
        return opt

    n = 3
    opt = get_opt()

    params = {
        "a": tc.backend.ones([4, n], dtype="float32"),
        "b": tc.backend.ones([4, n], dtype="float32"),
    }

    for _ in range(20):
        loss, grads = vgs(params, n)
        params = opt.update(grads, params)
        print(loss)

    assert loss < -0.7

    def f2(params, n):
        c = tc.Circuit(n)
        c = tc.templates.blocks.example_block(c, params)
        return tc.backend.real(c.expectation([tc.gates.x(), [n // 2]]))

    vgs2 = tc.backend.jit(tc.backend.value_and_grad(f2, argnums=0), static_argnums=1)

    params = tc.backend.ones([4, n], dtype="float32")
    opt = get_opt()

    for _ in range(20):
        loss, grads = vgs2(params, n)
        params = opt.update(grads, params)
        print(loss)

    assert loss < -0.7


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_hessian(backend):
    # hessian support is now very fragile and especially has potential issues on tf backend
    def f(param):
        return tc.backend.sum(param**2)

    hf = tc.backend.hessian(f)
    param = tc.backend.ones([2])
    np.testing.assert_allclose(hf(param), 2 * tc.backend.eye(2), atol=1e-5)

    param = tc.backend.ones([2, 2])
    assert list(hf(param).shape) == [2, 2, 2, 2]  # possible tf retracing?

    g = tc.templates.graphs.Line1D(5)

    def circuit_f(param):
        c = tc.Circuit(5)
        c = tc.templates.blocks.example_block(c, param, nlayers=1)
        return tc.templates.measurements.heisenberg_measurements(c, g)

    param = tc.backend.ones([10])
    hf = tc.backend.hessian(circuit_f)
    print(hf(param))  # still upto a conjugate for jax and tf backend.


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_nested_vmap(backend):
    def f(x, w):
        c = tc.Circuit(4)
        for i in range(4):
            c.rx(i, theta=x[i])
            c.ry(i, theta=w[i])
        return tc.backend.stack([c.expectation_ps(z=[i]) for i in range(4)])

    def fa1(*args):
        r = tc.backend.vmap(f, vectorized_argnums=1)(*args)
        return r

    def fa2(*args):
        r = tc.backend.vmap(fa1, vectorized_argnums=0)(*args)
        return r

    fa2jit = tc.backend.jit(fa2)

    ya = fa2(tc.backend.ones([3, 4]), tc.backend.ones([7, 4]))
    yajit = fa2jit(tc.backend.ones([3, 4]), tc.backend.ones([7, 4]))

    def fb1(*args):
        r = tc.backend.vmap(f, vectorized_argnums=0)(*args)
        return r

    def fb2(*args):
        r = tc.backend.vmap(fb1, vectorized_argnums=1)(*args)
        return r

    fb2jit = tc.backend.jit(fb2)

    yb = fb2(tc.backend.ones([3, 4]), tc.backend.ones([7, 4]))
    ybjit = fb2jit(tc.backend.ones([3, 4]), tc.backend.ones([7, 4]))

    np.testing.assert_allclose(ya, tc.backend.transpose(yb, [1, 0, 2]), atol=1e-5)
    np.testing.assert_allclose(ya, yajit, atol=1e-5)
    np.testing.assert_allclose(yajit, tc.backend.transpose(ybjit, [1, 0, 2]), atol=1e-5)
