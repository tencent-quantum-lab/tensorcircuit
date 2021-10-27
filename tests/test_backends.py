import sys
import os
from functools import partial
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf

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
        return (a + y)[0]

    dp_vvag = tc.backend.vvag(dict_plus, vectorized_argnums=1, argnums=0)
    x = {"a": tc.backend.ones([1])}
    y = tc.backend.ones([20, 1])
    v, g = dp_vvag(x, y)
    assert np.allclose(v.shape, [20])
    assert np.allclose(g["a"], 20.0, atol=1e-4)
