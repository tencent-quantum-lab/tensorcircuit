# pylint: disable=invalid-name

import sys
import os
from functools import partial
import numpy as np
import tensorflow as tf
import pytest
from pytest_lazyfixture import lazy_fixture as lf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc
from tensorcircuit import experimental
from tensorcircuit.quantum import PauliString2COO, PauliStringSum2COO
from tensorcircuit.applications.vqes import construct_matrix_v2
from tensorcircuit.applications.physics.baseline import TFIM1Denergy, Heisenberg1Denergy

i, x, y, z = [t.tensor for t in tc.gates.pauli_gates]

# note i is in use!

check_pairs = [
    ([0, 0], np.eye(4)),
    ([0, 1], np.kron(i, x)),
    ([2, 1], np.kron(y, x)),
    ([3, 1], np.kron(z, x)),
    ([3, 2, 2, 0], np.kron(np.kron(np.kron(z, y), y), i)),
    ([0, 1, 1, 1], np.kron(np.kron(np.kron(i, x), x), x)),
]


def test_about():
    print(tc.about())


def test_ps2coo(tfb):
    for l, a in check_pairs:
        r1 = PauliString2COO(tf.constant(l, dtype=tf.int64))
        np.testing.assert_allclose(tc.backend.to_dense(r1), a, atol=1e-5)


def test_pss2coo(tfb):
    l = [t[0] for t in check_pairs[:4]]
    a = sum([t[1] for t in check_pairs[:4]])
    r1 = PauliStringSum2COO(tf.constant(l, dtype=tf.int64))
    np.testing.assert_allclose(tc.backend.to_dense(r1), a, atol=1e-5)
    l = [t[0] for t in check_pairs[4:]]
    a = sum([t[1] for t in check_pairs[4:]])
    r1 = PauliStringSum2COO(tf.constant(l, dtype=tf.int64), weight=[0.5, 1])
    a = check_pairs[4][1] * 0.5 + check_pairs[5][1] * 1.0
    np.testing.assert_allclose(tc.backend.to_dense(r1), a, atol=1e-5)


def test_sparse(benchmark, tfb):
    def sparse(h):
        return PauliStringSum2COO(h)

    h = [[1 for _ in range(12)], [2 for _ in range(12)]]
    h = tf.constant(h, dtype=tf.int64)
    sparse(h)
    benchmark(sparse, h)


def test_dense(benchmark, tfb):
    def dense(h):
        return construct_matrix_v2(h, dtype=tf.complex64)

    h = [[1 for _ in range(12)], [2 for _ in range(12)]]
    h = [[1.0] + hi for hi in h]
    dense(h)
    benchmark(dense, h)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_adaptive_vmap(backend):
    def f(x):
        return x**2

    x = tc.backend.ones([30, 2])

    vf = experimental.adaptive_vmap(f, chunk_size=6)
    np.testing.assert_allclose(vf(x), tc.backend.ones([30, 2]), atol=1e-5)

    vf2 = experimental.adaptive_vmap(f, chunk_size=7)
    np.testing.assert_allclose(vf2(x), tc.backend.ones([30, 2]), atol=1e-5)

    def f2(x):
        return tc.backend.sum(x)

    vf3 = experimental.adaptive_vmap(f2, chunk_size=7)
    np.testing.assert_allclose(vf3(x), 2 * tc.backend.ones([30]), atol=1e-5)

    vf3_jit = tc.backend.jit(vf3)
    np.testing.assert_allclose(vf3_jit(x), 2 * tc.backend.ones([30]), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_adaptive_vmap_mul_io(backend):
    def f(x, y, a):
        return x + y + a

    vf = experimental.adaptive_vmap(f, chunk_size=6, vectorized_argnums=(0, 1))
    x = tc.backend.ones([30, 2])
    a = tc.backend.ones([2])
    # jax vmap has some weird behavior in terms of keyword arguments...
    # TODO(@refraction-ray): further investigate jax vmap behavior with kwargs
    np.testing.assert_allclose(vf(x, x, a), 3 * tc.backend.ones([30, 2]), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_qng(backend):
    n = 6

    def f(params):
        params = tc.backend.reshape(params, [4, n])
        c = tc.Circuit(n)
        c = tc.templates.blocks.example_block(c, params)
        return c.state()

    params = tc.backend.ones([4 * n])
    fim = experimental.qng(f)(params)
    assert tc.backend.shape_tuple(fim) == (4 * n, 4 * n)
    print(experimental.dynamics_matrix(f)(params))


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_dynamic_rhs(backend):
    h1 = tc.array_to_tensor(tc.gates._z_matrix)

    def f(param):
        c = tc.Circuit(1)
        c.rx(0, theta=param)
        return c.state()

    rhsf = experimental.dynamics_rhs(f, h1)
    np.testing.assert_allclose(rhsf(tc.backend.ones([])), -np.sin(1.0) / 2, atol=1e-5)

    h2 = tc.backend.coo_sparse_matrix(
        indices=tc.array_to_tensor(np.array([[0, 0], [1, 1]]), dtype="int64"),
        values=tc.array_to_tensor(np.array([1, -1])),
        shape=[2, 2],
    )

    rhsf = experimental.dynamics_rhs(f, h2)
    np.testing.assert_allclose(rhsf(tc.backend.ones([])), -np.sin(1.0) / 2, atol=1e-5)


@pytest.mark.parametrize("backend", ["tensorflow", "jax"])
def test_two_qng_approaches(backend):
    n = 6
    nlayers = 2
    with tc.runtime_backend(backend) as K:
        with tc.runtime_dtype("complex128"):

            def state(params):
                params = K.reshape(params, [2 * nlayers, n])
                c = tc.Circuit(n)
                c = tc.templates.blocks.example_block(c, params, nlayers=nlayers)
                return c.state()

            params = K.ones([2 * nlayers * n])
            params = K.cast(params, "float32")
            n1 = experimental.qng(state)(params)
            n2 = experimental.qng2(state)(params)
            np.testing.assert_allclose(n1, n2, atol=1e-7)


def test_arg_alias():
    @partial(tc.utils.arg_alias, alias_dict={"theta": ["alpha", "gamma"]})
    def f(theta: float, beta: float) -> float:
        """
        f doc

        :param theta: theta angle
        :type theta: float
        :param beta: beta angle
        :type beta: float
        :return: sum angle
        :rtype: float
        """
        return theta + beta

    np.testing.assert_allclose(f(beta=0.2, alpha=0.1), 0.3, atol=1e-5)
    print(f.__doc__)
    assert len(f.__doc__.strip().split("\n")) == 12


def test_finite_difference_tf(tfb):
    def f(param1, param2):
        n = 4
        c = tc.Circuit(n)
        for i in range(n):
            c.rx(i, theta=param1[i])
        for i in range(n - 1):
            c.cx(i, i + 1)
        for i in range(n - 1):
            c.rzz(i, i + 1, theta=param2[i])
        r = [c.expectation_ps(z=[i]) for i in range(n)]
        return tc.backend.stack(r)

    def fsum(param1, param2):
        return tc.backend.mean(f(param1, param2))

    p1 = tf.ones([4])
    p2 = tf.ones([3])
    g1, g2 = tc.backend.value_and_grad(fsum)(p1, p2)

    f1 = experimental.finite_difference_differentiator(
        f, argnums=(0, 1), shifts=(np.pi / 2, 2)
    )

    def fsum1(param1, param2):
        return tc.backend.mean(f1(param1, param2))

    g3, g4 = tc.backend.value_and_grad(fsum1)(p1, p2)

    np.testing.assert_allclose(g1, g3, atol=1e-5)
    np.testing.assert_allclose(g2, g4, atol=1e-5)


def test_evol(jaxb):
    def h_square(t, b):
        return (tc.backend.sign(t - 1.0) + 1) / 2 * b * tc.gates.x().tensor

    c = tc.Circuit(3)
    c.x(0)
    c.cx(0, 1)
    c.h(2)
    c = experimental.evol_local(
        c, [1], h_square, 2.0, tc.backend.convert_to_tensor(0.2)
    )
    c.rx(1, theta=np.pi - 0.4)
    np.testing.assert_allclose(c.expectation_ps(z=[1]), 1.0, atol=1e-5)

    ixi = tc.quantum.PauliStringSum2COO([[0, 1, 0]])

    def h_square_sparse(t, b):
        return (tc.backend.sign(t - 1.0) + 1) / 2 * b * ixi

    c = tc.Circuit(3)
    c.x(0)
    c.cx(0, 1)
    c.h(2)
    c = experimental.evol_global(
        c, h_square_sparse, 2.0, tc.backend.convert_to_tensor(0.2)
    )
    c.rx(1, theta=np.pi - 0.4)
    np.testing.assert_allclose(c.expectation_ps(z=[1]), 1.0, atol=1e-5)


def test_energy_baseline():
    print(TFIM1Denergy(10))
    print(Heisenberg1Denergy(10))
