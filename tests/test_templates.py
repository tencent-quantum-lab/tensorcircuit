# pylint: disable=invalid-name

import os
import sys

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


def test_any_measurement():
    c = tc.Circuit(2)
    c.H(0)
    c.H(1)
    mea = np.array([1, 1])
    r = tc.templates.measurements.any_measurements(c, mea, onehot=True)
    np.testing.assert_allclose(r, 1.0, atol=1e-5)
    mea2 = np.array([3, 0])
    r2 = tc.templates.measurements.any_measurements(c, mea2, onehot=True)
    np.testing.assert_allclose(r2, 0.0, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb")])
def test_parameterized_local_measurement(backend):
    c = tc.Circuit(3)
    c.X(0)
    c.cnot(0, 1)
    c.H(-1)
    basis = tc.backend.convert_to_tensor(np.array([3, 3, 1]))
    r = tc.templates.measurements.parameterized_local_measurements(
        c, structures=basis, onehot=True
    )
    np.testing.assert_allclose(r, np.array([-1, -1, 1]), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_sparse_expectation(backend):
    ham = tc.backend.coo_sparse_matrix(
        indices=[[0, 1], [1, 0]], values=tc.backend.ones([2]), shape=(2, 2)
    )

    def f(param):
        c = tc.Circuit(1)
        c.rx(0, theta=param[0])
        c.H(0)
        return tc.templates.measurements.sparse_expectation(c, ham)

    fvag = tc.backend.jit(tc.backend.value_and_grad(f))
    param = tc.backend.zeros([1])
    print(fvag(param))


def test_bell_block():
    c = tc.Circuit(4)
    c = tc.templates.blocks.Bell_pair_block(c)
    for _ in range(10):
        s = c.perfect_sampling()[0]
        assert s[0] != s[1]
        assert s[2] != s[3]


def test_qft_block() -> None:
    c = tc.Circuit(4)
    c = tc.templates.blocks.qft(c, 0, 1, 2, 3)
    s = c.perfect_sampling()
    assert s[1] - 0.0624999 < 10e-6


def test_grid_coord():
    cd = tc.templates.graphs.Grid2DCoord(3, 2)
    assert cd.all_cols() == [(0, 3), (1, 4), (2, 5)]
    assert cd.all_rows() == [(0, 1), (1, 2), (3, 4), (4, 5)]


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_qaoa_template(backend):
    cd = tc.templates.graphs.Grid2DCoord(3, 2)
    g = cd.lattice_graph(pbc=False)
    for e1, e2 in g.edges:
        g[e1][e2]["weight"] = np.random.uniform()

    def forward(paramzz, paramx):
        c = tc.Circuit(6)
        for i in range(6):
            c.H(i)
        c = tc.templates.blocks.QAOA_block(c, g, paramzz, paramx)
        return tc.templates.measurements.spin_glass_measurements(c, g)

    fvag = tc.backend.jit(tc.backend.value_and_grad(forward, argnums=(0, 1)))
    paramzz = tc.backend.real(tc.backend.ones([1]))
    paramx = tc.backend.real(tc.backend.ones([1]))
    _, gr = fvag(paramzz, paramx)
    np.testing.assert_allclose(gr[1].shape, [1])
    paramzz = tc.backend.real(tc.backend.ones([7]))
    paramx = tc.backend.real(tc.backend.ones([1]))
    _, gr = fvag(paramzz, paramx)
    np.testing.assert_allclose(gr[0].shape, [7])
    paramzz = tc.backend.real(tc.backend.ones([1]))
    paramx = tc.backend.real(tc.backend.ones([6]))
    _, gr = fvag(paramzz, paramx)
    np.testing.assert_allclose(gr[0].shape, [1])
    np.testing.assert_allclose(gr[1].shape, [6])


def test_state_wrapper():
    Bell_pair_block_state = tc.templates.blocks.state_centric(
        tc.templates.blocks.Bell_pair_block
    )
    s = Bell_pair_block_state(np.array([1.0, 0, 0, 0]))
    np.testing.assert_allclose(
        s, np.array([0.0, 0.70710677 + 0.0j, -0.70710677 + 0.0j, 0]), atol=1e-5
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_amplitude_encoding(backend):
    batched_amplitude_encoding = tc.backend.vmap(
        tc.templates.dataset.amplitude_encoding, vectorized_argnums=0
    )
    figs = np.stack([np.eye(2), np.ones([2, 2])])
    figs = tc.array_to_tensor(figs)
    states = batched_amplitude_encoding(figs, 3)
    # note that you cannot use nqubits=3 here for jax backend
    # see this issue: https://github.com/google/jax/issues/7465
    np.testing.assert_allclose(states[1], np.array([0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0]))
    states = batched_amplitude_encoding(
        figs, 2, tc.array_to_tensor(np.array([0, 3, 1, 2]), dtype="int32")
    )
    np.testing.assert_allclose(states[0], 1 / np.sqrt(2) * np.array([1, 1, 0, 0]))


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_mpo_measurement(backend):
    def f(theta):
        mpo = tc.quantum.QuOperator.from_local_tensor(
            tc.array_to_tensor(tc.gates._x_matrix), [2, 2, 2], [0]
        )
        c = tc.Circuit(3)
        c.ry(0, theta=theta)
        c.H(1)
        c.H(2)
        e = tc.templates.measurements.mpo_expectation(c, mpo)
        return e

    v, g = tc.backend.jit(tc.backend.value_and_grad(f))(tc.backend.ones([]))

    np.testing.assert_allclose(v, 0.84147, atol=1e-4)
    np.testing.assert_allclose(g, 0.54032, atol=1e-4)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_operator_measurement(backend):
    mpo = tc.quantum.QuOperator.from_local_tensor(
        tc.array_to_tensor(tc.gates._x_matrix), [2, 2], [0]
    )
    dense = tc.array_to_tensor(np.kron(tc.gates._x_matrix, np.eye(2)))
    sparse = tc.quantum.PauliString2COO([1, 0])

    for h in [dense, sparse, mpo]:

        def f(theta):
            c = tc.Circuit(2)
            c.ry(0, theta=theta)
            c.H(1)
            e = tc.templates.measurements.operator_expectation(c, h)
            return e

        v, g = tc.backend.jit(tc.backend.value_and_grad(f))(tc.backend.ones([]))

        np.testing.assert_allclose(v, 0.84147, atol=1e-4)
        np.testing.assert_allclose(g, 0.54032, atol=1e-4)


@pytest.fixture
def symmetric_matrix():
    matrix = np.array([[-5.0, -2.0], [-2.0, 6.0]])
    nsym_matrix = np.array(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0], [8.0, 7.0, 6.0]]
    )
    return matrix, nsym_matrix


def test_QUBO_to_Ising(symmetric_matrix):
    matrix1, matrix2 = symmetric_matrix
    pauli_terms, weights, offset = tc.templates.conversions.QUBO_to_Ising(matrix1)
    n = matrix1.shape[0]
    expected_num_terms = n + n * (n - 1) // 2
    assert len(pauli_terms) == expected_num_terms
    assert len(weights) == expected_num_terms
    assert isinstance(pauli_terms, list)
    assert isinstance(weights, np.ndarray)
    assert isinstance(offset, float)
    assert pauli_terms == [
        [1, 0],
        [0, 1],
        [1, 1],
    ]
    assert all(weights == np.array([3.5, -2.0, -1.0]))
    assert offset == -0.5

    with pytest.raises(ValueError):
        tc.templates.conversions.QUBO_to_Ising(matrix2)
