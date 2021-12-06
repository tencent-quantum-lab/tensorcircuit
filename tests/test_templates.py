import sys
import os
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


def test_grid_coord():
    cd = tc.templates.graphs.Grid2DCoord(3, 2)
    assert cd.all_cols() == [(0, 3), (1, 4), (2, 5)]
    assert cd.all_rows() == [(0, 1), (1, 2), (3, 4), (4, 5)]
