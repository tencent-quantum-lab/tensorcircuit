import sys
import os
import numpy as np

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


def test_rgate(highp):
    np.testing.assert_almost_equal(
        tc.gates.rgate(1, 2, 3).tensor, tc.gates.rgate_theoretical(1, 2, 3).tensor
    )


def test_exp_gate():
    c = tc.Circuit(2)
    c.exp(
        0,
        1,
        unitary=tc.gates.array_to_tensor(
            np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        ),
        theta=tc.gates.num_to_tensor(np.pi / 2),
    )
    assert np.allclose(c.wavefunction()[0], -1j)


def test_any_gate():
    c = tc.Circuit(2)
    c.any(0, unitary=np.eye(2))
    assert np.allclose(c.expectation((tc.gates.z(), [0])), 1.0)
