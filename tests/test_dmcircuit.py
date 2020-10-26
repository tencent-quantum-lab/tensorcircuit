import sys
import os
import numpy as np

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


def test_gate_dm():
    c = tc.DMCircuit(3)
    c.H(0)
    c.rx(1, theta=tc.num_to_tensor(np.pi))
    assert np.allclose(c.expectation((tc.gates.z(), [0])), 0.0)
    assert np.allclose(c.expectation((tc.gates.z(), [1])), -1.0)


def test_gate_depolarizing():
    c = tc.DMCircuit(2)
    c.H(0)
    c.apply_general_kraus(tc.channels.depolarizingchannel(0.1, 0.1, 0.1), [(1,)])
    np.testing.assert_allclose(
        c.densitymatrix(check=True),
        np.array(
            [[0.4, 0, 0.4, 0], [0, 0.1, 0, 0.1], [0.4, 0, 0.4, 0], [0, 0.1, 0, 0.1]]
        ),
        atol=1e-5,
    )


def test_channel_identity():
    cs = tc.channels.depolarizingchannel(0.1, 0.15, 0.2)
    tc.channels.single_qubit_kraus_identity_check(cs)
