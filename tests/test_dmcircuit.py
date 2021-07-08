import sys
import os
import numpy as np
import pytest
import tensorflow as tf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc
from .conftest import tfb


def test_gate_dm():
    c = tc.DMCircuit(3)
    c.H(0)
    c.rx(1, theta=tc.num_to_tensor(np.pi))
    assert np.allclose(c.expectation((tc.gates.z(), [0])), 0.0)
    assert np.allclose(c.expectation((tc.gates.z(), [1])), -1.0)


def test_dm_inputs():
    w = np.zeros([8])
    w[1] = 1.0
    c = tc.DMCircuit(3, inputs=w)
    c.cnot(2, 1)
    assert np.allclose(c.expectation((tc.gates.z(), [1])), -1.0)
    assert np.allclose(c.expectation((tc.gates.z(), [2])), -1.0)
    assert np.allclose(c.expectation((tc.gates.z(), [0])), 1.0)


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


@pytest.mark.parametrize("backend", [None, tfb])
def test_channel_identity(backend):
    cs = tc.channels.depolarizingchannel(0.1, 0.15, 0.2)
    tc.channels.single_qubit_kraus_identity_check(cs)
    cs = tc.channels.amplitudedampingchannel(0.25, 0.3)
    tc.channels.single_qubit_kraus_identity_check(cs)
    cs = tc.channels.phasedampingchannel(0.6)
    tc.channels.single_qubit_kraus_identity_check(cs)
    cs = tc.channels.resetchannel()
    tc.channels.single_qubit_kraus_identity_check(cs)


def test_ad_channel(tfb):
    p = tf.Variable(initial_value=0.1, dtype=tf.float32)
    theta = tf.Variable(initial_value=0.1, dtype=tf.complex64)
    with tf.GradientTape() as tape:
        tape.watch(p)
        c = tc.DMCircuit(3)
        c.rx(1, theta=theta)
        c.apply_general_kraus(tc.channels.depolarizingchannel(p, 0.2 * p, p), [(1,)])
        c.H(0)
        loss = c.expectation((tc.gates.z(), [1]))
        g = tape.gradient(loss, p)
    np.testing.assert_allclose(loss.numpy(), 0.7562, atol=1e-4)
    np.testing.assert_allclose(g.numpy(), -2.388, atol=1e-4)
