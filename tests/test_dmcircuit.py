import sys
import os
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf
import tensorflow as tf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc
from tensorcircuit.channels import *
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
    c.apply_general_kraus(depolarizingchannel(0.1, 0.1, 0.1), [(1,)])
    np.testing.assert_allclose(
        c.densitymatrix(check=True),
        np.array(
            [[0.4, 0, 0.4, 0], [0, 0.1, 0, 0.1], [0.4, 0, 0.4, 0], [0, 0.1, 0, 0.1]]
        ),
        atol=1e-5,
    )

    c = tc.DMCircuit2(2)
    c.H(0)
    kraus = depolarizingchannel(0.1, 0.1, 0.1)
    # kraus = [k.tensor for k in kraus]
    c.apply_general_kraus(kraus, 1)
    np.testing.assert_allclose(
        c.densitymatrix(check=True),
        np.array(
            [[0.4, 0, 0.4, 0], [0, 0.1, 0, 0.1], [0.4, 0, 0.4, 0], [0, 0.1, 0, 0.1]]
        ),
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb")])
def test_mult_qubit_kraus(backend):
    xx = np.array(
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.complex64
    ) / np.sqrt(2)
    yz = (
        np.array(
            [[0, 0, -1.0j, 0], [0, 0, 0, 1.0j], [1.0j, 0, 0, 0], [0, -1.0j, 0, 0]],
            dtype=np.complex64,
        )
        / np.sqrt(2)
    )

    def forward(theta):
        c = tc.DMCircuit(3)
        c.H(0)
        c.rx(1, theta=theta)
        c.apply_general_kraus(
            [
                tc.gates.Gate(xx.reshape([2, 2, 2, 2])),
                tc.gates.Gate(yz.reshape([2, 2, 2, 2])),
            ],
            [(0, 1), (0, 1)],
        )
        c.H(1)
        return tc.backend.real(tc.backend.sum(c.densitymatrix()))

    theta = tc.num_to_tensor(0.2)
    vag = tc.backend.value_and_grad(forward)
    v1, g1 = vag(theta)
    assert np.allclose(tc.backend.numpy(g1), 0.199, atol=1e-2)

    def forward2(theta):
        c = tc.DMCircuit2(3)
        c.H(0)
        c.rx(1, theta=theta)
        c.apply_general_kraus(
            [tc.gates.Gate(xx), tc.gates.Gate(yz)],
            0,
            1,
        )
        c.H(1)
        return tc.backend.real(tc.backend.sum(c.densitymatrix()))

    theta = tc.num_to_tensor(0.2)
    vag2 = tc.backend.value_and_grad(forward2)
    v2, g2 = vag2(theta)
    assert np.allclose(tc.backend.numpy(g2), 0.199, atol=1e-2)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb")])
def test_channel_identity(backend):
    cs = depolarizingchannel(0.1, 0.15, 0.2)
    tc.channels.single_qubit_kraus_identity_check(cs)
    cs = amplitudedampingchannel(0.25, 0.3)
    tc.channels.single_qubit_kraus_identity_check(cs)
    cs = phasedampingchannel(0.6)
    tc.channels.single_qubit_kraus_identity_check(cs)
    cs = resetchannel()
    tc.channels.single_qubit_kraus_identity_check(cs)


def test_ad_channel(tfb):
    p = tf.Variable(initial_value=0.1, dtype=tf.float32)
    theta = tf.Variable(initial_value=0.1, dtype=tf.complex64)
    with tf.GradientTape() as tape:
        tape.watch(p)
        c = tc.DMCircuit(3)
        c.rx(1, theta=theta)
        c.apply_general_kraus(depolarizingchannel(p, 0.2 * p, p), [(1,)])
        c.H(0)
        loss = c.expectation((tc.gates.z(), [1]))
        g = tape.gradient(loss, p)
    np.testing.assert_allclose(loss.numpy(), 0.7562, atol=1e-4)
    np.testing.assert_allclose(g.numpy(), -2.388, atol=1e-4)
