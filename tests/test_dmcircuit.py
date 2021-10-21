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


def test_state_inputs():
    w = np.zeros([8])
    w[1] = 1.0
    c = tc.DMCircuit(3, inputs=w)
    c.cnot(2, 1)
    assert np.allclose(c.expectation((tc.gates.z(), [1])), -1.0)
    assert np.allclose(c.expectation((tc.gates.z(), [2])), -1.0)
    assert np.allclose(c.expectation((tc.gates.z(), [0])), 1.0)

    s2 = np.sqrt(2.0)
    w = np.array([1 / s2, 0, 0, 1.0j / s2])
    c = tc.DMCircuit(2, inputs=w)
    c.Y(0)
    answer = np.array(
        [[0, 0, 0, 0], [0, 0.5, -0.5j, 0], [0, 0.5j, 0.5, 0], [0, 0, 0, 0]]
    )
    print(c.densitymatrix())
    assert np.allclose(c.densitymatrix(), answer)

    c = tc.DMCircuit2(2, inputs=w)
    c.Y(0)
    print(c.densitymatrix())
    assert np.allclose(c.densitymatrix(), answer)


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb"), lf("tfb")])
def test_dm_inputs(backend):
    rho0 = np.array([[0, 0, 0, 0], [0, 0.5, 0, -0.5j], [0, 0, 0, 0], [0, 0.5j, 0, 0.5]])
    b1 = np.array([[0, 1.0j], [0, 0]])
    b2 = np.array([[0, 0], [1.0j, 0]])
    ib1 = np.kron(np.eye(2), b1)
    ib2 = np.kron(np.eye(2), b2)
    rho1 = ib1 @ rho0 @ np.transpose(np.conj(ib1)) + ib2 @ rho0 @ np.transpose(
        np.conj(ib2)
    )
    iy = np.kron(np.eye(2), np.array([[0, -1.0j], [1.0j, 0]]))
    rho2 = iy @ rho1 @ np.transpose(np.conj(iy))
    rho0 = rho0.astype(np.complex64)
    b1 = b1.astype(np.complex64)
    b2 = b2.astype(np.complex64)
    c = tc.DMCircuit2(nqubits=2, dminputs=rho0)
    c.apply_general_kraus([tc.gates.Gate(b1), tc.gates.Gate(b2)], 1)
    assert np.allclose(c.densitymatrix(), rho1, atol=1e-4)
    c.y(1)
    assert np.allclose(c.densitymatrix(), rho2, atol=1e-4)


def test_inputs_and_kraus():
    rho0 = np.array([[0, 0, 0, 0], [0, 0.5, 0, -0.5j], [0, 0, 0, 0], [0, 0.5j, 0, 0.5]])
    b1 = np.array([[0, 1.0j], [0, 0]])
    b2 = np.array([[0, 0], [1.0j, 0]])
    single_qubit_kraus_identity_check([tc.gates.Gate(b1), tc.gates.Gate(b2)])
    ib1 = np.kron(np.eye(2), b1)
    ib2 = np.kron(np.eye(2), b2)
    rho1 = ib1 @ rho0 @ np.transpose(np.conj(ib1)) + ib2 @ rho0 @ np.transpose(
        np.conj(ib2)
    )
    iy = np.kron(np.eye(2), np.array([[0, -1.0j], [1.0j, 0]]))
    rho2 = iy @ rho1 @ np.transpose(np.conj(iy))
    s2 = np.sqrt(2.0)
    w = np.array([1 / s2, 0, 0, 1.0j / s2])

    c = tc.DMCircuit2(2, inputs=w)
    c.y(0)
    c.cnot(0, 1)
    assert np.allclose(c.densitymatrix(), rho0, atol=1e-4)
    c.apply_general_kraus([tc.gates.Gate(b1), tc.gates.Gate(b2)], 1)
    assert np.allclose(c.densitymatrix(), rho1, atol=1e-4)
    c.y(1)
    assert np.allclose(c.densitymatrix(), rho2, atol=1e-4)

    c = tc.DMCircuit(2, inputs=w)
    c.y(0)
    c.cnot(0, 1)
    assert np.allclose(c.densitymatrix(), rho0, atol=1e-4)
    c.apply_general_kraus([tc.gates.Gate(b1), tc.gates.Gate(b2)], [(1,)])
    assert np.allclose(c.densitymatrix(), rho1, atol=1e-4)
    c.y(1)
    assert np.allclose(c.densitymatrix(), rho2, atol=1e-4)


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

    c = tc.DMCircuit(2)
    c.H(0)
    kraus = depolarizingchannel(0.1, 0.1, 0.1)
    # kraus = [k.tensor for k in kraus]
    c.depolarizing(1, px=0.1, py=0.1, pz=0.1)
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

    c = tc.DMCircuit2(2)
    c.H(0)
    kraus = depolarizingchannel(0.1, 0.1, 0.1)
    # kraus = [k.tensor for k in kraus]
    c.depolarizing(1, px=0.1, py=0.1, pz=0.1)
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


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb")])
def test_noise_param_ad(backend):
    def forward(p):
        c = tc.DMCircuit2(2)
        c.X(1)
        c.depolarizing(1, px=p, py=p, pz=p)
        return tc.backend.real(
            c.expectation(
                (
                    tc.gates.z(),
                    [
                        1,
                    ],
                )
            )
        )

    theta = tc.num_to_tensor(0.1)
    vag = tc.backend.value_and_grad(forward)
    v, g = vag(theta)
    assert np.allclose(v, -0.6, atol=1e-2)
    assert np.allclose(g, 4, atol=1e-2)


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
