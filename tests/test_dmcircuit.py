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
from tensorcircuit.channels import (
    depolarizingchannel,
    single_qubit_kraus_identity_check,
)


def test_gate_dm():
    c = tc.DMCircuit(3)
    c.H(0)
    c.rx(1, theta=tc.num_to_tensor(np.pi))
    np.testing.assert_allclose(c.expectation((tc.gates.z(), [0])), 0.0, atol=1e-4)
    np.testing.assert_allclose(c.expectation((tc.gates.z(), [1])), -1.0, atol=1e-4)


def test_state_inputs():
    w = np.zeros([8])
    w[1] = 1.0
    c = tc.DMCircuit(3, inputs=w)
    c.cnot(2, 1)
    np.testing.assert_allclose(c.expectation((tc.gates.z(), [1])), -1.0)
    np.testing.assert_allclose(c.expectation((tc.gates.z(), [2])), -1.0)
    np.testing.assert_allclose(c.expectation((tc.gates.z(), [0])), 1.0)

    s2 = np.sqrt(2.0)
    w = np.array([1 / s2, 0, 0, 1.0j / s2])
    c = tc.DMCircuit(2, inputs=w)
    c.Y(0)
    answer = np.array(
        [[0, 0, 0, 0], [0, 0.5, -0.5j, 0], [0, 0.5j, 0.5, 0], [0, 0, 0, 0]]
    )
    print(c.densitymatrix())
    np.testing.assert_allclose(c.densitymatrix(), answer)

    c = tc.DMCircuit(2, inputs=w)
    c.Y(0)
    print(c.densitymatrix())
    np.testing.assert_allclose(c.densitymatrix(), answer)


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
    c = tc.DMCircuit(nqubits=2, dminputs=rho0)
    c.apply_general_kraus([tc.gates.Gate(b1), tc.gates.Gate(b2)], [(1,)])
    np.testing.assert_allclose(c.densitymatrix(), rho1, atol=1e-4)
    c.y(1)
    np.testing.assert_allclose(c.densitymatrix(), rho2, atol=1e-4)


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

    c = tc.DMCircuit(2, inputs=w)
    c.y(0)
    c.cnot(0, 1)
    np.testing.assert_allclose(c.densitymatrix(), rho0, atol=1e-4)
    c.apply_general_kraus([tc.gates.Gate(b1), tc.gates.Gate(b2)], 1)
    np.testing.assert_allclose(c.densitymatrix(), rho1, atol=1e-4)
    c.y(1)
    np.testing.assert_allclose(c.densitymatrix(), rho2, atol=1e-4)

    c = tc.DMCircuit(2, inputs=w)
    c.y(0)
    c.cnot(0, 1)
    np.testing.assert_allclose(c.densitymatrix(), rho0, atol=1e-4)
    c.apply_general_kraus([tc.gates.Gate(b1), tc.gates.Gate(b2)], [(1,)])
    np.testing.assert_allclose(c.densitymatrix(), rho1, atol=1e-4)
    c.y(1)
    np.testing.assert_allclose(c.densitymatrix(), rho2, atol=1e-4)


def test_gate_depolarizing():
    ans = np.array(
        [[0.4, 0, 0.4, 0], [0, 0.1, 0, 0.1], [0.4, 0, 0.4, 0], [0, 0.1, 0, 0.1]]
    )

    def check_template(c, api="v1"):
        c.H(0)
        if api == "v1":
            kraus = depolarizingchannel(0.1, 0.1, 0.1)
            c.apply_general_kraus(kraus, [(1,)])
        else:
            c.depolarizing(1, px=0.1, py=0.1, pz=0.1)
        np.testing.assert_allclose(
            c.densitymatrix(check=True),
            ans,
            atol=1e-5,
        )

    for c, v in zip([tc.DMCircuit(2), tc.DMCircuit_reference(2)], ["v1", "v2"]):
        check_template(c, v)


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb")])
def test_mult_qubit_kraus(backend):
    xx = np.array(
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.complex64
    ) / np.sqrt(2)
    yz = np.array(
        [[0, 0, -1.0j, 0], [0, 0, 0, 1.0j], [1.0j, 0, 0, 0], [0, -1.0j, 0, 0]],
        dtype=np.complex64,
    ) / np.sqrt(2)

    def forward(theta):
        c = tc.DMCircuit_reference(3)
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
    vg = tc.backend.value_and_grad(forward)
    _, g1 = vg(theta)
    np.testing.assert_allclose(tc.backend.numpy(g1), 0.199, atol=1e-2)

    def forward2(theta):
        c = tc.DMCircuit(3)
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
    vg2 = tc.backend.value_and_grad(forward2)
    _, g2 = vg2(theta)
    np.testing.assert_allclose(tc.backend.numpy(g2), 0.199, atol=1e-2)


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb")])
def test_noise_param_ad(backend):
    def forward(p):
        c = tc.DMCircuit(2)
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
    vg = tc.backend.value_and_grad(forward)
    v, g = vg(theta)
    np.testing.assert_allclose(v, -0.6, atol=1e-2)
    np.testing.assert_allclose(g, 4, atol=1e-2)


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


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb"), lf("torchb")])
def test_inputs_pipeline(backend):
    dm = 0.25 * np.eye(4)
    c = tc.DMCircuit(2, dminputs=dm)
    c.H(0)
    c.general_kraus(
        [np.sqrt(0.5) * tc.gates._i_matrix, np.sqrt(0.5) * tc.gates._x_matrix], 1
    )
    r = c.expectation_ps(z=[0])
    np.testing.assert_allclose(r, 0.0, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_measure(backend):
    key = tc.backend.get_random_state(42)

    @tc.backend.jit
    def r(key):
        tc.backend.set_random_state(key)
        c = tc.DMCircuit(2)
        c.depolarizing(0, px=0.2, py=0.2, pz=0.2)
        rs = c.measure(0, with_prob=True)
        return rs

    key1, key2 = tc.backend.random_split(key)
    rs1, rs2 = r(key1), r(key2)
    assert rs1[0] != rs2[0]
    print(rs1[1], rs2[1])


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_mpo_dm(backend, highp):
    c = tc.DMCircuit(3)
    c.H(1)
    c.depolarizing(0, px=0.05, py=0.02, pz=0.02)
    c.multicontrol(1, 0, 2, ctrl=[1, 1], unitary=tc.gates.x())
    r = tc.backend.real(c.expectation_ps(z=[2]))
    np.testing.assert_allclose(r, 0.93, atol=1e-8)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_to_circuit(backend):
    c = tc.DMCircuit(2)
    c.X(0)
    c.depolarizing(0, px=0.1, py=0.1, pz=0.1)
    c.cnot(0, 1)
    np.testing.assert_allclose(
        tc.backend.real(c.expectation_ps(z=[1])), -0.6, atol=1e-5
    )
    c2 = c.to_circuit()
    np.testing.assert_allclose(
        tc.backend.real(c2.expectation_ps(z=[1])), -1.0, atol=1e-5
    )


def test_dmcircuit_inverse():
    c = tc.DMCircuit2(3)
    c.h(0)
    c.rx(1, theta=0.5)
    c.amplitudedamping(1, gamma=0.1, p=0.9)
    c.amplitudedamping(2, gamma=0.1, p=0.9)
    c.rzz(0, 2, theta=-1.0)
    ci = c.inverse()
    r = tc.backend.real(ci.expectation_ps(z=[2]))
    c2 = tc.DMCircuit2(3)
    c2.rzz(0, 2, theta=1.0)
    c2.rx(1, theta=-0.5)
    c2.h(0)
    r2 = tc.backend.real(c2.expectation_ps(z=[2]))
    np.testing.assert_allclose(r, r2, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_perfect_sampling_with_status(backend):
    n = 3

    @tc.backend.jit
    def m(s):
        c = tc.DMCircuit(n)
        for i in range(n):
            c.H(i)
        for i in range(n):
            c.amplitudedamping(i, p=1.0, gamma=0.1)
        return c.perfect_sampling(s)

    s, p = m(tc.backend.convert_to_tensor(np.array([0.9, 0.5, 0.7])))
    np.testing.assert_allclose(s, np.array([1, 0, 1]))
    np.testing.assert_allclose(p, 0.111375, atol=1e-5)


def test_dm_circuit_draw():
    c = tc.DMCircuit(3)
    c.H(0)
    c.cnot(0, 2)
    c.depolarizing(1, px=0.1, py=0.1, pz=0.1)
    c.rxx(1, 2, theta=0.5)
    print("\n")
    print(c.draw())


def test_dm_qiskit():
    try:
        import qiskit

        print(qiskit.__version__)
    except ImportError:
        pytest.skip("qiskit is not installed")
    n = 4
    c = tc.DMCircuit(n)
    c.ryy(0, 1, theta=1.3)
    c.td(1)
    c.h(2)
    c.cnot(3, 0)
    c.cry(0, 1, theta=0.3)
    c.multicontrol(1, 2, 0, 3, ctrl=[0, 1], unitary=tc.gates._zz_matrix)
    c.cswap(1, 2, 3)
    qc = c.to_qiskit()
    c1 = tc.DMCircuit.from_qiskit(qc)
    r0 = sum(
        [c.expectation_ps(z=[i]) for i in range(n)]
        + [c.expectation_ps(y=[i]) for i in range(n)]
    )
    r1 = sum(
        [c1.expectation_ps(z=[i]) for i in range(n)]
        + [c1.expectation_ps(y=[i]) for i in range(n)]
    )
    np.testing.assert_allclose(r0, r1, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_dmcircuit_split(backend):
    n = 4

    def f(param, max_singular_values=None, max_truncation_err=None, fixed_choice=None):
        if (max_singular_values is None) and (max_truncation_err is None):
            split = None
        else:
            split = {
                "max_singular_values": max_singular_values,
                "max_truncation_err": max_truncation_err,
                "fixed_choice": fixed_choice,
            }
        c = tc.DMCircuit(
            n,
            split=split,
        )
        for i in range(n):
            c.H(i)
        for j in range(2):
            for i in range(n - 1):
                c.exp1(i, i + 1, theta=param[2 * j, i], unitary=tc.gates._zz_matrix)
            for i in range(n):
                c.rx(i, theta=param[2 * j + 1, i])
        loss = c.expectation_ps(z=[1, 2])
        return tc.backend.real(loss)

    s1 = f(tc.backend.ones([4, n]))
    s2 = f(tc.backend.ones([4, n]), max_truncation_err=1e-5)
    s3 = f(tc.backend.ones([4, n]), max_singular_values=2, fixed_choice=1)
    np.testing.assert_allclose(s1, s2, atol=1e-5)
    np.testing.assert_allclose(s1, s3, atol=1e-5)

    # np.testing.assert_allclose(s1, s2, atol=1e-5)
    np.testing.assert_allclose(s1, s3, atol=1e-5)

    f_vg = tc.backend.jit(
        tc.backend.value_and_grad(f, argnums=0), static_argnums=(1, 2, 3)
    )

    s1, g1 = f_vg(tc.backend.ones([4, n]))
    s3, g3 = f_vg(tc.backend.ones([4, n]), max_singular_values=2, fixed_choice=1)

    np.testing.assert_allclose(s1, s3, atol=1e-5)
    np.testing.assert_allclose(g1, g3, atol=1e-5)


def test_dm_amplitude():
    c = tc.DMCircuit(2)
    c.H(0)
    c.cnot(0, 1)
    np.testing.assert_allclose(c.amplitude("11"), 0.5, atol=1e-5)
    c.depolarizing(1, px=0.2, py=0, pz=0)
    np.testing.assert_allclose(c.amplitude("11"), 0.4, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_dm_amplitude_jit(backend):
    @tc.backend.jit
    def m(s):
        c = tc.DMCircuit(2)
        c.H(0)
        c.cnot(0, 1)
        c.depolarizing(1, px=0.2, py=0.0, pz=0.0)
        return c.amplitude(s)

    np.testing.assert_allclose(m(tc.array_to_tensor([1, 1])), 0.4, atol=1e-5)
    np.testing.assert_allclose(m(tc.array_to_tensor([1, 0])), 0.1, atol=1e-5)


def test_sample():
    c = tc.DMCircuit(2)
    c.H(0)
    c.cnot(0, 1)
    c.depolarizing(1, px=0.2, py=0.0, pz=0.0)
    print(c.sample(batch=10))
    print(c.sample(batch=20, allow_state=True))


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_dm_mps_inputs(backend):
    ns = [tc.gates.Gate(np.array([0.0, 1.0])) for _ in range(3)]
    mps = tc.quantum.QuVector([n[0] for n in ns])
    c = tc.DMCircuit(3, mps_inputs=mps)
    c.cnot(0, 1)
    c.depolarizing(1, px=0.2, py=0, pz=0)
    np.testing.assert_allclose(c.expectation_ps(z=[1]), 0.6, atol=1e-5)
    np.testing.assert_allclose(c.expectation_ps(z=[2]), -1.0, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_dm_mpo_inputs(backend):
    n = 4
    nodes = [tc.Gate(np.eye(2) / 2.0) for _ in range(n)]
    mpo = tc.quantum.QuOperator([n[0] for n in nodes], [n[1] for n in nodes])
    print(len(mpo.nodes))
    np.testing.assert_allclose(tc.backend.trace(mpo.eval_matrix()), 1.0, atol=1e-5)
    c = tc.DMCircuit(4, mpo_dminputs=mpo)
    for i in range(n):
        c.x(i)
        c.depolarizing(i, px=0.1, py=0.1, pz=0.1)
    np.testing.assert_allclose(c.state(), np.eye(2**n) / 2**n, atol=1e-5)
    print(len(c._nodes), len(c._front))


def test_dm_cond_measure():
    c = tc.DMCircuit(2)
    c.H(0)
    np.testing.assert_allclose(c.expectation_ps(x=[0]), 1.0, atol=1e-5)
    c.cond_measure(0)
    np.testing.assert_allclose(c.expectation_ps(x=[0]), 0.0, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_prepend_dmcircuit(backend):
    c = tc.DMCircuit(2)
    c.H(0)
    c1 = tc.DMCircuit(2)
    c1.cnot(0, 1)
    c2 = c1.append(c)
    c3 = c2.prepend(c)
    qir = c3.to_qir()
    for n, n0 in zip(qir, ["h", "cnot", "h"]):
        assert n["name"] == n0
    s = c3.wavefunction()
    np.testing.assert_allclose(s[0], s[1], atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_dm_sexpps(backend):
    c = tc.DMCircuit(1, inputs=1 / np.sqrt(2) * np.array([1.0, 1.0j]))
    y = c.sample_expectation_ps(y=[0])
    ye = c.expectation_ps(y=[0])
    np.testing.assert_allclose(y, 1.0, atol=1e-5)
    np.testing.assert_allclose(ye, 1.0, atol=1e-5)

    c = tc.DMCircuit(4)
    c.H(0)
    c.H(1)
    c.X(2)
    c.Y(3)
    c.cnot(0, 1)
    c.depolarizing(1, px=0.05, py=0.05, pz=0.1)
    c.rx(1, theta=0.3)
    c.ccnot(2, 3, 1)
    c.depolarizing(0, px=0.05, py=0.0, pz=0.1)
    c.rzz(0, 3, theta=0.5)
    c.ry(3, theta=2.2)
    c.amplitudedamping(2, gamma=0.1, p=0.95)
    c.s(1)
    c.td(2)
    c.cswap(3, 0, 1)
    y = c.sample_expectation_ps(x=[1], y=[0], z=[2, 3])
    ye = c.expectation_ps(x=[1], y=[0], z=[2, 3])
    np.testing.assert_allclose(ye, y, atol=1e-5)
    y2 = c.sample_expectation_ps(x=[1], y=[0], z=[2, 3], shots=81920)
    print(y, y2)
    assert np.abs(y2 - y) < 0.015


def test_dm_sexpps_jittable_vamppable(jaxb):
    n = 4
    m = 2

    def f(param, key):
        c = tc.DMCircuit(n)
        for j in range(m):
            for i in range(n - 1):
                c.cnot(i, i + 1)
            for i in range(n):
                c.rx(i, theta=param[i, j])
        return tc.backend.real(
            c.sample_expectation_ps(y=[n // 2], shots=8192, random_generator=key)
        )

    vf = tc.backend.jit(tc.backend.vmap(f, vectorized_argnums=(0, 1)))
    r = vf(
        tc.backend.ones([2, n, m], dtype="float32"),
        tc.backend.stack(
            [
                tc.backend.get_random_state(42),
                tc.backend.get_random_state(43),
            ]
        ),
    )
    assert np.abs(r[0] - r[1]) > 1e-4

    print(r)


def test_dm_sexpps_jittable_vamppable_tf(tfb):
    # finally giving up backend agnosticity
    # and not sure the effciency and the safety of vmap random in tf
    n = 4
    m = 2

    def f(param):
        c = tc.DMCircuit(n)
        for j in range(m):
            for i in range(n - 1):
                c.cnot(i, i + 1)
            for i in range(n):
                c.rx(i, theta=param[i, j])
        return tc.backend.real(c.sample_expectation_ps(y=[n // 2], shots=8192))

    vf = tc.backend.jit(tc.backend.vmap(f, vectorized_argnums=0))
    r = vf(tc.backend.ones([2, n, m]))
    r1 = vf(tc.backend.ones([2, n, m]))
    assert np.abs(r[0] - r[1]) > 1e-5
    assert np.abs(r[0] - r1[0]) > 1e-5
    print(r, r1)
