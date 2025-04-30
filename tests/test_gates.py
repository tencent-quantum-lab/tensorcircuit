import sys
import os
import numpy as np

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


def test_rgate(highp):
    np.testing.assert_almost_equal(
        tc.gates.r_gate(1, 2, 3).tensor, tc.gates.rgate_theoretical(1, 2, 3).tensor
    )


def test_phase_gate():
    c = tc.Circuit(1)
    c.h(0)
    c.phase(0, theta=np.pi / 2)
    np.testing.assert_allclose(c.state()[1], 0.7071j, atol=1e-4)


def test_cu_gate():
    c = tc.Circuit(2)
    c.cu(0, 1, theta=np.pi / 2, phi=-np.pi / 4, lbd=np.pi / 4)
    m = c.matrix()
    print(m)
    np.testing.assert_allclose(m[2:, 2:], tc.gates._wroot_matrix, atol=1e-5)
    np.testing.assert_allclose(m[:2, :2], np.eye(2), atol=1e-5)


def test_get_u_parameter(highp):
    for _ in range(6):
        hermitian = np.random.uniform(size=[2, 2])
        hermitian += np.conj(np.transpose(hermitian))
        unitary = tc.backend.expm(hermitian * 1.0j)
        params = tc.gates.get_u_parameter(unitary)
        unitary2 = tc.gates.u_gate(theta=params[0], phi=params[1], lbd=params[2])
        ans = unitary2.tensor
        unitary = unitary / np.exp(1j * np.angle(unitary[0, 0]))
        np.testing.assert_allclose(unitary, ans, atol=1e-3)


def test_ided_gate():
    g = tc.gates.rx.ided()
    np.testing.assert_allclose(
        tc.backend.reshapem(g(theta=0.3).tensor),
        np.kron(np.eye(2), tc.gates.rx(theta=0.3).tensor),
        atol=1e-5,
    )
    g1 = tc.gates.rx.ided(before=False)
    np.testing.assert_allclose(
        tc.backend.reshapem(g1(theta=0.3).tensor),
        np.kron(tc.gates.rx(theta=0.3).tensor, np.eye(2)),
        atol=1e-5,
    )


def test_fsim_gate():
    theta = 0.2
    phi = 0.3
    c = tc.Circuit(2)
    c.iswap(0, 1, theta=-theta)
    c.cphase(0, 1, theta=-phi)
    m = c.matrix()
    ans = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.95105654 + 0.0j, 0.0 - 0.309017j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 - 0.309017j, 0.95105654 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.9553365 - 0.29552022j],
        ]
    )
    np.testing.assert_allclose(m, ans, atol=1e-5)
    print(m)


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
    np.testing.assert_allclose(c.wavefunction()[0], -1j)


def test_any_gate():
    c = tc.Circuit(2)
    c.any(0, unitary=np.eye(2))
    np.testing.assert_allclose(c.expectation((tc.gates.z(), [0])), 1.0)


def test_iswap_gate():
    t = tc.gates.iswap_gate().tensor
    ans = np.array([[1.0, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1.0]])
    np.testing.assert_allclose(t, ans.reshape([2, 2, 2, 2]), atol=1e-5)
    t = tc.gates.iswap_gate(theta=0).tensor
    np.testing.assert_allclose(t, np.eye(4).reshape([2, 2, 2, 2]), atol=1e-5)


def test_gate_list():
    assert tc.Circuit.sgates == tc.abstractcircuit.sgates


def test_controlled():
    xgate = tc.gates.x
    cxgate = xgate.controlled()
    ccxgate = cxgate.controlled()
    assert ccxgate.n == "ccx"
    assert ccxgate.ctrl == [1, 1]
    np.testing.assert_allclose(
        ccxgate().tensor, tc.backend.reshape2(tc.gates._toffoli_matrix)
    )
    ocxgate = cxgate.ocontrolled()
    c = tc.Circuit(3)
    c.x(0)
    c.any(1, 0, 2, unitary=ocxgate())
    np.testing.assert_allclose(c.expectation([tc.gates.z(), [2]]), -1, atol=1e-5)
    print(c.to_qir()[1])


def test_variable_controlled():
    crxgate = tc.gates.rx.controlled()
    c = tc.Circuit(2)
    c.x(0)
    tc.Circuit.crx_my = tc.Circuit.apply_general_variable_gate_delayed(crxgate)
    c.crx_my(0, 1, theta=0.3)
    np.testing.assert_allclose(
        c.expectation([tc.gates.z(), [1]]), 0.95533645, atol=1e-5
    )
    assert c.to_qir()[1]["name"] == "crx"


def test_adjoint_gate():
    np.testing.assert_allclose(
        tc.gates.sd().tensor, tc.backend.adjoint(tc.gates._s_matrix)
    )
    assert tc.gates.td.n == "td"


def test_rxx_gate():
    c1 = tc.Circuit(3)
    c1.rxx(0, 1, theta=1.0)
    c1.ryy(0, 2, theta=0.5)
    c1.rzz(0, 1, theta=-0.5)
    c2 = tc.Circuit(3)
    c2.exp1(0, 1, theta=1.0 / 2, unitary=tc.gates._xx_matrix)
    c2.exp1(0, 2, theta=0.5 / 2, unitary=tc.gates._yy_matrix)
    c2.exp1(0, 1, theta=-0.5 / 2, unitary=tc.gates._zz_matrix)
    np.testing.assert_allclose(c1.state(), c2.state(), atol=1e-5)
