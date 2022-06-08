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


def test_iswap_gate():
    t = tc.gates.iswap_gate().tensor
    ans = np.array([[1.0, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1.0]])
    np.testing.assert_allclose(t, ans.reshape([2, 2, 2, 2]), atol=1e-5)
    t = tc.gates.iswap_gate(theta=0).tensor
    np.testing.assert_allclose(t, np.eye(4).reshape([2, 2, 2, 2]), atol=1e-5)


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
    tc.Circuit.crx_my = c.apply_general_variable_gate_delayed(crxgate)
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
    c2.exp1(0, 1, theta=1.0, unitary=tc.gates._xx_matrix)
    c2.exp1(0, 2, theta=0.5, unitary=tc.gates._yy_matrix)
    c2.exp1(0, 1, theta=-0.5, unitary=tc.gates._zz_matrix)
    np.testing.assert_allclose(c1.state(), c2.state(), atol=1e-5)
