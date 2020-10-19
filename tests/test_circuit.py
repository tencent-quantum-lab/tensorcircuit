import sys
import os
import numpy as np

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


def test_wavefunction():
    qc = tc.Circuit(2)
    qc.apply_double_gate(
        tc.gates.Gate(np.arange(16).reshape(2, 2, 2, 2).astype(np.complex64)), 0, 1
    )
    assert np.real(qc.wavefunction()[0][2]) == 8
    qc = tc.Circuit(2)
    qc.apply_double_gate(
        tc.gates.Gate(np.arange(16).reshape(2, 2, 2, 2).astype(np.complex64)), 1, 0
    )
    qc.wavefunction()
    assert np.real(qc.wavefunction()[0][2]) == 4
    qc = tc.Circuit(2)
    qc.apply_single_gate(
        tc.gates.Gate(np.arange(4).reshape(2, 2).astype(np.complex64)), 0
    )
    qc.wavefunction()
    assert np.real(qc.wavefunction()[0][2]) == 2


def test_basics():
    c = tc.Circuit(2)
    c.x(0)
    assert np.allclose(c.amplitude("10"), np.array(1.0))
    c.CNOT(0, 1)
    assert np.allclose(c.amplitude("11"), np.array(1.0))


def test_measure():
    c = tc.Circuit(3)
    c.H(0)
    c.h(1)
    c.toffoli(0, 1, 2)
    assert c.measure(2)[0] in ["0", "1"]


def test_expectation():
    c = tc.Circuit(2)
    c.H(0)
    assert np.allclose(c.expectation((tc.gates.z(), [0])), 0, atol=1e-7)


def test_qcode():
    qcode = """
4
x 0
cnot 0 1
r 2 theta 1.0 alpha 1.57
"""
    c = tc.Circuit.from_qcode(qcode)
    assert c.measure(1)[0] == "1"
    assert c.to_qcode() == qcode[1:]


def universal_ad():
    @tc.backend.jit
    def forward(theta):
        c = tc.Circuit(2)
        c.R(0, theta=theta, alpha=0.5, phi=0.8)
        return tc.backend.real(c.expectation((tc.gates.z(), [0])))

    gg = tc.backend.grad(forward)
    gg = tc.backend.jit(gg)
    theta = tc.gates.num_to_tensor(1.0)
    return gg(theta)


def test_ad():
    # this amazingly shows how to code once and run in very different AD-ML engines
    tc.set_backend("tensorflow")
    universal_ad()
    tc.set_backend("jax")
    universal_ad()
    tc.set_backend("numpy")
