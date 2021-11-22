import sys
import os
import numpy as np
import scipy
import pytest
from pytest_lazyfixture import lazy_fixture as lf


thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc
tc.set_dtype('complex128')

#TODO: make everything compatible to different backends


def reproducible_unitary(n):
    A = np.arange(n**2).reshape((n, n))
    A = A + np.sin(A) * 1j
    A = A - A.conj().T
    return scipy.linalg.expm(A).astype(tc.dtypestr)


O1 = tc.gates.any(reproducible_unitary(2).reshape((2, 2)))
O2 = tc.gates.any(reproducible_unitary(4).reshape((2, 2, 2, 2)))

# Construct a complicated circuit by Circuit and MPSCircuit and compare

N = 16
D = 100
c = tc.Circuit(N)
c.H(0)

# create as much correlation as possible
for j in range(N // 2):
    for i in range(j, N - 1 - j):
        c.apply(O2.copy(), i, i + 1)
        c.apply(O1.copy(), i)
# test non-adjacent double gates
c.apply(O2.copy(), N // 2 - 1, N // 2 + 1)
c.apply(O2.copy(), N // 2 - 2, N // 2 + 2)
c.cz(2, 3)
w_c = c.wavefunction().flatten()

mps = tc.MPSCircuit(16)
mps.set_truncation_rule(max_singular_values=D)
mps.H(0)
for j in range(N // 2):
    for i in range(j, N - 1 - j):
        mps.apply(O2.copy(), i, i + 1)
        mps.apply(O1.copy(), i)
mps.apply(O2.copy(), N // 2 - 1, N // 2 + 1)
mps.apply(O2.copy(), N // 2 - 2, N // 2 + 2)
mps.cz(2, 3)
w_mps = mps.wavefunction().flatten()

mps_exact = tc.MPSCircuit(16)
mps_exact.set_truncation_rule()
mps_exact.H(0)
for j in range(N // 2):
    for i in range(j, N - 1 - j):
        mps_exact.apply(O2.copy(), i, i + 1)
        mps_exact.apply(O1.copy(), i)
mps_exact.apply(O2.copy(), N // 2 - 1, N // 2 + 1)
mps_exact.apply(O2.copy(), N // 2 - 2, N // 2 + 2)
mps_exact.cz(2, 3)
w_mps_exact = mps_exact.wavefunction().flatten()


def test_wavefunction():
    # the wavefuntion is exact if there's no truncation
    np.allclose(w_mps_exact, w_c, atol=1e-7)


def test_truncation():
    # compare with a precalculated value
    real_fedility_ref = 0.9998649404355184
    real_fedility = np.abs(w_mps.conj().dot(w_c)) ** 2
    assert np.isclose(real_fedility, real_fedility_ref)
    estimated_fedility_ref = 0.9999264292880625
    estimated_fedility = mps._fidelity
    assert np.isclose(estimated_fedility, estimated_fedility_ref)


def test_amplitude():
    # compare with wavefunction
    s = "01" * (N // 2)
    sint = int(s, 2)  # binary to decimal
    err_amplitude = np.abs(mps.amplitude(s) - w_mps[sint])
    assert np.isclose(err_amplitude, 0)


def test_expectation():
    for site in range(N):
        exp_mps = mps_exact.expectation_single_gate(tc.gates.z(), site)
        exp_mps_general = mps_exact.general_expectation([tc.gates.z(), [site]])
        exp_c = c.expectation((tc.gates.z(), [site]), reuse=False)
        assert np.isclose(exp_mps, exp_c)
        # the general expectation of a double qubit gate would be non-exact because of truncation, which could also be manually disabled (currently not implemented)
        assert np.isclose(exp_mps_general, exp_c)


# create a fixed wavefunction and create the corresponding MPS
w_external = np.abs(np.sin(np.arange(2**N) % np.exp(1))).astype(tc.dtypestr)  # Just want to find a function that is so strange that the correlation is strong enough
w_external /= np.linalg.norm(w_external)
mps_external = tc.MPSCircuit.from_wavefunction(w_external, max_singular_values=D)
mps_external_exact = tc.MPSCircuit.from_wavefunction(w_external)


def test_fromwavefunction():
    assert np.allclose(mps_external_exact.wavefunction(), w_external)
    # compare fidelity of truncation with theoretical limit obtained by SVD
    real_fedility = np.abs(mps_external.wavefunction().conj().flatten().dot(w_external)) ** 2
    s = np.linalg.svd(w_external.reshape((2**(N // 2), 2**(N // 2))))[1]
    theoretical_upper_limit = np.sum(s[0:D]**2)
    relative_err = np.log((1 - real_fedility) / (1 - theoretical_upper_limit))
    assert np.isclose(relative_err, 0.1185, atol=1e-4)


def test_proj():
    # compare projection value with wavefunction calculated results
    proj = mps.proj_with_mps(mps_external)
    proj_ref = mps_external.wavefunction().conj().dot(w_mps)
    np.isclose(proj, proj_ref, atol=1e-6)
