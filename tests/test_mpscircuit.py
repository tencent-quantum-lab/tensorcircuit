# pylint: disable=unused-variable
# pylint: disable=invalid-name

from typing import Tuple, Any
import sys
import os
import numpy as np
import scipy
import pytest
from pytest_lazyfixture import lazy_fixture as lf

Tensor = Any

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


N = 16
D = 100
rules = tc.truncation_rules(max_singular_values=D)
type_test_circuits = Tuple[tc.Circuit, Tensor, tc.MPSCircuit, Tensor, tc.MPSCircuit, Tensor]


def get_test_circuits(full) -> type_test_circuits:
    def reproducible_unitary(n):
        A = np.arange(n**2).reshape((n, n))
        A = A + np.sin(A) * 1j
        A = A - A.conj().T
        return scipy.linalg.expm(A).astype(tc.dtypestr)

    O1 = tc.gates.any(reproducible_unitary(2).reshape((2, 2)))
    O2 = tc.gates.any(reproducible_unitary(4).reshape((2, 2, 2, 2)))

    # Construct a complicated circuit by Circuit and MPSCircuit and compare

    c = tc.Circuit(N)
    c.H(0)

    if full:

        def rangei(j, N):
            return range(0, N - 1)

        # rangei = lambda j, N: range(0, N - 1)
    else:

        def rangei(j, N):
            return range(j, N - 1 - j)

        # rangei = lambda j, N: range(j, N - 1 - j)

    # create as much correlation as possible
    for j in range(N // 2):
        for i in rangei(j, N):
            c.apply(O2.copy(), i, i + 1)
            c.apply(O1.copy(), i)
    # test non-adjacent double gates
    c.apply(O2.copy(), N // 2 - 1, N // 2 + 1)
    c.apply(O2.copy(), N // 2 - 2, N // 2 + 2)
    c.cz(2, 3)
    w_c = c.wavefunction()

    mps = tc.MPSCircuit(N, **rules)
    mps.H(0)
    for j in range(N // 2):
        for i in rangei(j, N):
            mps.apply(O2.copy(), i, i + 1)
            mps.apply(O1.copy(), i)
    mps.apply(O2.copy(), N // 2 - 1, N // 2 + 1)
    mps.apply(O2.copy(), N // 2 - 2, N // 2 + 2)
    mps.cz(2, 3)
    w_mps = mps.wavefunction()

    mps_exact = tc.MPSCircuit(N)
    mps_exact.H(0)
    for j in range(N // 2):
        for i in rangei(j, N):
            mps_exact.apply(O2.copy(), i, i + 1)
            mps_exact.apply(O1.copy(), i)
    mps_exact.apply(O2.copy(), N // 2 - 1, N // 2 + 1)
    mps_exact.apply(O2.copy(), N // 2 - 2, N // 2 + 2)
    mps_exact.cz(2, 3)
    w_mps_exact = mps_exact.wavefunction()

    return [c, w_c, mps, w_mps, mps_exact, w_mps_exact]


def do_test_canonical(test_circucits: type_test_circuits):
    (
        c,
        w_c,
        mps,
        w_mps,
        mps_exact,
        w_mps_exact,
    ) = test_circucits
    np.testing.assert_allclose(np.abs(mps._mps.check_canonical()), 0, atol=1e-12)
    np.testing.assert_allclose(np.abs(mps_exact._mps.check_canonical()), 0, atol=1e-12)


def do_test_wavefunction(test_circucits: type_test_circuits):
    (
        c,
        w_c,
        mps,
        w_mps,
        mps_exact,
        w_mps_exact,
    ) = test_circucits
    # the wavefuntion is exact if there's no truncation
    np.testing.assert_allclose(tc.backend.numpy(w_mps_exact), tc.backend.numpy(w_c))


def do_test_truncation(test_circucits: type_test_circuits, real_fedility_ref, estimated_fedility_ref):
    (
        c,
        w_c,
        mps,
        w_mps,
        mps_exact,
        w_mps_exact,
    ) = test_circucits
    # compare with a precalculated value
    real_fedility = (
        np.abs(tc.backend.numpy(w_mps).conj().dot(tc.backend.numpy(w_c))) ** 2
    )
    np.testing.assert_allclose(real_fedility, real_fedility_ref)
    # np.testing.assert_allclose(real_fedility, 0.9998648317622654)
    estimated_fedility = mps._fidelity
    np.testing.assert_allclose(estimated_fedility, estimated_fedility_ref)
    # np.testing.assert_allclose(estimated_fedility, 0.9999264292512574)


def do_test_amplitude(test_circucits: type_test_circuits):
    (
        c,
        w_c,
        mps,
        w_mps,
        mps_exact,
        w_mps_exact,
    ) = test_circucits
    # compare with wavefunction
    s = "01" * (N // 2)
    sint = int(s, 2)  # binary to decimal
    err_amplitude = tc.backend.abs(mps.amplitude(s) - w_mps[sint])
    np.testing.assert_allclose(err_amplitude, 0, atol=1e-12)


def do_test_expectation(test_circucits: type_test_circuits):
    (
        c,
        w_c,
        mps,
        w_mps,
        mps_exact,
        w_mps_exact,
    ) = test_circucits
    gates_sites = (tc.gates.z(), [3])
    exp_mps = mps_exact.expectation(gates_sites)
    exp_c = c.expectation(gates_sites, reuse=False)
    np.testing.assert_allclose(exp_mps, exp_c, atol=1e-7)

    gates_sites = (tc.gates.cnot(), [3, 7])
    exp_mps = mps_exact.expectation(gates_sites)
    exp_c = c.expectation(gates_sites, reuse=False)
    np.testing.assert_allclose(exp_mps, exp_c, atol=1e-7)


def external_wavefunction():
    # create a fixed wavefunction and create the corresponding MPS
    w_external = np.abs(np.sin(np.arange(2**N) % np.exp(1))).astype(
        tc.dtypestr
    )  # Just want to find a function that is so strange that the correlation is strong enough
    w_external /= np.linalg.norm(w_external)
    w_external = tc.backend.convert_to_tensor(w_external)
    mps_external = tc.MPSCircuit(N, wavefunction=w_external, **rules)
    mps_external_exact = tc.MPSCircuit(N, wavefunction=w_external)
    return w_external, mps_external, mps_external_exact


def do_test_fromwavefunction(external_wavefunction):
    (
        w_external,
        mps_external,
        mps_external_exact,
    ) = external_wavefunction
    np.testing.assert_allclose(mps_external_exact.wavefunction(), w_external, atol=1e-7)
    # compare fidelity of truncation with theoretical limit obtained by SVD
    w_external = tc.backend.numpy(w_external)
    real_fedility = (
        np.abs(tc.backend.numpy(mps_external.wavefunction()).conj().dot(w_external))
        ** 2
    )
    s = np.linalg.svd(w_external.reshape((2 ** (N // 2), 2 ** (N // 2))))[1]
    theoretical_upper_limit = np.sum(s[0:D] ** 2)
    relative_err = np.log((1 - real_fedility) / (1 - theoretical_upper_limit))
    np.testing.assert_allclose(relative_err, 0.11, atol=1e-2)


def do_test_proj(test_circucits: type_test_circuits, external_wavefunction):
    (
        c,
        w_c,
        mps,
        w_mps,
        mps_exact,
        w_mps_exact,
    ) = test_circucits
    (
        w_external,
        mps_external,
        mps_external_exact,
    ) = external_wavefunction
    # compare projection value with wavefunction calculated results
    proj = mps.proj_with_mps(mps_external)
    proj_ref = (
        tc.backend.numpy(mps_external.wavefunction())
        .conj()
        .dot(tc.backend.numpy(w_mps))
    )
    np.testing.assert_allclose(proj, proj_ref, atol=1e-12)


def do_test_tensor_input(test_circucits: type_test_circuits):
    (
        c,
        w_c,
        mps,
        w_mps,
        mps_exact,
        w_mps_exact,
    ) = test_circucits
    newmps = tc.MPSCircuit(mps._nqubits, tensors=mps.tensors, center_position=mps.center_position)
    for t1, t2 in zip(newmps.tensors, mps.tensors):
        np.testing.assert_allclose(tc.backend.numpy(t1), tc.backend.numpy(t2), atol=1e-12)


@pytest.mark.parametrize(
    "backend, dtype", [(lf("tfb"), lf("highp")), (lf("jaxb"), lf("highp"))]
)
def test_circuits_1(backend, dtype):
    import time
    begin = time.time()
    circuits = get_test_circuits(False)
    do_test_canonical(circuits)
    do_test_wavefunction(circuits)
    do_test_truncation(circuits, 0.9998648317622654, 0.9999264292512574)
    do_test_amplitude(circuits)
    do_test_expectation(circuits)
    external = external_wavefunction()
    do_test_fromwavefunction(external)
    do_test_proj(circuits, external)
    do_test_tensor_input(circuits)


def test_circuits_2(highp):
    circuits = get_test_circuits(True)
    do_test_truncation(circuits, 0.9705050538783289, 0.984959108658121)
