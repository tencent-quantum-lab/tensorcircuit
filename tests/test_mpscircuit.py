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
split = tc.cons.split_rules(max_singular_values=D)
type_test_circuits = Tuple[
    tc.Circuit, Tensor, tc.MPSCircuit, Tensor, tc.MPSCircuit, Tensor
]


def reproducible_unitary(n):
    A = np.arange(n**2).reshape((n, n))
    A = A + np.sin(A) * 1j
    A = A - A.conj().T
    return scipy.linalg.expm(A).astype(tc.dtypestr)


def get_test_circuits(full) -> type_test_circuits:
    O1 = tc.gates.any(reproducible_unitary(2).reshape((2, 2)))
    O2 = tc.gates.any(reproducible_unitary(4).reshape((2, 2, 2, 2)))
    O3 = tc.gates.any(reproducible_unitary(8).reshape((2, 2, 2, 2, 2, 2)))

    # Construct a complicated circuit by Circuit and MPSCircuit and compare

    if full:

        def rangei(j, N):
            return range(0, N - 1)

    else:

        def rangei(j, N):
            return range(j, N - 1 - j)

    def simulate(c):
        c.H(0)
        # create as much correlation as possible
        for j in range(N // 2):
            for i in rangei(j, N):
                c.apply(O2.copy(), i, i + 1)
                c.apply(O1.copy(), i)
        # test non-adjacent double gates
        c.apply(O2.copy(), N // 2 - 1, N // 2 + 1)
        c.apply(O3.copy(), int(N * 0.2), int(N * 0.4), int(N * 0.6))
        if isinstance(c, tc.MPSCircuit):
            np.testing.assert_allclose(np.abs(c._mps.check_canonical()), 0, atol=1e-12)
        c.apply(O2.copy(), N // 2 - 2, N // 2 + 2)
        c.apply(O3.copy(), int(N * 0.4), int(N * 0.6), int(N * 0.8))
        if isinstance(c, tc.MPSCircuit):
            np.testing.assert_allclose(np.abs(c._mps.check_canonical()), 0, atol=1e-12)
        c.cz(2, 3)

    c = tc.Circuit(N)
    simulate(c)
    w_c = c.wavefunction()

    mps = tc.MPSCircuit(N, split=split)
    simulate(mps)
    do_test_norm(mps)
    mps.normalize()
    w_mps = mps.wavefunction()

    mps_exact = tc.MPSCircuit(N)
    simulate(mps_exact)
    w_mps_exact = mps_exact.wavefunction()

    return [c, w_c, mps, w_mps, mps_exact, w_mps_exact]


def do_test_norm(mps: tc.MPSCircuit):
    norm1 = mps.get_norm()
    norm2 = tc.backend.norm(mps.wavefunction())
    np.testing.assert_allclose(norm1, norm2, atol=1e-12)


def do_test_canonical(test_circuits: type_test_circuits):
    (
        c,
        w_c,
        mps,
        w_mps,
        mps_exact,
        w_mps_exact,
    ) = test_circuits
    np.testing.assert_allclose(np.abs(mps._mps.check_canonical()), 0, atol=1e-12)
    np.testing.assert_allclose(np.abs(mps_exact._mps.check_canonical()), 0, atol=1e-12)


def do_test_wavefunction(test_circuits: type_test_circuits):
    (
        c,
        w_c,
        mps,
        w_mps,
        mps_exact,
        w_mps_exact,
    ) = test_circuits
    # the wavefuntion is exact if there's no truncation
    np.testing.assert_allclose(tc.backend.numpy(w_mps_exact), tc.backend.numpy(w_c))


def do_test_truncation(
    test_circuits: type_test_circuits, real_fedility_ref, estimated_fedility_ref
):
    (
        c,
        w_c,
        mps,
        w_mps,
        mps_exact,
        w_mps_exact,
    ) = test_circuits
    # compare with a precalculated value
    real_fedility = (
        np.abs(tc.backend.numpy(w_mps).conj().dot(tc.backend.numpy(w_c))) ** 2
    )
    estimated_fedility = tc.backend.numpy(mps._fidelity)
    print(real_fedility)
    print(estimated_fedility)
    np.testing.assert_allclose(real_fedility, real_fedility_ref, atol=1e-8)
    np.testing.assert_allclose(estimated_fedility, estimated_fedility_ref, atol=1e-8)


def do_test_amplitude(test_circuits: type_test_circuits):
    (
        c,
        w_c,
        mps,
        w_mps,
        mps_exact,
        w_mps_exact,
    ) = test_circuits
    # compare with wavefunction
    s = "01" * (N // 2)
    sint = int(s, 2)  # binary to decimal
    err_amplitude = tc.backend.abs(mps.amplitude(s) - w_mps[sint])
    np.testing.assert_allclose(err_amplitude, 0, atol=1e-12)


def do_test_expectation(test_circuits: type_test_circuits):
    (
        c,
        w_c,
        mps,
        w_mps,
        mps_exact,
        w_mps_exact,
    ) = test_circuits

    single_gate = (tc.gates.z(), [11])
    tensor = (np.sin(np.arange(16)) + np.cos(np.arange(16)) * 1j).reshape((2, 2, 2, 2))
    double_gate_nonunitary = (
        tc.gates.Gate(tc.backend.convert_to_tensor(tensor)),
        [2, 6],
    )
    double_gate = (tc.gates.cnot(), [2, 6])
    triple_gate = (tc.gates.toffoli(), [1, 5, 9])
    gates = [single_gate, double_gate_nonunitary, triple_gate]

    exp_mps = mps_exact.expectation(*gates)
    exp_c = c.expectation(*gates, reuse=False)
    np.testing.assert_allclose(exp_mps, exp_c, atol=1e-7)

    # ps
    x = [0, 2]
    y = [1, 3, 5]
    z = [6, 8, 10]
    exp_mps = mps_exact.expectation_ps(x=x, y=y, z=z)
    exp_c = c.expectation_ps(x=x, y=y, z=z)
    np.testing.assert_allclose(exp_mps, exp_c, atol=1e-7)


def external_wavefunction():
    # create a fixed wavefunction and create the corresponding MPS
    w_external = np.abs(np.sin(np.arange(2**N) % np.exp(1))).astype(
        tc.dtypestr
    )  # Just want to find a function that is so strange that the correlation is strong enough
    w_external /= np.linalg.norm(w_external)
    w_external = tc.backend.convert_to_tensor(w_external)
    mps_external = tc.MPSCircuit(N, wavefunction=w_external, split=split)
    mps_external_exact = tc.MPSCircuit(N, wavefunction=w_external)
    return w_external, mps_external, mps_external_exact


def do_test_fromwavefunction(external_wavefunction, relative_err_ref):
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
    print(relative_err)
    np.testing.assert_allclose(relative_err, 0.11, atol=1e-2)


def do_test_proj(test_circuits: type_test_circuits, external_wavefunction):
    (
        c,
        w_c,
        mps,
        w_mps,
        mps_exact,
        w_mps_exact,
    ) = test_circuits
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


def do_test_tensor_input(test_circuits: type_test_circuits):
    (
        c,
        w_c,
        mps,
        w_mps,
        mps_exact,
        w_mps_exact,
    ) = test_circuits
    newmps = tc.MPSCircuit(
        mps._nqubits,
        tensors=mps.get_tensors(),
        center_position=mps.get_center_position(),
    )
    for t1, t2 in zip(newmps.get_tensors(), mps.get_tensors()):
        np.testing.assert_allclose(
            tc.backend.numpy(t1), tc.backend.numpy(t2), atol=1e-12
        )


def do_test_measure(test_circuits: type_test_circuits):
    (
        c,
        w_c,
        mps,
        w_mps,
        mps_exact,
        w_mps_exact,
    ) = test_circuits
    index = [6, 5, 2, 9]
    status = tc.backend.convert_to_tensor([0.1, 0.3, 0.5, 0.7])
    result_c = c.measure(*index, with_prob=True, status=status)
    result_mps = mps.measure(*index, with_prob=True, status=status)
    result_mps_exact = mps_exact.measure(*index, with_prob=True, status=status)
    np.testing.assert_allclose(result_mps[0], result_c[0], atol=1e-8)
    np.testing.assert_allclose(result_mps_exact[0], result_c[0], atol=1e-8)
    np.testing.assert_allclose(result_mps[1], result_c[1], atol=1e-4)
    np.testing.assert_allclose(result_mps_exact[1], result_c[1], atol=1e-8)


def test_MPO_conversion(highp, tfb):
    O3 = tc.backend.convert_to_tensor(
        reproducible_unitary(8).reshape((2, 2, 2, 2, 2, 2))
    )
    I = tc.backend.convert_to_tensor(np.eye(2).astype("complex128"))
    gate = tc.gates.Gate(O3)

    MPO3, _ = tc.MPSCircuit.gate_to_MPO(gate, 2, 3, 4)
    tensor3 = tc.MPSCircuit.MPO_to_gate(MPO3).tensor
    tensor3 = tc.backend.numpy(tensor3)
    tensor3_ref = tc.backend.numpy(O3)
    np.testing.assert_allclose(tensor3, tensor3_ref, atol=1e-12)

    MPO4, _ = tc.MPSCircuit.gate_to_MPO(gate, 1, 3, 4)
    tensor4 = tc.MPSCircuit.MPO_to_gate(MPO4).tensor
    tensor4_ref = tc.backend.einsum("ijkabc,pq->ipjkaqbc", O3, I)
    tensor4 = tc.backend.numpy(tensor4)
    tensor4_ref = tc.backend.numpy(tensor4_ref)
    np.testing.assert_allclose(tensor4, tensor4_ref, atol=1e-12)


@pytest.mark.parametrize(
    "backend, dtype", [(lf("tfb"), lf("highp")), (lf("jaxb"), lf("highp"))]
)
def test_circuits_1(backend, dtype):
    import time

    begin = time.time()
    circuits = get_test_circuits(False)
    print("time", time.time() - begin)
    do_test_canonical(circuits)
    do_test_wavefunction(circuits)
    do_test_truncation(circuits, 0.9987293417932497, 0.999440840610151)
    do_test_amplitude(circuits)
    do_test_expectation(circuits)
    external = external_wavefunction()
    do_test_fromwavefunction(external, 0.1185)
    do_test_proj(circuits, external)
    do_test_tensor_input(circuits)
    do_test_measure(circuits)
    print("time", time.time() - begin)


def test_circuits_2(highp):
    circuits = get_test_circuits(True)
    do_test_truncation(circuits, 0.9401410770899974, 0.9654331011546374)
