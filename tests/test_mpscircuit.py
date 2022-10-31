# pylint: disable=unused-variable
# pylint: disable=invalid-name

from typing import Tuple, Any
import sys
import os
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf

Tensor = Any

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


N = 8
D = 6
split = tc.cons.split_rules(max_singular_values=D)
type_test_circuits = Tuple[
    tc.Circuit, Tensor, tc.MPSCircuit, Tensor, tc.MPSCircuit, Tensor
]


def reproducible_unitary(n, param):
    exp2n = 2**n
    A = tc.backend.cast(
        tc.backend.reshape(tc.backend.arange(exp2n**2), (exp2n, exp2n)), tc.dtypestr
    )
    A = A + tc.backend.sin(A) * param * 1j
    A = A - tc.backend.conj(tc.backend.transpose(A))
    return tc.backend.reshape(tc.backend.expm(A), (2,) * n * 2)


def simulate(c, check=True, params=None):
    if params is None:
        params = tc.backend.ones((3,), dtype=tc.dtypestr)
    O1 = tc.gates.any(reproducible_unitary(1, params[0]))
    O2 = tc.gates.any(reproducible_unitary(2, params[1]))
    O3 = tc.gates.any(reproducible_unitary(3, params[2]))

    # Construct a complicated circuit by Circuit and MPSCircuit and compare

    c.H(0)
    # create as much correlation as possible
    for i in range(0, N - 1, 2):
        c.apply(O2.copy(), i, i + 1)
        c.apply(O1.copy(), i)
    c.apply(O3.copy(), int(N * 0.1), int(N * 0.5), int(N * 0.9))
    c.apply(O2.copy(), 1, N - 2)
    if check and isinstance(c, tc.MPSCircuit):
        np.testing.assert_allclose(np.abs(c._mps.check_canonical()), 0, atol=1e-12)
    c.cz(2, 3)


def get_test_circuits() -> type_test_circuits:
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
    print(real_fedility, estimated_fedility)
    if real_fedility_ref is not None:
        np.testing.assert_allclose(real_fedility, real_fedility_ref, atol=1e-5)
    if estimated_fedility_ref is not None:
        np.testing.assert_allclose(
            estimated_fedility, estimated_fedility_ref, atol=1e-5
        )


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

    single_gate = (tc.gates.z(), [3])
    tensor = (np.sin(np.arange(16)) + np.cos(np.arange(16)) * 1j).reshape((2, 2, 2, 2))
    double_gate_nonunitary = (
        tc.gates.Gate(tc.backend.convert_to_tensor(tensor)),
        [2, 6],
    )
    double_gate = (tc.gates.cnot(), [2, 6])
    triple_gate = (tc.gates.toffoli(), [7, 1, 5])
    gates = [single_gate, double_gate_nonunitary, triple_gate]

    exp_mps = mps_exact.expectation(*gates)
    exp_c = c.expectation(*gates, reuse=False)
    np.testing.assert_allclose(exp_mps, exp_c, atol=1e-7)

    # ps
    x = [0, 2]
    y = [5, 3, 1]
    z = [6, 4]
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
    if relative_err_ref is not None:
        np.testing.assert_allclose(relative_err, relative_err_ref, atol=1e-4)


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
    index = [6, 5, 2, 1]
    status = tc.backend.convert_to_tensor([0.1, 0.3, 0.7, 0.9])
    result_c = c.measure(*index, with_prob=True, status=status)
    result_mps_exact = mps_exact.measure(*index, with_prob=True, status=status)
    np.testing.assert_allclose(result_mps_exact[0], result_c[0], atol=1e-8)
    np.testing.assert_allclose(result_mps_exact[1], result_c[1], atol=1e-8)


def test_MPO_conversion(highp, tfb):
    O3 = reproducible_unitary(3, 1.0)
    I = tc.backend.eye(2, dtype=tc.dtypestr)
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
def test_circuits(backend, dtype):
    circuits = get_test_circuits()
    do_test_canonical(circuits)
    do_test_wavefunction(circuits)
    do_test_truncation(circuits, 0.902663090851, 0.910305380327)
    do_test_amplitude(circuits)
    do_test_expectation(circuits)
    external = external_wavefunction()
    do_test_fromwavefunction(external, 0.276089)
    do_test_proj(circuits, external)
    do_test_tensor_input(circuits)
    do_test_measure(circuits)


@pytest.mark.parametrize("backend, dtype", [(lf("tfb"), lf("highp"))])
def test_circuits_jit(backend, dtype):
    def expec(params):
        mps = tc.MPSCircuit(N, split=split)
        simulate(mps, check=False, params=params)
        x = [0, 2]
        y = [5, 3, 1]
        z = [6, 4]
        exp = mps.expectation_ps(x=x, y=y, z=z)
        return tc.backend.real(exp)

    params = tc.backend.ones((3,), dtype=tc.dtypestr)
    expec_vg = tc.backend.value_and_grad(expec)
    expec_vg_jit = tc.backend.jit(expec_vg)
    exp = expec(params)
    exp_jit, exp_grad_jit = expec_vg_jit(params)
    dir = tc.backend.convert_to_tensor(np.array([1.0, 2.0, 3.0], dtype=tc.dtypestr))
    epsilon = 1e-6
    exp_p = expec(params + dir * epsilon)
    exp_m = expec(params - dir * epsilon)
    exp_grad_dir_numerical = (exp_p - exp_m) / (epsilon * 2)
    exp_grad_dir_jit = tc.backend.real(tc.backend.sum(exp_grad_jit * dir))
    np.testing.assert_allclose(
        tc.backend.numpy(exp), tc.backend.numpy(exp_jit), atol=1e-10
    )
    np.testing.assert_allclose(
        tc.backend.numpy(exp_grad_dir_numerical),
        tc.backend.numpy(exp_grad_dir_jit),
        atol=1e-6,
    )
