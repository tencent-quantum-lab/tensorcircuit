import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf
import tensorflow as tf

try:
    import openfermion as _
except ModuleNotFoundError:
    pytestmark = pytest.mark.skip("skip fgs test due to missing openfermion module")

import tensorcircuit as tc

F = tc.fgs.FGSSimulator
FT = tc.fgs.FGSTestSimulator


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_cmatrix(backend, highp):
    c = tc.fgs.FGSSimulator(4, [0])
    c1 = tc.fgs.FGSTestSimulator(4, [0])
    c.evol_hp(0, 1, 1.2)
    c1.evol_hp(0, 1, 1.2)
    c.evol_icp(2, 0.5)
    c1.evol_icp(2, 0.5)
    c.evol_cp(0, -0.8)
    c1.evol_cp(0, -0.8)
    c.evol_sp(1, 0, 0.5)
    c1.evol_sp(1, 0, 0.5)
    c.evol_sp(1, 2, 1.5)
    c1.evol_sp(1, 2, 1.5)
    c.evol_ihamiltonian(c.chemical_potential(1.0, 1, 4))
    c1.evol_ihamiltonian(c1.chemical_potential_jw(1.0, 1, 4))
    np.testing.assert_allclose(
        c1.get_cmatrix_majorana(), c.get_cmatrix_majorana(), atol=1e-5
    )
    np.testing.assert_allclose(c1.get_cmatrix(), c.get_cmatrix(), atol=1e-5)
    np.testing.assert_allclose(c1.entropy([0, 2]), c.entropy([0, 2]), atol=1e-5)
    np.testing.assert_allclose(
        c1.renyi_entropy(2, [1]), c.renyi_entropy(2, [1]), atol=1e-5
    )


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_fgs_ad(backend, highp):
    import optax

    N = 18

    def f(chi):
        c = tc.fgs.FGSSimulator(N, [0])
        for i in range(N - 1):
            c.evol_hp(i, i + 1, chi[i])
        cm = c.get_cmatrix()
        return -tc.backend.real(1 - cm[N, N])

    chi_vg = tc.backend.jit(tc.backend.value_and_grad(f))

    if tc.backend.name == "tensorflow":
        opt = tc.backend.optimizer(tf.keras.optimizers.Adam(1e-2))
    else:
        opt = tc.backend.optimizer(optax.adam(1e-2))

    param = tc.backend.ones([N - 1], dtype="float64")
    for _ in range(300):
        vs, gs = chi_vg(param)
        param = opt.update(gs, param)

    np.testing.assert_allclose(vs, -0.9986, atol=1e-2)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_hamiltonian_generation(backend):
    hc = F.chemical_potential(0.8, 0, 3)
    h1 = FT.get_hmatrix(hc, 3) + 0.4 * tc.backend.eye(2**3)
    # constant shift
    h2 = FT.chemical_potential_jw(0.8, 0, 3)
    np.testing.assert_allclose(h1, h2, atol=1e-5)

    hc = F.hopping(0.8j, 0, 1, 3)
    h1 = FT.get_hmatrix(hc, 3)
    h2 = FT.hopping_jw(0.8j, 0, 1, 3)
    np.testing.assert_allclose(h1, h2, atol=1e-5)

    hc = F.sc_pairing(0.8, 1, 0, 3)
    h1 = FT.get_hmatrix(hc, 3)
    h2 = FT.sc_pairing_jw(0.8, 1, 0, 3)
    np.testing.assert_allclose(h1, h2, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_ground_state(backend, highp):
    N = 3
    hc = (
        F.hopping(1.0, 0, 1, N)
        + F.hopping(1.0, 1, 2, N)
        + F.chemical_potential(0.4, 0, N)
        - F.chemical_potential(0.4, 1, N)
        + F.sc_pairing(0.8, 0, 1, N)
    )

    c = F(N, hc=hc)
    c1 = FT(N, hc=hc)

    np.testing.assert_allclose(c.get_cmatrix(), c1.get_cmatrix(), atol=1e-6)
    np.testing.assert_allclose(c1.entropy([0]), c.entropy([0]), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_post_select(backend, highp):
    for ind in [0, 1, 2]:
        for keep in [0, 1]:
            c = F(3, filled=[0, 2])
            c.evol_hp(0, 1, 0.2)
            c.evol_cp(0, 0.5)
            c.evol_hp(1, 2, 0.3j)
            c.evol_cp(1, -0.8)
            c.evol_sp(0, 2, 0.3)
            c.post_select(ind, keep)
            c1 = FT(3, filled=[0, 2])
            c1.evol_hp(0, 1, 0.2)
            c1.evol_cp(0, 0.5)
            c1.evol_hp(1, 2, 0.3j)
            c1.evol_cp(1, -0.8)
            c1.evol_sp(0, 2, 0.3)
            c1.post_select(ind, keep)
            np.testing.assert_allclose(c.get_cmatrix(), c1.get_cmatrix(), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_jittable_measure(backend):
    @tc.backend.jit
    def get_cmatrix(status):
        r = []
        c = F(3, filled=[0])
        c.evol_hp(0, 1, 0.2)
        c.evol_cp(0, 0.5)
        c.evol_hp(1, 2, 0.3j)
        r.append(c.cond_measure(1, status[0]))
        c.evol_cp(1, -0.8)
        r.append(c.cond_measure(2, status[1]))
        r.append(c.cond_measure(1, status[3]))
        c.evol_sp(0, 2, 0.3)
        c.evol_hp(2, 1, 0.2)
        c.evol_hp(0, 1, 0.8)
        return c.get_cmatrix(), tc.backend.stack(r)

    def get_cmatrix_baseline(status):
        r = []
        c = FT(3, filled=[0])
        c.evol_hp(0, 1, 0.2)
        c.evol_cp(0, 0.5)
        c.evol_hp(1, 2, 0.3j)
        r.append(c.cond_measure(1, status[0]))
        c.evol_cp(1, -0.8)
        r.append(c.cond_measure(2, status[1]))
        r.append(c.cond_measure(1, status[3]))
        c.evol_sp(0, 2, 0.3)
        c.evol_hp(2, 1, 0.2)
        c.evol_hp(0, 1, 0.8)
        return c.get_cmatrix(), np.array(r)

    status = np.array([0.2, 0.83, 0.61, 0.07])
    m, his = get_cmatrix(tc.backend.convert_to_tensor(status))
    m1, his1 = get_cmatrix_baseline(status)
    np.testing.assert_allclose(m, m1, atol=1e-5)
    np.testing.assert_allclose(his, his1, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_exp_2body(backend):
    c = F(3, filled=[1])
    np.testing.assert_allclose(c.expectation_2body(4, 1), 1.0, atol=1e-5)
    np.testing.assert_allclose(c.expectation_2body(5, 2), 0.0, atol=1e-5)
    np.testing.assert_allclose(c.expectation_2body(1, 4), 0.0, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_exp_4body(backend):
    c = F(4, filled=[0, 2])
    c.evol_hp(0, 1, 0.3)
    c.evol_hp(2, 3, 0.3)
    c.evol_sp(0, 3, 0.5)
    c.evol_sp(0, 2, 0.9)
    c.evol_cp(0, -0.4)

    c1 = FT(4, filled=[0, 2])
    c1.evol_hp(0, 1, 0.3)
    c1.evol_hp(2, 3, 0.3)
    c1.evol_sp(0, 3, 0.5)
    c1.evol_sp(0, 2, 0.9)
    c1.evol_cp(0, -0.4)

    np.testing.assert_allclose(
        c.expectation_4body(0, 4, 1, 5), c.expectation_4body(0, 4, 1, 5), atol=1e-5
    )
    np.testing.assert_allclose(
        c.expectation_4body(0, 1, 4, 5), c.expectation_4body(0, 1, 4, 5), atol=1e-5
    )
    np.testing.assert_allclose(
        c.expectation_4body(0, 4, 2, 6), c.expectation_4body(0, 4, 2, 6), atol=1e-5
    )
    np.testing.assert_allclose(
        c.expectation_4body(0, 2, 3, 1), c.expectation_4body(0, 2, 3, 1), atol=1e-5
    )
    np.testing.assert_allclose(
        c.expectation_4body(0, 1, 4, 6), c.expectation_4body(0, 1, 4, 6), atol=1e-5
    )
    np.testing.assert_allclose(
        c.expectation_4body(1, 0, 6, 7), c.expectation_4body(1, 0, 6, 7), atol=1e-5
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_overlap(backend):
    def compute_overlap(FGScls):
        c = FGScls(3, filled=[0, 2])
        c.evol_hp(0, 1, 1.2)
        c.evol_hp(1, 2, 0.3)
        c.evol_cp(0, 0.5)
        c.evol_sp(0, 2, 0.3)
        c1 = FGScls(3, filled=[0, 2])
        c1.evol_hp(0, 1, 0.2)
        c1.evol_hp(1, 2, 0.6)
        c1.evol_sp(1, 0, -1.1)
        return tc.backend.abs(c.overlap(c1))

    np.testing.assert_allclose(
        compute_overlap(tc.FGSSimulator), compute_overlap(tc.fgs.FGSTestSimulator), 1e-5
    )
