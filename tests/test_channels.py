import sys
import os
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf


thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc
from tensorcircuit.channels import (
    depolarizingchannel,
    amplitudedampingchannel,
    phasedampingchannel,
    resetchannel,
)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_channel_identity(backend):
    cs = depolarizingchannel(0.1, 0.15, 0.2)
    tc.channels.single_qubit_kraus_identity_check(cs)
    cs = amplitudedampingchannel(0.25, 0.3)
    tc.channels.single_qubit_kraus_identity_check(cs)
    cs = phasedampingchannel(0.6)
    tc.channels.single_qubit_kraus_identity_check(cs)
    cs = resetchannel()
    tc.channels.single_qubit_kraus_identity_check(cs)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_dep(backend):

    cs = tc.channels.generaldepolarizingchannel(0.1, 1)
    tc.channels.kraus_identity_check(cs)

    cs = tc.channels.generaldepolarizingchannel([0.1, 0.1, 0.1], 1)
    tc.channels.kraus_identity_check(cs)

    cs = tc.channels.generaldepolarizingchannel(0.02, 2)
    tc.channels.kraus_identity_check(cs)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_rep_transformation(backend):

    kraus_set = []
    kraus_set.append(tc.channels.phasedampingchannel(0.2))
    kraus_set.append(tc.channels.resetchannel())
    kraus_set.append(tc.channels.generaldepolarizingchannel(0.1, 1))
    kraus_set.append(tc.channels.phasedampingchannel(0.5))

    density_set = []
    dx = np.array([[0.5, 0.5], [0.5, 0.5]])
    dy = np.array([[0.5, 0.5 * 1j], [-0.5 * 1j, 0.5]])
    density_set.append(dx)
    density_set.append(dy)
    density_set.append(0.1 * dx + 0.9 * dy)

    for density_matrix in density_set:
        for kraus in kraus_set:
            tc.channels.check_rep_transformation(kraus, density_matrix, verbose=False)

    kraus = tc.channels.generaldepolarizingchannel(0.01, 2)
    density_matrix = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    tc.channels.check_rep_transformation(kraus, density_matrix, verbose=False)

    # test
    choi = np.zeros([4, 4])
    kraus = tc.channels.choi_to_kraus(choi)
    np.testing.assert_allclose(kraus, [np.zeros([2, 2])], atol=1e-5)
