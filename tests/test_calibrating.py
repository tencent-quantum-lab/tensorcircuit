import sys
import os
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf
from scipy.optimize import curve_fit


thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


def exp_fit(x, y):
    def fit_function(x_values, y_values, function, init_params):
        fitparams, _ = curve_fit(function, x_values, y_values, init_params)
        return fitparams

    fit_params = fit_function(
        x, y, lambda x, A, C, T: (A * np.exp(-x / T) + C), [-3, 0, 100]
    )

    _, _, T = fit_params

    return T


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_calib_t1(backend):

    t1 = 300
    t2 = 100
    time = 100
    nstep = int(4 * t1 / time)

    # calibrating experiments
    pex = []
    for i in range(nstep):

        dmc = tc.DMCircuit(1)
        dmc.x(0)
        for _ in range(i):
            dmc.i(0)
            dmc.thermalrelaxation(
                0, t1=t1, t2=t2, time=time, method="AUTO", excitedstatepopulation=0
            )

        val = dmc.expectation_ps(z=[0])
        p = (1 - val) / 2.0
        pex.append(p)

    # data fitting
    x1 = np.array([i * time for i in range(nstep)])
    y1 = np.array(np.real(pex))

    T1 = exp_fit(x1, y1)
    np.testing.assert_allclose(t1, T1, atol=1e-1)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_calib_t2(backend):

    pex = []
    t1 = 300
    t2 = 280
    time = 50
    nstep = int(4 * t2 / time)

    for i in range(nstep):

        dmc = tc.DMCircuit(1)
        dmc.h(0)
        for _ in range(0, i):
            dmc.i(0)
            dmc.thermalrelaxation(
                0, t1=t1, t2=t2, time=time, method="AUTO", excitedstatepopulation=0
            )
        # dmc.rz(0,theta = i*np.pi/1.5)
        dmc.h(0)

        val = dmc.expectation_ps(z=[0])
        p = (1 - val) / 2.0
        pex.append(p)

    # data fitting
    x1 = np.array([i * time for i in range(nstep)])
    y1 = np.array(np.real(pex))

    T2 = exp_fit(x1, y1)
    np.testing.assert_allclose(t2, T2, atol=1e-1)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_calib_dep(backend):

    pex = []
    pz = 0.02
    nstep = 40
    for i in range(nstep):

        dmc = tc.DMCircuit(1)
        dmc.x(0)
        for _ in range(i):
            dmc.s(0)
            dmc.generaldepolarizing(0, p=pz, num_qubits=1)

        val = dmc.expectation_ps(z=[0])
        p = (1 - val) / 2.0
        if i % 2 == 0:
            pex.append(p)

    # data fitting
    x1 = np.array([i for i in range(0, nstep, 2)])
    y1 = np.array(np.real(pex))

    def fit_function(x_values, y_values, function, init_params):
        fitparams, _ = curve_fit(function, x_values, y_values, init_params)
        return fitparams

    fit_params = fit_function(x1, y1, lambda x, A, B, C: (A * B**x + C), [-0, 0, 0])

    _, B, _ = fit_params
    pz1 = (1 - B) / 4.0

    np.testing.assert_allclose(pz, pz1, atol=1e-1)
