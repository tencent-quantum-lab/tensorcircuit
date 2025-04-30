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


def fit_function(x_values, y_values, function, init_params):
    fitparams, _ = curve_fit(function, x_values, y_values, init_params)
    return fitparams


def T1_cali(t1, t2, time, method, excitedstatepopulation):
    # calibrating experiments
    nstep = int(4 * t1 / time)
    pex = []
    for i in range(nstep):
        dmc = tc.DMCircuit(1)
        dmc.x(0)
        for _ in range(i):
            dmc.i(0)
            dmc.thermalrelaxation(
                0,
                t1=t1,
                t2=t2,
                time=time,
                method=method,
                excitedstatepopulation=excitedstatepopulation,
            )

        val = dmc.expectation_ps(z=[0])
        p = (1 - val) / 2.0
        pex.append(p)

    timelist = np.array([i * time for i in range(nstep)])
    measurement = np.array(np.real(pex))

    return measurement, timelist


def T2_cali(t1, t2, time, method, excitedstatepopulation):
    # calibrating experiments
    nstep = int(4 * t2 / time)
    pex = []
    for i in range(nstep):
        dmc = tc.DMCircuit(1)
        dmc.h(0)
        for _ in range(0, i):
            dmc.i(0)
            dmc.thermalrelaxation(
                0,
                t1=t1,
                t2=t2,
                time=time,
                method=method,
                excitedstatepopulation=excitedstatepopulation,
            )
        # dmc.rz(0,theta = i*np.pi/1.5)
        dmc.h(0)

        val = dmc.expectation_ps(z=[0])
        p = (1 - val) / 2.0
        pex.append(p)

    timelist = np.array([i * time for i in range(nstep)])
    measurement = np.array(np.real(pex))

    return measurement, timelist


def dep_cali(dep, nqubit):
    pex = []
    nstep = 40
    for i in range(nstep):
        dmc = tc.DMCircuit(1)
        dmc.x(0)
        for _ in range(i):
            dmc.s(0)
            dmc.generaldepolarizing(0, p=dep, num_qubits=nqubit)

        val = dmc.expectation_ps(z=[0])
        p = (1 - val) / 2.0
        if i % 2 == 0:
            pex.append(p)

    timelist = np.array([i for i in range(0, nstep, 2)])
    measurement = np.array(np.real(pex))

    return measurement, timelist


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_cali_t1(backend):
    t1 = 300
    t2 = 100
    time = 100
    method = "AUTO"
    excitedstatepopulation = 0
    measurement, timelist = T1_cali(t1, t2, time, method, excitedstatepopulation)

    fit_params = fit_function(
        timelist, measurement, lambda x, A, C, T: (A * np.exp(-x / T) + C), [-3, 0, 100]
    )

    _, _, T = fit_params

    np.testing.assert_allclose(t1, T, atol=1e-1)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_cali_t2(backend):
    t1 = 300
    t2 = 280
    time = 50
    method = "AUTO"
    excitedstatepopulation = 0
    measurement, timelist = T2_cali(t1, t2, time, method, excitedstatepopulation)

    fit_params = fit_function(
        timelist, measurement, lambda x, A, C, T: (A * np.exp(-x / T) + C), [-3, 0, 100]
    )

    _, _, T = fit_params

    np.testing.assert_allclose(t2, T, atol=1e-1)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_cali_dep(backend):
    dep = 0.02
    nqubit = 1
    measurement, timelist = dep_cali(dep, nqubit)

    fit_params = fit_function(
        timelist, measurement, lambda x, A, B, C: (A * B**x + C), [-0, 0, 0]
    )

    _, B, _ = fit_params
    dep1 = (1 - B) / 4.0**nqubit

    np.testing.assert_allclose(dep, dep1, atol=1e-1)
