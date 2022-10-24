"""
1. Add readout error and mitigate readout error with two methods.
2. Add thermalrelaxition error and calibrate the thermalrelaxition error.
"""

import numpy as np
from scipy.optimize import minimize, curve_fit
import tensorcircuit as tc

# Add readout error and mitigate readout error with two methods.
def miti_readout_circ(nqubit):
    miticirc = []
    for i in range(2**nqubit):
        name = "{:0" + str(nqubit) + "b}"
        lisbs = [int(x) for x in name.format(i)]
        c = tc.Circuit(nqubit)
        for k in range(nqubit):
            if lisbs[k] == 1:
                c.X(k)
        miticirc.append(c)
    return miticirc


def probability_bs(bs):
    nqubit = len(list(bs.keys())[0])
    probability = [0] * 2**nqubit
    shots = sum([bs[s] for s in bs])
    for s in bs:
        probability[int(s, 2)] = bs[s] / shots
    return probability


def mitigate_probability(probability_noise, calmatrix, method="inverse"):
    if method == "inverse":
        X = np.linalg.inv(calmatrix)
        Y = probability_noise
        probability_cali = X @ Y
    else:  # method="square"

        def fun(x):
            return sum((probability_noise - calmatrix @ x) ** 2)

        x0 = np.random.rand(len(probability_noise))
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        bnds = tuple((0, 1) for x in x0)
        res = minimize(fun, x0, method="SLSQP", constraints=cons, bounds=bnds, tol=1e-6)
        probability_cali = res.x
    return probability_cali


def mitigate_readout(nqubit, circ, readout_error):

    K = tc.set_backend("tensorflow")

    key = K.get_random_state(42)
    keys = []
    for _ in range(2**nqubit):
        key, subkey = tc.backend.random_split(key)
        keys.append(subkey)

    # calibration matrix
    miticirc = miti_readout_circ(nqubit)
    shots = 100000
    calmatrix = np.zeros((2**nqubit, 2**nqubit))
    for i in range(2**nqubit):
        c = miticirc[i]
        bs = c.sample(
            batch=shots,
            allow_state=True,
            readout_error=readout_error,
            format_="count_dict_bin",
            random_generator=keys[i],
        )
        for s in bs:
            calmatrix[int(s, 2)][i] = bs[s] / shots

    key, subkey = tc.backend.random_split(key)
    bs = circ.sample(
        batch=shots, allow_state=True, format_="count_dict_bin", random_generator=subkey
    )
    probability_perfect = probability_bs(bs)
    print("probability_without_readouterror", probability_perfect)

    key, subkey = tc.backend.random_split(key)
    bs = circ.sample(
        batch=shots,
        allow_state=True,
        readout_error=readout_error,
        format_="count_dict_bin",
        random_generator=subkey,
    )
    probability_noise = probability_bs(bs)
    print("probability_with_readouterror", probability_noise)

    probability_miti = mitigate_probability(
        probability_noise, calmatrix, method="inverse"
    )
    print("mitigate_readouterror_method1", probability_miti)

    probability_miti = mitigate_probability(
        probability_noise, calmatrix, method="square"
    )
    print("mitigate_readouterror_method2", probability_miti)


def example_readout_mitigate():
    nqubit = 3
    c = tc.Circuit(nqubit)
    c.H(0)
    c.cnot(0, 1)
    c.X(2)

    readout_error = []
    readout_error.append([0.9, 0.75])  # readout error of qubit 0
    readout_error.append([0.4, 0.7])  # readout error of qubit 1
    readout_error.append([0.7, 0.9])  # readout error of qubit 2

    mitigate_readout(nqubit, c, readout_error)


# Thermalrelaxition calibration
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


def example_T1_cali():
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

    print("realistic_T1", t1)
    print("calibrated_T1", T)


def example_T2_cali():
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

    print("realistic_T2", t2)
    print("calibrated_T2", T)


if __name__ == "__main__":
    example_readout_mitigate()
    example_T1_cali()
    example_T2_cali()
