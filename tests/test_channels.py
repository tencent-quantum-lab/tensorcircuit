import sys
import os
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf
from scipy.optimize import minimize


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

    cs2 = tc.channels.isotropicdepolarizingchannel(0.02 * 15, 2)
    for c1, c2 in zip(cs, cs2):
        np.testing.assert_allclose(c1.tensor, c2.tensor)


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


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_thermal(backend):
    t2 = 100
    time = 100

    t1 = 180
    kraus = tc.channels.thermalrelaxationchannel(t1, t2, time, "AUTO", 0.1)
    supop1 = tc.channels.kraus_to_super(kraus)

    kraus = tc.channels.thermalrelaxationchannel(t1, t2, time, "ByKraus", 0.1)
    supop2 = tc.channels.kraus_to_super(kraus)

    np.testing.assert_allclose(supop1, supop2, atol=1e-5)

    t1 = 80
    kraus = tc.channels.thermalrelaxationchannel(t1, t2, time, "AUTO", 0.1)
    supop1 = tc.channels.kraus_to_super(kraus)

    kraus = tc.channels.thermalrelaxationchannel(t1, t2, time, "ByChoi", 0.1)
    supop2 = tc.channels.kraus_to_super(kraus)

    np.testing.assert_allclose(supop1, supop2, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_noisecircuit(backend):
    # Monte carlo simulation
    def noisecircuit(X):
        n = 1
        c = tc.Circuit(n)
        c.x(0)
        # noise = tc.channels.thermalrelaxationchannel(300, 400, 1000, "AUTO", 0)
        # c.general_kraus(noise, 0, status=X)
        c.thermalrelaxation(
            0,
            t1=300,
            t2=400,
            time=1000,
            method="ByChoi",
            excitedstatepopulation=0,
            status=X,
        )

        val = c.expectation_ps(z=[0])
        return val

    noisec_vmap = tc.backend.vmap(noisecircuit, vectorized_argnums=0)
    noisec_jit = tc.backend.jit(noisec_vmap)

    nmc = 10000
    X = tc.backend.implicit_randu(nmc)
    valuemc = sum(tc.backend.numpy(noisec_jit(X))) / nmc

    # Density matrix simulation
    def noisecircuitdm():
        n = 1
        dmc = tc.DMCircuit(n)
        dmc.x(0)
        dmc.thermalrelaxation(
            0, t1=300, t2=400, time=1000, method="ByChoi", excitedstatepopulation=0
        )
        val = dmc.expectation_ps(z=[0])
        return val

    noisec_jit = tc.backend.jit(noisecircuitdm)
    valuedm = noisec_jit()

    np.testing.assert_allclose(valuemc, valuedm, atol=1e-1)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_readout(backend):
    nqubit = 3
    c = tc.Circuit(nqubit)
    c.X(0)

    value = c.sample_expectation_ps(z=[0, 1, 2])
    valueaim = -1
    np.testing.assert_allclose(value, valueaim, atol=1e-3)

    readout_error = []
    readout_error.append([0.9, 0.75])  # readout error of qubit 0
    readout_error.append([0.4, 0.7])  # readout error of qubit 1
    readout_error.append([0.7, 0.9])  # readout error of qubit 2

    # readout_error is a list
    value = c.sample_expectation_ps(z=[0, 1, 2], readout_error=readout_error)
    valueaim = 0.04
    np.testing.assert_allclose(value, valueaim, atol=1e-1)

    # readout_error is a tensor
    readout_error = tc.array_to_tensor(readout_error)
    value = c.sample_expectation_ps(z=[0, 1, 2], readout_error=readout_error)
    valueaim = 0.04
    np.testing.assert_allclose(value, valueaim, atol=1e-1)

    # test jitble
    def jitest(readout_error):
        nqubit = 3
        c = tc.Circuit(nqubit)
        c.X(0)
        return c.sample_expectation_ps(z=[0, 1, 2], readout_error=readout_error)

    calvalue = tc.backend.jit(jitest)
    value = calvalue(readout_error)
    valueaim = 0.04
    np.testing.assert_allclose(value, valueaim, atol=1e-1)

    # test contractor time
    # start = timeit.default_timer()
    # def speed(nqubit):
    #     c = tc.Circuit(nqubit)
    #     c.X(0)
    #     readout_error = []
    #     for _ in range(nqubit):
    #         readout_error.append([0.9, 0.75])  # readout error of qubit 0
    #     value = c.sample_expectation_ps(
    #         z=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], readout_error=readout_error
    #     )
    #     return value

    # speed(10)
    # stop = timeit.default_timer()
    # print("Time: ", stop - start)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_noisesample(backend):
    readout_error = []
    readout_error.append([0.9, 0.75])  # readout error of qubit 0
    readout_error.append([0.4, 0.7])  # readout error of qubit 1
    readout_error.append([0.7, 0.9])  # readout error of qubit 2

    c = tc.Circuit(3)
    c.H(0)
    c.cnot(0, 1)
    print(c.sample(allow_state=True, readout_error=readout_error))
    print(c.sample(batch=8, allow_state=True, readout_error=readout_error))
    print(
        c.sample(
            batch=8,
            allow_state=True,
            readout_error=readout_error,
            random_generator=tc.backend.get_random_state(42),
        )
    )

    key = tc.backend.get_random_state(42)
    bs = c.sample(
        batch=1000, allow_state=True, format_="count_dict_bin", random_generator=key
    )
    print(bs)
    bs = c.sample(
        batch=1000,
        allow_state=True,
        readout_error=readout_error,
        format_="count_dict_bin",
        random_generator=key,
    )
    print(bs)

    # test jitble
    def jitest(readout_error):
        c = tc.Circuit(3)
        c.H(0)
        c.cnot(0, 1)
        return c.sample(batch=8, allow_state=True, format_="sample_int")

    calsample = tc.backend.jit(jitest)
    sampletest = calsample(readout_error)
    print(sampletest)


# mitigate readout error
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
    key = tc.backend.get_random_state(42)
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


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_readout_mitigate(backend):
    nqubit = 3
    c = tc.Circuit(nqubit)
    c.H(0)
    c.cnot(0, 1)
    c.X(2)

    readout_error = []
    readout_error.append([0.9, 0.75])  # readout error of qubit 0, p0|0=0.9, p1|1=0.75
    readout_error.append([0.4, 0.7])  # readout error of qubit 1
    readout_error.append([0.7, 0.9])  # readout error of qubit 2

    mitigate_readout(nqubit, c, readout_error)
