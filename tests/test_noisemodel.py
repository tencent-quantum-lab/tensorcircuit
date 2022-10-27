import pytest
from pytest_lazyfixture import lazy_fixture as lf


import tensorcircuit as tc
from tensorcircuit.noisemodel import (
    NoiseConf,
    circuit_with_noise,
    expectation_ps_noisfy,
    sample_expectation_ps_noisfy,
)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_noisemodel_expectvalue(backend):

    error1 = tc.channels.generaldepolarizingchannel(0.1, 1)
    # error3 =  tc.channels.resetchannel()
    error3 = tc.channels.thermalrelaxationchannel(300, 400, 100, "ByChoi", 0)

    noise_conf = NoiseConf()
    noise_conf.add_noise("rx", error1)
    noise_conf.add_noise("h", error3)

    readout_error = []
    readout_error.append([0.9, 0.75])  # readout error of qubit 0
    readout_error.append([0.4, 0.7])  # readout error of qubit 1

    noise_conf1 = NoiseConf()
    noise_conf1.add_noise("readout", readout_error)

    c = tc.Circuit(2)
    c.rx(0, theta=0.4)
    c.h(0)
    c.x(1)
    cnoise = circuit_with_noise(c, noise_conf, [0.1] * 2)
    value = cnoise.expectation_ps(z=[0, 1])
    print("noise_circuit_value", value)

    dmc = tc.DMCircuit(2)
    dmc.rx(0, theta=0.4)
    dmc.h(0)
    dmc.x(1)
    cnoise = circuit_with_noise(dmc, noise_conf)
    value = cnoise.expectation_ps(z=[0, 1])
    print("noise_circuit_value", value)

    value = expectation_ps_noisfy(c, z=[0, 1], nmc=10000)
    print("mc", value)

    value = expectation_ps_noisfy(dmc, z=[0, 1])
    print("dm", value)

    value = expectation_ps_noisfy(c, z=[0, 1], noise_conf=noise_conf, nmc=10000)
    print("mc_quantum", value)

    value = expectation_ps_noisfy(dmc, z=[0, 1], noise_conf=noise_conf)
    print("dm_quantum", value)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_noisemodel_sample(backend):
    c = tc.Circuit(2)
    c.rx(0, theta=0.4)
    c.h(0)
    c.x(1)
    error1 = tc.channels.generaldepolarizingchannel(0.1, 1)
    error3 = tc.channels.thermalrelaxationchannel(300, 400, 100, "ByChoi", 0)
    # error3 =  tc.channels.resetchannel()

    readout_error = []
    readout_error.append([0.9, 0.75])  # readout error of qubit 0
    readout_error.append([0.4, 0.7])  # readout error of qubit 1

    noise_conf = NoiseConf()
    noise_conf.add_noise("rx", error1)
    noise_conf.add_noise("h", error3)
    noise_conf.add_noise("readout", readout_error)

    noise_conf1 = NoiseConf()
    noise_conf1.add_noise("readout", readout_error)

    noise_conf2 = NoiseConf()
    noise_conf2.add_noise("rx", error1)
    noise_conf2.add_noise("h", error3)

    value = sample_expectation_ps_noisfy(
        c, z=[0, 1], noise_conf=noise_conf1, nmc=100000
    )
    print("noise_nmc_read", value)

    value = sample_expectation_ps_noisfy(c, z=[0, 1], noise_conf=noise_conf, nmc=10000)
    print("noise_nmc_read_quantum", value)

    value = sample_expectation_ps_noisfy(c, z=[0, 1], noise_conf=noise_conf2, nmc=10000)
    print("noise_nmc_quantum", value)

    dmc = tc.DMCircuit(2)
    dmc.rx(0, theta=0.4)
    dmc.h(0)
    dmc.x(1)

    value = sample_expectation_ps_noisfy(dmc, z=[0, 1], noise_conf=noise_conf1)
    print("noise_dm_read", value)

    value = sample_expectation_ps_noisfy(dmc, z=[0, 1], noise_conf=noise_conf)
    print("noise_dm_read_quantum", value)

    value = sample_expectation_ps_noisfy(dmc, z=[0, 1], noise_conf=noise_conf2)
    print("noise_nmc_quantum", value)
