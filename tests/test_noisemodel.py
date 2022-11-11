import pytest
from pytest_lazyfixture import lazy_fixture as lf
import numpy as np

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
    noise_conf1.add_noise("rx", error1)
    noise_conf1.add_noise("h", error3)
    noise_conf1.add_noise("readout", readout_error)

    c = tc.Circuit(2)
    c.rx(0, theta=0.4)
    c.h(0)
    c.x(1)
    cnoise = circuit_with_noise(c, noise_conf, [0.1] * 2)
    value = cnoise.expectation_ps(z=[0, 1])
    # print("noise_circuit_value", value)
    # np.testing.assert_allclose(value, -0.18, atol=1e-2)

    dmc = tc.DMCircuit(2)
    dmc.rx(0, theta=0.4)
    dmc.h(0)
    dmc.x(1)
    cnoise = circuit_with_noise(dmc, noise_conf)
    value = cnoise.expectation_ps(z=[0, 1])
    # print("noise_circuit_value", value)
    np.testing.assert_allclose(value, -0.28, atol=1e-2)

    value = expectation_ps_noisfy(c, z=[0, 1], nmc=10000)
    # print("mc", value)
    np.testing.assert_allclose(value, 0, atol=1e-2)

    value = expectation_ps_noisfy(dmc, z=[0, 1])
    # print("dm", value)
    np.testing.assert_allclose(value, 0, atol=1e-2)

    value = expectation_ps_noisfy(c, z=[0, 1], noise_conf=noise_conf, nmc=10000)
    # print("mc_quantum", value)
    np.testing.assert_allclose(value, -0.28, atol=1e-2)

    value = expectation_ps_noisfy(dmc, z=[0, 1], noise_conf=noise_conf)
    # print("dm_quantum", value)
    np.testing.assert_allclose(value, -0.28, atol=1e-2)

    value = expectation_ps_noisfy(c, z=[0, 1], noise_conf=noise_conf1, nmc=10000)
    # print("mc_readout", value)
    np.testing.assert_allclose(value, -0.28, atol=1e-2)


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
    # print("noise_nmc_read", value)
    np.testing.assert_allclose(value, -0.06, atol=1e-2)

    value = sample_expectation_ps_noisfy(c, z=[0, 1], noise_conf=noise_conf, nmc=10000)
    # print("noise_nmc_read_quantum", value)
    np.testing.assert_allclose(value, -0.133, atol=1e-2)

    value = sample_expectation_ps_noisfy(c, z=[0, 1], noise_conf=noise_conf2, nmc=10000)
    # print("noise_nmc_quantum", value)
    np.testing.assert_allclose(value, -0.28, atol=1e-2)

    dmc = tc.DMCircuit(2)
    dmc.rx(0, theta=0.4)
    dmc.h(0)
    dmc.x(1)

    value = sample_expectation_ps_noisfy(dmc, z=[0, 1], noise_conf=noise_conf1)
    # print("noise_dm_read", value)
    np.testing.assert_allclose(value, -0.06, atol=1e-2)

    value = sample_expectation_ps_noisfy(dmc, z=[0, 1], noise_conf=noise_conf)
    # print("noise_dm_read_quantum", value)
    np.testing.assert_allclose(value, -0.133, atol=1e-2)

    value = sample_expectation_ps_noisfy(dmc, z=[0, 1], noise_conf=noise_conf2)
    # print("noise_nmc_quantum", value)
    np.testing.assert_allclose(value, -0.28, atol=1e-2)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_noisemodel_qubit(backend):

    # noise_conf = NoiseConf()
    # noise_conf.add_noise("h1","t0")
    # noise_conf.add_noise("h1",["t1","t2"],[[0],[1]])
    # noise_conf.add_noise("h1",["t3"],[[0]])
    # noise_conf.add_noise("h2",["v1","v2"],[[0],[1]])
    # noise_conf.add_noise("h2",["v3"],[[0]])



    c = tc.Circuit(2)
    c.rx(0, theta=0.4)
    c.rx(1, theta=0.8)
    c.h(0)
    c.h(1)
    c.x(0)
    c.x(1)
    c.cnot(0,1)
    error1 = tc.channels.generaldepolarizingchannel(0.1, 1)
    error2 = tc.channels.generaldepolarizingchannel(0.01, 2)
    error3 = tc.channels.thermalrelaxationchannel(300, 400, 100, "ByChoi", 0)


    readout_error = []
    readout_error.append([0.9, 0.75])  # readout error of qubit 0
    readout_error.append([0.4, 0.7])  # readout error of qubit 1

    noise_conf = NoiseConf()
    noise_conf.add_noise("rx", error1)
    noise_conf.add_noise("rx", [error3],[[0]])
    noise_conf.add_noise("h",[error3,error1],[[0],[1]])
    noise_conf.add_noise("x",[error3],[[0]])
    noise_conf.add_noise("cnot", [error2],[[0,1]])
    # #noise_conf.add_noise("readout", readout_error)


    cnoise = circuit_with_noise(c, noise_conf, [0.1] * 7)
    value = cnoise.expectation_ps(z=[0, 1])
    print(value)

    # value = expectation_ps_noisfy(c, z=[0, 1], noise_conf=noise_conf, nmc=10000)
    # print(value)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_dep(backend):
    c = tc.Circuit(2)
    c.cnot(0,1)


    error1 = tc.channels.generaldepolarizingchannel(0.1, 1)
    error2 = tc.channels.generaldepolarizingchannel(0.01, 2)


    noise_conf = NoiseConf()
    #noise_conf.add_noise("rx",[error1,error1],[[0],[1]])
    noise_conf.add_noise("cnot", [error2],[[0,1]])


    cnoise = circuit_with_noise(c, noise_conf, [0.1] * 7)
    value = cnoise.expectation_ps(z=[0, 1])
    print(value)

    value = expectation_ps_noisfy(c, z=[0, 1], noise_conf=noise_conf, nmc=10000)
    print(value)






    # value = sample_expectation_ps_noisfy(dmc, z=[0, 1], noise_conf=noise_conf)
    # print(value)


    kraus = tc.channels.generaldepolarizingchannel(0.1, 1)
    tc.channels.kraus_identity_check(kraus)

    c.general_kraus(kraus, 0, status=0.1)
    print("1",c.expectation_ps(z=[0,1]))


    c.unitary_kraus(kraus, 0,status=0.8)
    print("1",c.expectation_ps(z=[0,1]))

    dmc = tc.DMCircuit(2)
    dmc.cnot(0,1)

    dmc.general_kraus(kraus, 0)
    print("1",dmc.expectation_ps(z=[0,1]))

    dmc = tc.DMCircuit(2)
    dmc.cnot(0,1)

    dmc.generaldepolarizing(0,p=0.1, num_qubits=1)
    print("1",dmc.expectation_ps(z=[0,1]))



    kraus = tc.channels.generaldepolarizingchannel(0.01, 2)
 

    tc.channels.kraus_identity_check(kraus)

    c.general_kraus(kraus, 0,1, status=0.3)
    print("2",c.expectation_ps(z=[0,1]))


    c.unitary_kraus(kraus, 0,1,status=0.7)
    print("2",c.expectation_ps(z=[0,1]))


    dmc.general_kraus(kraus, 0,1)
    print("2",dmc.expectation_ps(z=[0,1]))

    dmc = tc.DMCircuit(2)
    dmc.cnot(0,1)

    dmc.generaldepolarizing(0,1,p=0.01, num_qubits=2)
    print("2",dmc.expectation_ps(z=[0,1]))





   











