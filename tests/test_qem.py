from functools import partial
import pytest
from pytest_lazyfixture import lazy_fixture as lf
import numpy as np
import networkx as nx

import tensorcircuit as tc
from tensorcircuit.noisemodel import NoiseConf
from tensorcircuit.results import qem
from tensorcircuit.results.qem import (
    zne_option,
    apply_zne,
    dd_option,
    apply_dd,
    apply_rc,
)
from tensorcircuit.results.qem import benchmark_circuits


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_benchmark_circuits(backend):
    # QAOA
    graph = [(2, 0), (0, 3), (1, 2)]
    weight = [1] * len(graph)
    params = np.array([[1, 1]])

    _ = benchmark_circuits.QAOA_circuit(graph, weight, params)

    # mirror circuit
    # return circuit and ideal counts {"01000":1}
    _, _ = benchmark_circuits.mirror_circuit(
        depth=5, two_qubit_gate_prob=1, connectivity_graph=nx.complete_graph(3), seed=20
    )

    # GHZ circuit
    _ = benchmark_circuits.generate_ghz_circuit(10)

    # Werner-state with linear complexity
    #  {'1000': 0.25, '0100': 0.25, '0010': 0.25, '0001': 0.25}
    _ = benchmark_circuits.generate_w_circuit(5)

    # RB cirucit
    _ = benchmark_circuits.generate_rb_circuits(2, 7)[0]


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_zne(backend):
    c = tc.Circuit(2)
    for _ in range(3):
        c.rx(range(2), theta=0.4)

    error1 = tc.channels.generaldepolarizingchannel(0.01, 1)
    noise_conf = NoiseConf()
    noise_conf.add_noise("rx", error1)

    def execute(circuit):
        value = circuit.expectation_ps(z=[0], noise_conf=noise_conf, nmc=10000)
        return value

    random_state = np.random.RandomState(0)
    noise_scaling_function = partial(
        zne_option.scaling.fold_gates_at_random,
        # fidelities = {"single": 1.0},
        random_state=random_state,
    )
    factory = zne_option.inference.PolyFactory(scale_factors=[1, 3, 5], order=1)
    # factory = zne_option.inference.ExpFactory(scale_factors=[1,1.5,2],asymptote=0.)
    # factory = zne_option.inference.RichardsonFactory(scale_factors=[1,1.5,2])
    # factory = zne_option.inference.AdaExpFactory(steps=5, asymptote=0.)

    result = apply_zne(
        circuit=c,
        executor=execute,
        factory=factory,
        scale_noise=noise_scaling_function,
        num_to_average=1,
    )

    ideal_value = c.expectation_ps(z=[0])
    mit_value = result

    np.testing.assert_allclose(ideal_value, mit_value, atol=4e-2)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_dd(backend):
    c = tc.Circuit(2)
    for _ in range(3):
        c.rx(range(2), theta=0.4)

    def execute(circuit):
        value = circuit.expectation_ps(z=[0])
        return value

    def execute2(circuit):
        key = tc.backend.get_random_state(42)
        count = circuit.sample(
            batch=1000, allow_state=True, format_="count_dict_bin", random_generator=key
        )
        return count

    _ = apply_dd(
        circuit=c,
        executor=execute,
        rule=["X", "X"],
        rule_args={"spacing": -1},
        full_output=True,
        ignore_idle_qubit=True,
        fulldd=False,
    )

    _ = apply_dd(
        circuit=c,
        executor=execute2,
        rule=dd_option.rules.xyxy,
        rule_args={"spacing": -1},
        full_output=True,
        ignore_idle_qubit=True,
        fulldd=True,
        iscount=True,
    )

    # wash circuit based on use_qubits and washout iden gates
    _ = qem.prune_ddcircuit(c, qlist=list(range(c.circuit_param["nqubits"])))


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_rc(backend):
    c = tc.Circuit(2)
    for _ in range(3):
        c.rx(range(2), theta=0.4)
        c.cnot(0, 1)

    def execute(circuit):
        value = circuit.expectation_ps(z=[0])
        return value

    def execute2(circuit):
        key = tc.backend.get_random_state(42)
        count = circuit.sample(
            batch=1000, allow_state=True, format_="count_dict_bin", random_generator=key
        )
        return count

    _ = apply_rc(circuit=c, executor=execute, num_to_average=6, simplify=False)

    _ = apply_rc(
        circuit=c, executor=execute2, num_to_average=6, simplify=True, iscount=True
    )

    # generate a circuit with rc
    _ = qem.rc_circuit(c)
