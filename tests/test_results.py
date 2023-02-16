import pytest
import numpy as np


import tensorcircuit as tc
from tensorcircuit.results import counts
from tensorcircuit.results.readout_mitigation import ReadoutMit

d = {"000": 2, "101": 3, "100": 4}


def test_marginal_count():
    assert counts.marginal_count(d, [1, 2])["00"] == 6
    assert counts.marginal_count(d, [1])["0"] == 9
    assert counts.marginal_count(d, [2, 1, 0])["001"] == 4


def test_count2vec():
    assert counts.vec2count(counts.count2vec(d, normalization=False), prune=True) == d


def test_kl():
    a = {"00": 512, "11": 512}
    assert counts.kl_divergence(a, a) == 0


def test_expectation():
    assert counts.expectation(d, [0, 1]) == -5 / 9
    assert counts.expectation(d, None, [[1, -1], [1, 0], [1, 1]]) == -5 / 9


def test_plot_histogram():
    d = {"00": 10, "01": 2, "11": 8}
    d1 = {"00": 11, "11": 9}
    print(counts.plot_histogram([d, d1]))


def test_readout():
    nqubit = 4
    shots = 4096
    c = tc.Circuit(nqubit)
    c.H(0)
    c.cnot(0, 1)
    c.x(3)

    idea_count = c.sample(batch=shots, allow_state=True, format="count_dict_bin")
    raw_count = run([c], shots)[0]

    # test "inverse",  "constrained_least_square", "M3"
    mit = ReadoutMit(execute=run)
    mit.cals_from_system([0, 1, 2, 3, 6], shots=10000, method="local")

    mit_count1 = mit.apply_correction(
        raw_count, [1, 3, 2], method="inverse"
    )  #  direct(Max2),iterative(Max3), inverse,square
    mit_count2 = mit.apply_correction(
        raw_count, [1, 3, 2], method="constrained_least_square"
    )
    idea_count2 = counts.marginal_count(idea_count, [1, 3, 2])

    assert counts.kl_divergence(idea_count2, mit_count1) < 0.05
    assert counts.kl_divergence(idea_count2, mit_count2) < 0.05

    # test "global" and "equal"
    mit = ReadoutMit(execute=run)
    mit.cals_from_system([0, 1, 2, 3], shots=100000, method="global")
    A_global = mit.get_matrix([1, 3, 2])
    mit_countg = mit.apply_correction(
        raw_count, [1, 3, 2], method="constrained_least_square"
    )

    mit = ReadoutMit(execute=run)
    mit.cals_from_system([0, 1, 2, 3], shots=100000, method="local")
    A_local = mit.get_matrix([1, 3, 2])
    mit_countl = mit.apply_correction(
        raw_count, [1, 3, 2], method="constrained_least_square"
    )

    np.testing.assert_allclose(A_global, A_local, atol=1e-2)
    assert counts.kl_divergence(mit_countg, mit_countl) < 0.05


def test_readout_masks():
    mit = ReadoutMit(execute=run)
    mit.cals_from_system(
        [1, 2, 4], shots=8192, method="local", masks=["01010", "10101", "11111"]
    )
    np.testing.assert_allclose(
        mit.single_qubit_cals[1][0, 0], 0.02 * np.sin(2) + 0.978, atol=1e-2
    )


def test_readout_expv():
    nqubit = 4
    c = tc.Circuit(nqubit)
    c.H(0)
    c.cnot(0, 1)
    c.x(3)

    idea_count = c.sample(batch=100000, allow_state=True, format="count_dict_bin")
    raw_count = run([c], 100000)[0]

    cal_qubits = [0, 1, 2, 3]
    use_qubits = [0, 1]

    # idea_value = c.expectation_ps(z=[0,1])
    idea_count2 = counts.marginal_count(idea_count, use_qubits)
    idea_value = counts.expectation(idea_count2, z=[0, 1])

    mit = ReadoutMit(execute=run)
    mit.cals_from_system(cal_qubits, shots=100000, method="local")
    mit_count = mit.apply_correction(raw_count, use_qubits, method="inverse")
    mit_value = counts.expectation(mit_count, z=[0, 1])

    mit = ReadoutMit(execute=run)
    mit.cals_from_system(cal_qubits, shots=100000, method="local")
    mit_value1 = mit.expectation(raw_count, z=[0, 1], method="inverse")

    mit = ReadoutMit(execute=run)
    mit.cals_from_system(cal_qubits, shots=100000, method="global")
    mit_value2 = mit.expectation(raw_count, z=[0, 1], method="square")

    np.testing.assert_allclose(idea_value, mit_value, atol=1e-2)
    np.testing.assert_allclose(idea_value, mit_value1, atol=1e-2)
    np.testing.assert_allclose(idea_value, mit_value2, atol=1e-2)

    # test large size
    nqubit = 20
    c = tc.Circuit(nqubit)
    c.H(0)
    for i in range(nqubit - 1):
        c.cnot(i, i + 1)
    c.rx(1, theta=0.9)

    idea_count = c.sample(batch=100000, allow_state=True, format="count_dict_bin")
    raw_count = run([c], 100000)[0]

    cal_qubits = list(range(nqubit))
    use_qubits = list(range(nqubit))

    # idea_value = c.expectation_ps(z=[0,1])
    idea_count2 = counts.marginal_count(idea_count, use_qubits)
    idea_value = counts.expectation(idea_count2, z=list(range(nqubit)))

    mit = ReadoutMit(execute=run)
    mit.cals_from_system(cal_qubits, shots=100000, method="local")
    mit_value1 = mit.expectation(raw_count, z=list(range(nqubit)), method="inverse")

    np.testing.assert_allclose(idea_value, mit_value1, atol=1e-1)


def test_M3():
    try:
        import mthree  # pylint: disable=unused-import
    except ImportError:
        pytest.skip("****** No mthree, skipping test suit *******")

    nqubit = 20
    c = tc.Circuit(nqubit)
    c.H(0)
    for i in range(nqubit - 1):
        c.cnot(i, i + 1)
    c.rx(1, theta=0.9)

    idea_count = c.sample(batch=100000, allow_state=True, format="count_dict_bin")
    raw_count = run([c], 100000)[0]

    cal_qubits = list(range(nqubit))
    use_qubits = list(range(nqubit))

    idea_count2 = counts.marginal_count(idea_count, use_qubits)
    idea_value = counts.expectation(idea_count2, z=list(range(nqubit)))

    mit = ReadoutMit(execute=run)
    mit.cals_from_system(cal_qubits, shots=100000, method="local")
    mit_count = mit.apply_correction(raw_count, use_qubits, method="M3_auto")
    mit_value = counts.expectation(mit_count, z=list(range(nqubit)))
    np.testing.assert_allclose(idea_value, mit_value, atol=1e-1)

    nqubit = 4
    shots = 4096
    c = tc.Circuit(nqubit)
    c.H(0)
    c.cnot(0, 1)
    c.x(3)

    idea_count = c.sample(batch=shots, allow_state=True, format="count_dict_bin")
    raw_count = run([c], shots)[0]

    mit_count3 = mit.apply_correction(raw_count, [1, 3, 2], method="M3_direct")
    mit_count4 = mit.apply_correction(raw_count, [1, 3, 2], method="M3_iterative")
    idea_count2 = counts.marginal_count(idea_count, [1, 3, 2])
    assert counts.kl_divergence(idea_count2, mit_count3) < 0.05
    assert counts.kl_divergence(idea_count2, mit_count4) < 0.05


def partial_sample(c, batch, readout_error=None):
    measure_index = []
    for inst in c._extra_qir:
        if inst["name"] == "measure":
            measure_index.append(inst["index"][0])
    if len(measure_index) == 0:
        measure_index = list(range(c._nqubits))

    ct = c.sample(
        allow_state=True,
        batch=batch,
        readout_error=readout_error,
        format="count_dict_bin",
    )
    return tc.results.counts.marginal_count(ct, measure_index)


def run(cs, shots):
    # customized backend for mitigation test
    ts = []
    for c in cs:
        count = simulator(c, shots)
        ts.append(count)
    return ts


def simulator(c, shots, logical_physical_mapping=None):
    # with readout_error noise
    nqubit = c._nqubits
    if logical_physical_mapping is None:
        logical_physical_mapping = {i: i for i in range(nqubit)}

    gg = []
    for i in range(200):
        gg.append(np.sin(i) * 0.02 + 0.978)
        # gg.append(0.98 - i * 0.01)
    readout_error = np.reshape(gg[0 : nqubit * 2], (nqubit, 2))
    mapped_readout_error = [[1, 1]] * nqubit
    for lq, phyq in logical_physical_mapping.items():
        mapped_readout_error[lq] = readout_error[phyq]
    return partial_sample(c, shots, mapped_readout_error)


def test_mapping():
    nqubit = 15
    shots = 100000
    c = tc.Circuit(nqubit)
    c.H(4)
    c.cnot(4, 5)
    c.cnot(5, 6)
    c.cnot(6, 7)
    c.rx(4, theta=0.8)
    c.rx(7, theta=1.8)
    c.measure_instruction(4)
    c.measure_instruction(5)
    c.measure_instruction(6)
    c.measure_instruction(7)

    mit = ReadoutMit(execute=run)
    mit.cals_from_system(list(range(15)), shots=100000, method="local")

    show_qubits = [6, 7, 5]

    idea_count = c.sample(batch=shots, allow_state=True, format="count_dict_bin")
    idea_count1 = counts.marginal_count(idea_count, show_qubits)

    def miti_kl_mean(logical_physical_mapping):
        ls = []
        for _ in range(10):
            raw_count = simulator(c, shots, logical_physical_mapping)
            mit_count1 = mit.apply_correction(
                raw_count,
                qubits=show_qubits,
                positional_logical_mapping={1: 5, 0: 4, 2: 6, 3: 7},
                logical_physical_mapping=logical_physical_mapping,
                method="square",
            )
            ls.append(counts.kl_divergence(idea_count1, mit_count1))
        # print("std", np.std(listtt), np.mean(listtt)) # smaller error rate and larger shots, better mititation.
        np.testing.assert_allclose(np.mean(ls), 0.01, atol=1e-2)

    logical_physical_mapping = {4: 0, 6: 2, 7: 3, 5: 1}
    miti_kl_mean(logical_physical_mapping)

    logical_physical_mapping = {4: 4, 5: 5, 6: 6, 7: 7}
    miti_kl_mean(logical_physical_mapping)

    logical_physical_mapping = {4: 8, 5: 9, 6: 10, 7: 11}
    miti_kl_mean(logical_physical_mapping)


def test_readout_expv_map():
    shots = 100000
    nqubit = 7
    c = tc.Circuit(nqubit)
    c.H(3)
    c.cnot(3, 4)
    c.cnot(4, 5)
    c.rx(3, theta=0.8)
    c.rx(4, theta=1.2)
    c.measure_instruction(3)
    c.measure_instruction(4)
    c.measure_instruction(5)

    idea_count = c.sample(batch=100000, allow_state=True, format="count_dict_bin")
    idea_value = counts.expectation(idea_count, z=[4, 5])

    # logical_physical_mapping = {3: 3, 4: 4, 5: 5}
    logical_physical_mapping = {3: 1, 5: 3, 4: 6}
    positional_logical_mapping = {1: 4, 0: 3, 2: 5}

    raw_count = simulator(c, shots, logical_physical_mapping)

    cal_qubits = list(range(nqubit))

    mit = ReadoutMit(execute=run)
    mit.cals_from_system(cal_qubits, shots=100000, method="local")
    mit_value1 = mit.expectation(
        raw_count,
        z=[4, 5],
        positional_logical_mapping=positional_logical_mapping,
        logical_physical_mapping=logical_physical_mapping,
        method="inverse",
    )

    mit_count = mit.apply_correction(
        raw_count,
        qubits=[3, 4, 5],
        positional_logical_mapping=positional_logical_mapping,
        logical_physical_mapping=logical_physical_mapping,
        method="inverse",
    )
    mit_value = counts.expectation(mit_count, z=[1, 2])
    # print("idea", idea_value)
    # print("mit", mit_value)

    mit = ReadoutMit(execute=run)
    mit.cals_from_system(cal_qubits, shots=100000, method="global")
    mit_value2 = mit.expectation(
        raw_count,
        z=[4, 5],
        positional_logical_mapping=positional_logical_mapping,
        logical_physical_mapping=logical_physical_mapping,
        method="inverse",
    )

    # print("mit1", mit_value1)
    # print("mit2", mit_value2)
    np.testing.assert_allclose(idea_value, mit_value, atol=3 * 1e-2)
    np.testing.assert_allclose(idea_value, mit_value1, atol=3 * 1e-2)
    np.testing.assert_allclose(idea_value, mit_value2, atol=3 * 1e-2)
