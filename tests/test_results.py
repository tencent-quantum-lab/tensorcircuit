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


def run(cs, shots):
    # customized backend for mitigation test
    nqubit = cs[0]._nqubits
    gg = []
    for i in range(2 * nqubit):
        gg.append(np.sin(i) * 0.02 + 0.978)
    readout_error = np.reshape(gg, (nqubit, 2))

    ts = []
    for c in cs:
        count = c.sample(
            batch=shots,
            allow_state=True,
            readout_error=readout_error,
            format="count_dict_bin",
        )
        ts.append(count)
    return ts


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
