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


def test_readout():
    def run(cs, shots):
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

    nqubit = 4
    shots = 4096
    c = tc.Circuit(nqubit)
    c.H(0)
    c.cnot(0, 1)
    c.x(3)

    idea_count = c.sample(batch=shots, allow_state=True, format="count_dict_bin")
    raw_count = run([c], shots)[0]

    mit = ReadoutMit(execute=run)
    mit.cals_from_system([0, 1, 2, 3, 6], shots=10000, method="local")

    # TODO(@yutuer21): add m3 method
    mit_count1 = mit.apply_correction(
        raw_count, [1, 3, 2], method="inverse"
    )  #  direct(Max2),iterative(Max3), inverse,square
    mit_count2 = mit.apply_correction(raw_count, [1, 3, 2], method="square")
    idea_count2 = counts.marginal_count(idea_count, [1, 3, 2])

    assert counts.kl_divergence(idea_count2, mit_count1) < 0.05
    assert counts.kl_divergence(idea_count2, mit_count2) < 0.05
