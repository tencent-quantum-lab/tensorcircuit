from tensorcircuit.results import counts


d = {"000": 2, "101": 3, "100": 4}


def test_marginal_count():
    assert counts.marginal_count(d, [1, 2])["00"] == 6
    assert counts.marginal_count(d, [1])["0"] == 9
    assert counts.marginal_count(d, [2, 1, 0])["001"] == 4


def test_count2vec():
    assert counts.vec2count(counts.count2vec(d, normalization=False), prune=True) == d
