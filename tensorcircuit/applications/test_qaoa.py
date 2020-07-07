from .qaoa import evaluate_vag
from .graphdata import get_graph
from .layers import *
from ..cons import set_backend


def test_vag():
    set_backend("tensorflow")
    set_choice([Hlayer, rxlayer, zzlayer])
    expene, ene, eneg, p = evaluate_vag(
        np.array([0.0, 0.3, 0.5, 0.7, -0.8]),
        [0, 2, 1, 2, 1],
        get_graph("10A"),
        lbd=0,
        overlap_threhold=11,
    )
    assert np.allclose(ene.numpy(), -7.01, rtol=1e-2)
