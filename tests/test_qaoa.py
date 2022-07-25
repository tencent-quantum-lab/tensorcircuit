import sys
import os

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)

import numpy as np

from tensorcircuit.applications.dqas import set_op_pool
from tensorcircuit.applications.graphdata import get_graph
from tensorcircuit.applications.layers import Hlayer, rxlayer, zzlayer
from tensorcircuit.applications.vags import evaluate_vag


def test_vag(tfb):
    set_op_pool([Hlayer, rxlayer, zzlayer])
    expene, ene, eneg, p = evaluate_vag(
        np.array([0.0, 0.3, 0.5, 0.7, -0.8]),
        [0, 2, 1, 2, 1],
        get_graph("10A"),
        lbd=0,
        overlap_threhold=11,
    )
    print(expene, eneg, p)
    np.testing.assert_allclose(ene.numpy(), -7.01, rtol=1e-2)
