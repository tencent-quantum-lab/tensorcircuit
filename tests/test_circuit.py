import sys
import os
import numpy as np

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


def test_basics():
    c = tc.Circuit(2)
    c.x(0)
    assert np.allclose(c.amplitude("10"), np.array(1.0))
