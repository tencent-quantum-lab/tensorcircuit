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
    c.CNOT(0, 1)
    assert np.allclose(c.amplitude("11"), np.array(1.0))


def test_perfect_sampling():
    c = tc.Circuit(3)
    c.H(0)
    c.h(1)
    c.toffoli(0, 1, 2)
    c.perfect_sampling()
