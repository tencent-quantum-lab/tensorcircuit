import sys
import os
import numpy as np

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


def test_any_measurement():
    c = tc.Circuit(2)
    c.H(0)
    c.H(1)
    mea = np.array([1, 1])
    r = tc.templates.measurements.any_measurements(c, mea, onehot=True)
    np.testing.assert_allclose(r, 1.0, atol=1e-5)
    mea2 = np.array([3, 0])
    r2 = tc.templates.measurements.any_measurements(c, mea2, onehot=True)
    np.testing.assert_allclose(r2, 0.0, atol=1e-5)
