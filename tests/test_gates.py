import sys
import os
import numpy as np

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


def test_rgate():
    tc.set_dtype("complex128")
    np.testing.assert_almost_equal(
        tc.gates.rgate(1, 2, 3).tensor, tc.gates.rgate_theoretical(1, 2, 3).tensor
    )
