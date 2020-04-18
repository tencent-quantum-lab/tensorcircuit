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


def test_measure():
    c = tc.Circuit(3)
    c.H(0)
    c.h(1)
    c.toffoli(0, 1, 2)
    assert c.measure(2)[0] in ["0", "1"]


def test_expectation():
    c = tc.Circuit(2)
    c.H(0)
    assert np.allclose(c.expectation(tc.gates.z(), 0), 0)


def test_qcode():
    qcode = """
4
x 0
cnot 0 1
r 2 theta 1.0 alpha 1.57
"""
    c = tc.Circuit.from_qcode(qcode)
    assert c.measure(1)[0] == "1"
    print(c.to_qcode())
    assert c.to_qcode() == qcode[1:]
