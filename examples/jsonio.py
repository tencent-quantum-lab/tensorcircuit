"""
example showcasing how circuit can be load from and dump to json:
useful for storage or restful api
"""
import numpy as np
import tensorcircuit as tc

tc.set_dtype("complex128")


def make_circuit():
    c = tc.Circuit(3)
    c.h(0)
    c.H(2)
    c.CNOT(1, 2)
    c.rxx(0, 2, theta=0.3)
    c.crx(0, 1, theta=-0.8)
    c.r(1, theta=tc.backend.ones([]), alpha=0.2)
    c.toffoli(0, 2, 1)
    c.ccnot(0, 1, 2)
    c.any(0, 1, unitary=tc.gates._xx_matrix)
    c.multicontrol(1, 2, 0, ctrl=[0, 1], unitary=tc.gates._x_matrix)
    return c


if __name__ == "__main__":
    c = make_circuit()
    s = c.to_json()
    print(s)
    c.to_json(file="circuit.json")
    # load from json string
    c2 = tc.Circuit.from_json(s)
    print("\n", c2.draw())
    np.testing.assert_allclose(c.state(), c2.state(), atol=1e-5)
    print("test correctness 1")
    # load from json file
    c3 = tc.Circuit.from_json_file("circuit.json")
    print("\n", c3.draw())
    np.testing.assert_allclose(c.state(), c3.state(), atol=1e-5)
    print("test correctness 2")
