"""
benchmark sparse hamiltonian building
"""
import time
import numpy as np
import quimb
import tensorcircuit as tc


nwires = 20
# tc outperforms quimb in large nwires


# 1. tc approach for TFIM

g = tc.templates.graphs.Line1D(nwires, pbc=False)
time0 = time.time()
h1 = tc.quantum.heisenberg_hamiltonian(
    g, hzz=1, hxx=0, hyy=0, hz=0, hx=-1, hy=0, sparse=True, numpy=True
)
time1 = time.time()

print("tc time: ", time1 - time0)

# 2. quimb approach for TFIM

builder = quimb.tensor.tensor_gen.SpinHam()
# spin operator instead of Pauli matrix
builder += 4, "Z", "Z"
builder += -2, "X"
time0 = time.time()
h2 = builder.build_sparse(nwires)
h2 = h2.tocoo()
time1 = time.time()

print("quimb time: ", time1 - time0)


def assert_equal(h1, h2):
    np.testing.assert_allclose(h1.row, h2.row, atol=1e-5)
    np.testing.assert_allclose(h1.col, h2.col, atol=1e-5)
    np.testing.assert_allclose(h1.data, h2.data, atol=1e-5)


assert_equal(h1, h2)
