"""
benchmark sparse hamiltonian building
"""

import time

import jax
import numpy as np
import quimb
import scipy
import tensorflow as tf
import torch

import tensorcircuit as tc

tc.set_dtype("complex128")

nwires = 20
# tc with numpy backend outperforms quimb in large nwires

print("--------------------")

# 1.1 tc approach for TFIM (numpy backend)

tc.set_backend("numpy")
print("hamiltonian building with tc (numpy backend)")
print("numpy version: ", np.__version__)
print("scipy version: ", scipy.__version__)

g = tc.templates.graphs.Line1D(nwires, pbc=False)
time0 = time.perf_counter()
h11 = tc.quantum.heisenberg_hamiltonian(
    g, hzz=1, hxx=0, hyy=0, hz=0, hx=-1, hy=0, sparse=True, numpy=True
)
time1 = time.perf_counter()

print("tc (numpy) time: ", time1 - time0)
print("--------------------")

# 1.2 tc approach for TFIM (jax backend)

tc.set_backend("jax")
print("hamiltonian building with tc (jax backend)")
print("jax version: ", jax.__version__)

g = tc.templates.graphs.Line1D(nwires, pbc=False)
time0 = time.perf_counter()
h12 = tc.quantum.heisenberg_hamiltonian(
    g, hzz=1, hxx=0, hyy=0, hz=0, hx=-1, hy=0, sparse=True
)
time1 = time.perf_counter()

print("tc (jax) time: ", time1 - time0)
print("--------------------")

# 1.3 tc approach for TFIM (tensorflow backend)

tc.set_backend("tensorflow")
print("hamiltonian building with tc (tensorflow backend)")
print("tensorflow version: ", tf.__version__)

g = tc.templates.graphs.Line1D(nwires, pbc=False)
time0 = time.perf_counter()
h13 = tc.quantum.heisenberg_hamiltonian(
    g, hzz=1, hxx=0, hyy=0, hz=0, hx=-1, hy=0, sparse=True
)
time1 = time.perf_counter()

print("tc (tensorflow) time: ", time1 - time0)
print("--------------------")


# 2. quimb approach for TFIM

print("hamiltonian building with quimb")
print("quimb version: ", quimb.__version__)

builder = quimb.tensor.SpinHam1D()
# spin operator instead of Pauli matrix
builder += 4, "Z", "Z"
builder += -2, "X"
time0 = time.perf_counter()
h2 = builder.build_sparse(nwires)
h2 = h2.tocoo()
time1 = time.perf_counter()

print("quimb time: ", time1 - time0)


def assert_equal(h1, h2):
    np.testing.assert_allclose(h1.row, h2.row, atol=1e-5)
    np.testing.assert_allclose(h1.col, h2.col, atol=1e-5)
    np.testing.assert_allclose(h1.data, h2.data, atol=1e-5)


# numpy
assert_equal(h11, h2)

# jax
scipy_coo = scipy.sparse.coo_matrix(
    (
        h12.data,
        (h12.indices[:, 0], h12.indices[:, 1]),
    ),
    shape=h12.shape,
)
assert_equal(scipy_coo, h2)

# tensorflow
scipy_coo = scipy.sparse.coo_matrix(
    (
        h13.values,
        (h13.indices[:, 0], h13.indices[:, 1]),
    ),
    shape=h13.shape,
)
assert_equal(scipy_coo, h2)
