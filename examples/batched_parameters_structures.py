"""
VQE optimization over different parameter initializations and different circuit structures
"""

import tensorflow as tf
import numpy as np
import tensorcircuit as tc

K = tc.set_backend("tensorflow")

n = 6
lattice = tc.templates.graphs.Line1D(n, pbc=False)
h = tc.quantum.heisenberg_hamiltonian(
    lattice, hzz=1, hxx=0, hyy=0, hx=-1, hy=0, hz=0, sparse=False
)


def gate_list(param):
    l = [
        tc.gates.Gate(np.eye(4)),
        tc.gates.Gate(np.kron(tc.gates._x_matrix, np.eye(2))),
        tc.gates.Gate(np.kron(tc.gates._y_matrix, np.eye(2))),
        tc.gates.Gate(np.kron(tc.gates._z_matrix, np.eye(2))),
        tc.gates.Gate(np.kron(tc.gates._h_matrix, np.eye(2))),
        tc.gates.exp1_gate(theta=param, unitary=tc.gates._xx_matrix),
        tc.gates.exp1_gate(theta=param, unitary=tc.gates._yy_matrix),
        tc.gates.exp1_gate(theta=param, unitary=tc.gates._zz_matrix),
    ]
    return [tc.backend.reshape2(m.tensor) for m in l if isinstance(m, tc.gates.Gate)]


def makec(param, structure):
    c = tc.Circuit(n)
    for i in range(structure.shape[0]):
        for j in range(n):
            c.select_gate(structure[i, j], gate_list(param[i, j]), j, (j + 1) % n)
    return c


def vqef(param, structure):
    c = makec(param, structure)
    e = tc.templates.measurements.operator_expectation(c, h)
    return e


vqegf = tc.backend.jit(
    tc.backend.vmap(
        tc.backend.vvag(vqef, argnums=0, vectorized_argnums=0),
        vectorized_argnums=(0, 1),
    )
)

batch_structure = 2
batch_weights = 8
depth = 2
structure1 = np.array([[0, 1, 0, 5, 0, 6], [6, 0, 6, 0, 6, 0]])
structure2 = np.array([[0, 1, 0, 5, 0, 6], [5, 0, 5, 0, 5, 0]])
structure = tc.backend.stack([structure1, structure2])
structure = tc.backend.cast(structure, "int32")
weights = tc.backend.implicit_randn(shape=[batch_structure, batch_weights, depth, n])
opt = tc.backend.optimizer(tf.keras.optimizers.Adam(1e-2))

for _ in range(100):
    v, g = vqegf(weights, structure)
    print("energy: ", v)
    weights = opt.update(g, weights)
