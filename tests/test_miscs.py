import sys
import os
import numpy as np
import tensorflow as tf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc
from tensorcircuit.quantum import (
    PauliString2COO,
    PauliStringSum2COO,
    densify,
)
from tensorcircuit.applications.vqes import construct_matrix_v2

i, x, y, z = [t.tensor for t in tc.gates.pauli_gates]

# note i is in use!

check_pairs = [
    ([0, 0], np.eye(4)),
    ([0, 1], np.kron(i, x)),
    ([2, 1], np.kron(y, x)),
    ([3, 1], np.kron(z, x)),
    ([3, 2, 2, 0], np.kron(np.kron(np.kron(z, y), y), i)),
    ([0, 1, 1, 1], np.kron(np.kron(np.kron(i, x), x), x)),
]


def test_ps2coo():
    for l, a in check_pairs:
        r1 = PauliString2COO(tf.constant(l, dtype=tf.int64))
        np.testing.assert_allclose(densify(r1), a, atol=1e-5)


def test_pss2coo():
    l = [t[0] for t in check_pairs[:4]]
    a = sum([t[1] for t in check_pairs[:4]])
    r1 = PauliStringSum2COO(tf.constant(l, dtype=tf.int64))
    np.testing.assert_allclose(densify(r1), a, atol=1e-5)
    l = [t[0] for t in check_pairs[4:]]
    a = sum([t[1] for t in check_pairs[4:]])
    r1 = PauliStringSum2COO(tf.constant(l, dtype=tf.int64), weight=[0.5, 1])
    a = check_pairs[4][1] * 0.5 + check_pairs[5][1] * 1.0
    np.testing.assert_allclose(densify(r1), a, atol=1e-5)


def test_sparse(benchmark, tfb):
    def sparse(h):
        return PauliStringSum2COO(h)

    h = [[1 for _ in range(12)], [2 for _ in range(12)]]
    h = tf.constant(h, dtype=tf.int64)
    sparse(h)
    benchmark(sparse, h)


def test_dense(benchmark, tfb):
    def dense(h):
        return construct_matrix_v2(h, dtype=tf.complex64)

    h = [[1 for _ in range(12)], [2 for _ in range(12)]]
    h = [[1.0] + hi for hi in h]
    dense(h)
    benchmark(dense, h)
