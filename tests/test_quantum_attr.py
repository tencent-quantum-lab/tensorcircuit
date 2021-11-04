import sys
import os

import numpy as np
import tensorflow as tf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
from tensorcircuit.applications.vags import double_state, reduced_density_matrix


def test_double_state():
    s = double_state(tf.constant([[1.0, 0], [0, -1.0]]), beta=2.0)
    assert np.allclose(np.linalg.norm(s.numpy()), 1.0)
    assert np.allclose(
        s.numpy(),
        np.array(
            [
                [
                    np.exp(-1) / np.sqrt(np.exp(2) + np.exp(-2)),
                    0,
                    0,
                    np.exp(1) / np.sqrt(np.exp(2) + np.exp(-2)),
                ]
            ]
        ),
    )
    s2 = double_state(tf.constant([[0.0, 1.0], [1.0, 0.0]]), beta=1.0)
    assert np.allclose(np.linalg.norm(s2.numpy()), 1.0)
    em = np.exp(-0.5)
    ep = np.exp(0.5)
    ans = np.array([em + ep, em - ep, em - ep, em + ep])
    ans /= np.linalg.norm(ans)
    assert np.allclose(s2.numpy(), ans)


def test_reduced_dm():
    rho = reduced_density_matrix(
        tf.random.normal(shape=[128]), freedom=7, cut=[1, 3, 5]
    )
    assert np.allclose(np.trace(rho.numpy()), 1)
