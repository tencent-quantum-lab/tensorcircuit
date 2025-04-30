import sys
import os
import itertools

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)

import numpy as np
import tensorflow as tf


from tensorcircuit.applications.van import MaskedLinear, ResidualBlock, MADE, PixelCNN


def test_masklinear():
    ml = MaskedLinear(5, 10, 3, mask=tf.zeros([10, 3, 5, 3]))
    ml.set_weights([tf.ones([10, 3, 5, 3]), tf.ones([10, 3])])
    tf.debugging.assert_near(ml(tf.ones([1, 5, 3])), tf.ones([1, 10, 3]))

    mask = np.zeros([10, 3, 5, 3])
    mask[3, 2, 1, 0] = 1.0
    ml = MaskedLinear(5, 10, 3, mask=tf.constant(mask, dtype=tf.float32))
    ml.set_weights([tf.ones([10, 3, 5, 3]), tf.zeros([10, 3])])
    assert tf.reduce_sum(ml(tf.ones([5, 3]))[3, :]) == 1.0

    w = tf.random.uniform(shape=[10, 3, 5, 3])
    b = tf.random.uniform(shape=[10, 3])
    w_m = tf.reshape(w, [30, 15])
    b_m = tf.reshape(b, [30, 1])
    inputs = tf.ones([5, 3])
    inputs_m = tf.reshape(inputs, [15, 1])
    r_m = w_m @ inputs_m + b_m
    r = tf.reshape(r_m, [10, 3])
    ml = MaskedLinear(5, 10, 3)
    ml.set_weights([w, b])
    tf.debugging.assert_near(ml(inputs), r)


def test_residual_block():
    dense1 = tf.keras.layers.Dense(10, use_bias=False)
    dense1.build([1, 1])
    dense1.set_weights([np.ones([1, 10])])
    dense2 = tf.keras.layers.Dense(1, use_bias=False)
    dense2.build([1, 10])
    dense2.set_weights([np.ones([10, 1])])
    m = ResidualBlock([dense1, dense2])
    assert m(tf.ones([1, 1])) == 11.0


def test_made():
    import itertools

    m = MADE(2, 2, 6, 3, 3, nonmerge=False)
    l = []
    for i in itertools.product(*[list(range(3)) for _ in range(2)]):
        l.append(list(i))
    basis = tf.constant(l, dtype=tf.int32)
    print(basis)
    ptot = tf.reduce_sum(tf.exp(m.log_prob(tf.one_hot(basis, depth=3))))
    np.testing.assert_allclose(ptot.numpy(), 1.0, atol=1e-5)

    s, logp = m.sample(10)
    print(logp)
    assert s.shape == (10, 2, 3)


def test_made_fit_peak():
    opt = tf.optimizers.Adam(learning_rate=0.01)
    m = MADE(5, 5, 4, 3, 2)
    for step in range(100):
        with tf.GradientTape() as t:
            loss = -tf.reduce_sum(
                m.log_prob(
                    tf.one_hot(
                        [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 1, 1, 0]], depth=3
                    )
                )
            )
            gr = t.gradient(loss, m.variables)
        if step % 20 == 0:
            print(
                tf.exp(
                    m.log_prob(
                        tf.one_hot(
                            [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 1, 1, 0]], depth=3
                        )
                    )
                ).numpy()
            )
        opt.apply_gradients(zip(gr, m.variables))


def test_pixelcnn():
    m = PixelCNN(3, 5, 8)
    l = []
    for i in itertools.product(*[list(range(3)) for _ in range(4)]):
        l.append(list(i))
    basis = tf.constant(tf.reshape(l, [-1, 2, 2]), dtype=tf.int32)
    ptot = tf.reduce_sum(tf.exp(m.log_prob(tf.one_hot(basis, depth=3))))
    np.testing.assert_allclose(ptot.numpy(), 1.0, atol=1e-5)
