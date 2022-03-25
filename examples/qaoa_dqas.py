"""
Old-fashioned DQAS code on QAOA ansatz design, now deprecated.
"""

# pylint: disable=wildcard-import

import sys

sys.path.insert(0, "../")

from collections import namedtuple
from pickle import dump
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import tensorcircuit as tc
from tensorcircuit.applications.dqas import *
from tensorcircuit.applications.vags import *
from tensorcircuit.applications.layers import *
from tensorcircuit.applications.graphdata import regular_graph_generator

# qaoa_block_vag_energy = partial(qaoa_block_vag, f=(_identity, _neg))

tc.set_backend("tensorflow")


def main_layerwise_encoding():
    p = 5
    c = 7

    def noise():
        n = np.random.normal(loc=0.0, scale=0.002, size=[p, c])
        return tf.constant(n, dtype=tf.float32)

    def penalty_gradient(stp, nnp, lbd=0.15, lbd2=0.01):
        c = stp.shape[1]
        p = stp.shape[0]
        cost = tf.constant(
            [1.0, 1.0, 1.0, 1.0, 27 / 2 * 2.0, 27 / 2 * 2.0, 15 / 2.0], dtype=tf.float32
        )
        with tf.GradientTape() as t:
            prob = tf.math.exp(stp) / tf.tile(
                tf.math.reduce_sum(tf.math.exp(stp), axis=1)[:, tf.newaxis], [1, c]
            )
            penalty = 0.0
            for i in range(p - 1):
                penalty += lbd * tf.tensordot(prob[i], prob[i + 1], 1)
                penalty += lbd2 * tf.tensordot(cost, prob[i], [0, 0])
            penalty += lbd2 * tf.tensordot(cost, prob[p - 1], [0, 0])
            penalty_g = t.gradient(penalty, stp)
        return penalty_g

    learning_rate_sch = tf.keras.optimizers.schedules.InverseTimeDecay(0.12, 1, 0.01)

    DQAS_search(
        qaoa_vag_energy,
        g=regular_graph_generator(n=8, d=3),
        p=p,
        batch=8,
        prethermal=0,
        prethermal_preset=[0, 6, 1, 6, 1],
        epochs=3,
        parallel_num=2,
        pertubation_func=noise,
        nnp_initial_value=np.random.normal(loc=0.23, scale=0.06, size=[p, c]),
        stp_regularization=penalty_gradient,
        network_opt=tf.keras.optimizers.Adam(learning_rate=0.03),
        prethermal_opt=tf.keras.optimizers.Adam(learning_rate=0.04),
        structure_opt=tf.keras.optimizers.SGD(learning_rate=learning_rate_sch),
    )


result = namedtuple("result", ["epoch", "cand", "loss"])


def main_block_encoding():
    learning_rate_sch = tf.keras.optimizers.schedules.InverseTimeDecay(0.3, 1, 0.01)
    p = 6
    c = 8

    def record():
        return result(
            get_var("epoch"), get_var("cand_preset_repr"), get_var("avcost1").numpy()
        )

    def noise():
        # p = 6
        # c = 6
        n = np.random.normal(loc=0.0, scale=0.2, size=[2 * p, c])
        return tf.constant(n, dtype=tf.float32)

    stp, nnp, h = DQAS_search(
        qaoa_block_vag_energy,
        g=regular_graph_generator(n=8, d=3),
        batch=8,
        prethermal=0,
        prethermal_preset=[0, 6, 1, 6, 1],
        epochs=3,
        parallel_num=2,
        pertubation_func=noise,
        p=p,
        history_func=record,
        nnp_initial_value=np.random.normal(loc=0.23, scale=0.06, size=[2 * p, c]),
        network_opt=tf.keras.optimizers.Adam(learning_rate=0.04),
        prethermal_opt=tf.keras.optimizers.Adam(learning_rate=0.04),
        structure_opt=tf.keras.optimizers.SGD(learning_rate=learning_rate_sch),
    )

    with open("qaoa_block.result", "wb") as f:
        dump([stp.numpy(), nnp.numpy(), h], f)

    epochs = np.arange(len(h))
    data = np.array([r.loss for r in h])
    plt.plot(epochs, data)
    plt.xlabel("epoch")
    plt.ylabel("objective")
    plt.savefig("qaoa_block.pdf")


layer_pool = [Hlayer, rxlayer, rylayer, rzlayer, xxlayer, yylayer, zzlayer]
block_pool = [
    Hlayer,
    rx_zz_block,
    zz_ry_block,
    zz_rx_block,
    zz_rz_block,
    xx_rz_block,
    yy_rx_block,
    rx_rz_block,
]
set_op_pool(block_pool)

if __name__ == "__main__":
    # main_layerwise_encoding()
    main_block_encoding()
