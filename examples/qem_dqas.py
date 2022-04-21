"""
DQAS for QFT QEM circuit design, deprecated DQAS implementation
"""
import sys

sys.path.insert(0, "../")

from functools import partial
import cirq
import numpy as np
import tensorflow as tf
from tensorcircuit.applications.vags import qft_qem_vag
from tensorcircuit.applications.dqas import (
    set_op_pool,
    DQAS_search,
    verbose_output,
)


def main_3():
    qft_3 = partial(qft_qem_vag, n=3)

    set_op_pool([cirq.X, cirq.Y, cirq.Z, cirq.I, cirq.T, cirq.S])
    DQAS_search(
        qft_3,
        nnp_initial_value=tf.zeros([6, 6]),
        p=6,
        prethermal=0,
        batch=32,
        verbose=False,
        epochs=30,
    )


def main_4():
    qft_4 = partial(qft_qem_vag, n=4)

    set_op_pool(
        [
            cirq.I,
            cirq.X,
            cirq.Y,
            cirq.Z,
            cirq.H,
            cirq.rx(np.pi / 3),
            cirq.rx(np.pi * 2.0 / 3),
            cirq.rz(np.pi / 3),
            cirq.rz(np.pi * 2.0 / 3),
            cirq.S,
            cirq.T,
        ]
    )
    DQAS_search(
        qft_4,
        nnp_initial_value=tf.zeros([12, 11]),
        p=12,
        prethermal=0,
        batch=32,
        verbose=False,
        verbose_func=verbose_output,
        epochs=6,
    )


if __name__ == "__main__":
    # main_3()
    main_4()
