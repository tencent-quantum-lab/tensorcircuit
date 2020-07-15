import sys

sys.path.insert(0, "../")
import cirq
import tensorflow as tf
from tensorcircuit.applications.qaoa import (
    set_op_pool,
    qft_qem_vag,
    DQAS_search,
)


def main():
    set_op_pool([cirq.X, cirq.Y, cirq.Z, cirq.I, cirq.T, cirq.S])
    DQAS_search(
        qft_qem_vag,
        nnp_initial_value=tf.zeros([6, 6]),
        p=6,
        prethermal=0,
        batch=256,
        verbose=False,
        epochs=3000,
    )


if __name__ == "__main__":
    main()
