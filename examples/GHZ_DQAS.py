import sys

sys.path.insert(0, "../")
import tensorcircuit as tc
import numpy as np
import tensorflow as tf
import cirq

from tensorcircuit.applications.qaoa import (
    set_op_pool,
    get_preset,
    GHZ_vag,
    GHZ_vag_tfq,
    DQAS_search,
    double_qubits_initial,
)

tc.set_backend("tensorflow")


def main_tn():
    """
    DQAS with tensorcircuit engine backend by TensorNetwork

    :return:
    """
    # multi start may be necessary
    ghz_pool = [
        ("ry", 0),
        ("ry", 1),
        ("ry", 2),
        ("CNOT", 0, 1),
        ("CNOT", 1, 0),
        ("CNOT", 0, 2),
        ("CNOT", 2, 0),
        ("H", 0),
        ("H", 1),
        ("H", 2),
    ]
    set_op_pool(ghz_pool)
    c = len(ghz_pool)
    p = 4
    stp, nnp = DQAS_search(
        GHZ_vag,
        p=p,
        batch=128,
        epochs=200,
        verbose=True,
        parallel_num=0,
        nnp_initial_value=np.zeros([p, c]),
        structure_opt=tf.keras.optimizers.Adam(learning_rate=0.15),
    )
    preset = get_preset(stp).numpy()
    GHZ_vag(None, nnp, preset, verbose=True)


def main_tfq():
    """
    DQAS with tensorflow quantum engine

    :return:
    """
    p = 4
    cset = [
        cirq.H(cirq.GridQubit(0, 0)),
        cirq.H(cirq.GridQubit(1, 0)),
        cirq.CNOT(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
        cirq.CNOT(cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)),
        cirq.X(cirq.GridQubit(0, 0)),
        cirq.X(cirq.GridQubit(1, 0)),
    ]
    set_op_pool(cset)
    c = len(cset)
    stp, nnp = DQAS_search(
        GHZ_vag_tfq,
        g=double_qubits_initial(),
        p=p,
        batch=64,
        epochs=50,
        verbose=False,
        parallel_num=0,
        nnp_initial_value=np.zeros([p, c]),
        structure_opt=tf.keras.optimizers.Adam(learning_rate=0.15),
    )
    preset = get_preset(stp).numpy()
    GHZ_vag_tfq(double_qubits_initial().send(None), nnp, preset, verbose=True)


if __name__ == "__main__":
    main_tfq()
