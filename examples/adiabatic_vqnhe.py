"""
Calculate the molecule dissociation curve using VQNHE.
"""
from functools import partial
import sys

sys.path.insert(0, "../")

import numpy as np
import tensorflow as tf

import tensorcircuit as tc
from tensorcircuit.applications.vqes import VQNHE, JointSchedule

tc.set_backend("tensorflow")
tc.set_dtype("complex128")


def initial_param(t, last=None, lastlast=None):
    if ((t % 3 == 1) and last) or ((t % 3 == 2) and lastlast):
        if t % 3 == 2:
            last = lastlast
        qw = last[-1]
        qw = tf.Variable(
            qw.numpy() + np.random.uniform(low=-0.1, high=0.1, size=qw.numpy().shape)
        )
        cw = last[-2]
        for i, b in enumerate(cw):
            cw[i] = b + np.random.uniform(low=-0.1, high=0.1, size=t.shape)
        return {"c": cw, "q": qw}

    return {}


def adiabatic_range(hm, history):
    if len(history) > 0:
        last = sorted(
            [
                (
                    r["energy"],
                    r["quantum_energy"],
                    r["model_weights"],
                    r["circuit_weights"],
                )
                for r in history[-1]
            ],
            key=lambda s: s[0],
        )[0]
    else:
        last = None
    if len(history) > 1:
        lastlast = sorted(
            [
                (
                    r["energy"],
                    r["quantum_energy"],
                    r["model_weights"],
                    r["circuit_weights"],
                )
                for r in history[-1]
            ],
            key=lambda s: s[0],
        )[0]
    else:
        lastlast = None
    print("begin calculation on new")
    vqeinstance = VQNHE(
        4,
        hm,
        {"max_value": 5, "init_value": 1.0, "min_value": 0.1},
        {"filled_qubit": [0]},
    )

    def learn_q():
        return JointSchedule(180, 0.009, 800, 0.001, 800)

    def learn_c():
        return JointSchedule(160, 0.002, 10000, 0.2, 1500)

    rs = vqeinstance.multi_training(
        tries=2,
        maxiter=150,  # 10000
        threshold=0.2 * 1e-8,
        optq=learn_q,  # JointSchedule(2800, 0.009, 800, 0.002, 100),
        optc=learn_c,
        initialization_func=partial(initial_param, last=last, lastlast=lastlast),
    )
    print(
        sorted(
            [(r["energy"], r["quantum_energy"], r["iterations"]) for r in rs],
            key=lambda s: s[0],
        )
    )
    return rs


if __name__ == "__main__":
    history = []
    lihh = np.load("data_file")
    for h in lihh[3:6]:
        history.append(adiabatic_range(h.tolist(), history))
    print(history)
    # vqeinstance = VQNHE(
    #     4,
    #     lihh,
    #     {"max_value": 5, "init_value": 1.0, "min_value": 0.1},
    #     {"filled_qubit": [0]},
    # )

    # def learn_q():
    #     return JointSchedule(180, 0.009, 800, 0.001, 800)

    # def learn_c():
    #     return JointSchedule(160, 0.002, 10000, 0.2, 1500)

    # rs = vqeinstance.multi_training(
    #     tries=10,
    #     maxiter=15000,
    #     threshold=0.2 * 1e-8,
    #     learn_q=learn_q,  # JointSchedule(2800, 0.009, 800, 0.002, 100),
    #     learn_c=learn_c,
    # )
    # print(rs)
    # print(
    #     sorted(
    #         [(r["energy"], r["quantum_energy"], r["iterations"]) for r in rs],
    #         key=lambda s: s[0],
    #     )
    # )
