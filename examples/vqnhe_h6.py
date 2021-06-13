import cirq
import numpy as np
import scipy
import sympy as sy
import networkx as nx
import tensorflow_quantum as tfq
import tensorflow as tf
from functools import partial
from functools import lru_cache
import tensornetwork as tn
from itertools import product
import json
import sys

sys.path.insert(0, "../")
import tensorcircuit as tc
from tensorcircuit.applications.layers import *
from tensorcircuit.applications.van import *
from tensorcircuit.applications.graphdata import *
from tensorcircuit.applications.dqas import *
from tensorcircuit.applications.vags import *
from tensorcircuit.applications.vqes import *

tc.set_backend("tensorflow")
tc.set_dtype("complex128")

h6h = np.load("/PATH")  # reported in 0.99 A

vqeinstance = VQNHE(
    10,
    h6h.tolist(),
    {"width": 16, "stddev": 0.001, "choose": "complex-rbm"},  # model parameter
    {"filled_qubit": [0, 1, 3, 4, 5, 6, 8, 9], "epochs": 2},  # circuit parameter
    shortcut=True,  # enable shortcut for full Hamiltonian matrix evaluation
)
# 1110011100


def learn_q():
    return JointSchedule(200, 0.01, 800, 0.002, 800)


def learn_c():
    #     return 0.
    lr = np.random.choice([0.004, 0.006, 0.008, 0.1])
    return JointSchedule(200, 0.0006, 10000, lr, 5000)


rs = vqeinstance.multi_training(
    tries=10,
    maxiter=15000,  # 10000
    threshold=0.5e-8,
    learn_q=learn_q,
    learn_c=learn_c,
    onlyq=0,
    debug=200,
    checkpoints=[(900, -3.18), (2600, -3.19), (4500, -3.2)],
)
print(rs)
