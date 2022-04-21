"""
H6 molecule VQNHE with code from tc.application
"""
import sys

sys.path.insert(0, "../")
import numpy as np
import tensorcircuit as tc
from tensorcircuit.applications.vqes import VQNHE, JointSchedule, construct_matrix_v3

tc.set_backend("tensorflow")
tc.set_dtype("complex128")

h6h = np.load("./h6_hamiltonian.npy")  # reported in 0.99 A
hamiltonian = construct_matrix_v3(h6h.tolist())


vqeinstance = VQNHE(
    10,
    hamiltonian,
    {"width": 16, "stddev": 0.001, "choose": "complex-rbm"},  # model parameter
    {"filled_qubit": [0, 1, 3, 4, 5, 6, 8, 9], "epochs": 2},  # circuit parameter
    shortcut=True,  # enable shortcut for full Hamiltonian matrix evaluation
)
# 1110011100

rs = vqeinstance.multi_training(
    tries=2,  # 10
    maxiter=500,  # 10000
    threshold=0.5e-8,
    optq=JointSchedule(200, 0.01, 800, 0.002, 800),
    optc=JointSchedule(200, 0.0006, 10000, 0.008, 5000),
    onlyq=0,
    debug=200,
    checkpoints=[(900, -3.18), (2600, -3.19), (4500, -3.2)],
)
print(rs)
