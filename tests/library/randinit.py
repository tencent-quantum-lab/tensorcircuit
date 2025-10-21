from random import random

def qc_randinit(circuit):
    return
    # for i in range(0, circuit.num_qubits):
    for i in range(0, circuit.to_qiskit().num_qubits):
        if random() > 0.5:
            circuit.x(i)