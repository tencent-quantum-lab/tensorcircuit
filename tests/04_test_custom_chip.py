import sys
import os
import matplotlib.pyplot as plt

# Add the directory containing your module to Python's search path
module_path = ".."
sys.path.insert(0, module_path)

from tensorcircuit import Circuit, Param, gates, waveforms
from tensorcircuit.cloud.apis import submit_task, get_device, set_provider, set_token, list_devices, list_properties
from tensorcircuit.cloud.tencent import Topology
import re

# from dotenv import load_dotenv
# load_dotenv()

shots_const = 1000

print("✅ TEST FILE LOADED")
set_token(os.getenv("TOKEN"))
set_provider("tencent")
ds = list_devices()
print(ds)

def gen_custom_chip_circuit(t):
    qc = Circuit(3)

    # qc.h(0)
    qc.cnot(0, 2)
    # qc.cnot(1, 3)
    # qc.cnot(0, 3)

    # print(qc.to_tqasm())
    return qc


def run_circuit(qc):
    device_name = "tianji_s2"
    d = get_device(device_name)
# 以下是两种映射方式
    tp = Topology(d)
    # tp.map_qubits([6,3,4,1])
    # tp.map_qubits([6,3,4,1], [3,2,1,0])

    tp.map_qubit(0, 6)
    tp.map_qubit(1, 3)
    tp.map_qubit(2, 4)
    tp.map_qubit(3, 1)

    tp.pair_qubit(0, 1)
    tp.pair_qubit(0, 2)
    tp.pair_qubit(1, 3)
    t = submit_task(
        circuit=qc,
        shots=shots_const,
        device=d,
        topology = tp,
        enable_qos_gate_decomposition=False,
        enable_qos_qubit_mapping=False,
    )
    # print(qc.to_tqasm())
    # n = qc._nqubits
    rf = t.results()
    print(rf)

qc = gen_custom_chip_circuit(1.0)
result = run_circuit(qc)