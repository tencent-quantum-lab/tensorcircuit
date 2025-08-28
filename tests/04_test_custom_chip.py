import sys
import os
import matplotlib.pyplot as plt

# Add the directory containing your module to Python's search path
module_path = ".."
sys.path.insert(0, module_path)

from tensorcircuit import Circuit, Param, gates, waveforms
from tensorcircuit.cloud.apis import submit_task, get_device, set_provider, set_token, list_devices, list_properties
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
    qc = Circuit(2)

# 需添加 MapQubits(user_addr, chip_addr)，对应 #pragram的qubits.mapping
# 需添加 GridQubit(left_user_addr, right_user_addr)，对应 #pragram的qubits.coupling

# 以下是两种映射方式
    qc.map_qubits([4,9,8,14])
    qc.map_qubits([4,9,8,14], [3,2,1,0])


    qc.i(0)
    qc.cnot(0)

    # print(qc.to_tqasm())
    return qc



def gen_custom_chip_circuit(t):
    qc = Circuit(2)

# 需添加 MapQubit(user_addr, chip_addr)，对应 #pragram的qubits.mapping
# 需添加 GridQubit(left_user_addr, right_user_addr)，对应 #pragram的qubits.coupling


    qc.MapQubit(0, 4)
    qc.MapQubit(1, 9)
    qc.MapQubit(2, 8)
    qc.MapQubit(3, 14)

    qc.GridQubit(0, 1)
    qc.GridQubit(0, 2)
    qc.GridQubit(1, 3)

    qc.i(0)
    qc.cnot(0)

    # print(qc.to_tqasm())
    return qc


def run_circuit(qc):
    device_name = "tianji_s2"
    d = get_device(device_name)
    t = submit_task(
        circuit=qc,
        shots=shots_const,
        device=d,
        enable_qos_gate_decomposition=False,
        enable_qos_qubit_mapping=False,
    )
    # print(qc.to_tqasm())
    # n = qc._nqubits
    rf = t.results()
    print(rf)

qc = gen_gate_pulse_mix_circuit(1.0)
result = run_circuit(qc)