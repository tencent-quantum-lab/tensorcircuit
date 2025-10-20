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


def gen_multi_measure_circuit(t):
    qc = Circuit(6)

    qc.h(0)
    qc.cnot(0, 1)

    # 参数门在 to_tqasm 中的转换丢失了参数
    # qc.rxx(0, 5, theta=1.04632)

# 需添加测量指令
    qc.measz(0, 1)
    qc.cz(0, 1)
    qc.h(1)
    qc.measz(0, 1)


    print(qc.to_tqasm())
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
    rf = t.results()  # 需返回 multi_results
    return rf

qc = gen_multi_measure_circuit(1.0)
result = run_circuit(qc)
print(result)