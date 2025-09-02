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


def gen_gate_pulse_mix_circuit(t):
    qc = Circuit(2)

    param0 = Param("a")

    builder = qc.calibrate("basic_pulse", [param0])
    frame = builder.new_frame("drive_frame", param0)
    builder.play(frame, waveforms.CosineDrag(t, 0.2, 0.0, 0.0))

    builder.build()

    qc.add_calibration(builder, [0])

    qc.i(0)
    qc.x(0)

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
    return rf

qc = gen_gate_pulse_mix_circuit(1.0)
result = run_circuit(qc)
print(result)