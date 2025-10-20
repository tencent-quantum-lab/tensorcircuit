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


def gen_parametric_waveform_circuit(t):
    qc = Circuit(3)
    param0 = Param("a")
    builder0 = qc.calibrate("test01", [param0])
    frame0 = builder0.new_frame("drive_frame", param0)
    builder0.play(frame0, waveforms.CosineDrag(115.0, 0.2, 1.0, 0.0))
    builder0.build()
    param1 = Param("a")
    builder1 = qc.calibrate("test02", [param1])
    frame1 = builder1.new_frame("drive_frame", param1)
    builder1.play(frame1, waveforms.CosineDrag(115.0, 0.2, 1.0, 0.0))
    builder1.play(frame1, waveforms.Flattop(115.0, 0.2, 0.0))
    builder1.play(frame1, waveforms.Sine(115.0, 0.2, 0.1, 0.1, 0.0))
    
    builder1.build()
    param2 = Param("a")
    builder2 = qc.calibrate("test03", [param2])
    frame2 = builder2.new_frame("drive_frame", param2)
    builder2.play(frame2, waveforms.Gaussian(115.0, 0.2, 1.0, 0.0))
    builder2.build()

    qc.add_calibration(builder0, [0])  
    qc.add_calibration(builder1, [1]) 
    qc.add_calibration(builder2, [2]) 
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
    rf = t.results()
    # print(rf)

    
    return rf

qc = gen_parametric_waveform_circuit(1.0)
result = run_circuit(qc)
print(result)