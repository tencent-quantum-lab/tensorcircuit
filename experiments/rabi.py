import sys
import os
import matplotlib.pyplot as plt

# Add the directory containing your module to Python's search path
module_path = ".."
sys.path.insert(0, module_path)

from tensorcircuit import Circuit, Param, gates, waveforms
from tensorcircuit.cloud.apis import submit_task, get_device, set_provider, set_token, list_devices
import re

shots_const = 1000

print("✅ TEST FILE LOADED")
set_token("")
set_provider("tencent")
ds = list_devices()
print(ds)

# TQASM 0.2;
# QREG a[1];
# defcal rabi_test a {
# frame drive_frame = newframe(a); 
# play(drive_frame, cosine_drag($formatted_t, 0.2, 0.0, 0.0)); } 
# rabi_test a[0];
# MEASZ a[0];

def gen_parametric_waveform_circuit(t):
    qc = Circuit(2)

    param0 = Param("a")

    builder = qc.calibrate("rabi_test", [param0])
    builder.new_frame("drive_frame", param0)
    builder.play("drive_frame", waveforms.CosineDrag(t, 0.2, 0.0, 0.0))

    builder.build()
    qc.add_calibration(builder, ['q[0]']) //'rabi_test q[0]'
    
    tqasm_code = qc.to_tqasm()

    print(tqasm_code)
    return qc


def run_circuit(qc):
    device_name = "tianji_m2" 
    d = get_device(device_name)
    t = submit_task(
    circuit=qc,
    shots=shots_const,
    device=d,
    enable_qos_gate_decomposition=False,
    enable_qos_qubit_mapping=False,
    )
    rf = t.results()
    return rf



def exp_rabi():
    result_lst = []
    for t in range(1, 4, 2):
        qc = gen_parametric_waveform_circuit(t)
        result = run_circuit(qc)
        result['duration'] = t
        result_lst.append(result)
    return result_lst



def draw_rabi(result_lst):
    data = {
        'duration': [],
        '0': [],
        '1': []
    }
    
    for result in result_lst:
        data['0'].append(int(result['0']) / shots_const)
        data['1'].append(int(result['1']) / shots_const)
        data['duration'].append(result['duration'])


        

    plt.figure(figsize=(10,6))
    plt.plot(data['duration'], data['0'], 'b-o', label='State |0>')
    plt.plot(data['duration'], data['1'], 'r--s', label='State |1>')


    plt.title('Rabi Oscillation Experiment')
    plt.xlabel('Duration (dt)')
    plt.ylabel('Probability')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()


    plt.savefig('rabi.png', dpi=300)
    plt.show()


data = exp_rabi()
draw_rabi(data)
