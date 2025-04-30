import sys
import os
import matplotlib.pyplot as plt

# Add the directory containing your module to Python's search path
module_path = ".."
sys.path.insert(0, module_path)

from tensorcircuit import Circuit, Param, gates, waveforms
from tensorcircuit.cloud.apis import submit_task, get_device, set_provider, set_token, list_devices
import re

print("✅ TEST FILE LOADED")
set_token("abcdefg")
set_provider("tencent")
ds = list_devices()
print(ds)

def test_parametric_waveform(t):
    qc = Circuit(2)

    param0 = Param("a")
    param1 = Param("b")

    builder = qc.calibrate("my_gate", [param0, param1])
    builder.new_frame("f0", param0)
    builder.play("f0", waveforms.CosineDrag(t, 0.2, 0.0, 0.01))
    builder.new_frame("f1", param0)
    builder.play("f1", waveforms.CosineDrag(t, 0.2, 0.0, 0.01))
    builder.build()

    device_name = "tianji_m2" 
    d = get_device(device_name)
    t = submit_task(
    circuit=qc,
    shots=8192,
    device=d,
    enable_qos_gate_decomposition=False,
    enable_qos_qubit_mapping=False,
    )
    rf = t.results()
    return rf

data = {
    '00': [],
    '01': [],
    '10': [],
    '11': []
}
for i in range(1,5):
    result = test_parametric_waveform(i)
    print(i, result['00'], result['01'], result['10'], result['11'])
    sum = result['00'] + result['01'] + result['10'] + result['11']
    sum00 = result['00'] + result['01']
    sum01 = result['10'] + result['11']
    
    sum10 = result['00'] + result['10']
    sum11 = result['01'] + result['11']
    
    data['00'].append(sum00/sum)
    data['01'].append(sum01/sum)
    
    data['10'].append(sum10/sum)
    data['11'].append(sum11/sum)


    # for key in data:
    #     data[key].append(result[key]/sum)
    

plt.figure(figsize=(10, 6))

colors = ['blue', 'green', 'red', 'purple']  
marker_size = 4  

# for idx, (state, values) in enumerate(data.items()):
#     plt.plot(
#         range(1, 5), 
#         values,
#         marker='o',        
#         markersize=marker_size,
#         linestyle='-',     
#         linewidth=1.2,
#         color=colors[idx],
#         label=state
#     )

# plt.xlabel('Parameter i')
# plt.ylabel('Counts')
# plt.title('State Distribution by Parameter')
# plt.legend()
# plt.grid(True, alpha=0.3)


# 子图
for idx, (state, values) in enumerate(data.items()):
    plt.subplot(2, 2, idx+1) 
    
    plt.plot(
        range(1, 5),
        values,
        marker='o',
        markersize=4,
        linestyle='-',
        linewidth=1.2,
        color=colors[idx]
    )
    
    plt.xlabel('Parameter i')
    plt.ylabel('Counts')
    plt.title(f'State {state}')
    plt.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()