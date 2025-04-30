import sys
import os

# Add the directory containing your module to Python's search path
module_path = ".."
sys.path.insert(0, module_path)

from tensorcircuit import Circuit, Param, gates, waveforms
from tensorcircuit.cloud.apis import submit_task, get_device, set_provider, set_token, list_devices
import re

print("✅ TEST FILE LOADED")
set_token("xu1LTrkf0nP6sI8oh.bDPdk35RlOQZYy9hQPU6jK2J4d5AdINAOszCNPxTNGZ3-opBPhWcLcruYuSrvX8is1D9tKgw-O4Zg.Qf7fLp83AtSPP19jD6Na4piICkygomdfyxIjzhO6Zu-s5hgBu2709ZW=")
set_provider("tencent")
ds = list_devices()
print(ds)

def test_parametric_waveform():
    qc = Circuit(2)

    param0 = Param("param0")
    param1 = Param("param1")

    builder = qc.calibrate("my_gate", [param0, param1])
    builder.new_frame("f0", param0)
    builder.play("f0", waveforms.CosineDrag(10, 0.2, "0.5*PI", 0.01))
    builder.new_frame("f1", param1)
    builder.play("f1", waveforms.CosineDrag(20, 0.01, "0", 0))
    builder.build()

    tqasm_code = qc.to_tqasm()
    tqasm_code = tqasm_code + '\nmygate q[0], q[1]'
    print(tqasm_code)  

    assert "TQASM 0.2;" in tqasm_code
    assert "defcal my_gate param0, param1" in tqasm_code
    assert "frame f0 = newframe(param0);" in tqasm_code
    assert "play(f0, cosine_drag(10, 0.2, 0.5*PI, 0.01));" in tqasm_code
    assert "frame f1 = newframe(param1);" in tqasm_code
    assert "play(f1, cosine_drag(20, 0.01, 0, 0));" in tqasm_code
    assert re.search(r"defcal my_gate [^\)]* \s*\{", tqasm_code)
    
    
    #tc.cloud.apis.set_token("xu1LTrkf0nP6sI8oh.bDPdk35RlOQZYy9hQPU6jK2J4d5AdINAOszCNPxTNGZ3-opBPhWcLcruYuSrvX8is1D9tKgw-O4Zg.Qf7fLp83AtSPP19jD6Na4piICkygomdfyxIjzhO6Zu-s5hgBu2709ZW=")
    #tc.cloud.apis.set_provider("tencent")
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
    print(rf)


test_parametric_waveform()
