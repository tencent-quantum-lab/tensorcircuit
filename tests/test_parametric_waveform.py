import sys
import os

# Add the directory containing your module to Python's search path
module_path = "/Users/mac/tensorcircuit"
sys.path.insert(0, module_path)

from tensorcircuit import Circuit, Param, gates, waveforms
from tensorcircuit.cloud.apis import submit_task, get_device
import re

print("✅ TEST FILE LOADED")

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
    print(tqasm_code)  

    assert "OPENQASM 3;" in tqasm_code
    assert "defcal my_gate(param0, param1)" in tqasm_code
    assert "frame f0 = newframe(param0);" in tqasm_code
    assert "play(f0, cosine_drag(10, 0.2, 0.5*PI, 0.01));" in tqasm_code
    assert "frame f1 = newframe(param1);" in tqasm_code
    assert "play(f1, cosine_drag(20, 0.01, 0, 0));" in tqasm_code
    assert re.search(r"defcal my_gate\([^\)]*\)\s*\{", tqasm_code)
    
    return 


    device_name = "tianxuan_s1" 
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
