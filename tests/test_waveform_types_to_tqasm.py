import sys
import os

# Add the directory containing your module to Python's search path
module_path = "/Users/mac/tensorcircuit"
sys.path.insert(0, module_path)

from tensorcircuit import Circuit, Param, waveforms

def test_waveform_types_to_tqasm_():
    test_cases = [
        (waveforms.CosineDrag, [10, 0.2, "PI/2", 0.01], "cosine_drag"),
        (waveforms.Gaussian, [0.5, 20, 5], "gaussian"),
        (waveforms.GaussianSquare, [0.5, 20, 5, 10], "gaussian_square"),
        (waveforms.Drag, [0.4, 30, 10, 0.2], "drag"),
        (waveforms.Flattop, [1.0, 5, 20], "flattop"),
        (waveforms.Constant, [0.7, 15], "constant"),
        (waveforms.Sine, [0.3, "2*PI", 10], "sine"),
        (waveforms.Cosine, [0.6, "PI", 8], "cosine"),
    ]

    failed = False

    for waveform_class, args, expected_name in test_cases:
        qc = Circuit(1)
        p = Param("q0")

        builder = qc.calibrate("test_gate", [p])
        builder.new_frame("f0", p)
        builder.play("f0", waveform_class(*args))
        builder.build()

        qasm = qc.to_tqasm()
        print(f"--- {expected_name} ---\n{qasm}\n")

        if f"play(f0, {expected_name}(" not in qasm:
            print(f"❌ FAILED: waveform {waveform_class.__name__} expected '{expected_name}', but not found in QASM output.")
            failed = True
        else:
            print(f"✅ PASSED: waveform {expected_name}")

    if failed:
        raise AssertionError("Some waveform QASM outputs were incorrect.")
    else:
        print("All waveform QASM outputs passed.")

if __name__ == "__main__":
    test_waveform_types_to_tqasm_()