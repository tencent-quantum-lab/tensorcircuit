import sys
import os
import pytest

# from pytest_lazyfixture import lazy_fixture as lf


thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


def test_qsikit_compiler():
    try:
        import qiskit as _
    except ImportError:
        pytest.skip("qiskit is not installed")

    from tensorcircuit.compiler.qiskit_compiler import qiskit_compile

    c = tc.Circuit(2)
    c.x(1)
    c.cx(0, 1)

    c1, info = qiskit_compile(
        c,
        info=None,
        output="qasm",
        compiled_options={
            "basis_gates": ["cz", "rz", "h"],
            "optimization_level": 3,
            "coupling_map": [[0, 2], [2, 0], [1, 0], [0, 1]],
        },
    )
    assert "cz" in c1
    print(info["logical_physical_mapping"])

    c = tc.Circuit(2)
    c.x(1)
    c.cx(0, 1)
    c.measure_instruction(1)
    c1, info = qiskit_compile(
        c,
        info=None,
        output="tc",
        compiled_options={
            "basis_gates": ["cx", "rz", "h"],
            "optimization_level": 3,
            "coupling_map": [[0, 2], [2, 0], [1, 0], [0, 1]],
            "initial_layout": [1, 2],
        },
    )
    for inst in c1.to_qir():
        if inst["name"] == "h":
            assert inst["index"][0] == 2
    print(c1.draw())
    assert info["logical_physical_mapping"][1] in [0, 2]
    print(info)
    c2, info2 = qiskit_compile(
        c1,
        info=info,
        output="tc",
        compiled_options={
            "basis_gates": ["cx", "rz", "h"],
            "optimization_level": 3,
            "coupling_map": [[0, 2], [2, 0], [1, 0], [0, 1]],
        },
    )
    assert info2["positional_logical_mapping"] == {0: 1}
    print(c2.draw())


def test_composed_compiler():
    from tensorcircuit.compiler import default_compiler

    c = tc.Circuit(3)
    c.rx(0)
    c.cx(0, 1)
    c.cz(1, 0)
    c.rxx(0, 2, theta=0.2)
    c.measure_instruction(2)
    c.measure_instruction(0)
    c1, info = default_compiler(c)
    print(c1.draw())
    assert c1.gate_count_by_condition(lambda qir: qir["name"] == "cnot") == 4
    assert info["positional_logical_mapping"][0] == 2

    default_compiler.add_options(
        {
            "basis_gates": ["h", "rz", "cz"],
            "optimization_level": 2,
            "coupling_map": [[0, 1], [1, 2]],
        }
    )
    c1, info = default_compiler(c)
    assert c1.gate_count_by_condition(lambda qir: qir["name"] == "cnot") == 0
    print(info)
