"""
compilation utilities in tensorcircuit
"""

import tensorcircuit as tc


c = tc.Circuit(3)
c.rx(0, theta=0.2)
c.rz(0, theta=-0.3)
c.ry(1, theta=0.1)
c.h(2)
c.cx(0, 1)
c.cz(2, 1)
c.x(0)
c.y(0)
c.rxx(1, 2, theta=1.7)


c0, _ = tc.compiler.qiskit_compiler.qiskit_compile(
    c,
    compiled_options={"optimization_level": 0, "basis_gates": ["cx", "cz", "h", "rz"]},
)

c1, _ = tc.compiler.qiskit_compiler.qiskit_compile(
    c,
    compiled_options={"optimization_level": 1, "basis_gates": ["cx", "cz", "h", "rz"]},
)


c2, _ = tc.compiler.qiskit_compiler.qiskit_compile(
    c,
    compiled_options={"optimization_level": 2, "basis_gates": ["cx", "cz", "h", "rz"]},
)


c3, _ = tc.compiler.qiskit_compiler.qiskit_compile(
    c,
    compiled_options={"optimization_level": 3, "basis_gates": ["cx", "cz", "h", "rz"]},
)

print(
    "qiskit can become worse with higher level optimization when the target gate is not U3 but rz"
)
print("level 0:\n")
print(c0.draw())
print("level 1:\n")
print(c1.draw())
print("level 2:\n")
print(c2.draw())
print("level 3:\n")
print(c3.draw())


compiler_wo_mapping = tc.compiler.DefaultCompiler()
c4, _ = compiler_wo_mapping(c)
print(
    "compiled with tc default compiler: combining the good from qiskit and our tc own"
)
# we always uuggest using DefaultCompiler for tasks on qcloud
# internally we run optimized compiling using U3 basis with qiskit which has good performance
# and we unroll u3 with rz and apply replace/prune/merge loop developed in tc to further optimize the circuit
print(c4.draw())

print("gate number comparison (last ours vs before qiskit (0, 1, 2, 3))")
for c in [c0, c1, c2, c3, c4]:
    print(c.gate_count())

# if we want to apply routing/qubit mapping

compiler_w_mapping = tc.compiler.DefaultCompiler(
    {"coupling_map": [[0, 2], [2, 0], [1, 0], [0, 1]]}
)
c5, info = compiler_w_mapping(c)
print("circuit with qubit mapping")
print(c5.draw())
print(info)
