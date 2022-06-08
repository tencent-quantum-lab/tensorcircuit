import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np

nwires, nlayers = 6, 3
qubits = [cirq.GridQubit(0, i) for i in range(nwires)]
symbols = sympy.symbols("params_0:" + str(nlayers * nwires * 2))

circuit = cirq.Circuit()
for i in range(nwires):
    circuit.append(cirq.H(qubits[i]))
for j in range(nlayers):
    for i in range(nwires - 1):
        circuit.append(
            cirq.ZZPowGate(exponent=symbols[j * nwires * 2 + i])(
                qubits[i], qubits[(i + 1)]
            )
        )
    for i in range(nwires):
        circuit.append(cirq.rx(symbols[j * nwires * 2 + nwires + i])(qubits[i]))

circuit = tfq.convert_to_tensor([circuit])

hamiltonian = tfq.convert_to_tensor(
    [
        [
            sum(
                [cirq.Z(qubits[i]) * cirq.Z(qubits[i + 1]) for i in range(nwires - 1)]
                + [-1.0 * cirq.X(qubits[i]) for i in range(nwires)]
            )
        ]
    ]
)

ep = tfq.layers.Expectation()


@tf.function
def tf_vg(symbol_values):
    with tf.GradientTape() as g:
        g.watch(symbol_values)
        expectations = ep(
            circuit,
            symbol_names=symbols,
            symbol_values=symbol_values,
            operators=hamiltonian,
        )
    grads = g.gradient(expectations, [symbol_values])
    return expectations, grads


symbol_values = [np.random.normal(size=[nlayers * nwires * 2]).astype(np.float32)]
symbol_values = tf.Variable(tf.convert_to_tensor(symbol_values))
print(tf_vg(symbol_values))
