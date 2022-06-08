import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np

nwires, nlayers, nbatch = 6, 3, 16
qubits = [cirq.GridQubit(0, i) for i in range(nwires)]
my_symbol = sympy.symbols("params_0:" + str(nwires * 2 * nlayers))
model_circuit = cirq.Circuit()

for j in range(nlayers):
    for i in range(nwires - 1):
        model_circuit.append(
            cirq.ZZPowGate(exponent=my_symbol[j * nwires * 2 + i])(
                qubits[i], qubits[nwires - 1]
            )
        )
    for i in range(nwires):
        model_circuit.append(cirq.rx(my_symbol[j * nwires * 2 + nwires + i])(qubits[i]))

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        tfq.layers.PQC(model_circuit, cirq.Z(qubits[nwires - 1]) * 0.5 + 0.5),
    ]
)


def img2circuit(img):
    circuit = cirq.Circuit()
    for i in range(nwires - 1):
        circuit.append(cirq.rx(img[i])(qubits[i]))
    return circuit


img = np.random.normal(size=[nbatch, nwires]).astype(np.float32)
img = tfq.convert_to_tensor([img2circuit(x) for x in img])

print(model(img))
