import pennylane as qml
import tensorflow as tf

nwires, nlayers, nbatch = 6, 3, 16
dev = qml.device("default.qubit.tf", wires=nwires)


@tf.function
@qml.qnode(dev, interface="tf")
def yp(inputs, params):
    for i in range(nwires - 1):
        qml.RX(inputs[i], wires=i)
    for j in range(nlayers):
        for i in range(nwires - 1):
            qml.IsingZZ(params[i + j * 2 * nwires], wires=[i, nwires - 1])
        for i in range(nwires):
            qml.RX(params[nwires + i + j * 2 * nwires], wires=i)
    return qml.expval(qml.Hamiltonian([1.0], [qml.PauliZ(nwires - 1)], True))


model = qml.qnn.KerasLayer(yp, {"params": (nlayers * 2 * nwires)}, output_dim=1)

imgs = tf.random.normal([nbatch, nwires])

print(model(imgs))
