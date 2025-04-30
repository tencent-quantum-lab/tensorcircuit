import pennylane as qml
import tensorflow as tf

nwires, nlayers = 6, 3

coeffs = [-1.0] * nwires + [1.0] * (nwires - 1)
obs = [qml.PauliX(i) for i in range(nwires)] + [
    qml.PauliZ(i) @ qml.PauliZ((i + 1) % nwires) for i in range(nwires - 1)
]
Htfim = qml.Hamiltonian(coeffs, obs, True)

dev = qml.device("default.qubit.tf", wires=nwires)


@tf.function
@qml.qnode(dev, interface="tf")
def tf_expval(params):
    for i in range(nwires):
        qml.Hadamard(wires=i)
    for j in range(nlayers):
        for i in range(nwires - 1):
            qml.IsingZZ(params[i + j * 2 * nwires], wires=[i, i + 1])
        for i in range(nwires):
            qml.RX(params[nwires + i + j * 2 * nwires], wires=i)
    return qml.expval(Htfim)


params = tf.random.normal(shape=[nlayers * 2 * nwires])


@tf.function
def tf_vg(params):
    with tf.GradientTape() as t:
        t.watch(params)
        e = tf_expval(params)
    grad = t.gradient(e, [params])
    return e, grad


print(tf_vg(params))
