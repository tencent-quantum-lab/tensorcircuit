"""
benchmark on barren plateau using tfq and tc
"""

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import time
import tensorflow as tf
import cirq
import sympy
import numpy as np
import pennylane as qml
import tensorflow_quantum as tfq
import tensorcircuit as tc


def benchmark(f, *args, tries=3):
    time0 = time.time()
    f(*args)
    time1 = time.time()
    for _ in range(tries):
        print(f(*args))
    time2 = time.time()
    print("staging time: ", time1 - time0, "running time: ", (time2 - time1) / tries)


def tfq_approach(n_qubits=10, depth=10, n_circuits=100):
    """adopted from https://www.tensorflow.org/quantum/tutorials/barren_plateaus"""

    def generate_random_qnn(qubits, symbol, depth):
        circuit = cirq.Circuit()
        for qubit in qubits:
            circuit += cirq.ry(np.pi / 4.0)(qubit)
        j = 0
        for _ in range(depth):
            # Add a series of single qubit rotations.
            for qubit in qubits:
                random_n = np.random.uniform()
                if random_n > 2.0 / 3.0:
                    # Add a Z.
                    circuit += cirq.rz(symbol[j])(qubit)
                elif random_n > 1.0 / 3.0:
                    # Add a Y.
                    circuit += cirq.ry(symbol[j])(qubit)
                else:
                    # Add a X.
                    circuit += cirq.rx(symbol[j])(qubit)
                j += 1

            # Add CZ ladder.
            for src, dest in zip(qubits, qubits[1:]):
                circuit += cirq.CZ(src, dest)

        return circuit

    def process_batch(circuits, symbol, op):
        # Setup a simple layer to batch compute the expectation gradients.
        expectation = tfq.layers.Expectation()

        # Prep the inputs as tensors
        circuit_tensor = tfq.convert_to_tensor(circuits)
        values_tensor = tf.convert_to_tensor(
            np.random.uniform(0, 2 * np.pi, (n_circuits, 100)).astype(np.float32)
        )

        # Use TensorFlow GradientTape to track gradients.
        with tf.GradientTape() as g:
            g.watch(values_tensor)
            forward = expectation(
                circuit_tensor,
                operators=op,
                symbol_names=symbol,
                symbol_values=values_tensor,
            )

        # Return variance of gradients across all circuits.
        grads = g.gradient(forward, values_tensor)
        grad_var = tf.math.reduce_std(grads, axis=0)
        return grad_var.numpy()[0]

    qubits = cirq.GridQubit.rect(1, n_qubits)
    symbol = sympy.symbols("theta_{0:100}")
    circuits = [generate_random_qnn(qubits, symbol, depth) for _ in range(n_circuits)]
    op = cirq.Z(qubits[0]) * cirq.Z(qubits[1])
    theta_var = process_batch(circuits, symbol, op)
    return theta_var


benchmark(tfq_approach)


K = tc.set_backend("tensorflow")
Rx = tc.gates.rx
Ry = tc.gates.ry
Rz = tc.gates.rz


def op_expectation(params, seed, n_qubits, depth):
    paramsc = tc.backend.cast(params, dtype="float32")  # parameters of gates
    seedc = tc.backend.cast(seed, dtype="float32")
    # parameters of circuit structure

    c = tc.Circuit(n_qubits)
    for i in range(n_qubits):
        c.ry(i, theta=np.pi / 4)
    for l in range(depth):
        for i in range(n_qubits):
            c.unitary_kraus(
                [Rx(paramsc[i, l]), Ry(paramsc[i, l]), Rz(paramsc[i, l])],
                i,
                prob=[1 / 3, 1 / 3, 1 / 3],
                status=seedc[i, l],
            )
        for i in range(n_qubits - 1):
            c.cz(i, i + 1)

    return K.real(c.expectation_ps(z=[0, 1]))


op_expectation_vmap_vvag = K.jit(
    K.vvag(op_expectation, argnums=0, vectorized_argnums=(0, 1)),
    static_argnums=(2, 3),
)


def pennylane_approach(n_qubits=10, depth=10, n_circuits=100):
    dev = qml.device("lightning.qubit", wires=n_qubits)
    gate_set = [qml.RX, qml.RY, qml.RZ]

    @qml.qnode(dev)
    def rand_circuit(params, status):
        for i in range(n_qubits):
            qml.RY(np.pi / 4, wires=i)

        for j in range(depth):
            for i in range(n_qubits):
                gate_set[status[i, j]](params[j, i], wires=i)

            for i in range(n_qubits - 1):
                qml.CZ(wires=[i, i + 1])

        return qml.expval(qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(1)], True))

    gf = qml.grad(rand_circuit, argnum=0)
    params = np.random.uniform(0, 2 * np.pi, size=[n_circuits, depth, n_qubits])
    status = np.random.choice(3, size=[n_circuits, depth, n_qubits])

    g_results = []

    for i in range(n_circuits):
        g_results.append(gf(params[i], status[i]))

    g_results = np.stack(g_results)

    return np.std(g_results[:, 0, 0])


benchmark(pennylane_approach)


def tc_approach(n_qubits=10, depth=10, n_circuits=100):
    seed = tc.array_to_tensor(
        np.random.uniform(low=0.0, high=1.0, size=[n_circuits, n_qubits, depth]),
        dtype="float32",
    )
    params = tc.array_to_tensor(
        np.random.uniform(low=0.0, high=2 * np.pi, size=[n_circuits, n_qubits, depth]),
        dtype="float32",
    )

    _, grad = op_expectation_vmap_vvag(params, seed, n_qubits, depth)
    # the gradient variance of the first parameter
    grad_var = tf.math.reduce_std(K.numpy(grad), axis=0)

    return grad_var[0, 0]


benchmark(tc_approach)
