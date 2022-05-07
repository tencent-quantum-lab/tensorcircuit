"""
benchmark on barren plateau using tfq and tc
"""
import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import time
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
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

        for d in range(depth):
            # Add a series of single qubit rotations.
            for i, qubit in enumerate(qubits):
                random_n = np.random.uniform()
                random_rot = (
                    np.random.uniform() * 2.0 * np.pi if i != 0 or d != 0 else symbol
                )
                if random_n > 2.0 / 3.0:
                    # Add a Z.
                    circuit += cirq.rz(random_rot)(qubit)
                elif random_n > 1.0 / 3.0:
                    # Add a Y.
                    circuit += cirq.ry(random_rot)(qubit)
                else:
                    # Add a X.
                    circuit += cirq.rx(random_rot)(qubit)

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
            np.random.uniform(0, 2 * np.pi, (n_circuits, 1)).astype(np.float32)
        )

        # Use TensorFlow GradientTape to track gradients.
        with tf.GradientTape() as g:
            g.watch(values_tensor)
            forward = expectation(
                circuit_tensor,
                operators=op,
                symbol_names=[symbol],
                symbol_values=values_tensor,
            )

        # Return variance of gradients across all circuits.
        grads = g.gradient(forward, values_tensor)
        grad_var = tf.math.reduce_std(grads, axis=0)
        return grad_var.numpy()[0]

    qubits = cirq.GridQubit.rect(1, n_qubits)
    symbol = sympy.Symbol("theta")
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

# tfq mac cpu: 4.43
# jax mac cpu: 98.25, 0.078
# tf mac cpu: 132.93, 0.21
# tfq t4 cpu: 3.97
# tf mac gpu T4: 158, 0.15
# jax mac gpu T4: 53.8, 0.022
