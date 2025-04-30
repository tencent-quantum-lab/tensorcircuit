import sys
import tensorflow as tf
import cirq
import sympy
import numpy as np
import time
import datetime
import cpuinfo
import os
import uuid
import utils


def tfquantum_benchmark(
    uuid, nwires, nlayer, nitrs, timeLimit, isgpu, check_ps=False, minus=1
):
    meta = {}

    if isgpu == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        meta["isgpu"] = "off"

    else:
        gpu = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)
        meta["isgpu"] = "on"
        meta["Gpuinfo"] = utils.gpuinfo()

    import tensorflow_quantum as tfq

    meta["Software"] = "tensorflow quantum"
    meta["minus"] = minus
    meta["Cpuinfo"] = cpuinfo.get_cpu_info()["brand_raw"]
    meta["Version"] = {
        "sys": sys.version,
        "cirq": cirq.__version__,
        "tensorflow": tf.__version__,
        "tensorflow_quantum": tfq.__version__,
        "numpy": np.__version__,
    }
    meta["VQE test parameters"] = {
        "nQubits": nwires,
        "nlayer": nlayer,
        "nitrs": nitrs,
        "timeLimit": timeLimit,
    }
    meta["UUID"] = uuid
    meta["Benchmark Time"] = (
        datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    )
    meta["Results"] = {}
    qubits = [cirq.GridQubit(0, i) for i in range(nwires)]

    # using differentiable op interface
    my_op = tfq.get_expectation_op()
    adjoint_differentiator = tfq.differentiators.Adjoint()
    op = adjoint_differentiator.generate_differentiable_op(analytic_op=my_op)
    my_symbol = sympy.symbols("params_0:" + str(nlayer * nwires * 2))
    circuit = cirq.Circuit()
    for i in range(nwires):
        circuit.append(cirq.H(qubits[i]))
    for j in range(nlayer):
        for i in range(nwires - minus):
            circuit.append(
                cirq.ZZPowGate(exponent=my_symbol[j * nwires * 2 + i])(
                    qubits[i], qubits[(i + 1) % nwires]
                )
            )
        for i in range(nwires):
            circuit.append(cirq.rx(my_symbol[j * nwires * 2 + nwires + i])(qubits[i]))

    circuit = tfq.convert_to_tensor([circuit])
    psums = tfq.convert_to_tensor(
        [
            [
                sum(
                    [
                        cirq.Z(qubits[i]) * cirq.Z(qubits[(i + 1) % nwires])
                        for i in range(nwires - 1)
                    ]
                    + [-1.0 * cirq.X(qubits[i]) for i in range(nwires)]
                )
            ]
        ]
    )
    symbol_values = np.array(
        [np.random.normal(size=[nlayer * nwires * 2])], dtype=np.float32
    )
    # Calculate tfq gradient.
    symbol_values_t = tf.Variable(tf.convert_to_tensor(symbol_values))
    symbol_names = tf.convert_to_tensor(
        ["params_" + str(i) for i in range(nwires * 2 * nlayer)]
    )

    opt_tf = tf.keras.optimizers.Adam(learning_rate=0.1)

    @tf.function
    def f():
        with tf.GradientTape() as g:
            g.watch(symbol_values_t)
            expectations = op(circuit, symbol_names, symbol_values_t, psums)
        grads = g.gradient(expectations, [symbol_values_t])
        # opt_tf.apply_gradients(zip(grads, [symbol_values_t]))
        return expectations

    ct, it, Nitrs = utils.timing(f, nitrs, timeLimit)
    meta["Results"]["differentiable operation"] = {
        "Construction time": ct,
        "Iteration time": it,
        "# of actual iterations": Nitrs,
    }

    # using keras layer interface
    qubits = [cirq.GridQubit(0, i) for i in range(nwires)]
    my_symbol = sympy.symbols("params_0:" + str(nwires * 2 * nlayer))
    circuit = cirq.Circuit()
    for i in range(nwires):
        circuit.append(cirq.H(qubits[i]))
    for j in range(nlayer):
        for i in range(nwires - minus):
            circuit.append(
                cirq.ZZPowGate(exponent=my_symbol[j * nwires * 2 + i])(
                    qubits[i], qubits[(i + 1) % nwires]
                )
            )
        for i in range(nwires):
            circuit.append(cirq.rx(my_symbol[j * nwires * 2 + nwires + i])(qubits[i]))
    symbol_values = np.array(
        [np.random.normal(size=[nlayer * nwires * 2])], dtype=np.float32
    )
    # Calculate tfq gradient.
    symbol_values_t = tf.Variable(tf.convert_to_tensor(symbol_values))
    symbol_names = tf.convert_to_tensor(
        ["params_" + str(i) for i in range(nwires * 2 * nlayer)]
    )

    ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    oprs = [
        sum(
            [
                cirq.Z(qubits[i]) * cirq.Z(qubits[(i + 1) % nwires])
                for i in range(nwires - 1)
            ]
            + [-cirq.X(qubits[i]) for i in range(nwires)]
        )
    ]
    ep = tfq.layers.Expectation(
        dtype=tf.float32  # , differentiator=tfq.differentiators.ParameterShift()
    )
    opt_tf = tf.keras.optimizers.Adam(learning_rate=0.1)
    # tfq.convert_to_tensor is very time consuming, thus shall be done out of iteration loops
    cc = tfq.convert_to_tensor([circuit])

    @tf.function
    def f():
        with tf.GradientTape() as g:
            g.watch(symbol_values_t)
            expectations = ep(
                cc,
                symbol_names=symbol_names,
                symbol_values=symbol_values_t,
                operators=oprs,
            )
        grads = g.gradient(expectations, [symbol_values_t])
        # opt_tf.apply_gradients(zip(grads, [symbol_values_t]))
        return expectations

    ct, it, Nitrs = utils.timing(f, nitrs, timeLimit)
    meta["Results"]["keras layer & adjoint"] = {
        "Construction time": ct,
        "Iteration time": it,
        "# of actual iterations": Nitrs,
    }
    if check_ps:
        ep = tfq.layers.Expectation(
            dtype=tf.float32, differentiator=tfq.differentiators.ParameterShift()
        )
        symbol_values_t.assign(
            np.array([np.random.normal(size=[nlayer * nwires * 2])], dtype=np.float32)
        )

        @tf.function
        def f():
            with tf.GradientTape() as g:
                g.watch(symbol_values_t)
                expectations = ep(
                    cc,
                    symbol_names=symbol_names,
                    symbol_values=symbol_values_t,
                    operators=oprs,
                )
            grads = g.gradient(expectations, [symbol_values_t])
            # opt_tf.apply_gradients(zip(grads, [symbol_values_t]))
            return expectations

        ct, it, Nitrs = utils.timing(f, nitrs, timeLimit)
        meta["Results"]["keras layer & Parameter shift"] = {
            "Construction time": ct,
            "Iteration time": it,
            "# of actual iterations": Nitrs,
        }
    print(meta)
    return meta


if __name__ == "__main__":
    _uuid = str(uuid.uuid4())
    n, nlayer, nitrs, timeLimit, isgpu, minus, path, check = utils.arg(check=True)
    if check == 1:
        checkbool = True
    else:
        checkbool = False
    results = tfquantum_benchmark(
        _uuid, n, nlayer, nitrs, timeLimit, isgpu, check_ps=checkbool, minus=minus
    )
    utils.save(results, _uuid, path)
