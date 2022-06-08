import sys
import pennylane as qml
import tensorflow as tf
import numpy as np
import cpuinfo
import datetime
import os
import jax
import uuid
import utils


def H_tfim(J, h, nwires, bc=True):
    if bc:
        coeffs = [h] * nwires + [J] * nwires
        obs = [qml.PauliX(i) for i in range(nwires)] + [
            qml.PauliZ(i) @ qml.PauliZ((i + 1) % nwires) for i in range(nwires)
        ]
    else:
        coeffs = [h] * nwires + [J] * (nwires - 1)
        obs = [qml.PauliX(i) for i in range(nwires)] + [
            qml.PauliZ(i) @ qml.PauliZ((i + 1) % nwires) for i in range(nwires - 1)
        ]
    return qml.Hamiltonian(coeffs, obs, True)


def pennylane_benchmark(uuid, nwires, nlayer, nitrs, timeLimit, isgpu, minus=1):
    meta = {}

    if isgpu == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        meta["isgpu"] = "off"

    else:
        gpu = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)
        meta["isgpu"] = "on"
        meta["Gpuinfo"] = utils.gpuinfo()

    meta["Software"] = "pennylane"
    meta["minus"] = minus
    meta["Cpuinfo"] = cpuinfo.get_cpu_info()["brand_raw"]
    meta["Version"] = {
        "sys": sys.version,
        "tensorflow": tf.__version__,
        "pennylane": qml.__version__,
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

    Htfim = H_tfim(1.0, -1.0, nwires, False)

    # def circuit(params, wires):
    #     for i in range(nwires):
    #         qml.Hadamard(wires=i)
    #     for j in range(nlayer):
    #         for i in range(nwires-1):
    #             qml.IsingZZ(params[2 * nwires * j + i], wires=[i, (i + 1) % nwires])
    #         for i in range(nwires):
    #             qml.RX(params[2 * nwires * j + nwires + i], wires=i)

    dev = qml.device("default.qubit.jax", wires=nwires)

    @jax.jit
    @qml.qnode(dev, interface="jax")
    def jax_expval(params):
        for i in range(nwires):
            qml.Hadamard(wires=i)
        for j in range(nlayer):
            for i in range(nwires - minus):
                qml.IsingZZ(params[i + j * 2 * nwires], wires=[i, (i + 1) % nwires])
            for i in range(nwires):
                qml.RX(params[nwires + i + j * 2 * nwires], wires=i)
        return qml.expval(Htfim)

    vag = jax.value_and_grad(jax_expval)
    params = jax.numpy.array(np.random.normal(size=[nlayer * 2 * nwires]))

    def f():
        return vag(params)[0].block_until_ready()

    ct, it, Nitrs = utils.timing(f, nitrs, timeLimit)
    meta["Results"]["jax_expval"] = {
        "Construction time": ct,
        "Iteration time": it,
        "# of actual iterations": Nitrs,
    }

    # tf device below

    dev = qml.device("default.qubit.tf", wires=nwires)

    @tf.function
    @qml.qnode(dev, interface="tf")
    def tf_expval(params):
        for i in range(nwires):
            qml.Hadamard(wires=i)
        for j in range(nlayer):
            for i in range(nwires - minus):
                qml.IsingZZ(params[i + j * 2 * nwires], wires=[i, (i + 1) % nwires])
            for i in range(nwires):
                qml.RX(params[nwires + i + j * 2 * nwires], wires=i)
        return qml.expval(Htfim)

    params = tf.Variable(np.random.normal(size=[nlayer * 2 * nwires]), dtype=tf.float32)

    def f():
        with tf.GradientTape() as t:
            t.watch(params)
            e = tf_expval(params)
        grad = t.gradient(e, [params])
        # opt_tf.apply_gradients(zip(grad, [params]))
        return e

    ct, it, Nitrs = utils.timing(f, nitrs, timeLimit)
    meta["Results"]["tf_expval"] = {
        "Construction time": ct,
        "Iteration time": it,
        "# of actual iterations": Nitrs,
    }

    # using tf interface and expvalcost
    # params.assign(np.random.normal(size=[nlayer * 2 * nwires]))
    # cost_fn_tf = qml.ExpvalCost(circuit, Htfim, dev, interface="tf")

    # def f():
    #     with tf.GradientTape() as t:
    #         t.watch(params)
    #         e = cost_fn_tf(params)
    #     grad = t.gradient(e, [params])
    #     opt_tf.apply_gradients(zip(grad, [params]))

    # ct, it, Nitrs = utils.timing(f, nitrs, timeLimit)
    # meta["Results"]["tf_costfn"] = {
    #     "Construction time": ct,
    #     "Iteration time": it,
    #     "# of actual iterations": Nitrs,
    # }
    print(meta)
    return meta


if __name__ == "__main__":
    _uuid = str(uuid.uuid4())
    n, nlayer, nitrs, timeLimit, isgpu, minus, path = utils.arg()
    results = pennylane_benchmark(_uuid, n, nlayer, nitrs, timeLimit, isgpu, minus)
    utils.save(results, _uuid, path)
