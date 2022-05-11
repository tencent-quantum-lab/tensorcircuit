import sys
import tensorflow as tf
import cirq
import sympy
import numpy as np
import datetime
import cpuinfo
import os
import uuid
import utils


def tfquantum_benchmark(
    uuid,
    nwires,
    nlayer,
    nitrs,
    timeLimit,
    isgpu,
    train_img,
    test_img,
    train_lbl,
    test_lbl,
    nbatch,
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
    meta["Cpuinfo"] = cpuinfo.get_cpu_info()["brand_raw"]
    meta["Version"] = {
        "sys": sys.version,
        "cirq": cirq.__version__,
        "tensorflow": tf.__version__,
        "tensorflow_quantum": tfq.__version__,
        "numpy": np.__version__,
    }
    meta["QML test parameters"] = {
        "nQubits": nwires,
        "nlayer": nlayer,
        "nitrs": nitrs,
        "timeLimit": timeLimit,
        "nbatch": nbatch,
    }
    meta["UUID"] = uuid
    meta["Benchmark Time"] = (
        datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    )
    meta["Results"] = {}
    qubits = [cirq.GridQubit(0, i) for i in range(nwires)]

    my_symbol = sympy.symbols("params_0:" + str(nwires * 2 * nlayer))
    model_circuit = cirq.Circuit()
    for j in range(nlayer):
        for i in range(nwires - 1):
            model_circuit.append(
                cirq.ZZPowGate(exponent=my_symbol[j * nwires * 2 + i])(
                    qubits[i], qubits[nwires - 1]
                )
            )
        for i in range(nwires):
            model_circuit.append(
                cirq.rx(my_symbol[j * nwires * 2 + nwires + i])(qubits[i])
            )
    model = tf.keras.Sequential(
        [
            # The input is the data-circuit, encoded as a tf.string
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            # The PQC layer returns the expected value of the readout gate, range [0,1].
            tfq.layers.PQC(model_circuit, cirq.Z(qubits[nwires - 1]) * 0.5 + 0.5),
        ]
    )

    def img2circuit(img):
        circuit = cirq.Circuit()
        for i in range(nwires - 1):
            circuit.append(cirq.rx(img[i] * np.pi)(qubits[i]))
        return circuit

    utils.qml_data["train_img_tfq"] = tfq.convert_to_tensor(
        [img2circuit(x) for x in train_img]
    )
    print("img2circuit completed!")
    opt = tf.keras.optimizers.Adam(0.1)

    def vag_tfq(imgs, lbls):
        with tf.GradientTape() as t:
            t.watch(model.trainable_variables)
            loss = tf.reduce_mean((tf.reshape(model(imgs), [-1]) - lbls) ** 2)

        grad = t.gradient(loss, model.trainable_variables)
        return loss, grad

    def f(train_imgs, train_lbls):
        loss, grad = vag_tfq(train_imgs, train_lbls)
        opt.apply_gradients(zip(grad, model.trainable_variables))
        return loss

    ct, it, Nitrs = utils.qml_timing(f, nbatch, nitrs, timeLimit, tfq=True)
    meta["Results"]["keras layer"] = {
        "Construction time": ct,
        "Iteration time": it,
        "# of actual iterations": Nitrs,
    }

    # @tf.function
    # def vag_tfq(imgs, lbls):
    #     with tf.GradientTape() as t:
    #         t.watch(model.trainable_variables)
    #         loss = tf.reduce_mean((tf.reshape(model(imgs), [-1]) - lbls) ** 2)

    #     grad = t.gradient(loss, model.trainable_variables)
    #     return loss, grad

    # def f(train_imgs, train_lbls):
    #     loss, grad = vag_tfq(train_imgs, train_lbls)
    #     opt.apply_gradients(zip(grad, model.trainable_variables))
    #     return loss

    # ct, it, Nitrs = utils.qml_timing(f, nbatch, nitrs, timeLimit, tfq=True)
    # meta["Results"]["keras layer & tf function"] = {
    #     "Construction time": ct,
    #     "Iteration time": it,
    #     "# of actual iterations": Nitrs,
    # }

    print(meta)
    return meta


if __name__ == "__main__":
    _uuid = str(uuid.uuid4())
    n, nlayer, nitrs, timeLimit, isgpu, minus, path, nbatch = utils.arg(qml=True)

    train_img, test_img, train_lbl, test_lbl = utils.mnist_data_preprocessing(n - 1)
    results = tfquantum_benchmark(
        _uuid,
        n,
        nlayer,
        nitrs,
        timeLimit,
        isgpu,
        train_img,
        test_img,
        train_lbl,
        test_lbl,
        nbatch,
    )
    utils.save(results, _uuid, path)
