import sys
import pennylane as qml
import tensorflow as tf
import numpy as np
import cpuinfo
import datetime
import os
import jax
import uuid
import tensorcircuit as tc
import utils


def pennylane_benchmark(
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
    check_loop,
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

    meta["Software"] = "pennylane"
    meta["Cpuinfo"] = cpuinfo.get_cpu_info()["brand_raw"]
    meta["Version"] = {
        "sys": sys.version,
        "tensorflow": tf.__version__,
        "pennylane": qml.__version__,
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

    dev = qml.device("lightning.qubit", wires=nwires)

    K = tc.get_backend("tensorflow")

    @qml.qnode(dev, diff_method="adjoint", interface="tf")
    def lt_expval(img, params):
        for i in range(nwires - 1):
            qml.RX(img[i] * np.pi, wires=i)
        for j in range(nlayer):
            for i in range(nwires - 1):
                qml.IsingZZ(params[i + j * 2 * nwires], wires=[i, nwires - 1])
            for i in range(nwires):
                qml.RX(params[nwires + i + j * 2 * nwires], wires=i)
        return qml.expval(qml.Hamiltonian([1.0], [qml.PauliZ(nwires - 1)], True))

    def loss(imgs, lbls, param):
        params = K.stack([param for _ in range(lbls.shape[0])])
        params = K.transpose(params, [1, 0])
        return K.mean(
            (lbls - K.cast(lt_expval(imgs, params), "float32") * 0.5 - 0.5) ** 2
        )

    vag = K.value_and_grad(loss, argnums=2)
    param = K.convert_to_tensor((np.random.normal(size=[nlayer * 2 * nwires])))

    def f(train_imgs, train_lbls):
        e, grad = vag(
            K.transpose(
                K.convert_to_tensor(np.array(train_imgs).astype(np.float32)), (1, 0)
            ),
            K.reshape(
                K.convert_to_tensor(np.array(train_lbls).astype(np.float32)), (-1)
            ),
            param,
        )
        return e

    ct, it, Nitrs = utils.qml_timing(f, nbatch, nitrs, timeLimit)
    meta["Results"]["lightning"] = {
        "Construction time": ct,
        "Iteration time": it,
        "# of actual iterations": Nitrs,
    }

    print(meta["Results"]["lightning"])

    dev = qml.device("default.qubit.jax", wires=nwires)

    @qml.qnode(dev, interface="jax")
    def jax_expval(img, params):
        for i in range(nwires - 1):
            qml.RX(img[i] * np.pi, wires=i)
        for j in range(nlayer):
            for i in range(nwires - 1):
                qml.IsingZZ(params[i + j * 2 * nwires], wires=[i, nwires - 1])
            for i in range(nwires):
                qml.RX(params[nwires + i + j * 2 * nwires], wires=i)
        return qml.expval(qml.Hamiltonian([1.0], [qml.PauliZ(nwires - 1)], True))

    def loss(img, lbl, params):
        return (lbl - jax_expval(img, params) * 0.5 - 0.5) ** 2

    vag = jax.value_and_grad(loss, argnums=(2,))
    vag = jax.jit(vag)
    params = jax.numpy.array(np.random.normal(size=[nlayer * 2 * nwires]))

    def f(train_imgs, train_lbls):
        e = []
        for i in range(len(train_imgs)):
            ee, grad = vag(train_imgs[i], train_lbls[i], params)
            e.append(ee)
        return np.mean(e)

    if check_loop:
        ct, it, Nitrs = utils.qml_timing(f, nbatch, nitrs, timeLimit)
        meta["Results"]["jax_loop"] = {
            "Construction time": ct,
            "Iteration time": it,
            "# of actual iterations": Nitrs,
        }
    _vloss = jax.vmap(loss, (0, 0, None), 0)

    def vloss(img, lbl, params):
        return jax.numpy.mean(_vloss(img, lbl, params))

    vag = jax.value_and_grad(vloss, argnums=(2,))
    params = jax.numpy.array(np.random.normal(size=[nlayer * 2 * nwires]))

    # def f(train_imgs, train_lbls):
    #     e, grad = vag(jax.numpy.array(train_imgs), jax.numpy.array(train_lbls), params)
    #     return e

    # ct, it, Nitrs = utils.qml_timing(f, nbatch, nitrs, timeLimit)
    # meta["Results"]["jax_vmap"] = {
    #     "Construction time": ct,
    #     "Iteration time": it,
    #     "# of actual iterations": Nitrs,
    # }
    vag = jax.jit(vag)

    def f(train_imgs, train_lbls):
        e, grad = vag(jax.numpy.array(train_imgs), jax.numpy.array(train_lbls), params)
        return e

    ct, it, Nitrs = utils.qml_timing(f, nbatch, nitrs, timeLimit)
    meta["Results"]["jax_vmap"] = {
        "Construction time": ct,
        "Iteration time": it,
        "# of actual iterations": Nitrs,
    }

    print(meta)  # in case OOM in tf case

    # tf device below

    dev = qml.device("default.qubit.tf", wires=nwires)

    @qml.qnode(dev, interface="tf")
    def tf_value(img, params):
        for i in range(nwires - 1):
            qml.RX(img[i] * np.pi, wires=i)
        for j in range(nlayer):
            for i in range(nwires - 1):
                qml.IsingZZ(params[i + j * 2 * nwires], wires=[i, nwires - 1])
            for i in range(nwires):
                qml.RX(params[nwires + i + j * 2 * nwires], wires=i)
        return qml.expval(qml.Hamiltonian([1.0], [qml.PauliZ(nwires - 1)], True))

    @tf.function
    def tf_vag(img, lbl, params):
        with tf.GradientTape() as t:
            t.watch(params)
            loss = (tf_value(img, params) * 0.5 + 0.5 - lbl) ** 2
        return loss, t.gradient(loss, params)

    params = tf.Variable(np.random.normal(size=[nlayer * 2 * nwires]), dtype=tf.float64)
    a = tf.Variable(train_img[0])
    b = tf.Variable(train_lbl[0], dtype=tf.float64)
    opt = tf.keras.optimizers.Adam(0.01)

    def f(train_imgs, train_lbls):
        loss_ = []
        grad_ = []
        for i in range(len(train_imgs)):
            a.assign(train_imgs[i])
            b.assign(train_lbls[i])
            loss, grad = tf_vag(a, b, params)
            loss_.append(loss)
            grad_.append(grad)
        opt.apply_gradients(zip([tf.reduce_mean(grad_, axis=0)], [params]))
        return np.mean(loss_)

    if check_loop:
        ct, it, Nitrs = utils.qml_timing(f, nbatch, nitrs, timeLimit)
        meta["Results"]["tf_loop"] = {
            "Construction time": ct,
            "Iteration time": it,
            "# of actual iterations": Nitrs,
        }

    a = tf.Variable(train_img[:nbatch], dtype=tf.float64)
    b = tf.Variable(train_lbl[:nbatch], dtype=tf.float64)

    opt = tf.keras.optimizers.Adam(0.01)

    @tf.function
    @qml.qnode(dev, interface="tf")
    def tfvalue(img, lbl, params):
        for i in range(nwires - 1):
            qml.RX(img[i] * np.pi, wires=i)
        for j in range(nlayer):
            for i in range(nwires - 1):
                qml.IsingZZ(params[i + j * 2 * nwires], wires=[i, nwires - 1])
            for i in range(nwires):
                qml.RX(params[nwires + i + j * 2 * nwires], wires=i)
        return qml.expval(qml.Hamiltonian([1.0], [qml.PauliZ(nwires - 1)], True))

    def tf_loss(img, lbl, params):
        loss = (tf_value(img, params) * 0.5 + 0.5 - lbl) ** 2
        return loss

    tf_vvag = tf.function(
        tc.get_backend("tensorflow").vvag(tf_loss, vectorized_argnums=(0, 1), argnums=2)
    )

    def f(train_imgs, train_lbls):
        a.assign(train_imgs)
        b.assign(train_lbls)
        losses, grad = tf_vvag(a, b, params)
        opt.apply_gradients(zip([grad], [params]))
        return tf.reduce_mean(losses)

    ct, it, Nitrs = utils.qml_timing(f, nbatch, nitrs, timeLimit)
    meta["Results"]["tf_vmap"] = {
        "Construction time": ct,
        "Iteration time": it,
        "# of actual iterations": Nitrs,
    }
    print(meta)
    return meta


if __name__ == "__main__":
    _uuid = str(uuid.uuid4())
    n, nlayer, nitrs, timeLimit, isgpu, minus, path, check, nbatch = utils.arg(
        check=True, qml=True
    )
    train_img, test_img, train_lbl, test_lbl = utils.mnist_data_preprocessing(n - 1)
    if check == 1:
        checkbool = True
    else:
        checkbool = False
    results = pennylane_benchmark(
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
        checkbool,
    )
    utils.save(results, _uuid, path)
