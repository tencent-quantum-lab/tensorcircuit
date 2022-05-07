import tensorflow as tf
import numpy as np
import time
import utils
import datetime
import cpuinfo
import os
import sys
import uuid
import tensorcircuit as tc


def tensorcircuit_tf_benchmark(
    uuid,
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
    dt="32",
    check_loop=False,
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

    tc.set_backend("tensorflow")
    if dt == "32":
        tc.set_dtype("complex64")
        dtype = np.complex64
        tfdtype = tf.complex64
        vtype = np.float32
    else:  # dt == "64":
        tc.set_dtype("complex128")
        dtype = np.complex128
        tfdtype = tf.complex128
        vtype = np.float64
    # tc.set_contractor("plain")

    ii = np.eye(4, dtype=dtype)
    iir = tf.constant(ii.reshape([2, 2, 2, 2]), dtype=tfdtype)
    zz = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=dtype
    )
    zzr = tf.constant(zz.reshape([2, 2, 2, 2]), dtype=tfdtype)

    meta["Software"] = "tensorcircuit[tensorflow %s]" % dt
    meta["Cpuinfo"] = cpuinfo.get_cpu_info()["brand_raw"]
    meta["Version"] = {
        "sys": sys.version,
        "tensorflow": tf.__version__,
        "tensorcircuit": tc.__version__,
        "numpy": np.__version__,
    }
    meta["QML test parameters"] = {
        "nQubits": n,
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

    def expval(img, lbl, paramx, paramzz):
        c = tc.Circuit(n)

        for i in range(n - 1):
            c.rx(i, theta=tf.cast(img[i] * np.pi, tfdtype))

        for j in range(nlayer):
            for i in range(n - 1):
                c.exp1(i, n - 1, theta=tf.cast(paramzz[j * n + i], tfdtype), unitary=zz)
            for i in range(n):
                c.rx(i, theta=tf.cast(paramx[j * n + i], tfdtype))
        loss = tf.math.real(c.expectation((tc.gates.z(), [n - 1])) + 1.0) / 2
        loss = (loss - lbl) ** 2
        return tf.math.real(loss)

    paramx = tf.Variable(np.random.normal(size=[nlayer * n]), dtype=vtype)
    paramzz = tf.Variable(np.random.normal(size=[nlayer * n]), dtype=vtype)
    a = tf.Variable(train_img[:nbatch], dtype=vtype)
    b = tf.Variable(train_lbl[:nbatch], dtype=vtype)

    qml_vvag = tc.backend.vectorized_value_and_grad(
        expval, argnums=(2, 3), vectorized_argnums=(0, 1)
    )
    qml_vvag = tc.backend.jit(qml_vvag)
    opt = tf.keras.optimizers.Adam(0.01)

    def f(train_imgs, train_lbls):
        a.assign(train_imgs)
        b.assign(train_lbls)
        e, grad = qml_vvag(a, b, paramx, paramzz)
        opt.apply_gradients(zip(grad, [paramx, paramzz]))
        return tf.reduce_mean(e)

    ct, it, Nitrs = utils.qml_timing(f, nbatch, nitrs, timeLimit)
    meta["Results"]["vvag jit"] = {
        "Construction time": ct,
        "Iteration time": it,
        "# of actual iterations": Nitrs,
    }

    a = tf.Variable(train_img[0], dtype=vtype)
    b = tf.Variable(train_lbl[0], dtype=vtype)

    def vag(a, b, paramx, paramzz):
        with tf.GradientTape() as t:
            t.watch([paramx, paramzz])
            loss = expval(a, b, paramx, paramzz)
        grad = t.gradient(loss, [paramx, paramzz])
        return loss, grad

    if check_loop:
        vag = tf.function(vag)

        def f(train_imgs, train_lbls):
            loss_ = []
            grad0_ = []
            grad1_ = []

            for i in range(len(train_imgs)):
                a.assign(train_imgs[i])
                b.assign(train_lbls[i])
                loss, grad = vag(a, b, paramx, paramzz)
                #         print(b,loss)
                loss_.append(loss)
                grad0_.append(grad[0])
                grad1_.append(grad[1])
            opt.apply_gradients(
                zip(
                    [tf.reduce_mean(grad0_, axis=0), tf.reduce_mean(grad1_, axis=0)],
                    [paramx, paramzz],
                )
            )
            return tf.reduce_mean(loss_)

        ct, it, Nitrs = utils.qml_timing(f, nbatch, nitrs, timeLimit)
        meta["Results"]["naive loop + tf function"] = {
            "Construction time": ct,
            "Iteration time": it,
            "# of actual iterations": Nitrs,
        }
    print(meta)
    return meta


if __name__ == "__main__":
    _uuid = str(uuid.uuid4())
    n, nlayer, nitrs, timeLimit, isgpu, minus, path, dtype, check, nbatch = utils.arg(
        dtype=True, check=True, qml=True
    )
    train_img, test_img, train_lbl, test_lbl = utils.mnist_data_preprocessing(n - 1)
    if dtype == 1:
        dtypestr = "32"
    else:
        dtypestr = "64"
    if check == 1:
        checkbool = True
    else:
        checkbool = False
    results = tensorcircuit_tf_benchmark(
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
        dtypestr,
        check_loop=checkbool,
    )
    utils.save(results, _uuid, path)
