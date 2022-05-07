import os
from functools import lru_cache
from itertools import product
from datetime import datetime
import tensorflow as tf
import numpy as np
import sys
from jax.config import config
import utils
import jax
from jax import numpy as jnp

import tensorcircuit as tc
import time

import cpuinfo
import datetime
import sys
import uuid


def tensorcircuit_jax_benchmark(
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
    dtypestr,
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

    if dtypestr == "32":
        tc.set_dtype("complex64")
        dtype = np.complex64
    else:
        from jax.config import config

        config.update("jax_enable_x64", True)
        tc.set_dtype("complex128")
        dtype = np.complex128

    tc.set_backend("jax")

    ii = np.eye(4, dtype=dtype)
    iir = ii.reshape([2, 2, 2, 2])
    zz = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=dtype
    )
    zzr = zz.reshape([2, 2, 2, 2])

    meta["Software"] = "tensorcircuit[jax]"
    meta["Cpuinfo"] = cpuinfo.get_cpu_info()["brand_raw"]
    meta["Version"] = {
        "sys": sys.version,
        "jax": jax.__version__,
        "tensorcircuit": tc.__version__,
        "numpy": np.__version__,
    }
    meta["QML test parameters"] = {
        "nQubits": n,
        "nlayer": nlayer,
        "nitrs": nitrs,
        "timeLimit": timeLimit,
        "nbatch": nbatch,
        "dtype": dtypestr,
    }
    meta["UUID"] = uuid
    meta["Benchmark Time"] = (
        datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    )
    meta["Results"] = {}

    def expval(img, lbl, paramx, paramzz):
        c = tc.Circuit(n)

        for i in range(n - 1):
            c.rx(i, theta=tc.backend.cast(img[i] * np.pi, "complex128"))

        for j in range(nlayer):
            for i in range(n - 1):
                c.exp1(
                    i,
                    n - 1,
                    theta=tc.backend.cast(paramzz[j * n + i], "complex128"),
                    unitary=zz,
                )
            for i in range(n):
                c.rx(i, theta=tc.backend.cast(paramx[j * n + i], "complex128"))
        loss = jnp.real(c.expectation((tc.gates.z(), [n - 1])) + 1.0) / 2
        loss = (loss - lbl) ** 2
        return jnp.real(loss)

    paramx = jnp.ones([nlayer * n])
    paramzz = jnp.ones([nlayer * n])

    qml_vvag = tc.backend.vectorized_value_and_grad(
        expval, argnums=(2, 3), vectorized_argnums=(0, 1)
    )
    qml_vvag_jit = tc.backend.jit(qml_vvag)

    # def f(train_imgs, train_lbls):
    #     a = jnp.array(train_imgs)
    #     b = jnp.array(train_lbls)
    #     e, grad = qml_vvag(a, b, paramx, paramzz)
    #     return np.mean(e)

    # ct, it, Nitrs = utils.qml_timing(f, nbatch, nitrs, timeLimit)
    # meta["Results"]["tensorcircuit[jax] vvag"] = {
    #     "Construction time": ct,
    #     "Iteration time": it,
    #     "# of actual iterations": Nitrs,
    # }

    def f(train_imgs, train_lbls):
        a = jnp.array(train_imgs)
        b = jnp.array(train_lbls)
        e, grad = qml_vvag_jit(a, b, paramx, paramzz)
        return np.mean(e)

    ct, it, Nitrs = utils.qml_timing(f, nbatch, nitrs, timeLimit)
    meta["Results"]["tensorcircuit[jax] vvag jit"] = {
        "Construction time": ct,
        "Iteration time": it,
        "# of actual iterations": Nitrs,
    }
    print(meta)
    return meta


if __name__ == "__main__":
    _uuid = str(uuid.uuid4())
    n, nlayer, nitrs, timeLimit, isgpu, minus, path, dtype, nbatch = utils.arg(
        dtype=True, qml=True
    )
    train_img, test_img, train_lbl, test_lbl = utils.mnist_data_preprocessing(n - 1)
    if dtype == 1:
        dtypestr = "32"
    else:
        dtypestr = "64"
    results = tensorcircuit_jax_benchmark(
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
    )
    utils.save(results, _uuid, path)
