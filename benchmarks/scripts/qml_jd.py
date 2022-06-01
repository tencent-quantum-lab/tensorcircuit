import tedq as qai
import numpy as np
import cpuinfo
import tensorflow as tf
import datetime
import os
import torch
import jax
from jax import numpy as jnp
import uuid
import utils
import sys


def jd_benchmark(
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

    meta["Software"] = "JDQAI"
    meta["Cpuinfo"] = cpuinfo.get_cpu_info()["brand_raw"]
    meta["Version"] = {
        "sys": sys.version,
        "tensorflow": tf.__version__,
        "jdqai": None,
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

    def circuitDef(img, params):
        count = 0
        for i in range(nwires - 1):
            qai.RX(img[i] * np.pi, qubits=[i])
        for j in range(nlayer):
            for i in range(nwires - 1):
                qai.CNOT(qubits=[i, nwires - 1])
                qai.RZ(params[count], qubits=[nwires - 1])
                count += 1
                qai.CNOT(qubits=[i, nwires - 1])
            for i in range(nwires):
                qai.RX(params[count], qubits=[i])
                count += 1
        return qai.expval(qai.PauliZ(qubits=[nwires - 1]))

    circuit = qai.Circuit(
        circuitDef,
        nwires,
        torch.zeros([nwires]),
        torch.zeros([nlayer * nwires + nlayer * (nwires - 1)]),
    )
    compiled_circuit = circuit.compilecircuit(backend="pytorch")

    def loss(img, lbl, params):
        return (lbl - compiled_circuit(img, params) * 0.5 - 0.5) ** 2

    params = torch.from_numpy(
        np.random.normal(size=[nlayer * nwires + nlayer * (nwires - 1)])
    )
    params.requires_grad_(True)

    def f(train_imgs, train_lbls):
        train_imgs = torch.from_numpy(train_imgs)
        train_lbls = torch.from_numpy(np.array(train_lbls))
        e = torch.zeros([])
        for i in range(len(train_imgs)):
            e += loss(train_imgs[i], train_lbls[i], params)[0]
        e.backward()
        # print(params.grad)
        return e

    ct, it, Nitrs = utils.qml_timing(f, nbatch, nitrs, timeLimit)
    meta["Results[torch backend]"] = {
        "Construction time": ct,
        "Iteration time": it,
        "# of actual iterations": Nitrs,
    }

    """
    # jax backend doesn't work
    circuit = qai.Circuit(
        circuitDef,
        nwires,
        jnp.zeros([nwires]),
        jnp.zeros([nlayer * nwires + nlayer * (nwires - 1)]),
    )
    compiled_circuit = circuit.compilecircuit(backend="jax")

    def loss(img, lbl, params):
        return (lbl - compiled_circuit(img, params) * 0.5 - 0.5) ** 2

    params = jnp.array(np.random.normal(size=[nlayer * nwires + nlayer * (nwires - 1)]))
    jax_vg = jax.value_and_grad(loss, argnums=2)

    def f(train_imgs, train_lbls):
        train_imgs = jnp.array(train_imgs)
        train_lbls = jnp.array(train_lbls)
        e = []
        grad = []
        for i in range(len(train_imgs)):
            eitem, graditem = loss(train_imgs[i], train_lbls[i], params)
        e.append(eitem)
        grad.append(graditem)
        print(jnp.mean(grad))
        return jnp.mean(e)

    ct, it, Nitrs = utils.qml_timing(f, nbatch, nitrs, timeLimit)
    meta["Results[jax backend]"] = {
        "Construction time": ct,
        "Iteration time": it,
        "# of actual iterations": Nitrs,
    }
    """
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
    results = jd_benchmark(
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

# mac laptop cpu
# n=10, nlayer=3
# nbatch = 64, jd vs. tc(jax): 1.79s vs 0.013s
