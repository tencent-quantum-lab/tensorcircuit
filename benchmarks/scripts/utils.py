import sys
import argparse
import time
import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path


qml_data = {}


def arg(dtype=False, check=False, contractor=False, qml=False, mps=False, tc=False):
    parser = argparse.ArgumentParser(description="PQC setup parameters.")
    parser.add_argument(
        "-n", dest="n", type=int, nargs=1, help="# of Qubits", default=[10]
    )
    parser.add_argument(
        "-nlayer", dest="nlayer", type=int, nargs=1, help="# of layers", default=[3]
    )
    parser.add_argument(
        "-nitrs", dest="nitrs", type=int, nargs=1, help="# of iterations", default=[100]
    )
    parser.add_argument(
        "-t", dest="timeLimit", type=int, nargs=1, help="Time limit(s)", default=[60]
    )
    parser.add_argument(
        "-gpu", dest="isgpu", type=int, nargs=1, help="GPU available", default=[0]
    )

    parser.add_argument(
        "-nbatch", dest="nbatch", type=int, nargs=1, help="batch number", default=[100]
    )
    parser.add_argument(
        "-m",
        dest="minus",
        type=int,
        nargs=1,
        help="0 is expensive for loop ladder!",
        default=[1],
    )
    parser.add_argument(
        "-path",
        dest="path",
        type=str,
        nargs=1,
        help="output json dir path ended with /",
        default=[None],
    )
    if dtype:
        parser.add_argument(
            "-dtype",
            dest="dtype",
            type=int,
            nargs=1,
            help="32 as 1 and 64 as 2",
            default=[1],
        )
    if check:
        parser.add_argument(
            "-c",
            dest="check",
            type=int,
            nargs=1,
            help="0 false",
            default=[0],
        )
    if contractor:
        parser.add_argument(
            "-contractor",
            dest="contractor",
            type=str,
            nargs=1,
            help="contractor type",
            default=["auto"],
        )
    if mps:
        parser.add_argument(
            "-mpsd",
            dest="mpsd",
            type=int,
            nargs=1,
            help="bond dimension of MPS",
            default=[None],
        )
    if tc:
        parser.add_argument(
            "-tcbackend",
            dest="tcbackend",
            type=str,
            nargs=1,
            help="backend of tensorcircuit",
            default=["tensorflow"],
        )
    args = parser.parse_args()
    r = [
        args.n[0],
        args.nlayer[0],
        args.nitrs[0],
        args.timeLimit[0],
        args.isgpu[0],
        args.minus[0],
        args.path[0],
    ]
    if dtype:
        r.append(args.dtype[0])
    if check:
        r.append(args.check[0])
    if contractor:
        r.append(args.contractor[0])
    if qml:
        r.append(args.nbatch[0])
    if mps:
        r.append(args.mpsd[0])
    if tc:
        r.append(args.tcbackend[0])
    return r


def mnist_data_preprocessing(PCA_components=10):
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA

    if qml_data == {}:
        if Path("../dataset/mnist.npz").exists():
            print("load local dataset")
            # from https://www.kaggle.com/vikramtiwari/mnist-numpy
            def load_data(path):
                with np.load(path) as f:
                    x_train, y_train = f["x_train"], f["y_train"]
                    x_test, y_test = f["x_test"], f["y_test"]
                    return (x_train, y_train), (x_test, y_test)

            (train_img, train_lbl), (test_img, test_lbl) = load_data(
                "../dataset/mnist.npz"
            )
            train_img = train_img.reshape([-1, 28 * 28])
            test_img = test_img.reshape([-1, 28 * 28])
        else:
            mnist = fetch_openml("mnist_784")
            train_img, test_img, train_lbl, test_lbl = train_test_split(
                mnist.data, mnist.target, test_size=2 / 10, random_state=0
            )
            # train_img = train_img.values
            # test_img = test_img.values
            # train_lbl = train_lbl.values
            # test_lbl = test_lbl.values
            # very slow, but anyway, tired of modify the preprocessing code

        def processing(train_img, test_img, train_lbl, test_lbl):
            train_img_ = []
            test_img_ = []
            train_lbl_ = []
            test_lbl_ = []

            for i in range(np.size(train_lbl)):
                if (
                    (train_lbl[i] == "3")
                    | (train_lbl[i] == "6")
                    | (train_lbl[i] == 3)
                    | (train_lbl[i] == 6)
                ):
                    train_img_.append(train_img[i])
                    train_lbl_.append(float(train_lbl[i] == "3" or train_lbl[i] == 3))

            for i in range(np.size(test_lbl)):
                if (
                    (test_lbl[i] == "3")
                    | (test_lbl[i] == "6")
                    | (test_lbl[i] == 3)
                    | (test_lbl[i] == 6)
                ):
                    test_img_.append(test_img[i])
                    test_lbl_.append(
                        float(test_lbl[i] == "3") or float(test_lbl[i] == 3)
                    )
            return train_img_, test_img_, train_lbl_, test_lbl_

        train_img, test_img, train_lbl, test_lbl = processing(
            train_img, test_img, train_lbl, test_lbl
        )
        pca = PCA(n_components=PCA_components)
        pca.fit(train_img)
        train_img = pca.transform(train_img)
        test_img = pca.transform(test_img)
        train_img = np.array([x / np.sqrt(np.sum(x**2)) for x in train_img])
        test_img = np.array([x / np.sqrt(np.sum(x**2)) for x in test_img])
        qml_data["test_img"] = test_img
        qml_data["train_img"] = train_img
        qml_data["train_lbl"] = train_lbl
        qml_data["test_lbl"] = test_lbl
    return (
        qml_data["train_img"],
        qml_data["test_img"],
        qml_data["train_lbl"],
        qml_data["test_lbl"],
    )


def gpuinfo():
    try:
        ns = os.popen("nvidia-smi --query-gpu=gpu_name --format=csv")
        return ns.readlines()[-1].strip()
    except:
        pass


def save(data, _uuid, path):
    if path is None:
        return
    with open(path + _uuid + ".json", "w") as f:
        json.dump(
            data,
            f,
            indent=4,
        )


def timing(f, nitrs, timeLimit):
    t0 = time.time()
    print(f())
    t1 = time.time()
    Nitrs = 1e-8
    for i in range(nitrs):
        a = f()
        print(a)
        # if a != None:
        #    print(a)
        if time.time() - t1 > timeLimit:
            break
        else:
            Nitrs += 1
    t2 = time.time()
    return t1 - t0, (t2 - t1) / Nitrs, int(Nitrs)


def qml_timing(f, nbatch, nitrs, timeLimit, tfq=False):
    t0 = time.time()
    if tfq:
        img_t = qml_data["train_img_tfq"]
        lbl_t = qml_data["train_lbl"]
    else:
        img_t = qml_data["train_img"]
        lbl_t = qml_data["train_lbl"]
    l_t = len(img_t)
    batch = (l_t // nbatch) - 1
    f(img_t[:nbatch], lbl_t[:nbatch])
    t1 = time.time()
    Nitrs = 1e-8
    for i in range(nitrs):
        a = f(
            img_t[(i % batch) * nbatch : (i % batch + 1) * nbatch],
            lbl_t[(i % batch) * nbatch : (i % batch + 1) * nbatch],
        )
        if a != None:
            print(a)
        if time.time() - t1 > timeLimit:
            break
        else:
            Nitrs += 1
    t2 = time.time()
    return t1 - t0, (t2 - t1) / Nitrs, int(Nitrs)


class Opt:
    def __init__(self, f, params, lr=0.01):
        self.f = f
        self.params = [tf.Variable(param) for param in params]
        self.adam = tf.keras.optimizers.Adam(lr)

    def step(self):
        e, grad = self.f(*self.params)
        grad = [tf.convert_to_tensor(g) for g in grad]
        self.adam.apply_gradients(zip(grad, self.params))
        return e[()]
