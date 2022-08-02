"""
A collection of useful function snippets that irrelevant with the main modules
or await for further refactor
"""
import logging
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

try:
    import cirq
except ImportError as e:
    logger.warning(e)
    logger.warning("Therefore some functionality in %s may not work" % __name__)

from ..circuit import Circuit
from .. import gates as G
from ..gates import array_to_tensor, num_to_tensor

Array = Any
Tensor = Any
Graph = Any


class FakeModule:
    def __getattr__(self, name: str) -> str:
        return name


fake_module = FakeModule()


##############
## qml related functions to be refactored
##############

# tf specific


def amplitude_encoding(
    fig: Tensor,
    qubits: int,
    index: Optional[Sequence[int]] = None,
    index_func: Optional[Callable[[int, int], Sequence[int]]] = None,
) -> Tensor:
    if fig.shape[-1] == 1:
        fig = tf.reshape(fig, shape=fig.shape[:-1])
    if len(fig.shape) == 2:
        fig = fig[tf.newaxis, ...]
    fig = tf.reshape(fig, shape=(fig.shape[0], fig.shape[1] * fig.shape[2]))
    norm = tf.linalg.norm(fig, axis=1)
    norm = norm[..., tf.newaxis]
    fig = fig / norm
    if fig.shape[1] < 2**qubits:
        fig = tf.concat(
            [
                fig,
                tf.zeros([fig.shape[0], 2**qubits - fig.shape[1]], dtype=tf.float64),
            ],
            axis=1,
        )
    if index is None and index_func is not None:
        index = []
        for i in range(32):
            # magic number here, only valid for MNIST data
            for j in range(32):
                l = index_func(i, j)
                r = 0
                for p, q in enumerate(l):
                    r += q * 2 ** (9 - p)
                index.append(r)
    if index is not None:
        fig = tf.constant(fig.numpy()[:, index], dtype=fig.dtype)
    return fig


def recursive_index(x: int, y: int) -> Sequence[int]:
    rl = []
    for k in range(5):
        rl.append((x // (2 ** (4 - k))) % 2)
        rl.append((y // (2 ** (4 - k))) % 2)
    return rl


def mnist_amplitude_data(
    a: int,
    b: int,
    binarize: bool = False,
    index: Optional[Sequence[int]] = None,
    index_func: Optional[Callable[[int, int], Sequence[int]]] = None,
    loader: Any = None,
    threshold: float = 0.4,
) -> Tensor:
    def filter_pair(x: Tensor, y: Tensor, a: int, b: int) -> Tuple[Tensor, Tensor]:
        keep = (y == a) | (y == b)
        x, y = x[keep], y[keep]
        y = y == a
        return x, y

    if loader is None:
        loader = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = loader.load_data()
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

    if binarize:
        x_train[x_train > threshold] = 1.0
        x_train[x_train <= threshold] = 0.0
        x_test[x_test > threshold] = 1.0
        x_test[x_test <= threshold] = 0.0
    x_train, y_train = filter_pair(x_train, y_train, a, b)
    x_test, y_test = filter_pair(x_test, y_test, a, b)
    x_train = tf.image.pad_to_bounding_box(x_train, 2, 2, 32, 32)
    x_test = tf.image.pad_to_bounding_box(x_test, 2, 2, 32, 32)
    x_train_q = amplitude_encoding(x_train, 10, index=index, index_func=index_func)
    x_test_q = amplitude_encoding(x_test, 10, index=index, index_func=index_func)
    return (x_train_q, y_train), (x_test_q, y_test)


def mnist_generator(
    x_train: Tensor, y_train: Tensor, batch: int = 1, random: bool = True
) -> Iterator[Any]:
    if isinstance(x_train, tf.Tensor):
        x_train = x_train.numpy()
    if isinstance(y_train, tf.Tensor):
        y_train = y_train.numpy()
    i = np.array(list(range(batch)))
    while True:
        if random:
            i = np.random.randint(low=0, high=x_train.shape[0], size=batch)
        else:
            i += batch
            i = i % x_train.shape[0]
        yield tf.constant(x_train[i]), tf.constant(y_train[i])


def generate_random_circuit(
    inputs: Tensor, nqubits: int = 10, epochs: int = 3, layouts: Tensor = None
) -> Circuit:
    inputs = tf.cast(inputs, dtype=tf.complex64)
    c = Circuit(nqubits, inputs=inputs)
    if layouts is None:
        layouts = np.random.choice([0, 1, 2], size=[epochs, nqubits])
    layouts = tf.reshape(layouts, shape=[epochs, nqubits])
    for epoch in range(epochs):
        for i in range(nqubits):
            flg = layouts[epoch, i]
            if flg == 0:
                c.rx(i, theta=num_to_tensor(np.pi / 2))  # type: ignore
            elif flg == 1:
                c.ry(i, theta=num_to_tensor(np.pi / 2))  # type: ignore
            else:
                c.wroot(i)  # type: ignore
        for i in range(nqubits):  # entangling layer
            c.swap(i, (i + 1) % nqubits)  # type: ignore
    return c


# the following vags are not the recommended fashion to write qml tasks in tensorcircuit now.
# also the hardcoded circuit structure is a bad practice, they are here for backward compatibility
# instead use vvag or keras paradigm now


def naive_qml_vag(
    gdata: Graph,
    nnp: Tensor,
    preset: Sequence[int],
    nqubits: int = 10,
    epochs: int = 3,
    target: int = 0,
) -> Tuple[Tensor, Tensor]:
    # used for structure searching?
    xs, ys = gdata
    loss = 0.0
    for x, y in zip(xs, ys):
        circuit = generate_random_circuit(
            x, nqubits=nqubits, epochs=epochs, layouts=preset
        )
        value = circuit.expectation(
            (
                G.z(),  # type: ignore
                [
                    target,
                ],
            )
        )
        y = tf.cast(y, dtype=tf.complex64)
        y = 2 * y - 1
        loss += (value - y) ** 2
    return tf.constant(tf.cast(loss, dtype=tf.float32)), tf.zeros_like(nnp)  # MSE loss


def train_qml_vag(
    gdata: Graph,
    nnp: Tensor,
    preset: Optional[Sequence[int]] = None,
    nqubits: int = 10,
    epochs: int = 3,
    batch: int = 64,
    validation: bool = False,
) -> Any:
    xs, ys = gdata
    with tf.GradientTape() as tape:
        tape.watch(nnp)
        cnnp = tf.cast(nnp, dtype=tf.complex64)
        c = Circuit(nqubits, inputs=np.ones([1024], dtype=np.complex64) / 2**5)
        for epoch in range(epochs):
            for i in range(nqubits):
                c.rz(i, theta=cnnp[3 * epoch, i])  # type: ignore
            for i in range(nqubits):
                c.ry(i, theta=cnnp[3 * epoch + 1, i])  # type: ignore
            for i in range(0, nqubits, 2):
                # c.swap(i, (i + 1) % 10)  # type: ignore
                c.exp(  # type: ignore
                    i,
                    (i + 1) % 10,
                    unitary=array_to_tensor(G._swap_matrix),
                    theta=cnnp[3 * epoch + 2, i],
                )
            for i in range(1, nqubits, 2):
                # c.swap(i, (i + 1) % 10)  # type: ignore
                c.exp(  # type: ignore
                    i,
                    (i + 1) % 10,
                    unitary=array_to_tensor(G._swap_matrix),
                    theta=cnnp[3 * epoch + 2, i],
                )
        for i in range(nqubits):
            c.rx(i, theta=cnnp[3 * epochs, i])  # type: ignore
        count = 0
        loss = 0
        for x, y in zip(xs, ys):
            y = tf.cast(y, dtype=tf.float32)  # [1, 0] binary
            c.replace_inputs(tf.cast(x, dtype=tf.complex64))
            flg = False
            yp = 0.0

            for i in range(nqubits):
                yp += cnnp[3 * epochs + 1, i] * c.expectation(
                    (
                        G.z(),  # type: ignore
                        [
                            i,
                        ],
                    ),
                    reuse=flg,
                )

                flg = True

            yp = tf.cast(yp, dtype=tf.float32)
            yp = tf.math.sigmoid(
                (yp + tf.math.real(cnnp[3 * epochs + 2, 0])) * 15
            )  ## [0,1] intepreted as probability
            if tf.math.abs(yp - y) < 0.5:
                count += 1
            loss += (yp - y) ** 2
        # print("accuracy:", count / batch)
        gr = tape.gradient(loss, cnnp)
        if not validation:
            return tf.constant(count / batch, dtype=tf.float32), gr
        else:
            return count / batch  # evaluate as validation kernel


def validate_qml_vag(
    gdata: Graph,
    nnp: Tensor,
    preset: Optional[Sequence[int]] = None,
    nqubits: int = 10,
    epochs: int = 3,
    batch: int = 64,
) -> Any:
    xs, ys = gdata
    cnnp = tf.cast(nnp, dtype=tf.complex64)
    c = Circuit(nqubits, inputs=np.ones([1024], dtype=np.complex64) / 2**5)
    for epoch in range(epochs):
        for i in range(nqubits):
            c.rz(i, theta=cnnp[3 * epoch, i])  # type: ignore
        for i in range(nqubits):
            c.ry(i, theta=cnnp[3 * epoch + 1, i])  # type: ignore
        for i in range(0, nqubits, 2):
            # c.swap(i, (i + 1) % 10)  # type: ignore
            c.exp(  # type: ignore
                i,
                (i + 1) % 10,
                unitary=array_to_tensor(G._swap_matrix),
                theta=cnnp[3 * epoch + 2, i],
            )
        for i in range(1, nqubits, 2):
            # c.swap(i, (i + 1) % 10)  # type: ignore
            c.exp(  # type: ignore
                i,
                (i + 1) % 10,
                unitary=array_to_tensor(G._swap_matrix),
                theta=cnnp[3 * epoch + 2, i],
            )
    for i in range(nqubits):
        c.rx(i, theta=cnnp[3 * epochs, i])  # type: ignore
    count = 0
    loss = 0
    for x, y in zip(xs, ys):
        y = tf.cast(y, dtype=tf.float32)  # [1, 0] binary
        c.replace_inputs(tf.cast(x, dtype=tf.complex64))
        flg = False
        yp = 0.0

        for i in range(nqubits):
            yp += cnnp[3 * epochs + 1, i] * c.expectation(
                (
                    G.z(),  # type: ignore
                    [
                        i,
                    ],
                ),
                reuse=flg,
            )

            flg = True

        yp = tf.cast(yp, dtype=tf.float32)
        yp = tf.math.sigmoid(
            (yp + tf.math.real(cnnp[3 * epochs + 2, 0])) * 15
        )  ## [0,1] intepreted as probability
        if tf.math.abs(yp - y) < 0.5:
            count += 1
        loss += (yp - y) ** 2

    return {
        "val_loss": loss / batch,
        "val_accuracy": count / batch,
    }  # evaluate as validation kernel


##############
## really small and irrelavant pieces below
##############


def color_svg(circuit: cirq.Circuit, *coords: Tuple[int, int]) -> Any:
    """
    color cirq circuit SVG for given gates,
    a small tool to hack the cirq SVG

    :param circuit:
    :param coords: integer coordinate which gate is colored
    :return:
    """
    import xml
    from cirq.contrib.svg import SVGCircuit  # type: ignore

    svg_str = SVGCircuit(circuit)._repr_svg_()
    DOMTree = xml.dom.minidom.parseString(svg_str)  # type: ignore
    xpos = []
    ypos = []
    for r in DOMTree.getElementsByTagName("rect"):  # [0].setAttribute("fill", "gray")
        xpos.append(int(float(r.getAttribute("x"))))
        ypos.append(int(float(r.getAttribute("y"))))
    xpos = sorted(list(set(xpos)))
    ypos = sorted(list(set(ypos)))
    # xpos_dict = dict(zip(range(len(xpos)), xpos))
    # ypos_dict = dict(zip(range(len(ypos)), ypos))
    i_xpos_dict = dict(zip(xpos, range(len(xpos))))
    i_ypos_dict = dict(zip(ypos, range(len(ypos))))
    for r in DOMTree.getElementsByTagName("rect"):
        x, y = int(float(r.getAttribute("x"))), int(float(r.getAttribute("y")))
        if (i_xpos_dict[x], i_ypos_dict[y]) in coords:
            r.setAttribute("fill", "gray")
    return DOMTree.toxml()


def repr2array(inputs: str) -> Array:
    """
    transform repr form of an array to real numpy array

    :param inputs:
    :return:
    """
    # used to transform np.array in print form into live np.array
    inputs = inputs.split("]")  # type: ignore
    inputs = [l.strip().strip("[") for l in inputs if l.strip()]  # type: ignore
    outputs = []
    for l in inputs:
        o = [float(c.strip()) for c in l.split(" ") if c.strip()]
        outputs.append(o)
    return np.array(outputs)


def TFIM1Denergy(
    L: int, Jzz: float = 1.0, Jx: float = 1.0, Pauli: bool = True
) -> float:
    # PBC
    # nice tutorial: https://arxiv.org/pdf/2009.09208.pdf
    # further investigation on 1 TFIM solution structure is required
    # will fail on AFM phase Jzz>Jx and Jzz>0 and odd sites (frustration in boundary)
    e = 0
    if Pauli:
        Jx *= 2
        Jzz *= 4
    for i in range(L):
        q = np.pi * (2 * i - (1 + (-1) ** L) / 2) / L
        e -= np.abs(Jx) / 2 * np.sqrt(1 + Jzz**2 / 4 / Jx**2 - Jzz / Jx * np.cos(q))
    return e


def Heisenberg1Denergy(L: int, Pauli: bool = True, maxiters: int = 1000) -> float:
    # PBC
    error = 1e-15
    eps = 1e-20  # avoid zero division
    phi = np.zeros([L // 2, L // 2])
    phi2 = np.zeros([L // 2, L // 2])
    lamb = np.array([2 * i + 1 for i in range(L // 2)])
    for _ in range(maxiters):
        k = 1 / L * (2 * np.pi * lamb + np.sum(phi, axis=-1) - np.diag(phi))  # type: ignore
        for i in range(L // 2):
            for j in range(L // 2):
                phi2[i, j] = (
                    np.arctan(
                        2
                        / (
                            1 / (np.tan(k[i] / 2) + eps)
                            - 1 / (np.tan(k[j] / 2) + eps)
                            + eps
                        )
                    )
                    * 2
                )
        if np.allclose(phi, phi2, rtol=error):  # type: ignore # converged
            break
        phi = phi2.copy()
    else:
        raise ValueError(
            "the maxiters %s is too small for bethe ansatz to converge" % maxiters
        )
    e = -np.sum(1 - np.cos(k)) + L / 4
    if Pauli is True:
        e *= 4
    return e  # type: ignore
