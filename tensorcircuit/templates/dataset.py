"""
Quantum machine learning related data preprocessing and embedding
"""

from typing import Any, Optional, Sequence, Tuple

import numpy as np

from ..cons import backend, dtypestr
from ..gates import array_to_tensor

Tensor = Any


def amplitude_encoding(
    fig: Tensor, nqubits: int, index: Optional[Sequence[int]] = None
) -> Tensor:
    # non-batch version
    # [WIP]
    fig = backend.reshape(fig, shape=[-1])
    norm = backend.norm(fig)
    fig = fig / norm
    if backend.shape_tuple(fig)[0] < 2**nqubits:
        fig = backend.concat(
            [
                fig,
                backend.zeros(
                    [2**nqubits - backend.shape_tuple(fig)[0]], dtype=fig.dtype
                ),
            ],
        )
    if index is not None:
        index = array_to_tensor(index, dtype="int32")
        fig = backend.gather1d(fig, index)
    fig = backend.cast(fig, dtypestr)
    return fig


# batched_amplitude_encoding = backend.vmap(amplitude_encoding, vectorized_argnums=0)


def mnist_pair_data(
    a: int,
    b: int,
    binarize: bool = False,
    threshold: float = 0.4,
    loader: Any = None,
) -> Tensor:
    def filter_pair(x: Tensor, y: Tensor, a: int, b: int) -> Tuple[Tensor, Tensor]:
        keep = (y == a) | (y == b)
        x, y = x[keep], y[keep]
        y = y == a
        return x, y

    if loader is None:
        import tensorflow as tf

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
    return (x_train, y_train), (x_test, y_test)
