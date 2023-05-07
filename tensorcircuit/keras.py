"""
Keras layer for tc quantum function
"""

from typing import Any, Callable, List, Optional, Sequence, Text, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, constraints, regularizers

from .cons import rdtypestr, backend

# @tf.keras.utils.register_keras_serializable(
#     package="tensorcircuit"
# )


class QuantumLayer(Layer):  # type: ignore
    def __init__(
        self,
        f: Callable[..., Any],
        weights_shape: Optional[Sequence[Tuple[int, ...]]] = None,
        initializer: Union[Text, Sequence[Text]] = "glorot_uniform",
        constraint: Optional[Union[Text, Sequence[Text]]] = None,
        regularizer: Optional[Union[Text, Sequence[Text]]] = None,
        **kwargs: Any
    ) -> None:
        """
        `QuantumLayer` wraps the quantum function `f` as a `keras.Layer`
        so that tensorcircuit is better integrated with tensorflow.
        Note that the input of the layer can be tensors or even list/dict of tensors.

        :param f: Callabel function.
        :type f: Callable[..., Any]
        :param weights_shape: The shape of the weights.
        :type weights_shape: Sequence[Tuple[int, ...]]
        :param initializer: The initializer of the weights, defaults to "glorot_uniform"
        :type initializer: Union[Text, Sequence[Text]], optional
        :param constraint: [description], defaults to None
        :type constraint: Optional[Union[Text, Sequence[Text]]], optional
        :param initializer: The regularizer of the weights, defaults to None
        :type initializer: Union[Text, Sequence[Text]], optional
        """

        super().__init__(**kwargs)
        if weights_shape is not None and isinstance(weights_shape[0], int):
            weights_shape = [weights_shape]
        if weights_shape is not None:
            self.number_weights = len(weights_shape)
        else:
            self.number_weights = 0
        self.f = f
        self.weights_shape = weights_shape
        if not (isinstance(initializer, list) or isinstance(initializer, tuple)):
            initializer = [initializer for _ in range(self.number_weights)]  # type: ignore
        self.initializer = [
            initializers.get(item) if isinstance(item, str) else item
            for item in initializer
        ]
        if not (isinstance(constraint, list) or isinstance(constraint, tuple)):
            constraint = [constraint for _ in range(self.number_weights)]  # type: ignore
        self.constraint = [
            constraints.get(item) if isinstance(item, str) else item
            for item in constraint
        ]
        if not (isinstance(regularizer, list) or isinstance(regularizer, tuple)):
            regularizer = [regularizer for _ in range(self.number_weights)]  # type: ignore
        self.regularizer = [
            regularizers.get(item) if isinstance(item, str) else item
            for item in regularizer
        ]

    def build(self, input_shape: Optional[List[int]] = None) -> None:
        if input_shape is None:
            input_shape = [1, 1]
        super().build(input_shape)
        self.pqc_weights = []
        if self.weights_shape is not None:
            for i, (shape, init, cst, reg) in enumerate(
                zip(
                    self.weights_shape,
                    self.initializer,
                    self.constraint,
                    self.regularizer,
                )
            ):
                self.pqc_weights.append(
                    self.add_weight(
                        name="PQCweights%s" % i,
                        shape=shape,
                        dtype=getattr(np, rdtypestr),
                        trainable=True,
                        initializer=init,
                        constraint=cst,
                        regularizer=reg,
                    )
                )

    @tf.function  # type: ignore
    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None,
        **kwargs: Any
    ) -> tf.Tensor:
        # input_shape = list(inputs.shape)
        # inputs = tf.reshape(inputs, (-1, input_shape[-1]))
        if inputs is None:  # not possible
            result = self.f(*self.pqc_weights, **kwargs)
        elif (
            len(
                backend.tree_map(backend.shape_tuple, backend.tree_flatten(inputs))[0][
                    0
                ]
            )
            == 1
        ):
            result = self.f(inputs, *self.pqc_weights, **kwargs)
        else:
            result = tf.vectorized_map(
                lambda vec: self.f(vec, *self.pqc_weights, **kwargs),
                inputs,
            )
        return result

    # [WIP], possibly cannot work as a general python function is part of the layer, hard to serialization
    # def get_config(self) -> dict:
    #     """Returns the config of the layer.
    #     The same layer can be reinstantiated later
    #     (without its trained weights) from this configuration.
    #     Returns:
    #       Python dictionary containing the configuration of the layer.
    #     """
    #     config = {}
    #     # Get base config
    #     base_config = super().get_config()
    #     return dict(list(base_config.items()) + list(config.items()))


KerasLayer = QuantumLayer


class HardwareLayer(QuantumLayer):
    """
    Keras Layer wrapping quantum function with cloud qpu access
    (using :py:mod:`tensorcircuit.cloud` module)
    """

    @tf.autograph.experimental.do_not_convert  # type: ignore
    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None,
        **kwargs: Any
    ) -> tf.Tensor:
        if inputs is None:  # not possible
            result = self.f(*self.pqc_weights, **kwargs)
        elif (
            len(
                backend.tree_map(backend.shape_tuple, backend.tree_flatten(inputs))[0][
                    0
                ]
            )
            == 1
        ):
            result = self.f(inputs, *self.pqc_weights, **kwargs)
        else:
            result = []
            for inp in inputs:
                result.append(self.f(inp, *self.pqc_weights, **kwargs))
            result = tf.stack(result)
        return result


KerasHardwareLayer = HardwareLayer


def output_asis_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    The keras loss function that directly taking the model output as the loss.

    :param y_true: Ignoring this parameter.
    :type y_true: tf.Tensor
    :param y_pred: Model output.
    :type y_pred: tf.Tensor
    :return: Model output, which is y_pred.
    :rtype: tf.Tensor
    """
    return y_pred


def save_func(f: Callable[..., Any], path: str) -> None:
    r"""
    Save tf function in the file (``tf.savedmodel`` format).

    :Example:

    .. code-block:: python

        @tf.function
        def test_circuit(weights):
            param = tf.cast(weights, tf.complex64)
            c = tc.Circuit(2)
            c.H(0)
            c.rx(0, theta=param[0])
            c.ry(1, theta=param[1])
            return tf.math.real(c.expectation((tc.gates.x(), [1])))

    >>> test_circuit(weights=tf.ones([2]))
    tf.Tensor(0.84147096, shape=(), dtype=float32)
    >>> K.save_func(test_circuit, save_path)
    >>> circuit_loaded = K.load_func(save_path)
    >>> circuit_loaded(weights=tf.ones([2]))
    tf.Tensor(0.84147096, shape=(), dtype=float32)
    >>> os.system(f"tree {save_path}")
    ~/model
    │   saved_model.pb
    ├─assets
    └─variables
        variables.data-00000-of-00001
        variables.index

    :param f: ``tf.function`` ed function with graph building
    :type f: Callable[..., Any]
    :param path: the dir path to save the function
    :type path: str
    """
    m = tf.Module()
    m.f = f
    tf.saved_model.save(m, path)


# sometimes, save_func complains about ``return tuple(tensor.shape.as_list())``
# ``ValueError: as_list() is not defined on an unknown TensorShape.`` when with vmap
# solved by fix the batch dimension by providing signature of tf.function, i.e.
# batch = 20
# tf.function(
#    tc.backend.vectorized_value_and_grad(vqe_forward, argnums=1, vectorized_argnums=0),
#    input_signature=[tf.TensorSpec([batch, nwires, 4], tf.float32),tf.TensorSpec([2 * nlayers, nwires], tf.float32) ],
# )


def load_func(
    *path: str, fallback: Optional[Callable[..., Any]] = None
) -> Callable[..., Any]:
    """
    Load function from the files in the ``tf.savedmodel`` format.
    We can load several functions at the same time, as they can be the same function of different input shapes.

    :Example:

    .. code-block:: python

        @tf.function
        def test_circuit(weights):
            param = tf.cast(weights, tf.complex64)
            c = tc.Circuit(2)
            c.H(0)
            c.rx(0, theta=param[0])
            c.ry(1, theta=param[1])
            return tf.math.real(c.expectation((tc.gates.x(), [1])))

    >>> test_circuit(weights=tf.ones([2]))
    tf.Tensor(0.84147096, shape=(), dtype=float32)
    >>> K.save_func(test_circuit, save_path)
    >>> circuit_loaded = K.load_func(save_path)
    >>> circuit_loaded(weights=tf.ones([2]))
    tf.Tensor(0.84147096, shape=(), dtype=float32)

    :param fallback: The fallback function when all functions loaded are failed, defaults to None
    :type fallback: Optional[Callable[..., Any]], optional
    :raises ValueError: When there is not legal loaded function of the input shape and no fallback callable.
    :return: A function that tries all loaded function against the input until the first success one.
    :rtype: Callable[..., Any]
    """
    ms = [tf.saved_model.load(p) for p in path]

    def wrapper(*args: Any, **kws: Any) -> Any:
        try:
            for m in ms:
                return m.f(*args, **kws)
        except ValueError as e:
            if fallback is not None:
                return fallback(*args, **kws)
            else:
                raise e

    return wrapper
