""" 
keras layer for tc quantum function
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer  # type: ignore
from tensorflow.keras import activations, initializers, constraints
from typing import List, Optional, Text, Tuple, Callable, Any, Sequence, Union
import tensornetwork as tn
import numpy as np


# @tf.keras.utils.register_keras_serializable(
#     package="tensorcircuit"
# )
class QuantumLayer(Layer):  # type: ignore
    def __init__(
        self,
        f: Callable[..., Any],
        weights_shape: Sequence[Tuple[int, ...]],
        initializer: Union[Text, Sequence[Text]] = "glorot_uniform",
        constraint: Optional[Union[Text, Sequence[Text]]] = None,
        **kwargs: Any
    ) -> None:

        super().__init__(**kwargs)
        if isinstance(weights_shape[0], int):
            weights_shape = [weights_shape]
        self.number_weights = len(weights_shape)
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

    def build(self, input_shape: Optional[List[int]] = None) -> None:
        if input_shape is None:
            input_shape = [1, 1]
        super().build(input_shape)
        self.pqc_weights = []
        for i, (shape, init, cst) in enumerate(
            zip(self.weights_shape, self.initializer, self.constraint)
        ):
            self.pqc_weights.append(
                self.add_weight(
                    name="PQCweights%s" % i,
                    shape=shape,
                    dtype=getattr(np, rdtypestr),  # type: ignore
                    trainable=True,
                    initializer=init,
                    constraint=cst,
                )
            )

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None,
        **kwargs: Any
    ) -> tf.Tensor:
        # input_shape = list(inputs.shape)
        # inputs = tf.reshape(inputs, (-1, input_shape[-1]))
        if inputs is None:
            result = self.f(*self.pqc_weights, **kwargs)
        elif len(inputs.shape) == 1:
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


# TODO(@refraction-ray): more specific keras layer as provided by tfq


def output_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return y_pred
