"""
One-hot variational autoregressive models for multiple categorical choices beyond binary
"""

from typing import Any, Optional, Tuple, Union, List
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
import numpy as np


class MaskedLinear(Layer):  # type: ignore
    def __init__(
        self,
        input_space: int,
        output_space: int,
        spin_channel: int,
        mask: Optional[tf.Tensor] = None,
        dtype: Optional[tf.DType] = None,
    ):
        super().__init__()
        if dtype is None:
            self._dtype = tf.float32
        else:
            self._dtype = dtype
        self.w = self.add_weight(
            name="w",
            shape=(output_space, spin_channel, input_space, spin_channel),
            initializer="random_normal",
            trainable=True,
            dtype=self._dtype,
        )
        self.b = self.add_weight(
            name="b",  # every variables in customized layer must have a explicit name so that the model can be saved
            # see https://github.com/tensorflow/tensorflow/issues/26811
            shape=(output_space, spin_channel),
            initializer="random_normal",  # zero is not ok, as the first output will always equal probabilitied
            trainable=True,
            dtype=self._dtype,
        )
        self.input_space = input_space
        self.output_space = output_space
        self.spin_channel = spin_channel
        if mask is not None:
            self.mask = mask
        else:
            self.mask = tf.ones([output_space, spin_channel, input_space, spin_channel])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        w_mask = tf.multiply(self.mask, self.w)
        return tf.tensordot(inputs, w_mask, axes=[[-2, -1], [2, 3]]) + self.b

    def regularization(self, lbd_w: float = 1.0, lbd_b: float = 1.0) -> tf.Tensor:
        return lbd_w * tf.reduce_sum(self.w**2) + lbd_b * tf.reduce_sum(self.b**2)


class MADE(Model):  # type: ignore
    def __init__(
        self,
        input_space: int,
        output_space: int,
        hidden_space: int,
        spin_channel: int,
        depth: int,
        evenly: bool = True,
        dtype: Optional[tf.DType] = None,
        activation: Optional[Any] = None,
        nonmerge: bool = True,
        probamp: Optional[tf.Tensor] = None,
    ):
        super().__init__()
        if not dtype:
            self._dtype = tf.float32
        else:
            self._dtype = dtype
        self._m: List[np.array] = []  # type: ignore
        self._masks: List[tf.Tensor] = []
        self.ml_layer: List[Any] = []
        self.input_space = input_space
        self.spin_channel = spin_channel
        self._build_masks(
            input_space, output_space, hidden_space, spin_channel, depth, evenly
        )
        self._check_masks()
        self.depth = depth
        self.relu = []
        if activation is None:
            activation = tf.keras.layers.PReLU
        for _ in range(depth):
            self.relu.append(activation())
        self.nonmerge = nonmerge
        self.probamp = probamp

    def _build_masks(
        self,
        input_space: int,
        output_space: int,
        hidden_space: int,
        spin_channel: int,
        depth: int,
        evenly: bool,
    ) -> None:
        self._m.append(np.arange(1, input_space + 1))
        for i in range(1, depth + 1):
            if i == depth:  # the last layer
                # assign output layer units a number between 1 and D
                m = np.arange(1, input_space + 1)
                assert (
                    output_space % input_space == 0
                ), "output space must by multiple of input space"
                self._m.append(
                    np.hstack([m for _ in range(output_space // input_space)])
                )
            else:  # middle layer
                # assign hidden layer units a number between 1 and D-1
                if evenly:
                    assert (
                        hidden_space % (input_space - 1) == 0
                    ), "hidden space must be multiple of input space -1 when you set evenly as True"
                    m = np.arange(1, input_space)
                    # TODO: whether 0 is ok need further scrunity, this is a bit away from original MADE idea
                    self._m.append(
                        np.hstack([m for _ in range(hidden_space // (input_space - 1))])
                    )
                else:
                    self._m.append(np.random.randint(1, input_space, size=hidden_space))
            if i == depth:
                mask = self._m[i][None, :] > self._m[i - 1][:, None]  # type: ignore
            else:
                # input to hidden & hidden to hidden
                mask = self._m[i][None, :] >= self._m[i - 1][:, None]  # type: ignore
            # add two spin channel axis into mask
            mask = mask.T
            mask = mask[:, np.newaxis, :, np.newaxis]
            mask = np.tile(mask, [1, spin_channel, 1, spin_channel])
            self._masks.append(tf.constant(mask, dtype=self._dtype))
            if i == depth and depth == 1:
                # in case there is only one layer
                self.ml_layer.append(
                    MaskedLinear(
                        input_space,
                        output_space,
                        spin_channel,
                        mask=self._masks[i - 1],
                        dtype=self._dtype,
                    )
                )
            elif i == depth:
                self.ml_layer.append(
                    MaskedLinear(
                        hidden_space,
                        output_space,
                        spin_channel,
                        mask=self._masks[i - 1],
                        dtype=self._dtype,
                    )
                )
            elif i == 1:
                self.ml_layer.append(
                    MaskedLinear(
                        input_space,
                        hidden_space,
                        spin_channel,
                        mask=self._masks[i - 1],
                        dtype=self._dtype,
                    )
                )
            else:
                self.ml_layer.append(
                    MaskedLinear(
                        hidden_space,
                        hidden_space,
                        spin_channel,
                        mask=self._masks[i - 1],
                        dtype=self._dtype,
                    )
                )

    def _check_masks(self) -> None:
        pass

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.nonmerge:
            nonmerge_block = tf.roll(inputs, axis=-2, shift=1) * -10.0
            # PBC here
            # once the last pixel is i, the next one cannot be i
        for i in range(self.depth):
            inputs = self.ml_layer[i](inputs)
            inputs = self.relu[i](inputs)
        if self.probamp is not None:
            inputs = self.probamp + inputs
        if self.nonmerge:
            inputs = nonmerge_block + inputs
            # TODO: to be investigated whether the nonmerge block can affect on unity of probability
        outputs = tf.keras.layers.Softmax()(inputs)
        return outputs

    def model(self) -> Any:
        x = tf.keras.layers.Input(shape=(self.input_space, self.spin_channel))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def regularization(self, lbd_w: float = 1.0, lbd_b: float = 1.0) -> tf.Tensor:
        loss = 0.0
        for l in self.ml_layer:
            loss += l.regularization(lbd_w=lbd_w, lbd_b=lbd_b)
        return loss

    def sample(self, batch_size: int) -> Tuple[np.array, tf.Tensor]:  # type: ignore
        eps = 1e-10
        sample = np.zeros(
            [batch_size, self.input_space, self.spin_channel], dtype=np.float32
        )
        for i in range(self.input_space):
            x_hat = self.call(sample)
            ln_x_hat = tf.math.log(
                tf.reshape(x_hat, [batch_size * self.input_space, self.spin_channel])
                + eps
            )
            choice = tf.random.categorical(ln_x_hat, num_samples=1)
            choice = tf.reshape(choice, [choice.shape[0]])
            one_hot = tf.one_hot(choice, depth=self.spin_channel)
            one_hot = tf.reshape(
                one_hot, [batch_size, self.input_space, self.spin_channel]
            )
            sample[:, i, :] = one_hot[:, i, :]

        return sample, x_hat

    def _log_prob(self, sample: tf.Tensor, x_hat: tf.Tensor) -> tf.Tensor:
        eps = 1e-10
        probm = tf.multiply(x_hat, sample)
        probm = tf.reduce_sum(probm, axis=-1)
        lnprobm = tf.math.log(probm + eps)
        return tf.reduce_sum(lnprobm, axis=-1)

    def log_prob(self, sample: tf.Tensor) -> tf.Tensor:
        x_hat = self.call(sample)
        log_prob = self._log_prob(sample, x_hat)
        return log_prob


class MaskedConv2D(Layer):  # type: ignore
    def __init__(self, mask_type: str, **kwargs: Any):
        super().__init__()
        assert mask_type in {"A", "B"}, "mask_type must be in A or B"
        self.mask_type = mask_type
        self.conv = tf.keras.layers.Conv2D(**kwargs)

    def build(self, input_shape: Union[List[tf.TensorShape], tf.TensorShape]) -> None:
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.kernel = self.add_weight("kernel", self.conv.kernel.shape)
        self.bias = self.add_weight("bias", self.conv.bias.shape)

        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        self.conv.kernel = self.kernel * self.mask
        self.conv.bias = self.bias
        return self.conv(inputs)


class ResidualBlock(Layer):  # type: ignore
    def __init__(self, layers: List[Any]):
        super().__init__()
        self.layers = layers

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        y = inputs
        for layer in self.layers:
            y = layer(y)
        return y + inputs


class PixelCNN(Model):  # type: ignore
    def __init__(self, spin_channel: int, depth: int, filters: int):
        super().__init__()
        self.rb = []
        self.rb0 = MaskedConv2D(
            mask_type="A", filters=filters, kernel_size=3, padding="same"
        )

        self.relu = tf.keras.layers.PReLU()
        for _ in range(1, depth):
            self.rb.append(
                ResidualBlock(
                    [
                        MaskedConv2D(
                            mask_type="B",
                            filters=filters,
                            kernel_size=3,
                            padding="same",
                        ),
                        tf.keras.layers.PReLU(),
                    ]
                )
            )
        self.rbn = MaskedConv2D(
            mask_type="B", filters=spin_channel, kernel_size=1, padding="same"
        )
        self.depth = depth
        self.spin_channel = spin_channel

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.rb0(inputs)
        x = self.relu(x)
        for i in range(self.depth - 1):
            x = self.rb[i](x)
        x = self.rbn(x)
        x = tf.keras.layers.Softmax(axis=-1)(x)
        return x

    def sample(self, batch_size: int, h: int, w: int) -> Tuple[np.array, tf.Tensor]:  # type: ignore
        eps = 1e-10
        sample = np.zeros([batch_size, h, w, self.spin_channel], dtype=np.float32)
        for i in range(h):
            for j in range(w):
                x_hat = self.call(sample)
                ln_x_hat = tf.math.log(
                    tf.reshape(x_hat, [batch_size * h * w, self.spin_channel]) + eps
                )
                choice = tf.random.categorical(ln_x_hat, num_samples=1)
                choice = tf.reshape(choice, [choice.shape[0]])
                one_hot = tf.one_hot(choice, depth=self.spin_channel)
                one_hot = tf.reshape(one_hot, [batch_size, h, w, self.spin_channel])
                sample[:, i, j, :] = one_hot[:, i, j, :]

        return sample, x_hat

    def _log_prob(self, sample: tf.Tensor, x_hat: tf.Tensor) -> tf.Tensor:
        eps = 1e-10
        probm = tf.multiply(x_hat, sample)
        probm = tf.reduce_sum(probm, axis=-1)
        lnprobm = tf.math.log(probm + eps)
        return tf.reduce_sum(lnprobm, axis=[-1, -2])

    def log_prob(self, sample: tf.Tensor) -> tf.Tensor:
        x_hat = self.call(sample)
        log_prob = self._log_prob(sample, x_hat)
        return log_prob


class NMF(Model):  # type: ignore
    def __init__(
        self,
        spin_channel: int,
        *dimensions: int,
        _dtype: tf.DType = tf.float32,
        probamp: Optional[tf.Tensor] = None
    ):
        super().__init__()
        self.w = self.add_weight(
            name="meanfield-parameter",
            shape=list(dimensions) + [spin_channel],
            initializer="random_normal",
            trainable=True,
            dtype=_dtype,
        )
        self._dtype = _dtype
        self.dimensions = list(dimensions)
        self.D = len(self.dimensions)
        self.spin_channel = spin_channel
        self.probamp = probamp

    def call(self, inputs: Optional[tf.Tensor] = None) -> tf.Tensor:
        # x = tf.keras.layers.Softmax(axis=-1)(self.w)
        if self.probamp is None:
            return self.w
        else:
            return self.w + self.probamp

    def sample(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        x_hat = self.call()
        x_hat = x_hat[tf.newaxis, :]
        tile_shape = tuple([batch_size] + [1 for _ in range(self.D + 1)])
        x_hat = tf.tile(x_hat, tile_shape)
        totalsize = tf.reduce_prod(self.dimensions)
        logits = tf.reshape(x_hat, (batch_size * totalsize, self.spin_channel))
        sample = tf.random.categorical(logits, 1)  # [system size, batch size]
        # [ * dimensions, batch size, spin channel]
        sample = tf.one_hot(sample, depth=self.spin_channel)
        sample = tf.reshape(
            sample, [batch_size] + self.dimensions + [self.spin_channel]
        )
        x_hat = tf.keras.layers.Softmax(axis=-1)(x_hat)
        return sample, x_hat

    def _log_prob(self, sample: tf.Tensor, x_hat: tf.Tensor) -> tf.Tensor:
        eps = 1e-10
        probm = tf.multiply(x_hat, sample)
        probm = tf.reduce_sum(probm, axis=-1)
        lnprobm = tf.math.log(probm + eps)
        return tf.reduce_sum(lnprobm, axis=[-i - 1 for i in range(self.D)])

    def log_prob(self, sample: tf.Tensor) -> tf.Tensor:
        x_hat = self.call(sample)
        log_prob = self._log_prob(sample, x_hat)
        return log_prob
