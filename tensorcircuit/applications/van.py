import tensorflow as tf
import numpy as np


class MaskedLinear(tf.keras.layers.Layer):
    def __init__(self, input_space, output_space, spin_channel, mask=None, dtype=None):
        super().__init__()
        if dtype is None:
            self._dtype = tf.float32
        else:
            self._dtype = dtype
        self.w = self.add_weight(
            shape=(output_space, spin_channel, input_space, spin_channel),
            initializer="random_normal",
            trainable=True,
            dtype=self._dtype,
        )
        self.b = self.add_weight(
            shape=(output_space, spin_channel),
            initializer="zero",
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

    def call(self, inputs):
        w_mask = tf.multiply(self.mask, self.w)
        return tf.tensordot(inputs, w_mask, axes=[[-2, -1], [2, 3]]) + self.b


class MADE(tf.keras.Model):
    def __init__(
        self,
        input_space,
        output_space,
        hidden_space,
        spin_channel,
        depth,
        evenly=True,
        dtype=None,
        activation=None,
    ):
        super().__init__()
        if not dtype:
            self._dtype = tf.float32
        else:
            self._dtype = dtype
        self._m = []
        self._masks = []
        self.ml_layer = []
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
        for i in range(depth):
            self.relu.append(activation())

    def _build_masks(
        self, input_space, output_space, hidden_space, spin_channel, depth, evenly
    ):
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
                        hidden_space % input_space == 0
                    ), "hidden space must by multiple of input space when you set evenly as True"
                    m = np.arange(
                        0, input_space
                    )  # TODO: whether 0 is ok need further scrunity, this is a bit away from original MADE idea
                    self._m.append(
                        np.hstack([m for _ in range(hidden_space // input_space)])
                    )
                else:
                    self._m.append(np.random.randint(0, input_space, size=hidden_sapce))
            if i == depth:
                mask = self._m[i][None, :] > self._m[i - 1][:, None]
            else:
                # input to hidden & hidden to hidden
                mask = self._m[i][None, :] >= self._m[i - 1][:, None]
            # add two spin channel axis into mask
            mask = mask.T
            mask = mask[:, np.newaxis, :, np.newaxis]
            mask = np.tile(mask, [1, spin_channel, 1, spin_channel])
            self._masks.append(tf.constant(mask, dtype=self._dtype))
            if i == depth:
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

    def _check_masks(self):
        pass

    def call(self, inputs):
        for i in range(self.depth):
            inputs = self.ml_layer[i](inputs)
            inputs = self.relu[i](inputs)
        outputs = tf.keras.layers.Softmax()(inputs)
        return outputs

    def model(self):
        x = tf.keras.layers.Input(shape=(self.input_space, self.spin_channel))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def sample(self, batch_size):
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

    def _log_prob(self, sample, x_hat):
        eps = 1e-10
        probm = tf.multiply(x_hat, sample)
        probm = tf.reduce_sum(probm, axis=-1)
        lnprobm = tf.math.log(probm + eps)
        return tf.reduce_sum(lnprobm, axis=-1)

    def log_prob(self, sample):
        x_hat = self.call(sample)
        log_prob = self._log_prob(sample, x_hat)

        return log_prob


class MaskedConv2D(tf.keras.layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super().__init__()
        assert mask_type in {"A", "B"}, "mask_type must be in A or B"
        self.mask_type = mask_type
        self.conv = tf.keras.layers.Conv2D(**kwargs)

    def build(self, input_shape):
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

    def call(self, inputs):
        self.conv.kernel = self.kernel * self.mask
        self.conv.bias = self.bias
        return self.conv(inputs)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def call(self, inputs):
        y = inputs
        for layer in self.layers:
            y = layer(y)
        return y + inputs


class PixelCNN(tf.keras.Model):
    def __init__(self, spin_channel, depth, filters):
        super().__init__()
        self.rb = []
        self.rb0 = MaskedConv2D(
            mask_type="A", filters=filters, kernel_size=3, padding="same"
        )

        self.relu = tf.keras.layers.PReLU()
        for i in range(1, depth):
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

    def call(self, inputs):
        x = self.rb0(inputs)
        x = self.relu(x)
        for i in range(self.depth - 1):
            x = self.rb[i](x)
        x = self.rbn(x)
        x = tf.keras.layers.Softmax(axis=-1)(x)
        return x

    def sample(self, batch_size, h, w):
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

    def _log_prob(self, sample, x_hat):
        eps = 1e-10
        probm = tf.multiply(x_hat, sample)
        probm = tf.reduce_sum(probm, axis=-1)
        lnprobm = tf.math.log(probm + eps)

        return tf.reduce_sum(lnprobm, axis=[-1, -2])

    def log_prob(self, sample):
        x_hat = self.call(sample)
        log_prob = self._log_prob(sample, x_hat)

        return log_prob
