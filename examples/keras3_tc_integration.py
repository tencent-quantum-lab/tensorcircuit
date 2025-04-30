"""
keras3 is excellent to use together with tc, we will have unique features including:
1. turn OO paradigm to functional paradigm, i.e. reuse keras layer function in functional programming
2. batch on neural network weights
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
import keras_core as keras
import numpy as np
import optax
import tensorcircuit as tc

K = tc.set_backend("jax")

batch = 8
n = 6
layer = keras.layers.Dense(1, activation="sigmoid")
layer.build([batch, n])

data_x = np.random.choice([0, 1], size=batch * n).reshape([batch, n])
# data_y = np.sum(data_x, axis=-1) % 2
data_y = data_x[:, 0]
data_y = data_y.reshape([batch, 1])
data_x = data_x.astype(np.float32)
data_y = data_y.astype(np.float32)


print("data", data_x, data_y)


def loss(xs, ys, params, weights):
    c = tc.Circuit(n)
    c.rx(range(n), theta=xs)
    c.cx(range(n - 1), range(1, n))
    c.rz(range(n), theta=params)
    outputs = K.stack([K.real(c.expectation_ps(z=[i])) for i in range(n)])
    ypred, _ = layer.stateless_call(weights, [], outputs)
    return keras.losses.binary_crossentropy(ypred, ys), ypred


# common data batch practice
vgf = K.jit(
    K.vectorized_value_and_grad(
        loss, argnums=(2, 3), vectorized_argnums=(0, 1), has_aux=True
    )
)

params = K.implicit_randn(shape=[n])
w = K.implicit_randn(shape=[n, 1])
b = K.implicit_randn(shape=[1])
opt = K.optimizer(optax.adam(1e-2))
# seems that currently keras3'optimizer doesn't support nested list of variables

for i in range(100):
    (v, yp), gs = vgf(data_x, data_y, params, [w, b])
    params, [w, b] = opt.update(gs, (params, [w, b]))
    if i % 10 == 0:
        print(K.mean(v))

m = keras.metrics.BinaryAccuracy()
m.update_state(data_y, yp[:, None])
print("acc", m.result())


# data batch with batched and quantum neural weights

vgf2 = K.jit(
    K.vmap(
        K.vectorized_value_and_grad(
            loss, argnums=(2, 3), vectorized_argnums=(0, 1), has_aux=True
        ),
        vectorized_argnums=(2, 3),
    )
)

wbatch = 4
params = K.implicit_randn(shape=[wbatch, n])
w = K.implicit_randn(shape=[wbatch, n, 1])
b = K.implicit_randn(shape=[wbatch, 1])
opt = K.optimizer(optax.adam(1e-2))
# seems that currently keras3'optimizer doesn't support nested list of variables

for i in range(100):
    (v, yp), gs = vgf2(data_x, data_y, params, [w, b])
    params, [w, b] = opt.update(gs, (params, [w, b]))
    if i % 10 == 0:
        print(K.mean(v, axis=-1))

for i in range(wbatch):
    m = keras.metrics.BinaryAccuracy()
    m.update_state(data_y, yp[0, :, None])
    print("acc", m.result())
    m.reset_state()
