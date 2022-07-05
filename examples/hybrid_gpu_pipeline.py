"""
quantum part in tensorflow or jax, neural part in torch, both on GPU,
fantastic hybrid pipeline
"""

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import time
import numpy as np
import tensorflow as tf
import torch
import tensorcircuit as tc

K = tc.set_backend("tensorflow")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


print(device)

# dataset preparation

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., np.newaxis] / 255.0


def filter_pair(x, y, a, b):
    keep = (y == a) | (y == b)
    x, y = x[keep], y[keep]
    y = y == a
    return x, y


x_train, y_train = filter_pair(x_train, y_train, 1, 5)
x_train_small = tf.image.resize(x_train, (3, 3)).numpy()
x_train_bin = np.array(x_train_small > 0.5, dtype=np.float32)
x_train_bin = np.squeeze(x_train_bin).reshape([-1, 9])
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
x_train_torch = torch.tensor(x_train_bin)
x_train_torch = x_train_torch.to(device=device)
y_train_torch = y_train_torch.to(device=device)

n = 9
nlayers = 3

# We define the quantum function,
# note how this function is running on tensorflow


def qpreds(x, weights):
    c = tc.Circuit(n)
    for i in range(n):
        c.rx(i, theta=x[i])
    for j in range(nlayers):
        for i in range(n - 1):
            c.cnot(i, i + 1)
        for i in range(n):
            c.rx(i, theta=weights[2 * j, i])
            c.ry(i, theta=weights[2 * j + 1, i])

    return K.stack([K.real(c.expectation_ps(z=[i])) for i in range(n)])


# qpreds_vmap = K.vmap(qpreds, vectorized_argnums=0)
# qpreds_batch = tc.interfaces.torch_interface(qpreds_vmap, jit=True, enable_dlpack=True)

quantumnet = tc.TorchLayer(
    qpreds,
    weights_shape=[2 * nlayers, n],
    use_vmap=True,
    use_interface=True,
    use_jit=True,
    enable_dlpack=True,
)
# enable_dlpack = False for old version of ML libs


model = torch.nn.Sequential(quantumnet, torch.nn.Linear(9, 1), torch.nn.Sigmoid())
model = model.to(device=device)


criterion = torch.nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
nepochs = 300
nbatch = 32
times = []
for epoch in range(nepochs):
    index = np.random.randint(low=0, high=100, size=nbatch)
    # index = np.arange(nbatch)
    inputs, labels = x_train_torch[index], y_train_torch[index]
    opt.zero_grad()

    with torch.set_grad_enabled(True):
        time0 = time.time()
        yps = model(inputs)
        loss = criterion(
            torch.reshape(yps, [nbatch, 1]), torch.reshape(labels, [nbatch, 1])
        )
        loss.backward()
        if epoch % 100 == 0:
            print(loss)
        opt.step()
        time1 = time.time()
        times.append(time1 - time0)
print("training time per step: ", np.mean(times[1:]))
