"""
QML task on noisy PQC with vmapped Monte Carlo noise simulation
"""

import time
import tensorflow as tf
import numpy as np
import optax
import tensorcircuit as tc

K = tc.set_backend("jax")

# numpy data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., np.newaxis] / 255.0


def filter_pair(x, y, a, b):
    keep = (y == a) | (y == b)
    x, y = x[keep], y[keep]
    y = y == a
    return x, y


datapoints = 200
batch = 32
logfile = "qml_param_v2.npy"
n = 9
m = 4
maxiter = 5000

x_train, y_train = filter_pair(x_train, y_train, 0, 1)
x_train_small = tf.image.resize(x_train, (3, 3)).numpy()
x_train_bin = np.array(x_train_small > 0.5, dtype=np.float32)
x_train_bin = np.squeeze(x_train_bin).reshape([-1, 9])


mnist_data = (
    tf.data.Dataset.from_tensor_slices((x_train_bin[:datapoints], y_train[:datapoints]))
    .repeat(maxiter)
    .shuffle(datapoints)
    .batch(batch)
)


def f(param, seed, x, pn):
    c = tc.Circuit(n)
    px, py, pz = pn, pn, pn
    for i in range(n):
        c.rx(i, theta=x[i] * np.pi / 2)
    for j in range(m):
        for i in range(n - 1):
            c.cx(i, i + 1)
            c.depolarizing(i, px=px, py=py, pz=pz, status=seed[j, i, 0])
            c.depolarizing(i + 1, px=px, py=py, pz=pz, status=seed[j, i, 1])
        for i in range(n):
            c.rz(i, theta=param[j, i, 0])
        for i in range(n):
            c.rx(i, theta=param[j, i, 1])

    ypreds = K.convert_to_tensor([K.real(c.expectation_ps(z=[i])) for i in range(n)])

    return K.mean(ypreds)


vf = tc.utils.append(K.vmap(f, vectorized_argnums=1), K.mean)


def loss(param, scale, seeds, x, y, pn):
    ypred = vf(param, seeds, x, pn)
    ypred = K.sigmoid(scale * ypred)
    return (
        K.real(-y * K.log(ypred) - (1 - y) * K.log(1 - ypred)),
        ypred,
    )


def acc(yps, ys):
    yps = K.numpy(yps)
    ys = K.numpy(ys)
    yps = (np.sign(yps - 0.5) + 1) / 2
    return 1 - np.mean(np.logical_xor(ys, yps))


vgloss = K.jit(K.vvag(loss, argnums=(0, 1), vectorized_argnums=(3, 4), has_aux=True))
vloss = K.jit(K.vmap(loss, vectorized_argnums=(3, 4)))


def train(param=None, scale=None, noise=0, noc=1, fixed=True):
    times = []
    if param is None:
        param = K.implicit_randn([m, n, 2])
    if scale is None:
        scale = 15.0 * K.ones([], dtype="float32")
    else:
        scale *= K.ones([], dtype="float32")
    opt = K.optimizer(optax.adam(1e-2))
    opt2 = K.optimizer(optax.adam(5e-2))
    pn = noise * K.ones([])

    try:
        for i, (xs, ys) in zip(range(maxiter), mnist_data):  # using tf data loader here
            xs, ys = tc.array_to_tensor(xs.numpy(), ys.numpy())
            seeds = K.implicit_randu([noc, m, n, 2])
            _, grads = vgloss(param, scale, seeds, xs, ys, pn)
            param = opt.update(grads[0], param)
            if fixed is False:
                scale = opt2.update(grads[1], scale)
            if i % 40 == 0:
                times.append(time.time())
                print("%s round" % str(i))
                print("scale: ", K.numpy(scale))
                inference(param)
                if len(times) > 1:
                    print("running time est.: ", (times[-1] - times[0]) / i)
    except KeyboardInterrupt:
        pass

    np.save(logfile, K.numpy(param))
    return param


def inference(param=None, scale=None, noise=0, noc=1, debug=False):
    pn = noise * K.ones([])
    if param is None:
        param = np.load(logfile)
    if scale is None:
        scale = 15.0 * K.ones([], dtype="float32")
    else:
        scale *= K.ones([], dtype="float32")
    seeds = K.implicit_randu([noc, m, n, 2])
    vs, yps = vloss(
        param,
        scale,
        seeds,
        tc.array_to_tensor(x_train_bin[:datapoints]),
        tc.array_to_tensor(y_train[:datapoints]),
        pn,
    )
    if debug:
        print(yps)
    print("loss: ", K.mean(vs))
    print("acc on training: %.4f" % acc(yps, y_train[:datapoints]))


if __name__ == "__main__":
    train(noise=0.01, scale=40, noc=500, fixed=False)
    # inference(noise=0.03, noc=3000, scale=40, debug=True)
