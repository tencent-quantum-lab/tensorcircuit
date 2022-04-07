"""
QML task on noisy PQC with vmapped Monte Carlo noise simulation
"""

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# one need this for jax+gpu combination in some cases
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
data_preparation = "v1"

x_train, y_train = filter_pair(x_train, y_train, 0, 1)

if data_preparation == "v1":
    x_train = tf.image.resize(x_train, (int(np.sqrt(n)), int(np.sqrt(n)))).numpy()
    x_train = np.array(x_train > 0.5, dtype=np.float32)
    x_train = np.squeeze(x_train).reshape([-1, n])

else:  # "v2"
    from sklearn.decomposition import PCA

    x_train = PCA(n).fit_transform(x_train.reshape([-1, 28 * 28]))


mnist_data = (
    tf.data.Dataset.from_tensor_slices((x_train[:datapoints], y_train[:datapoints]))
    .repeat(maxiter)
    .shuffle(datapoints)
    .batch(batch)
)


def f(param, seed, x, pn):
    c = tc.Circuit(n)
    px, py, pz = pn, pn, pn
    for i in range(n):
        if data_preparation == "v1":
            c.rx(i, theta=x[i] * np.pi / 2)
        else:
            c.rx(i, theta=K.atan(x[i]))
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
    y = K.cast(y, "float32")
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


def train(param=None, scale=None, noise=0, noc=1, fixed=True, val_step=40):
    times = []
    val_times = []
    if param is None:
        param = K.implicit_randn([m, n, 2])
    if scale is None:
        scale = 15.0 * K.ones([], dtype="float32")
    else:
        scale *= K.ones([], dtype="float32")
    if K.name == "jax":
        opt = K.optimizer(optax.adam(1e-2))
        opt2 = K.optimizer(optax.adam(5e-2))
    else:
        opt = K.optimizer(tf.keras.optimizers.Adam(1e-2))
        opt2 = K.optimizer(tf.keras.optimizers.Adam(5e-2))
    pn = noise * K.ones([], dtype="float32")

    try:
        for i, (xs, ys) in zip(range(maxiter), mnist_data):  # using tf data loader here
            xs, ys = tc.array_to_tensor(xs.numpy(), ys.numpy())
            seeds = K.implicit_randu([noc, m, n, 2])
            time0 = time.time()
            _, grads = vgloss(param, scale, seeds, xs, ys, pn)
            time1 = time.time()
            times.append(time1 - time0)
            param = opt.update(grads[0], param)
            if fixed is False:
                scale = opt2.update(grads[1], scale)
            if i % val_step == 0:
                print("%s round" % str(i))
                print("scale: ", K.numpy(scale))
                time0 = time.time()
                inference(param)
                time1 = time.time()
                val_times.append(time1 - time0)
                if len(times) > 1:
                    print("batch running time est.: ", np.mean(times[1:]))
                    print("batch staging time est.: ", times[0])
                if len(val_times) > 1:
                    print("full set running time est.: ", np.mean(val_times[1:]))
                    print("full set staging time est.: ", val_times[0])
    except KeyboardInterrupt:
        pass

    np.save(logfile, K.numpy(param))
    return param


def inference(param=None, scale=None, noise=0, noc=1, debug=False):
    pn = noise * K.ones([], dtype="float32")
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
        tc.array_to_tensor(x_train[:datapoints]),
        tc.array_to_tensor(y_train[:datapoints]),
        pn,
    )
    if debug:
        print(yps)
    print("loss: ", K.mean(vs))
    print("acc on training: %.4f" % acc(yps, y_train[:datapoints]))


if __name__ == "__main__":
    train(noise=0.005, scale=30, noc=100, fixed=False)
    # inference(noise=0.01, noc=1000, scale=40, debug=True)
