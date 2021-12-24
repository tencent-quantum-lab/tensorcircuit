import sys

sys.path.insert(0, "../")
import tensorcircuit as tc
from tensorcircuit import experimental
import optax
import tensorflow as tf
import time

tc.set_backend("jax")  # tf, jax are both ok

n, nlayers = 8, 3
g = tc.templates.graphs.Line1D(n)


def state(params):
    params = tc.backend.reshape(params, [2 * nlayers, n])
    c = tc.Circuit(n)
    c = tc.templates.blocks.example_block(c, params, nlayers=nlayers)
    return c.state()


def energy(params):
    s = state(params)
    c = tc.Circuit(n, inputs=s)
    loss = tc.templates.measurements.heisenberg_measurements(
        c, g, hzz=1, hxx=0, hyy=0, hx=-1
    )
    return tc.backend.real(loss)


vags = tc.backend.jit(tc.backend.value_and_grad(energy))
lr = 1e-2
if tc.backend.name == "jax":
    oopt = optax.sgd(lr)
else:
    oopt = tf.keras.optimizers.SGD(lr)
opt = tc.backend.optimizer(oopt)

qng = tc.backend.jit(experimental.qng(state))


def train_loop(params, i):
    qmetric = qng(params)
    value, grad = vags(params)
    ngrad = tc.backend.solve(qmetric, grad, assume_a="sym")
    params = opt.update(ngrad, params)
    if i % 10 == 0:
        print(tc.backend.numpy(value))
    return params


def plain_train_loop(params, i):
    value, grad = vags(params)
    params = opt.update(grad, params)
    if i % 10 == 0:
        print(tc.backend.numpy(value))
    return params


def prof(f):
    # params = tc.backend.implicit_randn([2 * nlayers * n])
    params = 0.1 * tc.backend.ones([2 * nlayers * n])
    params = tc.backend.cast(params, "float32")
    time0 = time.time()
    params = f(params, 0)
    time1 = time.time()
    for i in range(100):
        params = f(params, i + 1)
    time2 = time.time()

    print("staging time: ", time1 - time0, "running time: ", (time2 - time1) / 100)


if __name__ == "__main__":
    print("quantum natural gradient descent")
    prof(train_loop)
    print("gradient descent")
    prof(plain_train_loop)
