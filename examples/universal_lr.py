"""
Backend agnostic linear regression with gradient descent optimization:
a demonstration on most of core features and paradigm of tensorcircuit
"""

# this script shows how backend agnostic magic works, no code change is required to switch backend
# we also include jit, vmap and AD features in this pure classical example
# this demonstrates that tensorcircuit can serve as a solid unified ML library without any "quantumness"

import sys

sys.path.insert(0, "../")
import numpy as np
import tensorcircuit as tc

# (x, y) data preparation

nsamples = 100
k0 = 1.0
b0 = 0.0

xs0 = np.random.uniform(low=-1, high=1, size=[nsamples])
ys0 = k0 * xs0 + b0 + np.random.normal(scale=0.1, size=[nsamples])


def lr(xs, ys):
    """
    fully ML backend agnostic linear regression implementation
    """
    # construct the loss
    def loss_pointwise(x, y, param):
        k, b = param["k"], param["b"]
        yp = k * x + b
        return (yp - y) ** 2

    # we suppose this loss function only works for scalar, so that we can show the usage of ``vmap``

    loss_vmap = tc.backend.vmap(loss_pointwise, vectorized_argnums=(0, 1))

    # now we define the total loss for all data

    def loss(xs, ys, param):
        losses = loss_vmap(xs, ys, param)
        return tc.backend.sum(losses)

    # we get the jitted function to evaluate loss and its derivatives wrt. param

    loss_and_grad = tc.backend.jit(tc.backend.value_and_grad(loss, argnums=2))

    # setup initial values and optimizers

    weight = {"k": tc.backend.implicit_randn(), "b": tc.backend.implicit_randn()}

    if tc.backend.name == "tensorflow":
        import tensorflow as tf

        opt = tc.backend.optimizer(tf.keras.optimizers.Adam(1e-2))
    elif tc.backend.name == "jax":
        import optax

        opt = tc.backend.optimizer(optax.adam(1e-2))
    else:
        raise ValueError("Unsupported backend")

    # gradient descent optimization loop
    maxstep = 500
    for i in range(maxstep):
        loss, grad = loss_and_grad(xs, ys, weight)
        weight = opt.update(grad, weight)
        if i % 100 == 0 or i == maxstep - 1:
            print("optimized MSE loss after %s round: " % i, tc.backend.numpy(loss))

    return tc.backend.numpy(weight["k"]), tc.backend.numpy(weight["b"])


if __name__ == "__main__":

    for n in ["tensorflow", "jax"]:
        with tc.runtime_backend(n):  # runtime backend switch with context manager
            print("~~~~~~~~ using %s backend ~~~~~~~~" % n)
            xs_tensor, ys_tensor = tc.array_to_tensor(xs0, ys0, dtype="float32")
            print("predicted coefficient", lr(xs_tensor, ys_tensor))
