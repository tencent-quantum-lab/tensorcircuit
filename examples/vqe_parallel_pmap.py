"""
jax pmap paradigm for vqe on multiple gpus
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
from functools import partial
import jax
import optax
import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_contractor("cotengra")


def vqef(param, measure, n, nlayers):
    c = tc.Circuit(n)
    c.h(range(n))
    for i in range(nlayers):
        c.rzz(range(n - 1), range(1, n), theta=param[i, 0])
        c.rx(range(n), theta=param[i, 1])
    return K.real(
        tc.templates.measurements.parameterized_measurements(c, measure, onehot=True)
    )


def get_tfim_ps(n):
    tfim_ps = []
    for i in range(n):
        tfim_ps.append(tc.quantum.xyz2ps({"x": [i]}, n=n))
    for i in range(n):
        tfim_ps.append(tc.quantum.xyz2ps({"z": [i, (i + 1) % n]}, n=n))
    return K.convert_to_tensor(tfim_ps)


vqg_vgf = jax.vmap(K.value_and_grad(vqef), in_axes=(None, 0, None, None))


@partial(
    jax.pmap,
    axis_name="pmap",
    in_axes=(0, 0, None, None),
    static_broadcasted_argnums=(2, 3),
)
def update(param, measure, n, nlayers):
    # Compute the gradients on the given minibatch (individually on each device).
    loss, grads = vqg_vgf(param, measure, n, nlayers)
    grads = K.sum(grads, axis=0)
    grads = jax.lax.psum(grads, axis_name="pmap")
    loss = K.sum(loss, axis=0)
    loss = jax.lax.psum(loss, axis_name="pmap")
    param = opt.update(grads, param)
    return param, loss


if __name__ == "__main__":
    n = 8
    nlayers = 4
    ndevices = 8
    m = get_tfim_ps(n)
    m = K.reshape(m, [ndevices, m.shape[0] // ndevices] + list(m.shape[1:]))
    param = K.stateful_randn(jax.random.PRNGKey(43), shape=[nlayers, 2, n], stddev=0.1)
    param = K.stack([param] * ndevices)
    opt = K.optimizer(optax.adam(1e-2))
    for _ in range(100):
        param, loss = update(param, m, n, nlayers)
        print(loss[0])
