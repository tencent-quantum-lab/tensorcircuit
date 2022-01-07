"""
some chaotic properties calculation from circuit state
"""
from functools import partial
import sys

sys.path.insert(0, "../")
import tensorflow as tf
import tensorcircuit as tc

K = tc.set_backend("tensorflow")


# build the circuit


@partial(K.jit, static_argnums=(0, 1))
def get_state(params, n, nlayers, inputs=None):
    c = tc.Circuit(n, inputs=inputs)  # inputs is for input state of the circuit
    for i in range(nlayers):
        for j in range(n):
            c.ry(j, theta=params[i, j])
        for j in range(n):
            c.cnot(j, (j + 1) % n)
    # one can further customize the layout and gate type above
    return c.state()  # output wavefunction


params = K.implicit_randn([5, 10])

s = get_state(params, n=10, nlayers=5)

print(s)  # output state in vector form
rm = tc.quantum.reduced_density_matrix(s, cut=10 // 2)
print(tc.quantum.entropy(rm))
# entanglement
# for more quantum quantities functions, please refer to tc.quantum module

# get \partial \psi_i/ \partial \params_j (Jacobian)

jac_func = K.jacfwd(partial(get_state, n=10, nlayers=5))
print(jac_func(params))

# correlation


def get_zz(params, n, nlayers, inputs=None):
    s = get_state(params, n, nlayers, inputs)
    c = tc.Circuit(n, inputs=s)
    z1z2 = c.expectation([tc.gates.z(), [1]], [tc.gates.z(), [2]])
    return K.real(
        z1z2
    )  # one can also add serveral correlations together as energy estimation


# optimization, suppose the energy we want to minimize is just z1z2 as above

vag_func = K.jit(K.value_and_grad(get_zz), static_argnums=(1, 2))
opt = K.optimizer(tf.keras.optimizers.Adam(1e-2))

for i in range(200):  # gradient descent
    energy, grads = vag_func(params, 10, 5)
    params = opt.update(grads, params)
    if i % 20 == 0:
        print(energy)  # see energy optimization dynamics
