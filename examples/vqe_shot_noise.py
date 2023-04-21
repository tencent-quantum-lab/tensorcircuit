"""
VQE with finite measurement shot noise
"""

from functools import partial
import numpy as np
from scipy import optimize
import optax
import tensorcircuit as tc
from tensorcircuit import experimental as E

K = tc.set_backend("jax")
# note this script only supports jax backend

n = 6
nlayers = 4

# We use OBC 1D TFIM Hamiltonian in this script

ps = []
for i in range(n):
    l = [0 for _ in range(n)]
    l[i] = 1
    ps.append(l)
    # X_i
for i in range(n - 1):
    l = [0 for _ in range(n)]
    l[i] = 3
    l[i + 1] = 3
    ps.append(l)
    # Z_i Z_i+1
w = [-1.0 for _ in range(n)] + [1.0 for _ in range(n - 1)]


def generate_circuit(param):
    # construct the circuit ansatz
    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)
    for j in range(nlayers):
        for i in range(n - 1):
            c.rzz(i, i + 1, theta=param[i, j, 0])
        for i in range(n):
            c.rx(i, theta=param[i, j, 1])
    return c


def ps2xyz(psi):
    # ps2xyz([1, 2, 2, 0]) = {"x": [0], "y": [1, 2], "z": []}
    xyz = {"x": [], "y": [], "z": []}
    for i, j in enumerate(psi):
        if j == 1:
            xyz["x"].append(i)
        if j == 2:
            xyz["y"].append(i)
        if j == 3:
            xyz["z"].append(i)
    return xyz


@partial(K.jit, static_argnums=(2))
def exp_val(param, key, shots=1024):
    # expectation with shot noise
    # ps, w: H = \sum_i w_i ps_i
    # describing the system Hamiltonian as a weighted sum of Pauli string
    c = generate_circuit(param)
    if isinstance(shots, int):
        shots = [shots for _ in range(len(ps))]
    loss = 0
    for psi, wi, shot in zip(ps, w, shots):
        key, subkey = K.random_split(key)
        xyz = ps2xyz(psi)
        loss += wi * c.sample_expectation_ps(**xyz, shots=shot, random_generator=subkey)
    return K.real(loss)


@K.jit
def exp_val_analytical(param):
    c = generate_circuit(param)
    loss = 0
    for psi, wi in zip(ps, w):
        xyz = ps2xyz(psi)
        loss += wi * c.expectation_ps(**xyz)
    return K.real(loss)


# 0. Exact result

hm = tc.quantum.PauliStringSum2COO(ps, w, numpy=True)
hm = K.to_dense(hm)
e, v = np.linalg.eigh(hm)
print("exact ground state energy: ", e[0])

# 1.1 VQE with numerically exact expectation: gradient free

print("VQE without shot noise")

exp_val_analytical_sp = tc.interfaces.scipy_interface(
    exp_val_analytical, shape=[n, nlayers, 2], gradient=False
)

r = optimize.minimize(
    exp_val_analytical_sp,
    np.zeros([n * nlayers * 2]),
    method="COBYLA",
    options={"maxiter": 5000},
)
print(r)


# 1.2 VQE with numerically exact expectation: gradient based

exponential_decay_scheduler = optax.exponential_decay(
    init_value=1e-2, transition_steps=500, decay_rate=0.9
)
opt = K.optimizer(optax.adam(exponential_decay_scheduler))
param = K.implicit_randn([n, nlayers, 2], stddev=0.1)  # zeros stall the gradient
exp_val_grad_analytical = K.jit(K.value_and_grad(exp_val_analytical))
for i in range(1000):
    e, g = exp_val_grad_analytical(param)
    param = opt.update(g, param)
    if i % 100 == 99:
        print(e)


# 2.1 VQE with finite shot noise: gradient free

print("VQE with shot noise")


rkey = K.get_random_state(42)


def exp_val_wrapper(param):
    global rkey
    rkey, skey = K.random_split(rkey)
    # maintain stateless randomness in scipy optimize interface
    return exp_val(param, skey, shots=1024)


exp_val_sp = tc.interfaces.scipy_interface(
    exp_val_wrapper, shape=[n, nlayers, 2], gradient=False
)

r = optimize.minimize(
    exp_val_sp,
    np.random.normal(scale=0.1, size=[n * nlayers * 2]),
    method="COBYLA",
    options={"maxiter": 5000},
)
print(r)

# the real energy position after optimization

print("converged as: ", exp_val_analytical_sp(r["x"]))


# 2.2 VQE with finite shot noise: gradient based

exponential_decay_scheduler = optax.exponential_decay(
    init_value=1e-2, transition_steps=500, decay_rate=0.9
)
opt = K.optimizer(optax.adam(exponential_decay_scheduler))
param = K.implicit_randn([n, nlayers, 2], stddev=0.1)  # zeros stall the gradient
exp_grad = E.parameter_shift_grad_v2(exp_val, argnums=0, random_argnums=1)
rkey = K.get_random_state(42)

for i in range(1000):
    rkey, skey = K.random_split(rkey)
    g = exp_grad(param, skey)
    param = opt.update(g, param)
    if i % 100 == 99:
        rkey, skey = K.random_split(rkey)
        print(exp_val(param, skey))

# the real energy position after optimization

print("converged as:", exp_val_analytical(param))
