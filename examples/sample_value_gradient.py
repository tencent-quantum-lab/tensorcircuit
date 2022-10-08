"""
Evaluate expectation and gradient on Pauli string sum with finite measurement shots
"""
from functools import partial
import numpy as np
import tensorcircuit as tc
from tensorcircuit import experimental as E

K = tc.set_backend("jax")

n = 5
nlayers = 4
ps = []
for i in range(n):
    l = [0 for _ in range(n)]
    l[i] = 1
    ps.append(l)
for i in range(n - 1):
    l = [0 for _ in range(n)]
    l[i] = 3
    l[i + 1] = 3
    ps.append(l)


w = [-1.0 for _ in range(n)] + [1.0 for _ in range(n - 1)]


def generate_circuit(param):
    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)
    for j in range(nlayers):
        for i in range(n - 1):
            c.rzz(i, i + 1, theta=param[i, j, 0])
        for i in range(n):
            c.rx(i, theta=param[i, j, 1])
    return c


def sample_exp(c, ps, w, shots, key):
    if isinstance(shots, int):
        shots = [shots for _ in range(len(ps))]
    loss = 0
    for psi, wi, shot in zip(ps, w, shots):
        key, subkey = K.random_split(key)
        xyz = {"x": [], "y": [], "z": []}
        for i, j in enumerate(psi):
            if j == 1:
                xyz["x"].append(i)
            if j == 2:
                xyz["y"].append(i)
            if j == 3:
                xyz["z"].append(i)
        loss += wi * c.sample_expectation_ps(**xyz, shots=shot, random_generator=subkey)
    return loss


@K.jit
def exp_val_analytical(param):
    c = generate_circuit(param)
    loss = 0
    for psi, wi in zip(ps, w):
        xyz = {"x": [], "y": [], "z": []}
        for i, j in enumerate(psi):
            if j == 1:
                xyz["x"].append(i)
            if j == 2:
                xyz["y"].append(i)
            if j == 3:
                xyz["z"].append(i)
        loss += wi * c.expectation_ps(**xyz)
    return K.real(loss)


@partial(K.jit, static_argnums=2)
def exp_val(param, key, shots=4096):
    # from circuit parameter to expectation
    c = generate_circuit(param)
    return sample_exp(c, ps, w, shots, key=key)


print("benchmarking sample expectation")
tc.utils.benchmark(
    exp_val, K.ones([n, nlayers, 2], dtype="float32"), K.get_random_state(42)
)
print("benchmarking analytical expectation")
tc.utils.benchmark(exp_val_analytical, K.ones([n, nlayers, 2], dtype="float32"))
r1 = exp_val(K.ones([n, nlayers, 2], dtype="float32"), K.get_random_state(42))
r2 = exp_val_analytical(K.ones([n, nlayers, 2], dtype="float32"))
np.testing.assert_allclose(r1, r2, atol=0.05, rtol=0.01)
print("correctness check passed for expectation value")
gradf1 = E.parameter_shift_grad_v2(exp_val, argnums=0, random_argnums=1)
gradf2 = K.jit(K.grad(exp_val_analytical))
print("benchmarking sample gradient")
tc.utils.benchmark(
    gradf1, K.ones([n, nlayers, 2], dtype="float32"), K.get_random_state(42)
)
# n=12, nlayers=4, 276s + 0.75s
print("benchmarking analytical gradient")
tc.utils.benchmark(gradf2, K.ones([n, nlayers, 2], dtype="float32"))
r1 = gradf1(K.ones([n, nlayers, 2], dtype="float32"), K.get_random_state(42))
r2 = gradf2(K.ones([n, nlayers, 2], dtype="float32"))
print("gradient with measurement shot and parameter shift")
print(r1)
print(r2)
np.testing.assert_allclose(r1, r2, atol=0.2, rtol=0.01)
print("correctness check passed for gradients")
