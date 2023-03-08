"""
Boost the Monte Carlo noise simulation (specifically the staging time)
on general error with circuit layerwise slicing: new paradigm,
essentially the same as v1, but much simpler
"""

import time
import sys

sys.path.insert(0, "../")
import tensorcircuit as tc

tc.set_backend("jax")

n = 6  # 10
nlayer = 5  # 4


def precompute(c):
    s = c.state()
    return tc.Circuit(c._nqubits, inputs=s)


def f1(key, param, n, nlayer):
    if key is not None:
        tc.backend.set_random_state(key)
    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)
    for j in range(nlayer):
        for i in range(n - 1):
            c.cnot(i, i + 1)
            c.apply_general_kraus(tc.channels.phasedampingchannel(0.15), i)
            c.apply_general_kraus(tc.channels.phasedampingchannel(0.15), i + 1)
        for i in range(n):
            c.rx(i, theta=param[j, i])
    return tc.backend.real(c.expectation((tc.gates.z(), [int(n / 2)])))


def f2(key, param, n, nlayer):
    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)
    for j in range(nlayer):
        for i in range(n - 1):
            c.cnot(i, i + 1)
            c = precompute(c)
            c.apply_general_kraus(tc.channels.phasedampingchannel(0.15), i)
            c = precompute(c)
            c.apply_general_kraus(tc.channels.phasedampingchannel(0.15), i + 1)
        for i in range(n):
            c.rx(i, theta=param[j, i])
    return tc.backend.real(c.expectation((tc.gates.z(), [int(n / 2)])))


vagf1 = tc.backend.jit(tc.backend.value_and_grad(f1, argnums=1), static_argnums=(2, 3))
vagf2 = tc.backend.jit(tc.backend.value_and_grad(f2, argnums=1), static_argnums=(2, 3))

param = tc.backend.ones([nlayer, n])


def benchmark(f, tries=3):
    time0 = time.time()
    key = tc.backend.get_random_state(42)
    print(f(key, param, n, nlayer)[0])
    time1 = time.time()
    for _ in range(tries):
        print(f(key, param, n, nlayer)[0])
    time2 = time.time()
    print(
        "staging time: ",
        time1 - time0,
        "running time: ",
        (time2 - time1) / tries,
    )


print("without layerwise slicing jit")
benchmark(vagf1)
print("=============================")
print("with layerwise slicing jit")
benchmark(vagf2)

# mac16 intel cpu: (6*5, jax) 1015, 0.0035; 31.68, 0.00082
