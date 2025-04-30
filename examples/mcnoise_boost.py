"""
Boost the Monte Carlo noise simulation (specifically the staging time)
on general error with circuit layerwise slicing
"""

import time
import sys

sys.path.insert(0, "../")
import tensorcircuit as tc

tc.set_backend("jax")

n = 3  # 10
nlayer = 2  # 4


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


def templatecnot(s, param, i):
    c = tc.Circuit(n, inputs=s)
    c.cnot(i, i + 1)
    return c.state()


def templatenoise(key, s, param, i):
    c = tc.Circuit(n, inputs=s)
    status = tc.backend.stateful_randu(key)[0]
    c.apply_general_kraus(tc.channels.phasedampingchannel(0.15), i, status=status)
    return c.state()


def templaterz(s, param, j):
    c = tc.Circuit(n, inputs=s)
    for i in range(n):
        c.rx(i, theta=param[j, i])
    return c.state()


def f2(key, param, n, nlayer):
    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)
    s = c.state()
    for j in range(nlayer):
        for i in range(n - 1):
            s = templatecnot(s, param, i)
            key, subkey = tc.backend.random_split(key)
            s = templatenoise(subkey, s, param, i)
            key, subkey = tc.backend.random_split(key)
            s = templatenoise(subkey, s, param, i + 1)
        s = templaterz(s, param, j)
    return tc.backend.real(tc.expectation((tc.gates.z(), [int(n / 2)]), ket=s))


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

# 10*4: jax*T4: 235/0.36 vs. 26/0.04
