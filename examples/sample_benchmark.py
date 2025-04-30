"""
perfect sampling vs. state sampling
the benchmark results show that only use perfect/tensor sampling when the wavefunction doesn't fit in memory
"""

import time
import numpy as np
import tensorcircuit as tc

K = tc.set_backend("jax")
# tf staging is too slow


def construct_circuit(n, nlayers):
    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)
    for _ in range(nlayers):
        for i in range(n):
            r = np.random.randint(n - 1) + 1
            c.cnot(i, (i + r) % n)
    return c


for n in [8, 10, 12, 14, 16]:
    for nlayers in [2, 6, 10]:
        print("n: ", n, " nlayers: ", nlayers)
        c = construct_circuit(n, nlayers)
        time0 = time.time()
        s = c.sample(allow_state=True)
        time1 = time.time()
        # print(smp)
        print("state sampling time: ", time1 - time0)
        time0 = time.time()
        smp = c.sample()
        # print(smp)
        time1 = time.time()
        print("nonjit tensor sampling time: ", time1 - time0)
        time0 = time.time()
        s = c.sample(allow_state=True, batch=10)
        time1 = time.time()
        print("batch state sampling time: ", (time1 - time0) / 10)

        @K.jit
        def f(key):
            K.set_random_state(key)
            return c.sample()

        key = K.get_random_state(42)
        key1, key2 = K.random_split(key)
        time0 = time.time()
        smp = f(key1)
        time1 = time.time()
        for _ in range(10):
            key1, key2 = K.random_split(key2)
            smp = f(key1)
            # print(smp)
        time2 = time.time()
        print("jittable tensor sampling staginging time: ", time1 - time0)
        print("jittable tensor sampling running time: ", (time2 - time1) / 10)
