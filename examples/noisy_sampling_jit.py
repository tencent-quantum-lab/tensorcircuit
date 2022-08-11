"""
For hardware simlation, only sample interface is available and Monte Carlo simulation is enough
"""

import tensorcircuit as tc

n = 6
m = 4
pn = 0.003

K = tc.set_backend("jax")


def make_noise_circuit(c, weights, status=None):
    for j in range(m):
        for i in range(n - 1):
            c.cnot(i, i + 1)
            if c.is_dm is False:
                c.depolarizing(i, px=pn, py=pn, pz=pn, status=status[0, i, j])
                c.depolarizing(i + 1, px=pn, py=pn, pz=pn, status=status[1, i, j])
            else:
                c.depolarizing(i, px=pn, py=pn, pz=pn)
                c.depolarizing(i + 1, px=pn, py=pn, pz=pn)
        for i in range(n):
            c.rx(i, theta=weights[i, j])
    return c


@K.jit
def noise_measurement(weights, status, key):
    c = tc.Circuit(n)
    c = make_noise_circuit(c, weights, status)
    return c.sample(allow_state=True, random_generator=key)


@K.jit
def exact_result(weights):
    c = tc.DMCircuit(n)
    c = make_noise_circuit(c, weights)
    return K.real(c.expectation_ps(z=[0, 1]))


weights = K.ones([n, m])
z0z1_exact = exact_result(weights)


tries = 2**15
status = K.implicit_randu([tries, 2, n, m])
subkey = K.get_random_state(42)

# a micro benchmarking

tc.utils.benchmark(noise_measurement, weights, status[0], subkey)


rs = []
for i in range(tries):
    # can also be vmapped, but a tradeoff between number of trials here for further jit
    key, subkey = K.random_split(subkey)
    r = noise_measurement(weights, status[i], key)
    rs.append(r[0])

rs = (K.stack(rs) - 0.5) * 2
z0z1_mc = K.mean(rs[:, 0] * rs[:, 1])

print(z0z1_exact, z0z1_mc)

assert abs(z0z1_exact - z0z1_mc) < 0.03
