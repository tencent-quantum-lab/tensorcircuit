import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tqdm import tqdm
import tensorcircuit as tc
import jax

tc.set_backend("jax")

n = 5
nlayer = 3
mctries = 100000

print(jax.devices())


def template(c):
    # dont jit me!
    for i in range(n):
        c.H(i)
    for i in range(n):
        c.rz(i, theta=tc.num_to_tensor(i))
    for _ in range(nlayer):
        for i in range(n - 1):
            c.cnot(i, i + 1)
        for i in range(n):
            c.rx(i, theta=tc.num_to_tensor(i))
        for i in range(n):
            c.apply_general_kraus(tc.channels.phasedampingchannel(0.15), i)
    return c.state()


@tc.backend.jit
def answer():
    c = tc.DMCircuit2(n)
    return template(c)


rho0 = answer()

print(rho0)


@tc.backend.jit
def f(key):
    if key is not None:
        tc.backend.set_random_state(key)
    c = tc.Circuit(n)
    return template(c)


key = jax.random.PRNGKey(42)
f(key).block_until_ready()  # build the graph

rho = 0.0

for i in tqdm(range(mctries)):
    key, subkey = jax.random.split(key)
    psi = f(subkey)  # [2**n,1]
    rho += 1 / mctries * tc.backend.transpose(psi) @ tc.backend.conj(psi)

print(rho)
print(tc.backend.trace(rho0))
print(tc.backend.trace(rho))
print("difference\n", tc.backend.abs(rho - rho0))
print("difference in total\n", tc.backend.sum(tc.backend.abs(rho - rho0)))
print("fidelity", tc.quantum.fidelity(rho, rho0))
print("trace distance", tc.quantum.trace_distance(rho, rho0))
