"""
Interplay between jit, vmap, randomness and backend
"""

import tensorcircuit as tc

K = tc.set_backend("tensorflow")
n = 10
batch = 100

print("tensorflow backend")
# has serialization issue for random generation


@K.jit
def f(a, key):
    return a + K.stateful_randn(key, [n])


vf = K.jit(K.vmap(f))

key = K.get_random_state(42)

r, _, _ = tc.utils.benchmark(f, K.ones([n], dtype="float32"), key)
print(r)

r, _, _ = tc.utils.benchmark(vf, K.ones([batch, n], dtype="float32"), key)
print(r[:2])


K = tc.set_backend("jax")

print("jax backend")


@K.jit
def f2(a, key):
    return a + K.stateful_randn(key, [n])


vf2 = K.jit(K.vmap(f2, vectorized_argnums=(0, 1)))


key = K.get_random_state(42)

r, _, _ = tc.utils.benchmark(f2, K.ones([n], dtype="float32"), key)
print(r)

keys = K.stack([K.get_random_state(i) for i in range(batch)])

r, _, _ = tc.utils.benchmark(vf2, K.ones([batch, n], dtype="float32"), keys)
print(r[:2])
