"""
matrix product: a new twist
rewrite matrix product in a vmap style
"""
from functools import partial

import numpy as np
import tensorcircuit as tc

for bk in ["jax", "tensorflow"]:
    with tc.runtime_backend(bk) as K:
        print("~~~~~~~~~~~~~~~~~~~~~")
        print(f"using {K.name} backend")

        @partial(K.jit, jit_compile=True)
        def mul(a, b):
            return a @ b

        def ij(i, j):
            """
            Inner product
            """
            return K.tensordot(i, j, 1)

        vij = K.vmap(ij, vectorized_argnums=1)
        vvij = K.vmap(vij, vectorized_argnums=0)

        @partial(K.jit, jit_compile=True)
        def mul2(a, b):
            b = K.transpose(b)
            return vvij(a, b)

        for shape in [(256, 4096), (4096, 256), (2048, 2048)]:
            print(shape)
            a = K.implicit_randn(shape)
            b = K.implicit_randn([shape[1], shape[0]])
            print("plain matprod")
            r1, _, _ = tc.utils.benchmark(mul, a, b, tries=10)
            print("vmap matprod")
            r2, _, _ = tc.utils.benchmark(mul2, a, b, tries=10)
            np.testing.assert_allclose(r1, r2, atol=1e-5)
