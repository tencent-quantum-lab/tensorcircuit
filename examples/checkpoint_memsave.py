"""
Some possible attempts to save memory from the state-like simulator with checkpoint tricks (jax support only).
"""

import time
import sys
import logging

import numpy as np
import jax
import cotengra as ctg

logger = logging.getLogger("tensorcircuit")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

sys.path.insert(0, "../")
sys.setrecursionlimit(10000)

import tensorcircuit as tc

optr = ctg.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    parallel=True,
    minimize="write",
    max_time=15,
    max_repeats=512,
    progbar=True,
)
tc.set_contractor("custom", optimizer=optr, preprocessing=True)
tc.set_dtype("complex64")
tc.set_backend("jax")


nwires, nlayers = 10, 36
sn = int(np.sqrt(nlayers))


def recursive_checkpoint(funs):
    if len(funs) == 1:
        return funs[0]
    elif len(funs) == 2:
        f1, f2 = funs
        return lambda s, param: f1(
            f2(s, param[: len(param) // 2]), param[len(param) // 2 :]
        )
    else:
        f1 = recursive_checkpoint(funs[len(funs) // 2 :])
        f2 = recursive_checkpoint(funs[: len(funs) // 2])
        return lambda s, param: f1(
            jax.checkpoint(f2)(s, param[: len(param) // 2]), param[len(param) // 2 :]
        )


# not suggest in general for recursive checkpoint: too slow for staging (compiling)

"""
test case:
def f(s, param):
    return s + param
fc = recursive_checkpoint([f for _ in range(100)])
print(fc(jnp.zeros([2]), jnp.array([[i, i] for i in range(100)])))
"""


@jax.checkpoint
@jax.jit
def zzxlayer(s, param):
    c = tc.Circuit(nwires, inputs=s)
    for i in range(0, nwires):
        c.exp1(
            i,
            (i + 1) % nwires,
            theta=param[0, i],
            unitary=tc.gates._zz_matrix,
        )
    for i in range(nwires):
        c.rx(i, theta=param[0, nwires + i])
    return c.state()


@jax.checkpoint
@jax.jit
def zzxsqrtlayer(s, param):
    for i in range(sn):
        s = zzxlayer(s, param[i : i + 1])
    return s


@jax.jit
def totallayer(s, param):
    for i in range(sn):
        s = zzxsqrtlayer(s, param[i * sn : (i + 1) * sn])
    return s


def vqe_forward(param):
    s = tc.backend.ones([2**nwires])
    s /= tc.backend.norm(s)
    s = totallayer(s, param)
    e = tc.expectation((tc.gates.x(), [1]), ket=s)
    return tc.backend.real(e)


def profile(tries=3):
    time0 = time.time()
    tc_vg = tc.backend.jit(tc.backend.value_and_grad(vqe_forward))
    param = tc.backend.cast(tc.backend.ones([nlayers, 2 * nwires]), "complex64")
    print(tc_vg(param))

    time1 = time.time()
    for _ in range(tries):
        print(tc_vg(param)[0])

    time2 = time.time()
    print(time1 - time0, (time2 - time1) / tries)


profile()
