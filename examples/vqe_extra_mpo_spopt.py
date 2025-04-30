"""
Demonstration of TFIM VQE with extra size in MPO formulation, with highly customizable scipy optimization interface
"""

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import time
import logging
import sys
import numpy as np
from scipy import optimize

logger = logging.getLogger("tensorcircuit")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

sys.setrecursionlimit(10000)

import tensornetwork as tn
import cotengra as ctg

import tensorcircuit as tc

optc = ctg.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    parallel="ray",
    minimize="combo",
    max_time=6,
    max_repeats=4096,
    progbar=True,
)


def opt_reconf(inputs, output, size, **kws):
    tree = optc.search(inputs, output, size)
    tree_r = tree.subtree_reconfigure_forest(
        parallel="ray",
        progbar=True,
        num_trees=6,
        num_restarts=6,
        subtree_weight_what=("size",),
    )
    return tree_r.path()


tc.set_contractor("custom", optimizer=opt_reconf, preprocessing=True)
tc.set_dtype("complex64")
tc.set_backend("jax")
# jax backend is incompatible with keras.save

dtype = np.complex64

nwires, nlayers = 8, 2  # 600, 7


Jx = np.array([1.0 for _ in range(nwires - 1)])  # strength of xx interaction (OBC)
Bz = np.array([-1.0 for _ in range(nwires)])  # strength of transverse field
hamiltonian_mpo = tn.matrixproductstates.mpo.FiniteTFI(
    Jx, Bz, dtype=dtype
)  # matrix product operator
hamiltonian_mpo = tc.quantum.tn2qop(hamiltonian_mpo)


def vqe_forward(param):
    print("compiling")
    split_conf = {
        "max_singular_values": 2,
        "fixed_choice": 1,
    }
    c = tc.Circuit(nwires, split=split_conf)
    for i in range(nwires):
        c.H(i)
    for j in range(nlayers):
        for i in range(0, nwires - 1):
            c.exp1(
                i,
                (i + 1) % nwires,
                theta=param[4 * j, i],
                unitary=tc.gates._xx_matrix,
            )

        for i in range(nwires):
            c.rz(i, theta=param[4 * j + 1, i])
        for i in range(nwires):
            c.ry(i, theta=param[4 * j + 2, i])
        for i in range(nwires):
            c.rz(i, theta=param[4 * j + 3, i])
    return tc.templates.measurements.mpo_expectation(c, hamiltonian_mpo)


def scipy_optimize(
    f, x0, method, jac=None, tol=1e-9, maxiter=10000, step=1000, threshold=1e-6
):
    epoch = 0
    loss_prev = 0
    threshold = 1e-6
    count = 0
    times = []
    while epoch < maxiter:
        time0 = time.time()
        r = optimize.minimize(
            f, x0=x0, method=method, tol=tol, jac=jac, options={"maxiter": step}
        )
        time1 = time.time()
        times.append(time1 - time0)
        loss = r["fun"]
        epoch += step
        x0 = r["x"]
        print(epoch, loss)
        print(r["message"])
        if len(times) > 2:
            running_time = np.mean(times[1:]) / step
            staging_time = times[0] - running_time * step
            print("staging time: ", staging_time)
            print("running time: ", running_time)
        if abs(loss - loss_prev) < threshold:
            count += 1
        loss_prev = loss
        if count > 5 + int(2000 / step):
            break
    return loss, x0, epoch


if __name__ == "__main__":
    param = tc.backend.implicit_randn(stddev=0.1, shape=[4 * nlayers, nwires])
    vqe_ng = tc.interfaces.scipy_optimize_interface(
        vqe_forward, shape=[4 * nlayers, nwires], gradient=False, jit=True
    )
    vqe_g = tc.interfaces.scipy_optimize_interface(
        vqe_forward, shape=[4 * nlayers, nwires], gradient=True, jit=True
    )
    scipy_optimize(
        vqe_ng, tc.backend.numpy(param), method="COBYLA", jac=False, maxiter=50000
    )
    scipy_optimize(
        vqe_g,
        tc.backend.numpy(param),
        method="L-BFGS-B",
        jac=True,
        tol=1e-6,
        maxiter=50000,
    )
    # for BFGS, large tol is necessary, see
    # https://stackoverflow.com/questions/34663539/scipy-optimize-fmin-l-bfgs-b-returns-abnormal-termination-in-lnsrch
