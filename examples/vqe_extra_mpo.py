"""
Demonstration of TFIM VQE with extra size in MPO formulation
"""

import time
import logging
import sys
import numpy as np

logger = logging.getLogger("tensorcircuit")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

sys.setrecursionlimit(10000)

import tensorflow as tf
import tensornetwork as tn
import cotengra as ctg
import optax

import tensorcircuit as tc

opt = ctg.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    parallel="ray",
    minimize="combo",
    max_time=360,
    max_repeats=4096,
    progbar=True,
)


def opt_reconf(inputs, output, size, **kws):
    tree = opt.search(inputs, output, size)
    tree_r = tree.subtree_reconfigure_forest(
        parallel="ray",
        progbar=True,
        num_trees=20,
        num_restarts=20,
        subtree_weight_what=("size",),
    )
    return tree_r.path()


tc.set_contractor("custom", optimizer=opt_reconf, preprocessing=True)
tc.set_dtype("complex64")
tc.set_backend("tensorflow")
# jax backend is incompatible with keras.save

dtype = np.complex64

nwires, nlayers = 150, 7  # 600, 7


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


if __name__ == "__main__":
    refresh = False

    time0 = time.time()
    if refresh:
        tc_vg = tf.function(
            tc.backend.value_and_grad(vqe_forward),
            input_signature=[tf.TensorSpec([4 * nlayers, nwires], tf.float32)],
        )
        tc.keras.save_func(tc_vg, "./funcs/%s_%s_tfim_mpo" % (nwires, nlayers))
        time1 = time.time()
        print("staging time: ", time1 - time0)

    tc_vg_loaded = tc.keras.load_func("./funcs/%s_%s_tfim_mpo" % (nwires, nlayers))

    lr1 = 0.008
    lr2 = 0.06
    steps = 2000
    switch = 400
    debug_steps = 20

    if tc.backend.name == "jax":
        opt = tc.backend.optimizer(optax.adam(lr1))
        opt2 = tc.backend.optimizer(optax.sgd(lr2))
    else:
        opt = tc.backend.optimizer(tf.keras.optimizers.Adam(lr1))
        opt2 = tc.backend.optimizer(tf.keras.optimizers.SGD(lr2))

    times = []
    param = tc.backend.implicit_randn(stddev=0.1, shape=[4 * nlayers, nwires])

    for j in range(steps):
        loss, gr = tc_vg_loaded(param)
        if j < switch:
            param = opt.update(gr, param)
        else:
            if j == switch:
                print("switching the optimizer")
            param = opt2.update(gr, param)
        if j % debug_steps == 0 or j == steps - 1:
            times.append(time.time())
            print("loss", tc.backend.numpy(loss))
            if j > 0:
                print("running time:", (times[-1] - times[0]) / j)


"""
# Baseline code: obtained from DMRG using quimb

import quimb

h = quimb.tensor.tensor_gen.MPO_ham_ising(nwires, 4, 2, cyclic=False)
dmrg = quimb.tensor.tensor_dmrg.DMRG2(m, bond_dims=[10, 20, 100, 100, 200], cutoffs=1e-10)
dmrg.solve(tol=1e-9, verbosity=1) # may require repetition of this API
"""
