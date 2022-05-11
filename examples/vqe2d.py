"""
VQE on 2D square lattice Heisenberg model with size n*m
"""

import tensorflow as tf
import tensorcircuit as tc

# import cotengra as ctg

# optr = ctg.ReusableHyperOptimizer(
#     methods=["greedy", "kahypar"],
#     parallel=True,
#     minimize="flops",
#     max_time=120,
#     max_repeats=4096,
#     progbar=True,
# )
# tc.set_contractor("custom", optimizer=optr, preprocessing=True)

tc.set_dtype("complex64")
tc.set_backend("tensorflow")


n, m, nlayers = 3, 2, 2
coord = tc.templates.graphs.Grid2DCoord(n, m)


def singlet_init(circuit):  # assert n % 2 == 0
    nq = circuit._nqubits
    for i in range(0, nq - 1, 2):
        j = (i + 1) % nq
        circuit.X(i)
        circuit.H(i)
        circuit.cnot(i, j)
        circuit.X(j)
    return circuit


def vqe_forward(param):
    paramc = tc.backend.cast(param, dtype="complex64")
    c = tc.Circuit(n * m)
    c = singlet_init(c)
    for i in range(nlayers):
        c = tc.templates.blocks.Grid2D_entangling(
            c, coord, tc.gates._swap_matrix, paramc[i]
        )
    loss = tc.templates.measurements.heisenberg_measurements(c, coord.lattice_graph())
    return loss


vgf = tc.backend.jit(
    tc.backend.value_and_grad(vqe_forward),
)
param = tc.backend.implicit_randn(stddev=0.1, shape=[nlayers, 2 * n * m])


if __name__ == "__main__":
    lr = tf.keras.optimizers.schedules.ExponentialDecay(0.01, 100, 0.9)
    opt = tc.backend.optimizer(tf.keras.optimizers.Adam(lr))
    for j in range(1000):
        loss, gr = vgf(param)
        param = opt.update(gr, param)
        if j % 50 == 0:
            print("loss", loss.numpy())
