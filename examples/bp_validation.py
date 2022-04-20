import numpy as np
from tqdm import tqdm
import tensorcircuit as tc

xx = tc.gates._xx_matrix
yy = tc.gates._yy_matrix
zz = tc.gates._zz_matrix

nqubits, nlayers = 8, 6
# number of qubits and number of even-odd brick layers in the circuit
K = tc.set_backend("tensorflow")


def ansatz(param, gate_param):
    # gate_param here is a 2D-vector for theta and phi in the XXZ gate
    c = tc.Circuit(nqubits)
    # customize the circuit structure as you like
    for j in range(nlayers):
        for i in range(nqubits):
            c.ry(i, theta=param[j, 0, i, 0])
            c.rz(i, theta=param[j, 0, i, 1])
            c.ry(i, theta=param[j, 0, i, 2])
        for i in range(0, nqubits - 1, 2):  # even brick
            c.exp(
                i,
                i + 1,
                theta=1.0,
                unitary=gate_param[0] * (xx + yy) + gate_param[1] * zz,
            )
        for i in range(nqubits):
            c.ry(i, theta=param[j, 1, i, 0])
            c.rz(i, theta=param[j, 1, i, 1])
            c.ry(i, theta=param[j, 1, i, 2])
        for i in range(1, nqubits - 1, 2):  # odd brick with OBC
            c.exp(
                i,
                i + 1,
                theta=1.0,
                unitary=gate_param[0] * (xx + yy) + gate_param[1] * zz,
            )
    return c


def measure_z(param, gate_param, i):
    c = ansatz(param, gate_param)
    # K.real(c.expectation_ps(x=[i, i+1])) for measure on <X_iX_i+1>
    return K.real(c.expectation_ps(z=[i]))


gradf = K.jit(K.vvag(measure_z, argnums=0, vectorized_argnums=0), static_argnums=2)
# vectorized parallel the computation on circuit gradients for different circuit parameters

if __name__ == "__main__":
    batch = 100
    # tune batch as large as possible if the memory allows
    reps = 20
    measure_on = nqubits // 2
    # which qubit the sigma_z observable is on

    # we will average `reps*batch` different unitaries
    rlist = []
    gate_param = np.array([0, np.pi / 4])
    gate_param = tc.array_to_tensor(gate_param)
    for _ in tqdm(range(reps)):
        param = np.random.uniform(0, 2 * np.pi, size=[batch, nlayers, 2, nqubits, 3])
        param = tc.array_to_tensor(param, dtype="float32")
        _, gs = gradf(param, gate_param, measure_on)
        # gs.shape = [batch, nlayers, 2, nqubits, 3]
        gs = K.abs(gs) ** 2
        rlist.append(gs)
    gs2 = K.stack(rlist)
    # gs2.shape = [reps, batch, nlayers, 2, nqubits, 3]
    gs2 = K.reshape(gs2, [-1, nlayers, 2, nqubits, 3])
    gs2_mean = K.numpy(
        K.mean(gs2, axis=0)
    )  # numpy array for the averaged abs square for gradient of each element
    print(gs2_mean)
