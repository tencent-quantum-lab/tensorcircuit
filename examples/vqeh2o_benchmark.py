"""
Time comparison for different evaluation approach on molecule VQE
"""

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# one need this for jax+gpu combination in some cases
import time
import numpy as np
from openfermion.chem import MolecularData
from openfermion.transforms import (
    get_fermion_operator,
    binary_code_transform,
    checksum_code,
    reorder,
)
from openfermion.chem import geometry_from_pubchem
from openfermion.utils import up_then_down
from openfermionpyscf import run_pyscf

import tensorcircuit as tc

K = tc.set_backend("tensorflow")


n = 12
nlayers = 2
multiplicity = 1
basis = "sto-3g"
# 14 spin orbitals for H2O
geometry = geometry_from_pubchem("h2o")
description = "h2o"
molecule = MolecularData(geometry, basis, multiplicity, description=description)
molecule = run_pyscf(molecule, run_mp2=True, run_cisd=True, run_ccsd=True, run_fci=True)
mh = molecule.get_molecular_hamiltonian()
fh = get_fermion_operator(mh)
b = binary_code_transform(reorder(fh, up_then_down), 2 * checksum_code(7, 1))
lsb, wb = tc.templates.chems.get_ps(b, 12)
print("%s terms in H2O qubit Hamiltonian" % len(wb))
mb = tc.quantum.PauliStringSum2COO_numpy(lsb, wb)
mbd = mb.todense()
mb = K.coo_sparse_matrix(
    np.transpose(np.stack([mb.row, mb.col])), mb.data, shape=(2**n, 2**n)
)
mbd = tc.array_to_tensor(mbd)


def ansatz(param):
    c = tc.Circuit(n)
    for i in [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]:
        c.X(i)
    for j in range(nlayers):
        for i in range(n - 1):
            c.cz(i, i + 1)
        for i in range(n):
            c.rx(i, theta=param[j, i])
    return c


def benchmark(vqef, tries=3):
    if tries < 0:
        vagf = K.value_and_grad(vqef)
    else:
        vagf = K.jit(K.value_and_grad(vqef))
    time0 = time.time()
    v, g = vagf(K.zeros([nlayers, n]))
    time1 = time.time()
    for _ in range(tries):
        v, g = vagf(K.zeros([nlayers, n]))
    time2 = time.time()
    print("staging time: ", time1 - time0)
    if tries > 0:
        print("running time: ", (time2 - time1) / tries)
    return (v, g), (time1 - time0, (time2 - time1) / tries)


# 1. plain Pauli sum


def vqe1(param):
    c = ansatz(param)
    loss = 0.0
    for ps, w in zip(lsb, wb):
        obs = []
        for i, p in enumerate(ps):
            if p == 1:
                obs.append([tc.gates.x(), [i]])
            elif p == 2:
                obs.append([tc.gates.y(), [i]])
            elif p == 3:
                obs.append([tc.gates.z(), [i]])

        loss += w * c.expectation(*obs)
    return K.real(loss)


# 2. vmap the Pauli sum


def measurement(s, structure):
    c = tc.Circuit(n, inputs=s)
    return tc.templates.measurements.parameterized_measurements(
        c, structure, onehot=True
    )


measurement = K.jit(K.vmap(measurement, vectorized_argnums=1))
structures = tc.array_to_tensor(lsb)
weights = tc.array_to_tensor(wb, dtype="float32")


def vqe2(param):
    c = ansatz(param)
    s = c.state()
    ms = measurement(s, structures)
    return K.sum(ms * weights)


# 3. dense matrix


def vqe3(param):
    c = ansatz(param)
    return tc.templates.measurements.operator_expectation(c, mbd)


# 4. sparse matrix


def vqe4(param):
    c = ansatz(param)
    return tc.templates.measurements.operator_expectation(c, mb)


# 5. mpo (ommited, since it is not that applicable for molecule/long range Hamiltonian
# due to large bond dimension)


if __name__ == "__main__":
    r0 = None
    des = [
        "plain Pauli sum",
        "vmap Pauli sum",
        "dense Hamiltonian matrix",
        "sparse Hamiltonian matrix",
    ]
    vqef = [vqe1, vqe2, vqe3, vqe4]
    tries = [-1, 2, 5, 5]
    for i in range(4):
        print(des[i])
        r1, _ = benchmark(vqef[i], tries=tries[i])
        # plain approach takes too long to jit
        if r0 is not None:
            np.testing.assert_allclose(r0[0], r1[0], atol=1e-5)
            np.testing.assert_allclose(r0[1], r1[1], atol=1e-5)
        r0 = r1
        print("------------------")
