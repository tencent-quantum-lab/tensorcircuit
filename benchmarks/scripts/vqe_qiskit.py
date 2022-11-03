import sys
import datetime
from functools import reduce
from operator import xor, add
import uuid
import numpy as np
import cpuinfo

import qiskit
from qiskit.opflow import I, X, Z, StateFn
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.opflow.gradients import Gradient

import utils


def qiskit_benchmark(uuid, n, nlayer, nitrs, timeLimit, minus=1):
    def tfim_hamiltonian(n, j=1.0, h=-1.0):
        hams = []
        for i in range(n):
            xi = [I for _ in range(n)]
            xi[i] = X
            hams.append(h * reduce(xor, xi))
        for i in range(n - 1):
            zzi = [I for _ in range(n)]
            zzi[i] = Z
            zzi[i + 1] = Z
            hams.append(j * reduce(xor, zzi))
        return reduce(add, hams)

    def gradf(values):
        hamiltonian = tfim_hamiltonian(n)
        c = QuantumCircuit(n)
        params = ParameterVector("theta", length=2 * n * nlayer)
        for i in range(n):
            c.h(i)
        for j in range(nlayer):
            for i in range(n - minus):
                c.rzz(
                    params[j * n * 2 + i * 2],
                    i,
                    (i + 1) % n,
                )
            for i in range(n):
                c.rx(params[j * n * 2 + i * 2 + 1], i)

        # Define the expectation value corresponding to the energy
        op = ~StateFn(hamiltonian) @ StateFn(c)
        grad = Gradient().convert(operator=op, params=params)

        value_dict = {params: values}
        exp_result = op.assign_parameters(value_dict).eval()
        grad_result = grad.assign_parameters(value_dict).eval()
        return np.real(exp_result), np.real(np.array(grad_result))

    meta = {}
    meta["Software"] = "qiskit"
    meta["minus"] = minus
    meta["Cpuinfo"] = cpuinfo.get_cpu_info()["brand_raw"]
    meta["Version"] = {
        "sys": sys.version,
        "numpy": np.__version__,
        "qiskit": qiskit.version.VERSION,
    }
    meta["VQE test parameters"] = {
        "nQubits": n,
        "nlayer": nlayer,
        "nitrs": nitrs,
        "timeLimit": timeLimit,
    }
    meta["UUID"] = uuid
    meta["Benchmark Time"] = (
        datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    )
    meta["Results"] = {}
    param = np.random.normal(scale=0.1, size=[2 * n * nlayer])
    opt = utils.Opt(gradf, param, tuning=True, backend="numpy")
    ct, it, Nitrs = utils.timing(opt.step, nitrs, timeLimit)
    meta["Results"]["with jit"] = {
        "Construction time": ct,
        "Iteration time": it,
        "# of actual iterations": Nitrs,
    }
    print(meta)
    return meta


if __name__ == "__main__":
    _uuid = str(uuid.uuid4())
    (
        n,
        nlayer,
        nitrs,
        timeLimit,
        isgpu,
        minus,
        path,
    ) = utils.arg()

    r = qiskit_benchmark(
        _uuid,
        n,
        nlayer,
        nitrs,
        timeLimit,
        minus=minus,
    )
    utils.save(r, _uuid, path)
