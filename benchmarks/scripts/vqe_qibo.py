import datetime
import sys
import uuid
import cpuinfo

import tensorcircuit as tc

tc.set_backend("tensorflow")
import tensorflow as tf
from functools import reduce
from operator import mul
import qibo

qibo.set_backend("tensorflow")
from qibo import gates, models, hamiltonians
from qibo.symbols import I, X, Z
import utils


def qibo_benchmark(uuid, n, nlayer, nitrs, timeLimit, minus=1):
    @tf.function
    def geth(s, n):
        ctc = tc.Circuit(n, inputs=s)
        loss = 0.0
        for i in range(n - 1):
            loss += ctc.expectation_ps(z=[i, i + 1])
        for i in range(n):
            loss -= ctc.expectation_ps(x=[i])
        return loss

    @tf.function  # jit will raise error
    def optimize(params):
        with tf.GradientTape() as tape:
            c = models.Circuit(n)
            for i in range(n):
                c.add(gates.H(i))
            for j in range(nlayer):
                for i in range(n - minus):
                    c.add(gates.CNOT(i, i + 1))
                    c.add(gates.RZ(i + 1, theta=params[j, i, 0]))
                    c.add(gates.CNOT(i, i + 1))
                for i in range(n):
                    c.add(gates.RX(0, theta=params[j, i, 1]))
            s = c().state()
            # h = hamiltonians.TFIM(n, h=-1)
            # loss = h.expectation(s)
            # failed in jit
            # qibo lacks the API for local expectation, using tc as assistance
            # we must ensure the observable is computed term by term
            # instead of the expectation for the Hamiltonian as a whole
            # to ensure the comparison is fair

            loss = geth(s, n)

        grads = tape.gradient(loss, params)
        return loss, grads

    meta = {}
    meta["Software"] = "qibo"
    meta["minus"] = minus
    meta["Cpuinfo"] = cpuinfo.get_cpu_info()["brand_raw"]
    meta["Version"] = {
        "sys": sys.version,
        "qibo": qibo.__version__,
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
    params = tf.Variable(tf.random.normal((nlayer, n, 2), dtype=tf.float64))
    opt = utils.Opt(optimize, params, tuning=True, backend="tensorflow")
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

    r = qibo_benchmark(
        _uuid,
        n,
        nlayer,
        nitrs,
        timeLimit,
        minus=minus,
    )
    utils.save(r, _uuid, path)
