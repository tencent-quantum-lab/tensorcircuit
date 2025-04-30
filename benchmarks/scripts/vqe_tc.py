import numpy as np
import utils
import datetime
import cpuinfo
import os
import sys
import uuid
import tensorcircuit as tc


def tensorcircuit_benchmark(
    uuid,
    tcbackend,
    n,
    nlayer,
    nitrs,
    timeLimit,
    isgpu,
    mpsd,
    dt="32",
    minus=1,
    contractor="auto",
):
    meta = {}

    if isgpu == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        meta["isgpu"] = "off"

    else:
        # gpu = tf.config.list_physical_devices("GPU")
        # tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)
        meta["isgpu"] = "on"
        meta["Gpuinfo"] = utils.gpuinfo()
    tc.set_backend(tcbackend)
    if dt == "32":
        tc.set_dtype("complex64")
        dtype = "complex64"
        vtype = "float32"
    else:  # dt == "64":
        tc.set_dtype("complex128")
        dtype = "complex128"
        vtype = "float64"
    if contractor not in ["auto", "cotengra"]:
        tc.set_contractor(contractor)
    elif contractor == "cotengra":
        import cotengra as ctg

        opt0 = ctg.ReusableHyperOptimizer(
            minimize="combo",
            max_repeats=512,
            max_time=90,
            progbar=True,
        )

        # def opt_reconf(inputs, output, size, **kws):
        #     tree = opt0.search(inputs, output, size)
        #     tree_r = tree.subtree_reconfigure_forest(
        #         progbar=True,
        #         num_trees=20,
        #         num_restarts=10,
        #         subtree_weight_what=("flops",),
        #     )
        #     return tree_r.get_path()

        tc.set_contractor(
            "custom",
            optimizer=opt0,
            contraction_info=True,
            preprocessing=True,
        )

    meta["contractor"] = contractor

    def tfi_energy(c, n, j=1.0, h=-1.0):
        e = 0.0
        for i in range(n):
            e += h * c.expectation((tc.gates.x(), [i]))
        for i in range(n - 1):
            e += j * c.expectation((tc.gates.z(), [i]), (tc.gates.z(), [(i + 1) % n]))
        return e

    meta["Software"] = "tensorcircuit[%s %s]" % (tcbackend, dt)
    meta["minus"] = minus
    meta["Cpuinfo"] = cpuinfo.get_cpu_info()["brand_raw"]
    meta["Version"] = {
        "sys": sys.version,
        "tensorcircuit": tc.__version__,
        "numpy": np.__version__,
    }
    meta["VQE test parameters"] = {
        "mpsd": mpsd,
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

    paramx = tc.backend.ones(nlayer * n, dtype=vtype)
    paramzz = tc.backend.ones(nlayer * n, dtype=vtype)

    def energy_raw(paramx, paramzz):
        if mpsd is None:
            c = tc.Circuit(n)
        else:
            c = tc.MPSCircuit(n)
            c.set_split_rules({"max_singular_values": mpsd})
        paramx = tc.backend.cast(paramx, dtype)
        paramzz = tc.backend.cast(paramzz, dtype)

        for i in range(n):
            c.H(i)
        for j in range(nlayer):
            for i in range(n - minus):
                c.exp1(
                    i,
                    (i + 1) % n,
                    theta=paramzz[j * n + i],
                    unitary=tc.gates._zz_matrix,
                )
            for i in range(n):
                c.rx(i, theta=paramx[j * n + i])
        if mpsd is None:
            e = tfi_energy(c, n)
            fd = 1.0
        else:
            e = tfi_energy(c, n)
            fd = c._fidelity
        # tensorflow only works for complex case, while jax only works for real case, don't know how to solve it
        # if tcbackend != "tensorflow":
        e = tc.backend.real(e)

        return e, fd

    energy_raw = tc.backend.value_and_grad(energy_raw, argnums=(0, 1), has_aux=True)
    energy_raw = tc.backend.jit(energy_raw)

    def energy(paramx, paramzz):
        # paramx = tc.backend.convert_to_tensor(paramx)
        # paramzz = tc.backend.convert_to_tensor(paramzz)
        (value, f), grads = energy_raw(paramx, paramzz)
        print("fidelity: ", f, value)
        # print(tc.backend.numpy(f), tc.backend.numpy(value))
        # value = tc.backend.numpy(tc.backend.real(value))
        # grads = [tc.backend.numpy(tc.backend.real(g)) for g in grads]
        return value, grads

    opt = utils.Opt(energy, [paramx, paramzz], tuning=True, backend=tcbackend)
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
        dtype,
        contractor,
        mpsd,
        tcbackend,
    ) = utils.arg(dtype=True, contractor=True, mps=True, tc=True)
    if dtype == 1:
        dtypestr = "32"
    else:
        dtypestr = "64"

    r = tensorcircuit_benchmark(
        _uuid,
        tcbackend,
        n,
        nlayer,
        nitrs,
        timeLimit,
        isgpu,
        mpsd,
        dtypestr,
        minus=minus,
        contractor=contractor,
    )
    utils.save(r, _uuid, path)
