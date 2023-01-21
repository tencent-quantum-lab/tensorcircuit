# Prerequisites for running this example:
# Step 1: install julia, see https://julialang.org/download/.
# Please install julia >= 1.8.5, the 1.6.7 LTS version raises:
# `Error in python: free(): invalid pointer`
# Step 2: add julia path to the PATH env variable so that we can find it
# Step 3: install julia package `OMEinsum`, `ArgParse` and `JSON`, this example was tested with OMEinsum v0.7.2,
# see https://docs.julialang.org/en/v1/stdlib/Pkg/ for more details on julia's package manager

import os
import sys
import json
import time
from typing import List, Set, Dict, Tuple
import tempfile
import subprocess
import cotengra as ctg
from omeinsum_treesa_optimizer import OMEinsumTreeSAOptimizer
import tensorcircuit as tc

sys.setrecursionlimit(10000)
tc.set_backend("tensorflow")


class OMEinsumTreeSAOptimizerSubprocess(OMEinsumTreeSAOptimizer):
    def __init__(
        self,
        sc_target: int = 20,
        betas: Tuple[float, float, float] = (0.01, 0.01, 15),
        ntrials: int = 10,
        niters: int = 50,
        sc_weight: float = 1.0,
        rw_weight: float = 0.2,
        kahypar_init: bool = False,
    ):
        super().__init__(sc_target, betas, ntrials, niters, sc_weight, rw_weight)
        self.kahypar_init = kahypar_init

    def __call__(
        self,
        inputs: List[Set[str]],
        output: Set[str],
        size: Dict[str, int],
        memory_limit=None,
    ) -> List[Tuple[int, int]]:
        inputs_omeinsum = list(map(list, inputs))
        output_omeinsum = list(output)

        fin = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        fout = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        fin.close()
        fout.close()
        einsum_json = {
            "inputs": inputs_omeinsum,
            "output": output_omeinsum,
            "size": list(size.items()),
        }
        with open(fin.name, "w") as fp:
            json.dump(einsum_json, fp)

        cmd = (
            "julia omeinsum.jl --einsum_json {0} "
            "--result_json {1} --sc_target {2} "
            "--beta_start {3} --beta_step {4} "
            "--beta_stop {5} --ntrials {6} "
            "--niters {7} --sc_weight {8} "
            "--rw_weight {9}".format(
                fin.name,
                fout.name,
                self.sc_target,
                self.betas[0],
                self.betas[1],
                self.betas[2],
                self.ntrials,
                self.niters,
                self.sc_weight,
                self.rw_weight,
            )
        )

        if self.kahypar_init:
            cmd += " --kahypar_init"

        print(cmd)
        p = subprocess.Popen(
            cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        t0 = time.time()
        _, stderr = p.communicate()
        running_time = time.time() - t0
        print(f"running_time: {running_time}")
        retcode = p.wait()
        if retcode:
            os.unlink(fin.name)
            os.unlink(fout.name)
            raise Exception("julia failed\nstderr:\n%s\n" % stderr.decode("utf-8"))

        with open(fout.name, "r") as f:
            contraction_tree = json.load(f)
        os.unlink(fin.name)
        os.unlink(fout.name)

        num_tensors = len(contraction_tree["inputs"])
        assert num_tensors == len(
            inputs
        ), "should have the same number of input tensors"
        queue = list(range(num_tensors))
        path = []
        self._contraction_tree_to_contraction_path(
            contraction_tree["tree"], queue, path, num_tensors
        )
        return path


if __name__ == "__main__":
    # For more random circuits, please refer to
    # https://datadryad.org/stash/dataset/doi:10.5061/dryad.k6t1rj8
    c = tc.Circuit.from_qsim_file("circuit_n12_m14_s0_e0_pEFGH.qsim")

    opt = ctg.ReusableHyperOptimizer(
        methods=["greedy", "kahypar"],
        parallel=True,
        minimize="flops",
        max_repeats=1024,
        progbar=False,
    )
    print("cotengra contractor")
    tc.set_contractor(
        "custom",
        optimizer=opt,
        preprocessing=True,
        contraction_info=True,
        debug_level=2,
    )
    c.expectation_ps(z=[0], reuse=False)

    print("OMEinsum contractor")
    opt_treesa = OMEinsumTreeSAOptimizerSubprocess(
        sc_target=30, sc_weight=0.0, rw_weight=0.0
    )
    tc.set_contractor(
        "custom",
        optimizer=opt_treesa,
        preprocessing=True,
        contraction_info=True,
        debug_level=2,
    )
    c.expectation_ps(z=[0], reuse=False)

    print("OMEinsum contractor with kahypar init")
    opt_treesa = OMEinsumTreeSAOptimizerSubprocess(
        sc_target=30, sc_weight=0.0, rw_weight=0.0, kahypar_init=True
    )
    tc.set_contractor(
        "custom",
        optimizer=opt_treesa,
        preprocessing=True,
        contraction_info=True,
        debug_level=2,
    )
    c.expectation_ps(z=[0], reuse=False)
