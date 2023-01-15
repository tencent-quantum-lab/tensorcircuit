import os
import json
import time
from typing import List, Set, Dict, Tuple
import tempfile
import warnings
import cotengra as ctg

# Prerequisites for running this example:
# Step 1: install julia, see https://julialang.org/download/,
# Please install julia >= 1.8.5, the 1.6.7 LTS version raises:
# `Error in python: free(): invalid pointer`
# Step 2: add julia path to the PATH env variable so that juliacall can find it
# Step 3: install juliacall via `pip install juliacall`, this example was tested with juliacall 0.9.9
# Step 4: install julia package `OMEinsum`, this example was tested with OMEinsum v0.7.2,
# see https://docs.julialang.org/en/v1/stdlib/Pkg/ for more details on julia's package manager
# Step 5: for julia multi-threading, set env variable PYTHON_JULIACALL_THREADS=<N|auto>.
# However, in order to use julia multi-threading in juliacall,
# we have to turn off julia GC at the risk of OOM.
# See see https://github.com/cjdoris/PythonCall.jl/issues/219 for more details.
from juliacall import Main as jl

jl.seval("using OMEinsum")

from omeinsum_treesa_optimizer import OMEinsumTreeSAOptimizer
import tensorcircuit as tc

tc.set_backend("tensorflow")


class OMEinsumTreeSAOptimizerJuliaCall(OMEinsumTreeSAOptimizer):
    def __init__(
        self,
        sc_target: float = 20,
        betas: Tuple[float, float, float] = (0.01, 0.01, 15),
        ntrials: int = 10,
        niters: int = 50,
        sc_weight: float = 1.0,
        rw_weight: float = 0.2,
    ):
        super().__init__(sc_target, betas, ntrials, niters, sc_weight, rw_weight)

    def __call__(
        self,
        inputs: List[Set[str]],
        output: Set[str],
        size: Dict[str, int],
        memory_limit=None,
    ) -> List[Tuple[int, int]]:
        inputs_omeinsum = tuple(map(tuple, inputs))
        output_omeinsum = tuple(output)

        eincode = jl.OMEinsum.EinCode(inputs_omeinsum, output_omeinsum)

        size_dict = jl.OMEinsum.uniformsize(eincode, 2)
        for k, v in size.items():
            size_dict[k] = v

        algorithm = jl.OMEinsum.TreeSA(
            sc_target=self.sc_target,
            βs=jl.range(self.betas[0], step=self.betas[1], stop=self.betas[2]),
            ntrials=self.ntrials,
            niters=self.niters,
            sc_weight=self.sc_weight,
            rw_weight=self.rw_weight,
        )

        nthreads = jl.Threads.nthreads()
        if nthreads > 1:
            warnings.warn(
                "Julia receives Threads.nthreads()={0}. "
                "However, in order to use julia multi-threading in juliacall, "
                "we have to turn off julia GC at the risk of OOM. "
                "That means you may need a large memory machine. "
                "Please see https://github.com/cjdoris/PythonCall.jl/issues/219 "
                "for more details.".format(nthreads)
            )
            jl.GC.enable(False)
        t0 = time.time()
        optcode = jl.OMEinsum.optimize_code(eincode, size_dict, algorithm)
        running_time = time.time() - t0
        print(f"running_time: {running_time}")
        if nthreads > 1:
            jl.GC.enable(True)
        # jl.println("time and space complexity computed by OMEinsum: ",
        #    jl.OMEinsum.timespace_complexity(optcode, size_dict))

        fp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        fp.close()
        jl.OMEinsum.writejson(fp.name, optcode)
        with open(fp.name, "r") as f:
            contraction_tree = json.load(f)
        os.unlink(fp.name)

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
    "custom", optimizer=opt, preprocessing=True, contraction_info=True, debug_level=2
)
c.expectation_ps(z=[0], reuse=False)

print("OMEinsum contractor")
opt_treesa = OMEinsumTreeSAOptimizerJuliaCall(
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
