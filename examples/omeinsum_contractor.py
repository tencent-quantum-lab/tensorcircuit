import os
import json
from typing import List, Set, Dict, Tuple
import tempfile
import cotengra as ctg

# Please install a julia >= 1.8.5, the 1.6.7 LTS version raises:
# Error in `python': free(): invalid pointer
from juliacall import Main as jl

# We assume OMEinsum package is installed in julia
jl.seval("using OMEinsum")

import tensorcircuit as tc

tc.set_backend("tensorflow")


class OMEinsumTreeSAOptimizer(object):
    def __init__(
        self,
        sc_target: float = 20,
        betas: Tuple[float, float, float] = (0.01, 0.01, 15),
        ntrials: int = 10,
        niters: int = 50,
        sc_weight: float = 1.0,
        rw_weight: float = 0.2,
    ):
        self.sc_target = sc_target
        self.betas = betas
        self.ntrials = ntrials
        self.niters = niters
        self.sc_weight = sc_weight
        self.rw_weight = rw_weight

    def _contraction_tree_to_contraction_path(self, ei, queue, path, idx):
        if ei["isleaf"]:
            # OMEinsum provide 1-based index
            # but in contraction path we want 0-based index
            ei["tensorindex"] -= 1
            return idx
        assert len(ei["args"]) == 2, "must be a binary tree"
        for child in ei["args"]:
            idx = self._contraction_tree_to_contraction_path(child, queue, path, idx)
            assert "tensorindex" in child

        lhs_args = sorted(
            [queue.index(child["tensorindex"]) for child in ei["args"]], reverse=True
        )
        for arg in lhs_args:
            queue.pop(arg)

        ei["tensorindex"] = idx
        path.append(lhs_args)
        queue.append(idx)
        return idx + 1

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
        optcode = jl.OMEinsum.optimize_code(eincode, size_dict, algorithm)
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
opt_treesa = OMEinsumTreeSAOptimizer(sc_target=30, sc_weight=0.0, rw_weight=0.0)
tc.set_contractor(
    "custom",
    optimizer=opt_treesa,
    preprocessing=True,
    contraction_info=True,
    debug_level=2,
)
c.expectation_ps(z=[0], reuse=False)
