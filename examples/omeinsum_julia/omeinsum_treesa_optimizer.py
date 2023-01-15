from typing import List, Set, Dict, Tuple


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
        raise NotImplementedError
