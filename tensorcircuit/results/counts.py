"""
dict related functionalities
"""
from typing import Any, Dict, Sequence

import numpy as np
import qiskit


Tensor = Any
ct = Dict[str, int]


def reverse_count(count: ct) -> ct:
    ncount = {}
    for k, v in count.items():
        ncount[k[::-1]] = v
    return ncount


def marginal_count(count: ct, keep_list: Sequence[int]) -> ct:
    count = reverse_count(count)
    ncount = qiskit.result.utils.marginal_distribution(count, keep_list)
    return reverse_count(ncount)


def count2vec(count: ct, normalization: bool = True) -> Tensor:
    nqubit = len(list(count.keys())[0])
    probability = [0] * 2**nqubit
    shots = sum([v for k, v in count.items()])
    for k, v in count.items():
        if normalization is True:
            v /= shots  # type: ignore
        probability[int(k, 2)] = v
    return np.array(probability)


def vec2count(vec: Tensor, prune: bool = False) -> ct:
    from ..quantum import count_vector2dict

    if isinstance(vec, list):
        vec = np.array(vec)
    n = int(np.log(vec.shape[0]) / np.log(2) + 1e-9)
    c = count_vector2dict(vec, n, key="bin")
    if prune is True:
        nc = c.copy()
        for k, v in c.items():
            if np.abs(v) < 1e-8:
                del nc[k]
        return nc
    return c
