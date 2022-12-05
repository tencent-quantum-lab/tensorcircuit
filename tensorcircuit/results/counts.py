"""
dict related functionalities
"""
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import qiskit


Tensor = Any
ct = Dict[str, int]


def reverse_count(count: ct) -> ct:
    ncount = {}
    for k, v in count.items():
        ncount[k[::-1]] = v
    return ncount


def normalized_count(count: ct) -> Dict[str, float]:
    shots = sum([v for k, v in count.items()])
    return {k: v / shots for k, v in count.items()}


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


def kl_divergence(c1: ct, c2: ct) -> float:
    eps = 1e-4  # typical value for inverse of the total shots
    c1 = normalized_count(c1)  # type: ignore
    c2 = normalized_count(c2)  # type: ignore
    kl = 0
    for k, v in c1.items():
        kl += v * (np.log(v) - np.log(c2.get(k, eps)))
    return kl


def correlation(
    count: ct, zlist: Sequence[int], values: Tuple[int, int] = (1, -1)
) -> float:
    map_dict = {"0": values[0], "1": values[1]}
    r = 0
    shots = 0
    for k, v in count.items():
        ct = 1.0
        for i in zlist:
            ct *= map_dict[k[i]]
        r += ct * v  # type: ignore
        shots += v
    return r / shots
