"""
dict related functionalities
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np


Tensor = Any
ct = Dict[str, int]

# TODO(@refraction-ray): merge_count


def reverse_count(count: ct) -> ct:
    ncount = {}
    for k, v in count.items():
        ncount[k[::-1]] = v
    return ncount


def sort_count(count: ct) -> ct:
    return {k: v for k, v in sorted(count.items(), key=lambda item: -item[1])}


def normalized_count(count: ct) -> Dict[str, float]:
    shots = sum([v for k, v in count.items()])
    return {k: v / shots for k, v in count.items()}


def marginal_count(count: ct, keep_list: Sequence[int]) -> ct:
    import qiskit

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


def expectation(
    count: ct, z: Optional[Sequence[int]] = None, diagonal_op: Optional[Tensor] = None
) -> float:
    """
    compute diagonal operator expectation value from bit string count dictionary

    :param count: count dict for bitstring histogram
    :type count: ct
    :param z: if defaults as None, then ``diagonal_op`` must be set
        a list of qubit that we measure Z op on
    :type z: Optional[Sequence[int]]
    :param diagoal_op: shape [n, 2], explicitly indicate the diagonal op on each qubit
        eg. [1, -1] for z [1, 1] for I, etc.
    :type diagoal_op: Tensor
    :return: the expectation value
    :rtype: float
    """
    if z is None and diagonal_op is None:
        raise ValueError("One of `z` and `diagonal_op` must be set")
    n = len(list(count.keys())[0])
    if z is not None:
        diagonal_op = [[1, -1] if i in z else [1, 1] for i in range(n)]
    r = 0
    shots = 0
    for k, v in count.items():
        cr = 1.0
        for i in range(n):
            cr *= diagonal_op[i][int(k[i])]  # type: ignore
        r += cr * v  # type: ignore
        shots += v
    return r / shots


def plot_histogram(data: Any, **kws: Any) -> Any:
    """
    See ``qiskit.visualization.plot_histogram``:
    https://qiskit.org/documentation/stubs/qiskit.visualization.plot_histogram.html

    interesting kw options include: ``number_to_keep`` (int)

    :param data: _description_
    :type data: Any
    :return: _description_
    :rtype: Any
    """
    from qiskit.visualization import plot_histogram

    return plot_histogram(data, **kws)
