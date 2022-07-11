"""
Useful utilities for quantum chemistry related task
"""

from typing import Any, Tuple

import numpy as np

Tensor = Any


def get_ps(qo: Any, n: int) -> Tuple[Tensor, Tensor]:
    """
    Get Pauli string array and weights array for a qubit Hamiltonian
    as a sum of Pauli strings defined in openfermion ``QubitOperator``.

    :param qo: ``openfermion.ops.operators.qubit_operator.QubitOperator``
    :type qo: ``openfermion.ops.operators.qubit_operator.QubitOperator``
    :param n: The number of qubits
    :type n: int
    :return: Pauli String array and weights array
    :rtype: Tuple[Tensor, Tensor]
    """
    value = {"X": 1, "Y": 2, "Z": 3}
    terms = qo.terms
    res = []
    wts = []
    for key in terms:
        bit = np.zeros(n, dtype=int)
        for i in range(len(key)):
            bit[key[i][0]] = value[key[i][1]]
        w = terms[key]
        res_t = tuple()  # type: ignore
        for i in range(n):
            res_t = res_t + (bit[i],)
        res.append(res_t)
        wts.append(w)
    return np.array(res), np.array(wts)
