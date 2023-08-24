"""
helper functions for conversions
"""

from typing import Any, Tuple, List

import numpy as np

Tensor = Any
Array = Any


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


def QUBO_to_Ising(Q: Tensor) -> Tuple[Tensor, List[float], float]:
    """
    Cnvert the Q matrix into a the indication of pauli terms, the corresponding weights, and the offset.
    The outputs are used to construct an Ising Hamiltonian for QAOA.

    :param Q: The n-by-n square and symmetric Q-matrix.
    :return pauli_terms: A list of 0/1 series, where each element represents a Pauli term.
    A value of 1 indicates the presence of a Pauli-Z operator, while a value of 0 indicates its absence.
    :return weights: A list of weights corresponding to each Pauli term.
    :return offset: A float representing the offset term of the Ising Hamiltonian.
    """

    # input is n-by-n symmetric numpy array corresponding to Q-matrix
    # output is the components of Ising Hamiltonian

    n = Q.shape[0]

    # square matrix check
    if Q[0].shape[0] != n:
        raise ValueError("Matrix is not a square matrix.")

    offset = (
        np.triu(Q, 0).sum() / 2
    )  # Calculate the offset term of the Ising Hamiltonian
    pauli_terms = []  # List to store the Pauli terms
    weights = (
        -np.sum(Q, axis=1) / 2
    )  # Calculate the weights corresponding to each Pauli term

    for i in range(n):
        term = np.zeros(n)
        term[i] = 1
        pauli_terms.append(
            term.tolist()
        )  # Add a Pauli term corresponding to a single qubit

    for i in range(n - 1):
        for j in range(i + 1, n):
            term = np.zeros(n)
            term[i] = 1
            term[j] = 1
            pauli_terms.append(
                term.tolist()
            )  # Add a Pauli term corresponding to a two-qubit interaction

            weight = (
                Q[i][j] / 2
            )  # Calculate the weight for the two-qubit interaction term
            weights = np.concatenate(
                (weights, weight), axis=None
            )  # Add the weight to the weights list

    return pauli_terms, weights, offset
