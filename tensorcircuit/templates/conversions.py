"""
helper functions for conversions
"""

from typing import Any, Tuple, List
from ..cons import backend

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


def QUBO_to_Ising(Q: List[list]) -> Tuple[List[list], list, float]:
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


def QUBO_from_portfolio(cov: Array, mean: Array, q: float, B: int, t: float) -> Tensor:
    """
    convert portfolio parameters to a Q matrix
    :param cov: n-by-n covariance numpy array
    :param mean: numpy array of means
    :param q: the risk preference of investor
    :param B: budget
    :param t: penalty factor
    :return Q: n-by-n symmetric Q matrix
    """
    n = cov.shape[0]
    R = np.diag(mean)
    S = np.ones((n, n)) - 2 * B * np.diag(np.ones(n))

    Q = q * cov - R + t * S
    return Q


class StockData:
    """
    A class for converting real-world stock data to an annualized covariance matrix and annualized return.

    Attributes:
    - data: A list of continuous stock data in the same time span.
    - n_stocks: The number of stocks in the data.
    - n_days: The number of trading days in the data.

    Methods:
    - __init__(self, data): Initializes the StockData object.
    - get_return(self, decimals=5): Calculates the annualized return.
    - get_covariance(self, decimals=5): Calculates the annualized covariance matrix.
    - get_penalty(self, cov, ret, risk_pre, budget, decimals=5): Calculates the penalty factor.
    """

    def __init__(self, data):
        """
        Initializes the StockData object.

        :param data: A list of continuous stock data in the same time span.
        """
        self.data = data
        self.n_stocks = len(data)
        
        # Check the number of days
        n_days = [len(i) for i in data]
        if max(n_days) != (sum(n_days) / len(n_days)):
            raise Exception("Timespan of stocks should be the same")
        self.n_days = len(data[1])

        # Calculate the daily percentage price change
        self.daily_change = []
        for i in range(self.n_stocks):
            each_stock = []
            for j in range(self.n_days - 1):
                each_stock.append((data[i][j + 1] - data[i][j]) / data[i][j + 1])
            self.daily_change.append(each_stock)

    def get_return(self, decimals=5):
        """
        Calculates the annualized return (mu).

        :param decimals: Number of decimal places to round the result to (default: 5).
        :return: The annualized return as an array rounded to the specified number of decimals.
        """
        change = [[j + 1 for j in i] for i in self.daily_change]
        ret = np.prod(change, axis=1) ** (252 / self.n_days)
        return ret.round(decimals)

    def get_covariance(self, decimals=5):
        """
        Calculates the annualized covariance matrix (sigma).

        :param decimals: Number of decimal places to round the result to (default: 5).
        :return: The annualized covariance matrix rounded to the specified number of decimals.
        """
        mean = np.mean(self.daily_change, axis=1)
        relative_change = [
            [j - mean[i] for j in self.daily_change[i]] for i in range(6)
        ]
        cov = 252 / self.n_days * np.dot(relative_change, np.transpose(relative_change))
        return cov.round(decimals)

    def get_penalty(self, cov, ret, risk_pre, budget, decimals=5):
        """
        Calculates the penalty factor.

        :param cov: The annualized covariance matrix.
        :param ret: The annualized return.
        :param risk_pre: The risk preference factor.
        :param budget: The budget (number of stocks to select).
        :param decimals: Number of decimal places to round the result to (default: 5).
        :return: The penalty factor rounded to the specified number of decimals.
        """
        # Get all feasible and unfeasible states
        self.f_state = []  # Feasible states (number of '1's equal to budget)
        self.uf_state = []  # Unfeasible states
        self.all_state = []
        for i in range(2 ** self.n_stocks):
            state = f"{bin(i)[2:]:0>{self.n_stocks}}"
            n_ones = 0
            for j in state:
                if j == "1":
                    n_ones += 1
            self.all_state.append(state)
            if n_ones == budget:
                self.f_state.append(state)
            else:
                self.uf_state.append(state)

        # Determine the penalty factor
        mark = False
        penalty = 0  # Initial value
        while mark == False:
            R = np.diag(ret)
            S = np.ones((self.n_stocks, self.n_stocks)) - 2 * budget * np.diag(
                np.ones(self.n_stocks)
            )
            Q = risk_pre * cov - R + penalty * S
            F = []
            for state in self.f_state:
                x = np.array([int(bit) for bit in state])
                F.append(np.dot(x, np.dot(Q, x)) + penalty * budget ** 2)
            Fmin = np.amin(F)
            Fbar = np.mean(F)
            F = []
            for state in self.uf_state:
                x = np.array([int(bit) for bit in state])
                F.append(np.dot(x, np.dot(Q, x)) + penalty * budget ** 2)
            Fmin_uf = np.amin(F)
            location = np.where(F == Fmin_uf)[0][0]
            if Fmin_uf < 0.5 * (Fmin + Fbar):
                n_ones = 0
                for j in self.uf_state[location]:
                    if j == "1":
                        n_ones += 1
                penalty += (0.5 * (Fmin + Fbar) - Fmin_uf) / (n_ones - budget) ** 2
            else:
                mark = True  # Ready to return the penalty
        return round(penalty, decimals)


