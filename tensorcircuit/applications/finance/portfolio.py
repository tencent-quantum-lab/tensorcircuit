"""
Supplementary functions for portfolio optimization
"""

from typing import Any, List

import numpy as np

Array = Any
Tensor = Any


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

    def __init__(self, data: Tensor):
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
                each_stock.append((data[i][j + 1] - data[i][j]) / data[i][j])
            self.daily_change.append(each_stock)

    def get_return(self, decimals: int = 5) -> List[float]:
        """
        Calculates the annualized return (mu).

        :param decimals: Number of decimal places to round the result to (default: 5).
        :return: The annualized return as an array rounded to the specified number of decimals.
        """
        change = [[j + 1 for j in i] for i in self.daily_change]
        ret = np.prod(change, axis=1) ** (252 / self.n_days)
        return ret.round(decimals)

    def get_covariance(self, decimals: int = 5) -> Tensor:
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

    def get_penalty(
        self,
        cov: Tensor,
        ret: List[float],
        risk_pre: float,
        budget: int,
        decimals: int = 5,
    ) -> float:
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
        for i in range(2**self.n_stocks):
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
                F.append(np.dot(x, np.dot(Q, x)) + penalty * budget**2)
            Fmin = np.amin(F)
            Fbar = np.mean(F)
            F = []
            for state in self.uf_state:
                x = np.array([int(bit) for bit in state])
                F.append(np.dot(x, np.dot(Q, x)) + penalty * budget**2)
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
