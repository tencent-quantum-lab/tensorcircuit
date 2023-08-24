"""
Supplementary functions for portfolio optimization
"""

from typing import Any

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
        self.daily_change = [
            np.diff(data[i][:]) / data[i][:-1] for i in range(self.n_stocks)
        ]

    def get_return(self, decimals: int = 5) -> Any:
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
