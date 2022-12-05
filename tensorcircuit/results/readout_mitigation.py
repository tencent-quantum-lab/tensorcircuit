"""
readout error mitigation functionalities
"""

from typing import Any, Callable, List, Union
import numpy as np
from scipy.optimize import minimize

from .counts import count2vec, vec2count, ct
from ..circuit import Circuit
from ..utils import is_sequence

Tensor = Any


class ReadoutCal:
    def __init__(self, cal: Union[Tensor, List[Tensor]]):
        self.cal = cal
        if is_sequence(cal):
            self.local = True
            self.n = len(cal)
        else:
            self.local = False
            self.n = int(np.log(cal.shape[0]) / np.log(2) + 1e-9)  # type: ignore

    def get_matrix(self) -> Tensor:
        # cache
        if getattr(self, "calmatrix", None) is not None:
            return self.calmatrix  # type: ignore
        if self.local is False:
            self.calmatrix = self.cal  # type: ignore
            return self.cal
        # self.local = True
        calmatrix = self.cal[0]
        for i in range(1, self.n):
            calmatrix = np.kron(calmatrix, self.cal[i])
        self.calmatrix = calmatrix
        return calmatrix


def local_miti_readout_circ(nqubit: int) -> List[Circuit]:
    miticirc = []
    c = Circuit(nqubit)
    miticirc.append(c)
    c = Circuit(nqubit)
    for i in range(nqubit):
        c.X(i)  # type: ignore
    miticirc.append(c)
    return miticirc


def global_miti_readout_circ(nqubit: int) -> List[Circuit]:
    miticirc = []
    for i in range(2**nqubit):
        name = "{:0" + str(nqubit) + "b}"
        lisbs = [int(x) for x in name.format(i)]
        c = Circuit(nqubit)
        for k in range(nqubit):
            if lisbs[k] == 1:
                c.X(k)  # type: ignore
        miticirc.append(c)
    return miticirc


def mitigate_probability(
    probability_noise: Tensor, readout_cal: ReadoutCal, method: str = "inverse"
) -> Tensor:
    calmatrix = readout_cal.get_matrix()
    if method == "inverse":
        X = np.linalg.inv(calmatrix)
        Y = probability_noise
        probability_cali = X @ Y
    else:  # method="square"

        def fun(x: Any) -> Any:
            return sum((probability_noise - calmatrix @ x) ** 2)

        x0 = np.random.rand(len(probability_noise))
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        bnds = tuple((0, 1) for x in x0)
        res = minimize(fun, x0, method="SLSQP", constraints=cons, bounds=bnds, tol=1e-6)
        probability_cali = res.x
    return probability_cali


def apply_readout_mitigation(
    raw_count: ct, readout_cal: ReadoutCal, method: str = "inverse"
) -> ct:
    probability = count2vec(raw_count)
    shots = sum([v for k, v in raw_count.items()])
    probability = mitigate_probability(probability, readout_cal, method=method)
    probability = probability * shots
    return vec2count(probability)


def get_readout_cal(
    nqubit: int,
    shots: int,
    execute_fun: Callable[..., List[ct]],
    miti_method: str = "local",
) -> ReadoutCal:
    # TODO(@refraction-ray): more general qubit list
    if miti_method == "local":
        miticirc = local_miti_readout_circ(nqubit)

        lbs = execute_fun(miticirc, shots)
        readoutlist = []
        for i in range(nqubit):
            error00 = 0
            for s in lbs[0]:
                if s[i] == "0":
                    error00 = error00 + lbs[0][s] / shots  # type: ignore

            error10 = 0
            for s in lbs[1]:
                if s[i] == "0":
                    error10 = error10 + lbs[1][s] / shots  # type: ignore
            readoutlist.append(
                np.array(
                    [
                        [error00, error10],
                        [1 - error00, 1 - error10],
                    ]
                )
            )

        return ReadoutCal(readoutlist)

    elif miti_method == "global":
        miticirc = global_miti_readout_circ(nqubit)
        calmatrix = np.zeros((2**nqubit, 2**nqubit))
        lbs = execute_fun(miticirc, shots)
        for i in range(len(miticirc)):
            for s in lbs[i]:
                calmatrix[int(s, 2)][i] = lbs[i][s] / shots

        return ReadoutCal(calmatrix)

    else:
        raise ValueError("Unrecognized `miti_method`: %s" % miti_method)
