"""
Demonstrate the failure of rem when qubit number is much larger than 1/p
"""

from functools import partial
import numpy as np
import tensorcircuit as tc


def simulate_engine(p, ans):
    fans = ""
    for a in ans:
        if p > np.random.uniform():
            if a == "0":
                fans += "1"
            else:
                fans += "0"
        else:
            fans += a
    return fans


def run(cs, shots, p=0.1):
    # we assume only all 0 and all 1 results for simplicity
    rds = []
    for c in cs:
        if len(c.to_qir()) < 2:
            ans = "0" * c._nqubits
        else:
            ans = "1" * c._nqubits
        rd = {}
        for _ in range(shots):
            r = simulate_engine(p, ans)
            rd[r] = rd.get(r, 0) + 1
        rds.append(rd)
    return rds


if __name__ == "__main__":
    for p in [0.1, 0.05, 0.02]:
        print(p)
        n = int(3 / p)
        c = tc.Circuit(n)
        c.x(range(n))
        runp = partial(run, p=p)
        r = runp([c], 8192)[0]
        mit = tc.results.rem.ReadoutMit(runp)
        mit.cals_from_system(n)
        for i in range(n):
            print(i, "\n", mit.single_qubit_cals[i])
        rs = []
        for i in range(n):
            rs.append([i, np.abs(mit.expectation(r, list(range(i))))])
            print(rs[i])
