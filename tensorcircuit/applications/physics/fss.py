"""
finite size scaling tools
"""
from typing import List, Tuple, Optional

import numpy as np


def data_collapse(
    n: List[int],
    p: List[float],
    obs: List[List[float]],
    pc: float,
    nu: float,
    beta: float = 0,
    obs_type: int = 1,
    fit_type: int = 0,
    dobs: Optional[List[List[float]]] = None,
) -> Tuple[List[float], List[List[float]], List[List[float]], float]:
    xL: List[List[float]] = []  # x=(p-pc)L^(1/\nv)
    yL: List[List[float]] = []  # y=S(p,L)-S(pc,L) or S(p,L)
    pc_list = []
    if not isinstance(p[0], list):
        p = [p for _ in n]  # type: ignore
    for n0 in range(len(n)):
        xL.append([])
        yL.append([])
        for p0 in range(len(p[n0])):  # type: ignore
            xL[n0].append((p[n0][p0] - pc) * (n[n0] ** (1 / nu)))  # type: ignore
            pc_L = pc_linear_interpolation(p[n0], obs[n0], pc)  # type: ignore
            if obs_type == 0:
                yL[n0].append((obs[n0][p0] - pc_L) * n[n0] ** beta)
                # entanglement with only collapse and no crossing
            else:
                yL[n0].append(obs[n0][p0] * n[n0] ** beta)
                # tripartite mutual information
        pc_list.append(pc_L)
    if fit_type == 0:
        xL_all = []
        for i in range(len(xL)):
            xL_all.extend(xL[i])
        yL_ave = []
        loss = []
        for x0 in range(len(xL_all)):
            ybar_list = []
            for n0 in range(len(n)):
                if xL_all[x0] >= xL[n0][0] and xL_all[x0] <= xL[n0][-1]:
                    yinsert = pc_linear_interpolation(xL[n0], yL[n0], xL_all[x0])
                    ybar_list.append(yinsert)

            ybar = np.mean(ybar_list)
            mean_squared = [(ybar_list[i] - ybar) ** 2 for i in range(len(ybar_list))]
            loss.append(np.sum(mean_squared))
            yL_ave.append(ybar)
        loss = np.sum(loss)

        return pc_list, xL, yL, loss  # type: ignore

    # fit_type == 1
    if dobs is None:
        raise ValueError("uncertainty of each y has to be specified in `dobs`")

    datas = []
    for n0 in range(len(n)):
        for i in range(len(xL[n0])):
            datas.append([xL[n0][i], yL[n0][i], dobs[n0][i]])
    datas = sorted(datas, key=lambda x: x[0])
    loss = _quality_objective_v2(datas)  # type: ignore
    return pc_list, xL, yL, loss  # type: ignore


def _quality_objective_v2(datas: List[List[float]]) -> float:
    #  https://journals.aps.org/prb/supplemental/10.1103/PhysRevB.101.060301/Supplement.pdf
    loss = []
    for i in range(len(datas) - 2):
        # i, i+1, i+2
        x, y, d = datas[i + 1]
        x1, y1, d1 = datas[i]
        x2, y2, d2 = datas[i + 2]
        if np.abs(x - x1) < 1e-4 or np.abs(x - x2) < 1e-4:
            continue
        ybar = ((x2 - x) * y1 - (x1 - x) * y2) / (x2 - x1)
        delta = (
            d**2
            + d1**2 * (x2 - x) ** 2 / (x2 - x1) ** 2
            + d2**2 * (x1 - x) ** 2 / (x2 - x1) ** 2
        )
        w = (y - ybar) ** 2 / delta
        loss.append(w)
    return np.mean(loss)  # type: ignore


def pc_linear_interpolation(p: List[float], SA: List[float], pc_input: float) -> float:
    if pc_input in p:
        pc_index = p.index(pc_input)
        pc = SA[pc_index]
    else:
        pr_index = 0
        for p0 in range(len(p)):
            if p[p0] > pc_input:
                pr_index = p0
                break

        x = [p[pr_index - 1], p[pr_index]]
        y = [SA[pr_index - 1], SA[pr_index]]
        linear_model = np.polyfit(x, y, 1)
        linear_model_fn = np.poly1d(linear_model)
        pc = linear_model_fn(pc_input)
    return pc
