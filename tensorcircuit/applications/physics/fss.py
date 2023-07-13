"""
finite size scaling tools
"""
from typing import List, Tuple

import numpy as np


def data_collapse(
    n: List[int],
    p: List[float],
    obs: List[List[float]],
    pc: float,
    nu: float,
    beta: float = 0,
    obs_type: int = 1,
) -> Tuple[List[float], List[List[float]], List[List[float]], float]:
    xL: List[List[float]] = []  # x=(p-pc)L^(1/\nv)
    yL: List[List[float]] = []  # y=S(p,L)-S(pc,L) or S(p,L)
    pc_list = []
    for n0 in range(len(n)):
        xL.append([])
        yL.append([])
        for p0 in range(len(p)):
            xL[n0].append((p[p0] - pc) * (n[n0] ** (1 / nu)))
            pc_L = pc_linear_interpolation(p, obs[n0], pc)
            if obs_type == 0:
                yL[n0].append((obs[n0][p0] - pc_L) * n[n0] ** (beta / nu))
                # entanglement with only collapse and no crossing
            else:
                yL[n0].append(obs[n0][p0] * n[n0] ** (beta / nu))
                # tripartite mutual information
        pc_list.append(pc_L)

    xL_all = np.reshape(xL, -1)
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

    return pc_list, xL, np.array(yL), loss  # type: ignore


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
