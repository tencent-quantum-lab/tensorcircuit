"""
baseline calculators for physical systems
"""

import numpy as np


def TFIM1Denergy(
    L: int, Jzz: float = 1.0, Jx: float = 1.0, Pauli: bool = True
) -> float:
    # PBC
    # nice tutorial: https://arxiv.org/pdf/2009.09208.pdf
    # further investigation on 1 TFIM solution structure is required
    # will fail on AFM phase Jzz>Jx and Jzz>0 and odd sites (frustration in boundary)
    e = 0
    if Pauli:
        Jx *= 2
        Jzz *= 4
    for i in range(L):
        q = np.pi * (2 * i - (1 + (-1) ** L) / 2) / L
        e -= np.abs(Jx) / 2 * np.sqrt(1 + Jzz**2 / 4 / Jx**2 - Jzz / Jx * np.cos(q))
    return e


def Heisenberg1Denergy(L: int, Pauli: bool = True, maxiters: int = 1000) -> float:
    # PBC
    error = 1e-15
    eps = 1e-20  # avoid zero division
    phi = np.zeros([L // 2, L // 2])
    phi2 = np.zeros([L // 2, L // 2])
    lamb = np.array([2 * i + 1 for i in range(L // 2)])
    for _ in range(maxiters):
        k = 1 / L * (2 * np.pi * lamb + np.sum(phi, axis=-1) - np.diag(phi))
        for i in range(L // 2):
            for j in range(L // 2):
                phi2[i, j] = (
                    np.arctan(
                        2
                        / (
                            1 / (np.tan(k[i] / 2) + eps)
                            - 1 / (np.tan(k[j] / 2) + eps)
                            + eps
                        )
                    )
                    * 2
                )
        if np.allclose(phi, phi2, rtol=error):  # converged
            break
        phi = phi2.copy()
    else:
        raise ValueError(
            "the maxiters %s is too small for bethe ansatz to converge" % maxiters
        )
    e = -np.sum(1 - np.cos(k)) + L / 4
    if Pauli is True:
        e *= 4
    return e  # type: ignore
