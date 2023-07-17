"""
Shortcuts for reusable circuit ansatz
"""

from typing import Any, List

from ..circuit import Circuit as Circ

Tensor = Any
Circuit = Any


def QAOA_ansatz_for_Ising(
    params: List[float],
    nlayers: int,
    pauli_terms: Tensor,
    weights: List[float],
    full_coupling: bool = False,
    mixer: str = "X",
) -> Circuit:
    """
    Construct the QAOA ansatz for the Ising Model.
    The number of qubits is determined by `pauli_terms`.

    :param params: A list of parameter values used in the QAOA ansatz.
    :param nlayers: The number of layers in the QAOA ansatz.
    :pauli_terms: A list of Pauli terms, where each term is represented as a list of 0/1 series.
    :param weights: A list of weights corresponding to each Pauli term.
    :param full_coupling (optional): A flag indicating whether to use all-to-all coupling in mixers. Default is False.
    :paran mixer (optional): The mixer operator to use. Default is "X". The other options are "XY" and "ZZ".
    :return: QAOA ansatz for Ising model.
    """
    nqubits = len(pauli_terms[0])
    c: Any = Circ(nqubits)
    for i in range(nqubits):
        c.h(i)  # Apply Hadamard gate to each qubit

    for j in range(nlayers):
        # cost terms
        for k, term in enumerate(pauli_terms):
            index_of_ones = []
            for l in range(len(term)):
                if term[l] == 1:
                    index_of_ones.append(l)
            if len(index_of_ones) == 1:
                c.rz(index_of_ones[0], theta=2 * weights[k] * params[2 * j])
                # Apply Rz gate with angle determined by weight and current parameter value
            elif len(index_of_ones) == 2:
                c.rzz(
                    index_of_ones[0],
                    index_of_ones[1],
                    theta=weights[k] * params[2 * j],
                )
                # Apply exp1 gate with a custom unitary (zz_matrix)
                # and angle determined by weight and current parameter value
            else:
                raise ValueError("Invalid number of Z terms")

        # prepare the coupling map for mixer terms
        pairs = []
        if full_coupling is False:
            for q0 in list(range(0, nqubits - 1, 2)) + list(range(1, nqubits - 1, 2)):
                pairs.append([q0, q0 + 1])
            pairs.append([nqubits - 1, 0])
        elif full_coupling is True:
            for q0 in range(nqubits - 1):
                for q1 in range(q0 + 1, nqubits):
                    pairs.append([q0, q1])
        else:
            raise ValueError("Invalid input.")

        # standard mixer
        if mixer == "X":
            for i in range(nqubits):
                c.rx(
                    i, theta=params[2 * j + 1]
                )  # Apply Rx gate with angle determined by current parameter value

        # XY mixer
        elif mixer == "XY":
            for [q0, q1] in pairs:
                c.rxx(q0, q1, theta=params[2 * j + 1])
                c.ryy(q0, q1, theta=params[2 * j + 1])

        # ZZ mixer
        elif mixer == "ZZ":
            for [q0, q1] in pairs:
                c.rzz(q0, q1, theta=params[2 * j + 1])

        else:
            raise ValueError("Invalid mixer type.")

    return c
