import sys
import os

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)

import numpy as np
import pytest

from tensorcircuit.applications.dqas import set_op_pool
from tensorcircuit.applications.graphdata import get_graph
from tensorcircuit.applications.layers import Hlayer, rxlayer, zzlayer
from tensorcircuit.applications.vags import evaluate_vag
from tensorcircuit.templates.ansatz import QAOA_ansatz_for_Ising
from tensorcircuit.circuit import Circuit


def test_vag(tfb):
    set_op_pool([Hlayer, rxlayer, zzlayer])
    expene, ene, eneg, p = evaluate_vag(
        np.array([0.0, 0.3, 0.5, 0.7, -0.8]),
        [0, 2, 1, 2, 1],
        get_graph("10A"),
        lbd=0,
        overlap_threhold=11,
    )
    print(expene, eneg, p)
    np.testing.assert_allclose(ene.numpy(), -7.01, rtol=1e-2)


cases = [
    ("X", True),
    ("X", False),
    ("XY", True),
    ("XY", False),
    ("ZZ", True),
    ("ZZ", False),
]


@pytest.fixture
def example_inputs():
    params = [0.1, 0.2, 0.3, 0.4]
    nlayers = 2
    pauli_terms = [[0, 1, 0], [1, 0, 1]]
    weights = [0.5, -0.5]
    return params, nlayers, pauli_terms, weights


@pytest.mark.parametrize("mixer, full_coupling", cases)
def test_QAOA_ansatz_for_Ising(example_inputs, full_coupling, mixer):
    params, nlayers, pauli_terms, weights = example_inputs
    circuit = QAOA_ansatz_for_Ising(
        params, nlayers, pauli_terms, weights, full_coupling, mixer
    )
    n = len(pauli_terms[0])
    assert isinstance(circuit, Circuit)
    assert circuit._nqubits == n

    if mixer == "X":
        assert circuit.gate_count() == n + nlayers * (len(pauli_terms) + n)
    elif mixer == "XY":
        if full_coupling is False:
            assert circuit.gate_count() == n + nlayers * (len(pauli_terms) + 2 * n)
        else:
            assert circuit.gate_count() == n + nlayers * (
                len(pauli_terms) + sum(range(n + 1))
            )
    else:
        if full_coupling is False:
            assert circuit.gate_count() == n + nlayers * (len(pauli_terms) + n)
        else:
            assert circuit.gate_count() == n + nlayers * (
                len(pauli_terms) + sum(range(n + 1)) / 2
            )


@pytest.mark.parametrize("mixer, full_coupling", [("AB", True), ("XY", 1), ("TC", 5)])
def test_QAOA_ansatz_errors(example_inputs, full_coupling, mixer):
    params, nlayers, pauli_terms, weights = example_inputs
    with pytest.raises(ValueError):
        QAOA_ansatz_for_Ising(
            params, nlayers, pauli_terms, weights, full_coupling, mixer
        )
