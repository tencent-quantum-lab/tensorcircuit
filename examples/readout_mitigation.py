"""
Demonstration on readout error mitigation usage
"""

import numpy as np

import tensorcircuit as tc
from tensorcircuit.results.readout_mitigation import ReadoutMit


def partial_sample(c, batch, readout_error=None):
    measure_index = []
    for inst in c._extra_qir:
        if inst["name"] == "measure":
            measure_index.append(inst["index"][0])
    if len(measure_index) == 0:
        measure_index = list(range(c._nqubits))

    ct = c.sample(
        allow_state=True,
        batch=batch,
        readout_error=readout_error,
        format="count_dict_bin",
    )
    return tc.results.counts.marginal_count(ct, measure_index)


def simulator(c, shots, logical_physical_mapping=None):
    # with readout_error noise
    nqubit = c._nqubits
    if logical_physical_mapping is None:
        logical_physical_mapping = {i: i for i in range(nqubit)}

    gg = []
    for i in range(200):
        gg.append(np.sin(i) * 0.02 + 0.978)
        # mocked readout error
    readout_error = np.reshape(gg[0 : nqubit * 2], (nqubit, 2))
    mapped_readout_error = [[1, 1]] * nqubit
    for lq, phyq in logical_physical_mapping.items():
        mapped_readout_error[lq] = readout_error[phyq]
    return partial_sample(c, shots, mapped_readout_error)


def run(cs, shots):
    # customized backend for mitigation test
    # a simulator with readout error and qubit mapping supporting batch submission
    ts = []
    for c in cs:
        count = simulator(c, shots)
        ts.append(count)
    return ts


def apply_readout_mitigation():
    nqubit = 4
    c = tc.Circuit(nqubit)
    c.H(0)
    c.cnot(0, 1)
    c.x(3)

    shots = 10000

    raw_count = run([c], shots)[0]

    cal_qubits = [0, 1, 2, 3]
    use_qubits = [0, 1]

    # idea_value = c.expectation_ps(z=[0,1])
    idea_count = partial_sample(c, batch=shots)
    idea_count = tc.results.counts.marginal_count(idea_count, use_qubits)
    idea_value = tc.results.counts.expectation(idea_count, z=[0, 1])

    # use case 1
    mit = ReadoutMit(execute=run)
    mit.cals_from_system(cal_qubits, shots=shots, method="local")
    mit_count = mit.apply_correction(raw_count, use_qubits, method="inverse")
    mit_value1 = tc.results.counts.expectation(mit_count, z=[0, 1])

    # use case 2: directly mitigation on expectation
    mit = ReadoutMit(execute=run)
    mit.cals_from_system(cal_qubits, shots=shots, method="local")
    mit_value2 = mit.expectation(raw_count, z=[0, 1])

    # use case 3: global calibriation
    mit = ReadoutMit(execute=run)
    mit.cals_from_system(cal_qubits, shots=shots, method="global")
    mit_value3 = mit.expectation(raw_count, z=[0, 1], method="square")

    print("idea expectation value:", idea_value)
    print("mitigated expectation value:", mit_value1, mit_value2, mit_value3)


if __name__ == "__main__":
    apply_readout_mitigation()
