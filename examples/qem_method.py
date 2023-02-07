import numpy as np

import tensorcircuit as tc
from tensorcircuit.results import counts
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


def run(cs, shots):
    # customized backend for mitigation test
    ts = []
    for c in cs:
        count = simulator(c, shots)
        ts.append(count)
    return ts


def simulator(c, shots, logical_physical_mapping=None):
    # with readout_error noise
    nqubit = c._nqubits
    if logical_physical_mapping is None:
        logical_physical_mapping = {i: i for i in range(nqubit)}

    gg = []
    for i in range(200):
        gg.append(np.sin(i) * 0.02 + 0.978)
        # gg.append(0.98 - i * 0.01)
    readout_error = np.reshape(gg[0 : nqubit * 2], (nqubit, 2))
    mapped_readout_error = [[1, 1]] * nqubit
    for lq, phyq in logical_physical_mapping.items():
        mapped_readout_error[lq] = readout_error[phyq]
    return partial_sample(c, shots, mapped_readout_error)


def apply_readout_mitigation():
    nqubit = 4
    c = tc.Circuit(nqubit)
    c.H(0)
    c.cnot(0, 1)
    c.x(3)

    shots=10000

    idea_count = c.sample(batch=shots, allow_state=True, format="count_dict_bin")
    raw_count = run([c], 100000)[0]

    cal_qubits = [0, 1, 2, 3]
    use_qubits = [0, 1]

    # idea_value = c.expectation_ps(z=[0,1])
    idea_count2 = counts.marginal_count(idea_count, use_qubits)
    idea_value = counts.expectation(idea_count2, z=[0, 1])

    # use case 1
    mit = ReadoutMit(execute=run)
    mit.cals_from_system(cal_qubits, shots=shots, method="local")
    mit_count = mit.apply_correction(raw_count, use_qubits, method="inverse")
    mit_value1 = counts.expectation(mit_count, z=[0, 1])

    # use case 2
    mit = ReadoutMit(execute=run)
    mit.cals_from_system(cal_qubits, shots=shots, method="local")
    mit_value2 = mit.expectation(raw_count, z=[0, 1], method="inverse")

    # use case 3
    mit = ReadoutMit(execute=run)
    mit.cals_from_system(cal_qubits, shots=shots, method="global")
    mit_value3 = mit.expectation(raw_count, z=[0, 1], method="square")

    print("idea expectation value:", idea_value)
    print("mitigated expectation value:", mit_value1, mit_value2, mit_value3)


if __name__ == "__main__":
    apply_readout_mitigation()
