import numpy as np
import stim

import tensorcircuit as tc

np.random.seed(0)

tc.set_dtype("complex128")

clifford_one_qubit_gates = ["h", "x", "y", "z", "s"]
clifford_two_qubit_gates = ["cnot"]
clifford_gates = clifford_one_qubit_gates + clifford_two_qubit_gates


def genpair(num_qubits, count):
    choice = list(range(num_qubits))
    for _ in range(count):
        np.random.shuffle(choice)
        x, y = choice[:2]
        yield (x, y)


def random_clifford_circuit_with_mid_measurement(num_qubits, depth):
    c = tc.Circuit(num_qubits)
    for _ in range(depth):
        for j, k in genpair(num_qubits, 2):
            c.cnot(j, k)
        for j in range(num_qubits):
            getattr(c, np.random.choice(clifford_one_qubit_gates))(j)
        measured_qubit = np.random.randint(0, num_qubits - 1)
        sample, p = c.measure_reference(measured_qubit, with_prob=True)
        # Check if there is a non-zero probability to measure "0" for post-selection
        if (sample == "0" and p > 0.0) or (sample == "1" and p < 1.0):
            c.mid_measurement(measured_qubit, keep=0)
            c.measure_instruction(measured_qubit)
    return c


def convert_tc_circuit_to_stim_circuit(tc_circuit):
    stim_circuit = stim.Circuit()
    for instruction in tc_circuit._qir:
        gate_name = instruction["gate"].name
        qubits = instruction.get("index", [])
        if gate_name == "x":
            stim_circuit.append("X", qubits)
        elif gate_name == "y":
            stim_circuit.append("Y", qubits)
        elif gate_name == "z":
            stim_circuit.append("Z", qubits)
        elif gate_name == "h":
            stim_circuit.append("H", qubits)
        elif gate_name == "cnot":
            stim_circuit.append("CNOT", qubits)
        elif gate_name == "s":
            stim_circuit.append("S", qubits)
        else:
            raise ValueError(f"Unsupported gate: {gate_name}")
    for measurement in tc_circuit._extra_qir:
        qubit = measurement["index"]
        stim_circuit.append("M", qubit)
    return stim_circuit


# ref: https://quantumcomputing.stackexchange.com/questions/16718/measuring-entanglement-entropy-using-a-stabilizer-circuit-simulator
def get_binary_matrix(z_stabilizers):
    N = len(z_stabilizers)
    binary_matrix = np.zeros((N, 2 * N))
    r = 0  # Row number
    for row in z_stabilizers:
        c = 0  # Column number
        for i in row:
            if i == 3:  # Pauli Z
                binary_matrix[r, N + c] = 1
            if i == 2:  # Pauli Y
                binary_matrix[r, N + c] = 1
                binary_matrix[r, c] = 1
            if i == 1:  # Pauli X
                binary_matrix[r, c] = 1
            c += 1
        r += 1

    return binary_matrix


def get_bipartite_binary_matrix(binary_matrix, cut):
    N = len(binary_matrix)
    cutMatrix = np.zeros((N, 2 * cut))

    cutMatrix[:, :cut] = binary_matrix[:, :cut]
    cutMatrix[:, cut:] = binary_matrix[:, N : N + cut]

    return cutMatrix


# ref: https://gist.github.com/StuartGordonReid/eb59113cb29e529b8105?permalink_comment_id=3268301#gistcomment-3268301
def gf2_rank(matrix):
    n = len(matrix[0])
    rank = 0
    for col in range(n):
        j = 0
        rows = []
        while j < len(matrix):
            if matrix[j][col] == 1:
                rows += [j]
            j += 1
        if len(rows) >= 1:
            for c in range(1, len(rows)):
                for k in range(n):
                    matrix[rows[c]][k] = (matrix[rows[c]][k] + matrix[rows[0]][k]) % 2
            matrix.pop(rows[0])
            rank += 1
    for row in matrix:
        if sum(row) > 0:
            rank += 1
    return rank


# ref: https://quantumcomputing.stackexchange.com/questions/27795/exact-probabilities-of-outcomes-for-clifford-circuits-with-mid-circuit-measureme
def simulate_stim_circuit_with_mid_measurement(stim_circuit):
    simulator = stim.TableauSimulator()

    for instruction in stim_circuit.flattened():
        if instruction.name == "M":
            for t in instruction.targets_copy():
                expectaction = simulator.peek_z(t.value)
                desired = 0
                if expectaction == -1:
                    desired = 1
                simulator.postselect_z(t.value, desired_value=desired)
        else:
            c = stim.Circuit()
            c.append(instruction)
            simulator.do_circuit(c)

    return simulator.current_inverse_tableau() ** -1


if __name__ == "__main__":
    num_qubits = 10
    depth = 8

    tc_circuit = random_clifford_circuit_with_mid_measurement(num_qubits, depth)
    print(tc_circuit.draw(output="text"))

    stim_circuit = convert_tc_circuit_to_stim_circuit(tc_circuit)

    # Entanglement entropy calculation using stabilizer formalism
    stabilizer_tableau = simulate_stim_circuit_with_mid_measurement(stim_circuit)
    zs = [stabilizer_tableau.z_output(k) for k in range(len(stabilizer_tableau))]
    binary_matrix = get_binary_matrix(zs)
    cut_matrix = get_bipartite_binary_matrix(binary_matrix, num_qubits // 2)
    custom_entropy = (gf2_rank(cut_matrix.tolist()) - num_qubits // 2) * np.log(2)
    print("Stim Entanglement Entropy:", custom_entropy)

    # Entanglement entropy calculation using TensorCircuit
    state_vector = tc_circuit.wavefunction()
    assert np.linalg.norm(state_vector) > 0
    state_vector /= np.linalg.norm(state_vector)
    baseline_entropy = tc.quantum.entanglement_entropy(state_vector, num_qubits // 2)
    print("TensorCircuit Entanglement Entropy:", baseline_entropy)

    np.testing.assert_allclose(custom_entropy, baseline_entropy, atol=1e-8)
