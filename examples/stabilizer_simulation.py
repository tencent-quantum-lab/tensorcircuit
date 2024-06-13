import numpy as np
import stim

import tensorcircuit as tc

np.random.seed(0)

tc.set_dtype("complex128")

clifford_one_qubit_gates = ["H", "X", "Y", "Z", "S"]
clifford_two_qubit_gates = ["CNOT"]
clifford_gates = clifford_one_qubit_gates + clifford_two_qubit_gates


def genpair(num_qubits, count):
    choice = list(range(num_qubits))
    for _ in range(count):
        np.random.shuffle(choice)
        x, y = choice[:2]
        yield (x, y)


def random_clifford_circuit_with_mid_measurement(num_qubits, depth):
    c = tc.Circuit(num_qubits)
    operation_list = []
    for _ in range(depth):
        for j, k in genpair(num_qubits, 2):
            c.cnot(j, k)
            operation_list.append(("CNOT", (j, k)))
        for j in range(num_qubits):
            gate_name = np.random.choice(clifford_one_qubit_gates)
            getattr(c, gate_name)(j)
            operation_list.append((gate_name, (j,)))
        measured_qubit = np.random.randint(0, num_qubits - 1)
        sample, p = c.measure_reference(measured_qubit, with_prob=True)
        # Check if there is a non-zero probability to measure "0" for post-selection
        if (sample == "0" and not np.isclose(p, 0.0)) or (
            sample == "1" and not np.isclose(p, 1.0)
        ):
            c.mid_measurement(measured_qubit, keep=0)
            operation_list.append(("M", (measured_qubit,)))
    return c, operation_list


def convert_operation_list_to_stim_circuit(operation_list):
    stim_circuit = stim.Circuit()
    for instruction in operation_list:
        gate_name = instruction[0]
        qubits = instruction[1]
        stim_circuit.append(gate_name, qubits)
    return stim_circuit


# ref: https://quantumcomputing.stackexchange.com/questions/16718/measuring-entanglement-entropy-using-a-stabilizer-circuit-simulator
def get_binary_matrix(z_stabilizers):
    N = len(z_stabilizers)
    binary_matrix = np.zeros((N, 2 * N))
    for row_idx, row in enumerate(z_stabilizers):
        for col_idx, col in enumerate(row):
            if col == 3:  # Pauli Z
                binary_matrix[row_idx, N + col_idx] = 1
            if col == 2:  # Pauli Y
                binary_matrix[row_idx, N + col_idx] = 1
                binary_matrix[row_idx, col_idx] = 1
            if col == 1:  # Pauli X
                binary_matrix[row_idx, col_idx] = 1
    return binary_matrix


def get_cut_binary_matrix(binary_matrix, cut):
    N = len(binary_matrix)
    new_indices = [i for i in range(N) if i not in set(cut)] + [
        i + N for i in range(N) if i not in set(cut)
    ]
    return binary_matrix[:, new_indices]


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
                expectaction_value = simulator.peek_z(t.value)  # 1, 0, -1
                # there is a non-zero probability to measure "0" if expectaction_value is not -1
                if expectaction_value != -1:
                    simulator.postselect_z(t.value, desired_value=0)
        else:
            simulator.do(instruction)

    return simulator.current_inverse_tableau() ** -1


if __name__ == "__main__":
    # Number of qubits
    num_qubits = 8
    # Depth of the circuit
    depth = 10
    # index list that is traced out to calculate the entanglement entropy
    cut = [i for i in range(num_qubits // 2)]

    tc_circuit, op_list = random_clifford_circuit_with_mid_measurement(
        num_qubits, depth
    )
    print(tc_circuit.draw(output="text"))

    stim_circuit = convert_operation_list_to_stim_circuit(op_list)

    # Entanglement entropy calculation using stabilizer formalism
    stabilizer_tableau = simulate_stim_circuit_with_mid_measurement(stim_circuit)
    zs = [stabilizer_tableau.z_output(k) for k in range(len(stabilizer_tableau))]
    binary_matrix = get_binary_matrix(zs)
    bipartite_matrix = get_cut_binary_matrix(binary_matrix, cut)
    stim_entropy = (gf2_rank(bipartite_matrix.tolist()) - len(cut)) * np.log(2)
    print("Stim Entanglement Entropy:", stim_entropy)

    # Entanglement entropy calculation using TensorCircuit
    state_vector = tc_circuit.wavefunction()
    assert np.linalg.norm(state_vector) > 0
    # Normalize the state vector because mid-measurement operation is not unitary
    state_vector /= np.linalg.norm(state_vector)
    tc_entropy = tc.quantum.entanglement_entropy(state_vector, cut)
    print("TensorCircuit Entanglement Entropy:", tc_entropy)

    # Check if the entanglement entropies are close
    np.testing.assert_allclose(stim_entropy, tc_entropy, atol=1e-8)
