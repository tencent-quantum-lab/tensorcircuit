import numpy as np
import tensorflow as tf
import tensorcircuit as tc
import stim

# Set the data type and backend for the circuit
ctype, rtype = tc.set_dtype("complex64")
K = tc.set_backend("tensorflow")

# Define the number of qubits and the number of layers in the circuit
n = 6
nlayers = 6

def create_stabilizer_circuit(n, nlayers):
    """
    Create a stabilizer circuit with specified number of qubits and layers.

    Args:
        n (int): Number of qubits in the circuit.
        nlayers (int): Number of layers in the circuit.

    Returns:
        tensorcircuit.Circuit: Stabilizer circuit.
    """
    c = tc.Circuit(n)
    for _ in range(nlayers):
        for i in range(n):
            c.H(i)
        for i in range(n - 1):
            c.CNOT(i, i + 1)
        for i in range(n):
            c.S(i)
    return c

def tc_to_stim(tc_circuit):
    """
    Transform a TensorCircuit to a Stim circuit.

    Args:
        tc_circuit (tensorcircuit.Circuit): Input TensorCircuit.

    Returns:
        stim.Circuit: Output Stim circuit.
    """
    stim_circuit = stim.Circuit()
    merged_qir = tc.translation._merge_extra_qir(tc_circuit._qir, tc_circuit._extra_qir)
    for instruction in merged_qir:
        gate_type = instruction["gate"]
        qubits = instruction.get("targets", [])
        if gate_type == 'H':
            stim_circuit.append_operation("H", qubits)
        elif gate_type == 'CNOT':
            stim_circuit.append_operation("CX", qubits)
        elif gate_type == 'S':
            stim_circuit.append_operation("S", qubits)
        elif gate_type == 'M':
            stim_circuit.append_operation("M", qubits)
    return stim_circuit

def compute_entanglement_entropy(stabilizer_tableau, traced_out_sites):
    """
    Compute the entanglement entropy of the stabilizer tableau.

    Args:
        stabilizer_tableau (numpy.ndarray): Stabilizer tableau.
        traced_out_sites (list): List of qubit indices to be traced out.

    Returns:
        int: Entanglement entropy.
    """
    num_rows = len(stabilizer_tableau)
    num_qubits = num_rows // 2
    binary_matrix = np.zeros((2 * num_qubits, 2 * num_qubits), dtype=int)
    for i in range(num_qubits):
        z_output = stabilizer_tableau.z_output(i)
        for j in range(num_qubits):
            if z_output[j]:
                binary_matrix[i, j + num_qubits] = 1
        x_output = stabilizer_tableau.x_output(i)
        for j in range(num_qubits):
            if x_output[j]:
                binary_matrix[i + num_qubits, j] = 1
    if np.count_nonzero(binary_matrix) == 0:
        return 0
    rank = np.linalg.matrix_rank(binary_matrix)
    entropy = len(traced_out_sites) - rank
    return entropy

def main():
    """Main function to execute the quantum computation and entropy calculation."""
    circuit = create_stabilizer_circuit(n, nlayers)
    stim_circuit = tc_to_stim(circuit)
    simulator = stim.TableauSimulator()
    for instruction in stim_circuit:
        simulator.do(instruction)
    stabilizer_tableau = simulator.current_inverse_tableau() ** -1

    traced_out_sites = list(range(n // 2))
    try:
        custom_entropy = compute_entanglement_entropy(stabilizer_tableau, traced_out_sites)
        print("Custom Entanglement Entropy:", custom_entropy)
    except Exception as e:
        print("Error in custom entanglement entropy calculation:", e)

    try:
        state_vector = circuit.wavefunction()
        baseline_entropy = tc.quantum.entanglement_entropy(state_vector, traced_out_sites)
        print("Baseline Entanglement Entropy:", baseline_entropy)
    except Exception as e:
        print("Error in baseline entanglement entropy calculation:", e)

if __name__ == "__main__":
    main()
