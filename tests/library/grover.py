from qiskit.circuit.library import GroverOperator
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.compiler import transpile
import tensorcircuit as tc

def gen_grover_circ(width):
    oracle = QuantumCircuit(width,name='q')
    oracle.z(width-1)
    full_circuit = GroverOperator(oracle, insert_barriers=False, name='q')
    full_circuit = dag_to_circuit(circuit_to_dag(full_circuit))
    full_circuit = full_circuit.decompose()
    full_circuit = transpile(full_circuit,optimization_level=3)
    dag = circuit_to_dag(full_circuit)
    for node in dag.op_nodes():        
        # calculate the replacement
        if node.op.name == "p":
            replacement = QuantumCircuit(1)
            angle = node.op.params[0]
            replacement.rz(angle,0)
            # replace the node with our new decomposition
            dag.substitute_node_with_dag(node, circuit_to_dag(replacement))
        elif node.op.name == "cu1":
            replacement = QuantumCircuit(2)
            angle = node.op.params[0]
            replacement.cp(angle, 0, 1)
            dag.substitute_node_with_dag(node, circuit_to_dag(replacement))
            
    return tc.Circuit.from_qiskit(dag_to_circuit(dag))
    # return dag_to_circuit(dag)