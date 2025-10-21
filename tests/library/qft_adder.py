# %%
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, transpile
# from qiskit.providers.aer import QasmSimulator
from qiskit.circuit.library import QFT
from qiskit.circuit.library import MCMT
from qiskit.circuit.library import PhaseGate
import sys
from math import pi
from qiskit import transpile
import tensorcircuit as tc

# %%
"""
phi adder to implement n_qubits = n+1 bits (n bit encoding the original data) addition phi(a+b). Return a phi adder circuit. 
The returned circuit 
"""
def phi_Adder(n_qubits, a):

    """
    Here, n_qubits represents n + 1 inputs qubits
    """
    
 
    if a > 2**n_qubits - 1 or a < 0:
        print('a is out of range')
        sys.exit()

    p_Adder = QuantumCircuit(n_qubits)
    phi_array = np.zeros(n_qubits, dtype = complex)
    for m in range(n_qubits):
        phi = 2 * pi * a / 2**(m+1)
        phi_array [m] = phi
        p_Adder.p(phi, m)
    
    return p_Adder

# %%
"""
controlled phi adder by a single ancilla qubits
"""
def c_phi_Adder(n_qubits, a):
    
    if a > 2**n_qubits - 1 or a < 0:
        print('a is out of range')
        sys.exit()

    p_Adder = QuantumCircuit(n_qubits + 1)
    phi_array = np.zeros(n_qubits, dtype = complex)
    for m in range(n_qubits):

        phi = 2 * pi * a / 2**(m+1)
        phi_array [m] = phi
        p_Adder.cp(phi, n_qubits, m)
    
    
    return p_Adder
    

# %%
"""
Mod N Adder to implement a+b mod N, where a is constant (a, b < N are both n+1 bits digits,  2^n < N < 2^n+1, n_qubits = n + 2)
"""
def mod_N_Adder(n_qubits, a, N):

    n_all_qubits = n_qubits  + 1 # an ancilla is needed
    mod_N_adder = QuantumCircuit(n_all_qubits)
    a_add = phi_Adder(n_qubits, a)
    a_sub = a_add.inverse()
    N_add = phi_Adder(n_qubits, N)
    N_sub = N_add.inverse()
    control_phi_adder = c_phi_Adder(n_qubits, N)
    qft = QFT(num_qubits=n_qubits, approximation_degree=0, do_swaps=False, inverse=False, insert_barriers=False, name='QFT')
    qft_inv = qft.inverse()
    
    mod_N_adder = mod_N_adder.compose(qft)
    mod_N_adder.barrier()
    mod_N_adder = mod_N_adder.compose(a_add)
    mod_N_adder = mod_N_adder.compose(N_sub)
    mod_N_adder.barrier()
    mod_N_adder = mod_N_adder.compose(qft_inv)
    mod_N_adder.cx(n_qubits - 1, n_qubits)        
    mod_N_adder = mod_N_adder.compose(qft)
    mod_N_adder.barrier()
    mod_N_adder = mod_N_adder.compose(control_phi_adder)
    mod_N_adder.barrier()
    mod_N_adder = mod_N_adder.compose(a_sub)
    mod_N_adder.barrier()
    mod_N_adder = mod_N_adder.compose(qft_inv)
    mod_N_adder.x(n_qubits - 1)
    mod_N_adder.cx(n_qubits - 1, n_qubits)   
    mod_N_adder.x(n_qubits - 1)
    mod_N_adder = mod_N_adder.compose(qft)
    mod_N_adder.barrier()
    mod_N_adder = mod_N_adder.compose(a_add)
    mod_N_adder.barrier()
    mod_N_adder = mod_N_adder.compose(qft_inv)
       
    return mod_N_adder

# %%
def qft_init(b,n_qubits):
    qft = QFT(num_qubits=n_qubits, approximation_degree=0, do_swaps=False, inverse=False, insert_barriers=False, name='QFT')
    c_init = QuantumCircuit(n_qubits)
    bin_list = [int(i) for i in list('{0:0b}'.format(b))]
    bin_list = bin_list[::-1]
    for i in range(len(bin_list)):
        if bin_list[i]:
            # c_init.p(pi,i)
            c_init.x(i)
    c_init = c_init.compose(qft)
    return c_init

# %%
def qft_adder(a,b,n_qubits):
    n_all_qubits = n_qubits
    a_add = phi_Adder(n_all_qubits, a)
    # a_sub = a_add.inverse()
    b_init = qft_init(b,n_all_qubits)
    # N_add = phi_Adder(n_qubits, N)
    # N_sub = N_add.inverse()
    # control_phi_adder = c_phi_Adder(n_qubits, N)
    qft = QFT(num_qubits=n_all_qubits, approximation_degree=0, do_swaps=False, inverse=False, insert_barriers=False, name='QFT')
    qft_inv = qft.inverse()

    c_add = QuantumCircuit(n_all_qubits)
    
    c_add = c_add.compose(b_init)
    # c_add.barrier()
    c_add = c_add.compose(a_add)
    # c_add.barrier()
    c_add = c_add.compose(qft_inv)

    c_add = transpile(circuits=c_add,
            basis_gates=['h', 'rz', 'x', 'y', 'z', 'cz', 'cx'])
    return tc.Circuit.from_qiskit(c_add)
    # return c_add.decompose()