from qiskit import QuantumCircuit, QuantumRegister
import sys
import math
import numpy as np


class Dynamics:
    """
    Class to implement the simulation of quantum dynamics as described
    in Section 4.7 of Nielsen & Chuang (Quantum computation and quantum
    information (10th anniv. version), 2010.)

    A circuit implementing the quantum simulation can be generated for a given
    problem Hamiltonian parameterized by calling the gen_circuit() method.

    Attributes
    ----------
    H : ??
        The given Hamiltonian whose dynamics we want to simulate
    barriers : bool
        should barriers be included in the generated circuit
    measure : bool
        should a ClassicalRegister and measurements be added to the circuit
    regname : str
        optional string to name the quantum and classical registers. This
        allows for the easy concatenation of multiple QuantumCircuits.
    qr : QuantumRegister
        Qiskit QuantumRegister holding all of the quantum bits
    circ : QuantumCircuit
        Qiskit QuantumCircuit that represents the uccsd circuit
    """

    def __init__(self, H, barriers=False, measure=False, regname=None):

        # Hamiltonian
        self.H = H

        # set flags for circuit generation
        self.barriers = barriers
        self.nq = self.get_num_qubits()

        # create a QuantumCircuit object
        if regname is None:
            self.qr = QuantumRegister(self.nq)
        else:
            self.qr = QuantumRegister(self.nq, name=regname)
        self.circ = QuantumCircuit(self.qr)

        # Create and add an ancilla register to the circuit
        self.ancQ = QuantumRegister(1, 'ancQ')
        self.circ.add_register(self.ancQ)


    def get_num_qubits(self):
        """
        Given the problem Hamiltonian, return the appropriate number of qubits
        needed to simulate its dynamics.

        This number does not include the single ancilla qubit that is added
        to the circuit.
        """
        numq = 0
        for term in self.H:
            if len(term) > numq:
                numq = len(term)
        return numq


    def compute_to_Z_basis(self, pauli_str):
        """
        Take the given pauli_str of the form ABCD and apply operations to the
        circuit which will take it from the ABCD basis to the ZZZZ basis

        Parameters
        ----------
        pauli_str : str
            string of the form 'p1p2p3...pN' where pK is a Pauli matrix
        """
        for i, pauli in enumerate(pauli_str):
            if pauli is 'X':
                self.circ.h(self.qr[i])
            elif pauli is 'Y':
                self.circ.h(self.qr[i])
                self.circ.s(self.qr[i])


    def uncompute_to_Z_basis(self, pauli_str):
        """
        Take the given pauli_str of the form ABCD and apply operations to the
        circuit which will take it from the ZZZZ basis to the ABCD basis

        Parameters
        ----------
        pauli_str : str
            string of the form 'p1p2p3...pN' where pK is a Pauli matrix
        """
        for i, pauli in enumerate(pauli_str):
            if pauli is 'X':
                self.circ.h(self.qr[i])
            elif pauli is 'Y':
                self.circ.sdg(self.qr[i])
                self.circ.h(self.qr[i])


    def apply_phase_shift(self, delta_t):
        """
        Simulate the evolution of exp(-i(dt)Z)
        """
        # apply CNOT ladder -> compute parity
        for i in range(self.nq):
            self.circ.cx(self.qr[i], self.ancQ[0])

        # apply phase shift to the ancilla
        # rz applies the unitary: exp(-i*theta*Z/2)
        self.circ.rz(2*delta_t, self.ancQ[0])

        # apply CNOT ladder -> uncompute parity
        for i in range(self.nq-1, -1, -1):
            self.circ.cx(self.qr[i], self.ancQ[0])


    def gen_circuit(self):
        """
        Create a circuit implementing the quantum dynamics simulation

        Returns
        -------
        QuantumCircuit
            QuantumCircuit object of size nq with no ClassicalRegister and
            no measurements
        """
        
        # generate a naive version of a simulation circuit

        for term in self.H:
            self.compute_to_Z_basis(term)
            if self.barriers:
                self.circ.barrier()
            self.apply_phase_shift(1)
            if self.barriers:
                self.circ.barrier()
            self.uncompute_to_Z_basis(term)
            if self.barriers:
                self.circ.barrier()

        # generate a commutation aware version of a simulation circuit
        # simulate all commuting terms simulataneously by using 1 ancilla per 
        # term that will encode the phase shift based on the parity of the term.

        return self.circ


def gen_dynamics(H, barriers=True, measure=False, regname=None):
    """
    Generate a circuit to simulate the dynamics of a given Hamiltonian
    """

    dynamics = quantum_dynamics.Dynamics(H, barriers=barriers, measure=measure,
                                         regname=regname)

    circ = dynamics.gen_circuit()

    return circ
