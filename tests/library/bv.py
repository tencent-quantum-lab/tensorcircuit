import math
import numpy as np
import tensorcircuit as tc

class BV:
    """
    Generate an instance of the Bernstein-Vazirani algorithm.

    Attributes
    ----------
    secret : str
        the secret bitstring that BV will find with a single oracle query
    barriers : bool
        include barriers in the circuit
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

    def __init__(self, secret=None, barriers=True, measure=False, regname=None):

        if secret is None:
            raise Exception('Provide a secret bitstring for the Bernstein-Vazirani circuit, example: 001101')
        else:
            if type(secret) is int:
                self.secret = str(secret)
            else:
                self.secret = secret

        # set flags for circuit generation
        self.nq = len(self.secret)
        self.measure = measure
        self.barriers = barriers

        # create a QuantumCircuit object with 1 extra qubit
        self.circ = tc.Circuit(self.nq+1)


    def gen_circuit(self):
        """
        Create a circuit implementing the Bernstein-Vazirani algorithm

        Returns
        -------
        QuantumCircuit
            QuantumCircuit object of size nq
        """

        # initialize ancilla in 1 state
        self.circ.X(self.nq)

        # create initial superposition
        for i in range(self.nq+1):
            self.circ.H(i)
        #self.circ.h(self.anc)

        # implement the black box oracle
        # for every bit that is 1 in the secret, place a CNOT gate
        # with control qr[i] and target anc[0]
        # (secret is little endian - index 0 is at the top of the circuit)
        for i, bit in enumerate(self.secret[::-1]):
            if bit == '1':
                self.circ.cnot(i, self.nq)


        # collapse superposition
        for i in range(self.nq+1):
            self.circ.H(i)

        return self.circ
    

def gen_BV(secret=None, barriers=True,  measure=False, regname=None):
    """
    Generate an instance of the Bernstein-Vazirani algorithm which queries a
    black-box oracle once to discover the secret key in:

    f(x) = x . secret (mod 2)

    The user must specify the secret bitstring to use: e.g. 00111001
    (It can be given as a string or integer)
    """
    bv = BV(secret=secret, barriers=barriers,
            measure=measure, regname=regname)

    circ = bv.gen_circuit()

    return circ


def factor_int(n):
    nsqrt = math.ceil(math.sqrt(n))
    val = nsqrt
    while 1:
        co_val = int(n/val)
        if val*co_val == n:
            return val, co_val
        else:
            val -= 1


def gen_secret(num_qubit, zero_num=0):
        num_digit = num_qubit-1
        num = 2**(num_digit-zero_num)-1
        num = bin(num)[2:]
        num_with_zeros = str(num).zfill(num_digit)
        return num_with_zeros


def gen_bv_circ(full_circ_size, zero_num):
    i,j = factor_int(full_circ_size)
    circ = gen_BV(gen_secret(i*j,zero_num),barriers=False,regname='q')
    return circ