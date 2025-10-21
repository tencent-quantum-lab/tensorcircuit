'''
Teague Tomesh - 2/10/2020

Implementation of an n-bit ripple-carry adder.

Based on the specification given in Cuccaro, Draper, Kutin, Moulton.
(https://arxiv.org/abs/quant-ph/0410184v1)
'''

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import tensorcircuit as tc
from .randinit import qc_randinit


class RCAdder:
    """
    An n-bit ripple-carry adder can be generated using an instance of the
    RCAdder class by calling the gen_circuit() method.

    This adder circuit uses 1 ancilla qubit to add together two values
        a = a_(n-1)...a_0    and   b = b_(n-1)...a_0
    and store their sum
        s = s_n...s_0
    in the registers which initially held the b value.

    The adder circuit uses 2 + binary_len(a) + binary_len(b) qubits.
    The initial carry value is stored in the qubit at index = 0.
    The binary value of a_i is stored in the qubit at index = 2*i + 2
    The binary value of b_i is stored in the qubit at index = 2*i + 1
    The high bit, s_n, is stored in the last qubit at index = num_qubits - 1

    Attributes
    ----------
    nbits : int
        size, in bits, of the numbers the adder can handle
    nq : int
        number of qubits needed to construct the adder circuit
    a, b : int
        optional parameters to specify the numbers the adder should add.
        Will throw an exception if the length of the bitstring representations
        of a or b are greater than nbits.
    use_toffoli : bool
        Should the toffoli gate be used in the generated circuit or should it
        first be decomposed
    barriers : bool
        should barriers be included in the generated circuit
    regname : str
        optional string to name the quantum and classical registers. This
        allows for the easy concatenation of multiple QuantumCircuits.
    qr : QuantumRegister
        Qiskit QuantumRegister holding all of the quantum bits
    circ : QuantumCircuit
        Qiskit QuantumCircuit that represents the uccsd circuit
    """

    def __init__(self, nbits=None, a=0, b=0, use_toffoli=False, barriers=False,
                 measure=False, regname=None):

        # number of bits the adder can handle
        if nbits is None:
            raise Exception('Number of bits must be specified')
        else:
            self.nbits = nbits

        # given nbits, compute the number of qubits the adder will need
        # number of qubits = 1 ancilla for the initial carry value
        #                    + 2*nbits to hold both a and b
        #                    + 1 more qubit to hold the high bit, s_n
        self.nq = 1 + 2*nbits + 1

        # set flags for circuit generation
        if len('{0:b}'.format(a)) > nbits or len('{0:b}'.format(b)) > nbits:
            raise Exception('Binary representations of a and b must be less than or equal to nbits')

        self.a = a
        self.b = b
        self.use_toffoli = use_toffoli
        self.barriers = barriers
        self.measure = measure

        # create a QuantumCircuit object
        if regname is None:
            self.qr = QuantumRegister(self.nq)
        else:
            self.qr = QuantumRegister(self.nq, name=regname)
        self.circ = QuantumCircuit(self.qr)
        qc_randinit(self.circ)

        # add ClassicalRegister if measure is True
        if self.measure:
            self.cr = ClassicalRegister(self.nq)
            self.circ.add_register(self.cr)


    def _initialize_value(self, indices, value):
        """
        Initialize the qubits at indices to the given value

        Parameters
        ----------
        indices : List[int]
            List of qubit indices
        value : int
            The desired initial value
        """
        binstr = '{0:b}'.format(value)
        for index, val in enumerate(reversed(binstr)):
            if val is '1':
                self.circ.x(indices[index])


    def _toffoli(self, x, y, z):
        """
        Implement the toffoli gate using 1 and 2 qubit gates
        """
        self.circ.h(z)
        self.circ.cx(y,z)
        self.circ.tdg(z)
        self.circ.cx(x,z)
        self.circ.t(z)
        self.circ.cx(y,z)
        self.circ.t(y)
        self.circ.tdg(z)
        self.circ.cx(x,z)
        self.circ.cx(x,y)
        self.circ.t(z)
        self.circ.h(z)
        self.circ.t(x)
        self.circ.tdg(y)
        self.circ.cx(x,y)


    def _MAJ(self, x, y, z):
        """
        Implement the MAJ (Majority) gate described in Cuccaro, Draper, Kutin,
        Moulton.
        """
        self.circ.cx(z,y)
        self.circ.cx(z,x)

        if self.use_toffoli:
            self.circ.ccx(x,y,z)
        else:
            # use a decomposed version of toffoli
            self._toffoli(x,y,z)


    def _UMA(self, x, y, z):
        """
        Implement the UMA (UnMajority and Add) gate described in Cuccaro,
        Draper, Kutin, Moulton.
        """
        self.circ.x(y)
        self.circ.cx(x,y)
        if self.use_toffoli:
            self.circ.ccx(x,y,z)
        else:
            # use a decomposed version of toffoli
            self._toffoli(x,y,z)
        self.circ.x(y)
        self.circ.cx(z,x)
        self.circ.cx(z,y)


    def gen_circuit(self):
        """
        Create a circuit implementing the ripple-carry adder

        Returns
        -------
        QuantumCircuit
            QuantumCircuit object of size self.nq
        """
        high_bit_index = self.nq-1

        # initialize the a and b registers
        a_indices = [2*i+2 for i in range(self.nbits)]
        b_indices = [2*i+1 for i in range(self.nbits)]
        for index_list, value in zip([a_indices, b_indices], [self.a, self.b]):
            self._initialize_value(index_list, value)

        # compute the carry bits, c_i, in order using the MAJ ladder
        for a_i in a_indices:
            self._MAJ(a_i-2, a_i-1, a_i)

        # write the final carry bit value to the high bit register
        self.circ.cx(a_indices[-1], high_bit_index)

        # erase the carry bits in reverse order using the UMA ladder
        for a_i in reversed(a_indices):
            self._UMA(a_i-2, a_i-1, a_i)

        return self.circ


def gen_adder_circ(nbits):
    adder = RCAdder(nbits, a=0, b=0)
    circ = adder.gen_circuit()
    return tc.Circuit.from_qiskit(circ)

