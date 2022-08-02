"""
Methods for abstract circuits independent of nodes, edges and contractions
"""
# pylint: disable=invalid-name

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from functools import reduce
from operator import add

import numpy as np
import tensornetwork as tn

from . import gates
from .cons import backend, dtypestr
from .vis import qir2tex
from .quantum import QuOperator


Gate = gates.Gate
Tensor = Any

sgates = (
    ["i", "x", "y", "z", "h", "t", "s", "td", "sd", "wroot"]
    + ["cnot", "cz", "swap", "cy", "iswap", "ox", "oy", "oz"]
    + ["toffoli", "fredkin"]
)
vgates = [
    "r",
    "cr",
    "rx",
    "ry",
    "rz",
    "rxx",
    "ryy",
    "rzz",
    "crx",
    "cry",
    "crz",
    "orx",
    "ory",
    "orz",
    "any",
    "exp",
    "exp1",
]
mpogates = ["multicontrol", "mpo"]
gate_aliases = [
    ["cnot", "cx"],
    ["fredkin", "cswap"],
    ["toffoli", "ccnot"],
    ["any", "unitary"],
]


class AbstractCircuit:
    _nqubits: int
    _qir: List[Dict[str, Any]]
    inputs: Tensor
    circuit_param: Dict[str, Any]
    is_mps: bool

    sgates = sgates
    vgates = vgates
    mpogates = mpogates
    gate_aliases = gate_aliases

    def apply_general_gate(
        self,
        gate: Union[Gate, QuOperator],
        *index: int,
        name: Optional[str] = None,
        split: Optional[Dict[str, Any]] = None,
        mpo: bool = False,
        ir_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        An implementation of this method should also append gate directionary to self._qir
        """
        raise NotImplementedError

    @staticmethod
    def apply_general_variable_gate_delayed(
        gatef: Callable[..., Gate],
        name: Optional[str] = None,
        mpo: bool = False,
    ) -> Callable[..., None]:
        if name is None:
            name = getattr(gatef, "n")

        def apply(self: "AbstractCircuit", *index: int, **vars: Any) -> None:
            split = None
            localname = name
            if "name" in vars:
                localname = vars["name"]
                del vars["name"]
            if "split" in vars:
                split = vars["split"]
                del vars["split"]
            gate_dict = {
                "gatef": gatef,
                "index": index,
                "name": localname,
                "split": split,
                "mpo": mpo,
                "parameters": vars,
            }
            # self._qir.append(gate_dict)
            gate = gatef(**vars)
            self.apply_general_gate(
                gate,
                *index,
                name=localname,
                split=split,
                mpo=mpo,
                ir_dict=gate_dict,
            )  # type: ignore

        return apply

    @staticmethod
    def apply_general_gate_delayed(
        gatef: Callable[[], Gate],
        name: Optional[str] = None,
        mpo: bool = False,
    ) -> Callable[..., None]:
        # it is more like a register instead of apply
        # nested function must be utilized, functools.partial doesn't work for method register on class
        # see https://re-ra.xyz/Python-中实例方法动态绑定的几组最小对立/
        if name is None:
            name = getattr(gatef, "n")
        defaultname = name

        def apply(
            self: "AbstractCircuit",
            *index: int,
            split: Optional[Dict[str, Any]] = None,
            name: Optional[str] = None,
        ) -> None:
            if name is not None:
                localname = name
            else:
                localname = defaultname  # type: ignore

            # split = None
            gate = gatef()
            gate_dict = {"gatef": gatef}

            self.apply_general_gate(
                gate,
                *index,
                name=localname,
                split=split,
                mpo=mpo,
                ir_dict=gate_dict,
            )

        return apply

    @classmethod
    def _meta_apply(cls) -> None:
        """
        The registration of gate methods on circuit class using reflection mechanism
        """
        for g in sgates:
            setattr(
                cls, g, cls.apply_general_gate_delayed(gatef=getattr(gates, g), name=g)
            )
            setattr(
                cls,
                g.upper(),
                cls.apply_general_gate_delayed(gatef=getattr(gates, g), name=g),
            )
            matrix = gates.matrix_for_gate(getattr(gates, g)())
            matrix = gates.bmatrix(matrix)
            doc = """
            Apply **%s** gate on the circuit.
            See :py:meth:`tensorcircuit.gates.%s_gate`.


            :param index: Qubit number that the gate applies on.
                The matrix for the gate is

                .. math::

                      %s

            :type index: int.
            """ % (
                g.upper(),
                g,
                matrix,
            )
            # docs = """
            # Apply **%s** gate on the circuit.

            # :param index: Qubit number that the gate applies on.
            # :type index: int.
            # """ % (
            #     g.upper()
            # )
            getattr(cls, g).__doc__ = doc
            getattr(cls, g.upper()).__doc__ = doc

        for g in vgates:
            setattr(
                cls,
                g,
                cls.apply_general_variable_gate_delayed(
                    gatef=getattr(gates, g), name=g
                ),
            )
            setattr(
                cls,
                g.upper(),
                cls.apply_general_variable_gate_delayed(
                    gatef=getattr(gates, g), name=g
                ),
            )
            doc = """
            Apply **%s** gate with parameters on the circuit.
            See :py:meth:`tensorcircuit.gates.%s_gate`.


            :param index: Qubit number that the gate applies on.
            :type index: int.
            :param vars: Parameters for the gate.
            :type vars: float.
            """ % (
                g.upper(),
                g,
            )
            getattr(cls, g).__doc__ = doc
            getattr(cls, g.upper()).__doc__ = doc

        for g in mpogates:
            setattr(
                cls,
                g,
                cls.apply_general_variable_gate_delayed(
                    gatef=getattr(gates, g), name=g, mpo=True
                ),
            )
            setattr(
                cls,
                g.upper(),
                cls.apply_general_variable_gate_delayed(
                    gatef=getattr(gates, g), name=g, mpo=True
                ),
            )
            doc = """
            Apply %s gate in MPO format on the circuit.
            See :py:meth:`tensorcircuit.gates.%s_gate`.

            :param index: Qubit number that the gate applies on.
            :type index: int.
            :param vars: Parameters for the gate.
            :type vars: float.
            """ % (
                g,
                g,
            )
            getattr(cls, g).__doc__ = doc
            getattr(cls, g.upper()).__doc__ = doc

        for gate_alias in gate_aliases:
            present_gate = gate_alias[0]
            for alias_gate in gate_alias[1:]:
                setattr(cls, alias_gate, getattr(cls, present_gate))

    def to_qir(self) -> List[Dict[str, Any]]:
        """
        Return the quantum intermediate representation of the circuit.

        :Example:

        .. code-block:: python

            >>> c = tc.Circuit(2)
            >>> c.CNOT(0, 1)
            >>> c.to_qir()
            [{'gatef': cnot, 'gate': Gate(
                name: 'cnot',
                tensor:
                    array([[[[1.+0.j, 0.+0.j],
                            [0.+0.j, 0.+0.j]],

                            [[0.+0.j, 1.+0.j],
                            [0.+0.j, 0.+0.j]]],


                        [[[0.+0.j, 0.+0.j],
                            [0.+0.j, 1.+0.j]],

                            [[0.+0.j, 0.+0.j],
                            [1.+0.j, 0.+0.j]]]], dtype=complex64),
                edges: [
                    Edge(Dangling Edge)[0],
                    Edge(Dangling Edge)[1],
                    Edge('cnot'[2] -> 'qb-1'[0] ),
                    Edge('cnot'[3] -> 'qb-2'[0] )
                ]), 'index': (0, 1), 'name': 'cnot', 'split': None, 'mpo': False}]

        :return: The quantum intermediate representation of the circuit.
        :rtype: List[Dict[str, Any]]
        """
        return self._qir

    @classmethod
    def from_qir(
        cls, qir: List[Dict[str, Any]], circuit_params: Optional[Dict[str, Any]] = None
    ) -> "AbstractCircuit":
        """
        Restore the circuit from the quantum intermediate representation.

        :Example:

        >>> c = tc.Circuit(3)
        >>> c.H(0)
        >>> c.rx(1, theta=tc.array_to_tensor(0.7))
        >>> c.exp1(0, 1, unitary=tc.gates._zz_matrix, theta=tc.array_to_tensor(-0.2), split=split)
        >>> len(c)
        7
        >>> c.expectation((tc.gates.z(), [1]))
        array(0.764842+0.j, dtype=complex64)
        >>> qirs = c.to_qir()
        >>>
        >>> c = tc.Circuit.from_qir(qirs, circuit_params={"nqubits": 3})
        >>> len(c._nodes)
        7
        >>> c.expectation((tc.gates.z(), [1]))
        array(0.764842+0.j, dtype=complex64)

        :param qir: The quantum intermediate representation of a circuit.
        :type qir: List[Dict[str, Any]]
        :param circuit_params: Extra circuit parameters.
        :type circuit_params: Optional[Dict[str, Any]]
        :return: The circuit have same gates in the qir.
        :rtype: Circuit
        """
        if circuit_params is None:
            circuit_params = {}
        if "nqubits" not in circuit_params:
            nqubits = 0
            for d in qir:
                if max(d["index"]) > nqubits:  # type: ignore
                    nqubits = max(d["index"])  # type: ignore
            nqubits += 1
            circuit_params["nqubits"] = nqubits

        c = cls(**circuit_params)  # type: ignore
        c = cls._apply_qir(c, qir)
        return c

    @staticmethod
    def _apply_qir(
        c: "AbstractCircuit", qir: List[Dict[str, Any]]
    ) -> "AbstractCircuit":
        for d in qir:
            if "parameters" not in d:
                c.apply_general_gate_delayed(d["gatef"], d["name"], mpo=d["mpo"])(  # type: ignore
                    c, *d["index"], split=d["split"]  # type: ignore
                )
            else:
                c.apply_general_variable_gate_delayed(d["gatef"], d["name"], mpo=d["mpo"])(  # type: ignore
                    c, *d["index"], **d["parameters"], split=d["split"]  # type: ignore
                )
        return c

    def inverse(
        self, circuit_params: Optional[Dict[str, Any]] = None
    ) -> "AbstractCircuit":
        """
        inverse the circuit, return a new inversed circuit

        :EXAMPLE:

        >>> c = tc.Circuit(2)
        >>> c.H(0)
        >>> c.rzz(1, 2, theta=0.8)
        >>> c1 = c.inverse()

        :param circuit_params: keywords dict for initialization the new circuit, defaults to None
        :type circuit_params: Optional[Dict[str, Any]], optional
        :return: the inversed circuit
        :rtype: Circuit
        """
        if circuit_params is None:
            circuit_params = {}
        if "nqubits" not in circuit_params:
            circuit_params["nqubits"] = self._nqubits

        c = type(self)(**circuit_params)  # type: ignore
        for d in reversed(self._qir):
            if "parameters" not in d:
                self.apply_general_gate_delayed(d["gatef"].adjoint(), d["name"], mpo=d["mpo"])(  # type: ignore
                    c, *d["index"], split=d["split"]  # type: ignore
                )
            else:
                self.apply_general_variable_gate_delayed(d["gatef"].adjoint(), d["name"], mpo=d["mpo"])(  # type: ignore
                    c, *d["index"], **d["parameters"], split=d["split"]  # type: ignore
                )

        return c

    def append_from_qir(self, qir: List[Dict[str, Any]]) -> None:
        """
        Apply the ciurict in form of quantum intermediate representation after the current cirucit.

        :Example:

        >>> c = tc.Circuit(3)
        >>> c.H(0)
        >>> c.to_qir()
        [{'gatef': h, 'gate': Gate(...), 'index': (0,), 'name': 'h', 'split': None, 'mpo': False}]
        >>> c2 = tc.Circuit(3)
        >>> c2.CNOT(0, 1)
        >>> c2.to_qir()
        [{'gatef': cnot, 'gate': Gate(...), 'index': (0, 1), 'name': 'cnot', 'split': None, 'mpo': False}]
        >>> c.append_from_qir(c2.to_qir())
        >>> c.to_qir()
        [{'gatef': h, 'gate': Gate(...), 'index': (0,), 'name': 'h', 'split': None, 'mpo': False},
         {'gatef': cnot, 'gate': Gate(...), 'index': (0, 1), 'name': 'cnot', 'split': None, 'mpo': False}]

        :param qir: The quantum intermediate representation.
        :type qir: List[Dict[str, Any]]
        """
        self._apply_qir(self, qir)

    def to_qiskit(self) -> Any:
        """
        Translate ``tc.Circuit`` to a qiskit QuantumCircuit object.

        :return: A qiskit object of this circuit.
        """
        from .translation import qir2qiskit

        qir = self.to_qir()
        return qir2qiskit(qir, n=self._nqubits)

    def draw(self, **kws: Any) -> Any:
        """
        Visualise the circuit.
        This method recevies the keywords as same as qiskit.circuit.QuantumCircuit.draw.
        More details can be found here: https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.draw.html.

        :Example:
        >>> c = tc.Circuit(3)
        >>> c.H(1)
        >>> c.X(2)
        >>> c.CNOT(0, 1)
        >>> c.draw(output='text')
        q_0: ───────■──
             ┌───┐┌─┴─┐
        q_1: ┤ H ├┤ X ├
             ├───┤└───┘
        q_2: ┤ X ├─────
             └───┘
        """
        return self.to_qiskit().draw(**kws)

    @classmethod
    def from_qiskit(
        cls, qc: Any, n: Optional[int] = None, inputs: Optional[List[float]] = None
    ) -> "AbstractCircuit":
        """
        Import Qiskit QuantumCircuit object as a ``tc.Circuit`` object.

        :Example:

        >>> from qiskit import QuantumCircuit
        >>> qisc = QuantumCircuit(3)
        >>> qisc.h(2)
        >>> qisc.cswap(1, 2, 0)
        >>> qisc.swap(0, 1)
        >>> c = tc.Circuit.from_qiskit(qisc)

        :param qc: Qiskit Circuit object
        :type qc: QuantumCircuit in Qiskit
        :param n: The number of qubits for the circuit
        :type n: int
        :param inputs: possible input wavefunction for ``tc.Circuit``, defaults to None
        :type inputs: Optional[List[float]], optional
        :return: The same circuit but as tensorcircuit object
        :rtype: Circuit
        """
        from .translation import qiskit2tc

        if n is None:
            n = qc.num_qubits

        return qiskit2tc(qc.data, n, inputs, is_dm=cls.is_dm)  # type: ignore

    def vis_tex(self, **kws: Any) -> str:
        """
        Generate latex string based on quantikz latex package

        :return: Latex string that can be directly compiled via, e.g. latexit
        :rtype: str
        """
        if (not self.is_mps) and (self.inputs is not None):
            init = ["" for _ in range(self._nqubits)]
            init[self._nqubits // 2] = "\psi"
            okws = {"init": init}
        else:
            okws = {"init": None}  # type: ignore
        okws.update(kws)
        return qir2tex(self._qir, self._nqubits, **okws)  # type: ignore

    tex = vis_tex

    def select_gate(self, which: Tensor, kraus: Sequence[Gate], *index: int) -> None:
        """
        Apply ``which``-th gate from ``kraus`` list, i.e. apply kraus[which]

        :param which: Tensor of shape [] and dtype int
        :type which: Tensor
        :param kraus: A list of gate in the form of ``tc.gate`` or Tensor
        :type kraus: Sequence[Gate]
        :param index: the qubit lines the gate applied on
        :type index: int
        """
        kraus = [k.tensor if isinstance(k, tn.Node) else k for k in kraus]
        kraus = [gates.array_to_tensor(k) for k in kraus]
        l = len(kraus)
        r = backend.onehot(which, l)
        r = backend.cast(r, dtype=dtypestr)
        tensor = reduce(add, [r[i] * kraus[i] for i in range(l)])
        self.any(*index, unitary=tensor)  # type: ignore

    conditional_gate = select_gate

    def cond_measurement(self, index: int) -> Tensor:
        """
        Measurement on z basis at ``index`` qubit based on quantum amplitude
        (not post-selection). The highlight is that this method can return the
        measured result as a int Tensor and thus maintained a jittable pipeline.

        :Example:

        >>> c = tc.Circuit(2)
        >>> c.H(0)
        >>> r = c.cond_measurement(0)
        >>> c.conditional_gate(r, [tc.gates.i(), tc.gates.x()], 1)
        >>> c.expectation([tc.gates.z(), [0]]), c.expectation([tc.gates.z(), [1]])
        # two possible outputs: (1, 1) or (-1, -1)

        .. note::

            In terms of ``DMCircuit``, this method returns nothing and the density
            matrix after this method is kept in mixed state without knowing the
            measuremet resuslts



        :param index: the qubit for the z-basis measurement
        :type index: int
        :return: 0 or 1 for z measurement on up and down freedom
        :rtype: Tensor
        """
        return self.general_kraus(  # type: ignore
            [np.array([[1.0, 0], [0, 0]]), np.array([[0, 0], [0, 1]])], index, name="measure"  # type: ignore
        )

    cond_measure = cond_measurement

    def prepend(self, c: "AbstractCircuit") -> "AbstractCircuit":
        """
        prepend circuit ``c`` before

        :param c: The other circuit to be prepended
        :type c: BaseCircuit
        :return: The composed circuit
        :rtype: BaseCircuit
        """
        qir1 = self.to_qir()
        qir0 = c.to_qir()
        newc = type(self).from_qir(qir0 + qir1, self.circuit_param)
        self.__dict__.update(newc.__dict__)
        return self

    def append(self, c: "AbstractCircuit") -> "AbstractCircuit":
        """
        append circuit ``c`` before

        :example:

        >>> c1 = tc.Circuit(2)
        >>> c1.H(0)
        >>> c1.H(1)
        >>> c2 = tc.Circuit(2)
        >>> c2.cnot(0, 1)
        >>> c1.append(c2)
        <tensorcircuit.circuit.Circuit object at 0x7f8402968970>
        >>> c1.draw()
            ┌───┐
        q_0:┤ H ├──■──
            ├───┤┌─┴─┐
        q_1:┤ H ├┤ X ├
            └───┘└───┘

        :param c: The other circuit to be appended
        :type c: BaseCircuit
        :return: The composed circuit
        :rtype: BaseCircuit
        """
        qir1 = self.to_qir()
        qir2 = c.to_qir()
        newc = type(self).from_qir(qir1 + qir2, self.circuit_param)
        self.__dict__.update(newc.__dict__)
        return self
