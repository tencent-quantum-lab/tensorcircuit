"""
Methods for abstract circuits independent of nodes, edges and contractions
"""

# pylint: disable=invalid-name

from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Tuple
from copy import deepcopy
from functools import reduce
from operator import add
import json
import logging

import numpy as np
import tensornetwork as tn

from . import gates
from .cons import backend, dtypestr
from .vis import qir2tex
from .quantum import QuOperator
from .utils import is_sequence

logger = logging.getLogger(__name__)

Gate = gates.Gate
Tensor = Any

sgates = (
    ["i", "x", "y", "z", "h", "t", "s", "td", "sd", "wroot"]
    + ["cnot", "cz", "swap", "cy", "ox", "oy", "oz"]
    + ["toffoli", "fredkin"]
)
vgates = [
    "r",
    "cr",
    "u",
    "cu",
    "rx",
    "ry",
    "rz",
    "phase",
    "rxx",
    "ryy",
    "rzz",
    "cphase",
    "crx",
    "cry",
    "crz",
    "orx",
    "ory",
    "orz",
    "iswap",
    "any",
    "exp",
    "exp1",
]
mpogates = ["multicontrol", "mpo"]
gate_aliases = [
    ["cnot", "cx"],
    ["fredkin", "cswap"],
    ["toffoli", "ccnot"],
    ["toffoli", "ccx"],
    ["any", "unitary"],
    ["sd", "sdg"],
    ["td", "tdg"],
]


class AbstractCircuit:
    _nqubits: int
    _qir: List[Dict[str, Any]]
    _extra_qir: List[Dict[str, Any]]
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
            )

        def apply_list(self: "AbstractCircuit", *index: int, **vars: Any) -> None:
            if isinstance(index[0], int):
                apply(self, *index, **vars)
            elif is_sequence(index[0]) or isinstance(index[0], range):
                for i, ind in enumerate(zip(*index)):
                    nvars = {}
                    for k, v in vars.items():
                        try:
                            nvars[k] = v[i]
                        except Exception:  # pylint: disable=W0703
                            # (TypeError, IndexError) is not enough,
                            # tensorflow.python.framework.errors_impl.InvalidArgumentError
                            # not ideally captured here
                            nvars[k] = v
                    apply(self, *ind, **nvars)
            else:
                raise ValueError("Illegal index specification")

        return apply_list

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
                localname = defaultname

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

        def apply_list(self: "AbstractCircuit", *index: int, **kws: Any) -> None:
            if isinstance(index[0], int):
                apply(self, *index, **kws)
            elif is_sequence(index[0]) or isinstance(index[0], range):
                for ind in zip(*index):
                    apply(self, *ind, **kws)
            else:
                raise ValueError("Illegal index specification")

        return apply_list

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
        # TODO(@refraction-ray): add extra_qir support
        if circuit_params is None:
            circuit_params = {}
        if "nqubits" not in circuit_params:
            nqubits = 0
            for d in qir:
                if max(d["index"]) > nqubits:
                    nqubits = max(d["index"])
            nqubits += 1
            circuit_params["nqubits"] = nqubits

        c = cls(**circuit_params)
        c = cls._apply_qir(c, qir)
        return c

    @staticmethod
    def _apply_qir(
        c: "AbstractCircuit", qir: List[Dict[str, Any]]
    ) -> "AbstractCircuit":
        for d in qir:
            if "parameters" not in d:
                c.apply_general_gate_delayed(d["gatef"], d["name"], mpo=d["mpo"])(
                    c, *d["index"], split=d["split"]
                )
            else:
                c.apply_general_variable_gate_delayed(
                    d["gatef"], d["name"], mpo=d["mpo"]
                )(c, *d["index"], **d["parameters"], split=d["split"])
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

        c = type(self)(**circuit_params)
        for d in reversed(self._qir):
            if "parameters" not in d:
                if d["gatef"].n in self.sgates and d["gatef"].n not in [
                    "wroot",
                    "sd",
                    "td",
                ]:
                    self.apply_general_gate_delayed(
                        d["gatef"], d["name"], mpo=d["mpo"]
                    )(c, *d["index"], split=d["split"])
                elif d["gatef"].n in ["sd", "td"]:
                    self.apply_general_gate_delayed(
                        getattr(gates, d["gatef"].n[:-1]), d["name"], mpo=d["mpo"]
                    )(c, *d["index"], split=d["split"])
                else:
                    self.apply_general_gate_delayed(
                        d["gatef"].adjoint(), d["name"], mpo=d["mpo"]
                    )(c, *d["index"], split=d["split"])
            else:
                if d["gatef"].n in ["r", "cr"]:
                    params = {k: v for k, v in d["parameters"].items()}
                    if "theta" in params:
                        params["theta"] = -params.get("theta", 0.0)
                    self.apply_general_variable_gate_delayed(
                        d["gatef"], d["name"], mpo=d["mpo"]
                    )(c, *d["index"], **params, split=d["split"])
                elif d["gatef"].n in ["u", "cu"]:
                    params = {k: v for k, v in d["parameters"].items()}
                    # deepcopy fail with tf+jit
                    params["lbd"] = -d["parameters"].get("phi", 0) + np.pi
                    params["phi"] = -d["parameters"].get("lbd", 0) + np.pi
                    self.apply_general_variable_gate_delayed(
                        d["gatef"], d["name"], mpo=d["mpo"]
                    )(c, *d["index"], **params, split=d["split"])
                elif d["gatef"].n in self.vgates and d["gatef"].n not in [
                    "any",
                    "exp",
                    "exp1",
                ]:
                    params = {k: v for k, v in d["parameters"].items()}
                    df = 0.0
                    if d["gatef"].n == "iswap":
                        df = 1.0
                    params["theta"] = -params.get("theta", df)
                    self.apply_general_variable_gate_delayed(
                        d["gatef"], d["name"], mpo=d["mpo"]
                    )(c, *d["index"], **params, split=d["split"])

                # TODO(@refraction-ray): multi control gate?

                else:
                    self.apply_general_variable_gate_delayed(
                        d["gatef"].adjoint(), d["name"], mpo=d["mpo"]
                    )(c, *d["index"], **d["parameters"], split=d["split"])

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

    def initial_mapping(
        self,
        logical_physical_mapping: Dict[int, int],
        n: Optional[int] = None,
        circuit_params: Optional[Dict[str, Any]] = None,
    ) -> "AbstractCircuit":
        """
        generate a new circuit with the qubit mapping given by ``logical_physical_mapping``

        :param logical_physical_mapping: how to map logical qubits to the physical qubits on the new circuit
        :type logical_physical_mapping: Dict[int, int]
        :param n: number of qubit of the new circuit, can be different from the original one, defaults to None
        :type n: Optional[int], optional
        :param circuit_params: _description_, defaults to None
        :type circuit_params: Optional[Dict[str, Any]], optional
        :return: _description_
        :rtype: AbstractCircuit
        """
        if circuit_params is None:
            circuit_params = {}
        if "nqubits" not in circuit_params:
            if n is not None:
                circuit_params["nqubits"] = n
            else:
                circuit_params["nqubits"] = self._nqubits

        c = type(self)(**circuit_params)

        for d in self.to_qir():
            mapped_index = [logical_physical_mapping[i] for i in d["index"]]

            if "parameters" not in d:
                c.apply_general_gate_delayed(d["gatef"], d["name"], mpo=d["mpo"])(
                    c, *mapped_index, split=d["split"]
                )
            else:
                c.apply_general_variable_gate_delayed(
                    d["gatef"], d["name"], mpo=d["mpo"]
                )(c, *mapped_index, **d["parameters"], split=d["split"])
        for d in self._extra_qir:
            mapped_index = [logical_physical_mapping[i] for i in d["index"]]
            dc = deepcopy(d)
            dc["index"] = mapped_index
            c._extra_qir.append(dc)

        return c

    def get_positional_logical_mapping(self) -> Dict[int, int]:
        """
        Get positional logical mapping dict based on measure instruction.
        This function is useful when we only measure part of the qubits in the circuit,
        to process the count result from partial measurement, we must be aware of the mapping,
        i.e. for each position in the count bitstring, what is the corresponding qubits (logical)
        defined on the circuit

        :return: ``positional_logical_mapping``
        :rtype: Dict[int, int]
        """
        id_mapping = {i: i for i in range(self._nqubits)}
        if len(self._extra_qir) == 0:
            return id_mapping
        measure_index = []
        for inst in self._extra_qir:
            if inst["name"] == "measure":
                measure_index.append(inst["index"][0])
        if len(measure_index) == 0:
            return id_mapping
        return {i: j for i, j in enumerate(measure_index)}

    @staticmethod
    def standardize_gate(name: str) -> str:
        """
        standardize the gate name to tc common gate sets

        :param name: non-standard gate name
        :type name: str
        :return: the standard gate name
        :rtype: str
        """
        name = name.lower()
        for g1, g2 in gate_aliases:
            if name == g2:
                name = g1
                break
        if name not in sgates + vgates + mpogates:
            logger.warning(
                "gate name %s not in the common gate set that tc supported" % name
            )
        return name

    def gate_count(self, gate_list: Optional[Union[str, Sequence[str]]] = None) -> int:
        """
        count the gate number of the circuit

        :Example:

        >>> c = tc.Circuit(3)
        >>> c.h(0)
        >>> c.multicontrol(0, 1, 2, ctrl=[0, 1], unitary=tc.gates._x_matrix)
        >>> c.toffolli(1, 2, 0)
        >>> c.gate_count()
        3
        >>> c.gate_count(["multicontrol", "toffoli"])
        2

        :param gate_list: gate name or gate name list to be counted, defaults to None (counting all gates)
        :type gate_list: Optional[Sequence[str]], optional
        :return: the total number of all gates or gates in the ``gate_list``
        :rtype: int
        """
        if gate_list is None:
            return len(self._qir)
        else:
            if isinstance(gate_list, str):
                gate_list = [gate_list]
            gate_list = [self.standardize_gate(g) for g in gate_list]
            c = 0
            for d in self._qir:
                if d["gatef"].n in gate_list:
                    c += 1
            return c

    def gate_count_by_condition(
        self, cond_func: Callable[[Dict[str, Any]], bool]
    ) -> int:
        """
        count the number of gates that satisfy certain condition

        :Example:

        >>> c = tc.Circuit(3)
        >>> c.x(0)
        >>> c.h(0)
        >>> c.multicontrol(0, 1, 2, ctrl=[0, 1], unitary=tc.gates._x_matrix)
        >>> c.gate_count_by_condition(lambda qir: qir["index"] == (0, ))
        2
        >>> c.gate_count_by_condition(lambda qir: qir["mpo"])
        1

        :param cond_func: the condition for counting the gate
        :type cond_func: Callable[[Dict[str, Any]], bool]
        :return: the total number of all gates which satisfy the ``condition``
        :rtype: int
        """
        count = 0
        for d in self._qir:
            if cond_func(d):
                count += 1
        return count

    def gate_summary(self) -> Dict[str, int]:
        """
        return the summary dictionary on gate type - gate count pair

        :return: the gate count dict by gate type
        :rtype: Dict[str, int]
        """
        summary: Dict[str, int] = {}
        for d in self._qir:
            summary[d["name"]] = summary.get(d["name"], 0) + 1
        return summary

    def measure_instruction(self, *index: int) -> None:
        """
        add a measurement instruction flag, no effect on numerical simulation

        :param index: the corresponding qubits
        :type index: int
        """
        l = len(self._qir)
        for ind in index:
            d = {
                "index": [ind],
                "name": "measure",
                "gatef": "measure",
                "instruction": True,
                "pos": l,
            }
            self._extra_qir.append(d)

    def reset_instruction(self, *index: int) -> None:
        """
        add a reset instruction flag, no effect on numerical simulation

        :param index: the corresponding qubits
        :type index: int
        """
        l = len(self._qir)
        for ind in index:
            d = {
                "index": [ind],
                "name": "reset",
                "gatef": "reset",
                "instruction": True,
                "pos": l,
            }
            self._extra_qir.append(d)

    def barrier_instruction(self, *index: List[int]) -> None:
        """
        add a barrier instruction flag, no effect on numerical simulation

        :param index: the corresponding qubits
        :type index: List[int]
        """
        l = len(self._qir)
        d = {
            "index": index,
            "name": "barrier",
            "gatef": "barrier",
            "instruction": True,
            "pos": l,
        }
        self._extra_qir.append(d)

    def to_cirq(self, enable_instruction: bool = False) -> Any:
        """
        Translate ``tc.Circuit`` to a cirq circuit object.

        :param enable_instruction: whether also export measurement and reset instructions
        :type enable_instruction: bool, defaults to False
        :return: A cirq circuit of this circuit.
        """
        from .translation import qir2cirq

        qir = self.to_qir()
        if enable_instruction is False:
            return qir2cirq(qir, n=self._nqubits)
        return qir2cirq(qir, n=self._nqubits, extra_qir=self._extra_qir)

    def to_qiskit(
        self, enable_instruction: bool = False, enable_inputs: bool = False
    ) -> Any:
        """
        Translate ``tc.Circuit`` to a qiskit QuantumCircuit object.

        :param enable_instruction: whether also export measurement and reset instructions
        :type enable_instruction: bool, defaults to False
        :param enable_inputs: whether also export the inputs
        :type enable_inputs: bool, defaults to False
        :return: A qiskit object of this circuit.
        """
        from .translation import qir2qiskit, perm_matrix

        qir = self.to_qir()
        if enable_instruction:
            extra_qir = self._extra_qir
        else:
            extra_qir = None
        if enable_inputs and self.circuit_param.get("inputs") is not None:
            initialization = perm_matrix(self._nqubits).T @ self.circuit_param["inputs"]
        else:
            initialization = None
        return qir2qiskit(
            qir,
            n=self._nqubits,
            extra_qir=extra_qir,
            initialization=initialization,
        )

    def to_openqasm(self, **kws: Any) -> str:
        """
        transform circuit to openqasm via qiskit circuit,
        see https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.qasm.html
        for usage on possible options for ``kws``

        :return: circuit representation in openqasm format
        :rtype: str
        """
        return self.to_qiskit(enable_instruction=True).qasm(**kws)  # type: ignore

    @classmethod
    def from_openqasm(
        cls,
        qasmstr: str,
        circuit_params: Optional[Dict[str, Any]] = None,
        keep_measure_order: bool = False,
    ) -> "AbstractCircuit":
        from qiskit.circuit import QuantumCircuit

        if keep_measure_order is True:
            from .translation import qiskit_from_qasm_str_ordered_measure

            qiskit_circ = qiskit_from_qasm_str_ordered_measure(qasmstr)
        else:
            qiskit_circ = QuantumCircuit.from_qasm_str(qasmstr)
        c = cls.from_qiskit(qiskit_circ, circuit_params=circuit_params)
        return c

    @classmethod
    def from_openqasm_file(
        cls,
        file: str,
        circuit_params: Optional[Dict[str, Any]] = None,
        keep_measure_order: bool = False,
    ) -> "AbstractCircuit":
        from qiskit.circuit import QuantumCircuit

        if keep_measure_order is True:
            from .translation import qiskit_from_qasm_str_ordered_measure

            with open(file) as f:
                qasmstr = f.read()
            qiskit_circ = qiskit_from_qasm_str_ordered_measure(qasmstr)
        else:
            qiskit_circ = QuantumCircuit.from_qasm_file(file)
        c = cls.from_qiskit(qiskit_circ, circuit_params=circuit_params)
        return c

    def draw(self, **kws: Any) -> Any:
        """
        Visualise the circuit.
        This method recevies the keywords as same as qiskit.circuit.QuantumCircuit.draw.
        More details can be found here: https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.draw.html.
        Interesting kws options include: ``idle_wires``(bool)

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
        return self.to_qiskit(enable_instruction=True).draw(**kws)

    @classmethod
    def from_qiskit(
        cls,
        qc: Any,
        n: Optional[int] = None,
        inputs: Optional[List[float]] = None,
        circuit_params: Optional[Dict[str, Any]] = None,
        binding_params: Optional[Union[Sequence[float], Dict[Any, float]]] = None,
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
        :param circuit_params: kwargs given in Circuit.__init__ construction function, default to None.
        :type circuit_params: Optional[Dict[str, Any]]
        :param binding_params: (variational) parameters for the circuit.
            Could be either a sequence or dictionary depending on the type of parameters in the Qiskit circuit.
            For ``ParameterVectorElement`` use sequence. For ``Parameter`` use dictionary
        :type binding_params: Optional[Union[Sequence[float], Dict[Any, float]]]
        :return: The same circuit but as tensorcircuit object
        :rtype: Circuit
        """
        from .translation import qiskit2tc

        if n is None:
            n = qc.num_qubits

        return qiskit2tc(  # type: ignore
            qc.data,
            n,
            inputs,
            circuit_constructor=cls,
            circuit_params=circuit_params,
            binding_params=binding_params,
        )

    def vis_tex(self, **kws: Any) -> str:
        """
        Generate latex string based on quantikz latex package

        :return: Latex string that can be directly compiled via, e.g. latexit
        :rtype: str
        """
        if (not self.is_mps) and (self.inputs is not None):
            init = ["" for _ in range(self._nqubits)]
            init[self._nqubits // 2] = r"\psi"
            okws = {"init": init}
        else:
            okws = {"init": None}  # type: ignore
        okws.update(kws)
        return qir2tex(self._qir, self._nqubits, **okws)  # type: ignore

    tex = vis_tex

    def to_json(self, file: Optional[str] = None, simplified: bool = False) -> Any:
        """
        circuit dumps to json

        :param file: file str to dump the json to, defaults to None, return the json str
        :type file: Optional[str], optional
        :param simplified: If False, keep all info for each gate, defaults to be False.
            If True, suitable for IO since less information is required
        :type simplified: bool
        :return: None if dumps to file otherwise the json str
        :rtype: Any
        """
        from .translation import qir2json

        tcqasm = qir2json(self.to_qir(), simplified=simplified)
        if file is not None:
            with open(file, "w") as f:
                json.dump(tcqasm, f)
        return json.dumps(tcqasm)

    @classmethod
    def from_qsim_file(
        cls, file: str, circuit_params: Optional[Dict[str, Any]] = None
    ) -> "AbstractCircuit":
        with open(file, "r") as f:
            lines = f.readlines()
        if circuit_params is None:
            circuit_params = {}
        if "nqubits" not in circuit_params:
            circuit_params["nqubits"] = int(lines[0])

        c = cls(**circuit_params)
        c = cls._apply_qsim(c, lines)
        return c

    @staticmethod
    def _apply_qsim(c: "AbstractCircuit", qsim_str: List[str]) -> "AbstractCircuit":
        def _convert_ints_and_floats(x: str) -> Union[str, int, float]:
            try:
                return int(x)
            except ValueError:
                pass

            try:
                return float(x)
            except ValueError:
                pass

            return x.lower()

        qsim_gates = [
            tuple(map(_convert_ints_and_floats, line.strip().split(" ")))
            for line in qsim_str[1:]
            if line
        ]
        # https://github.com/quantumlib/qsim/blob/master/docs/input_format.md
        # https://github.com/jcmgray/quimb/blob/master/quimb/tensor/circuit.py#L241
        for gate in qsim_gates:
            if gate[1] == "h":
                getattr(c, "H")(gate[2])
            elif gate[1] == "x":
                getattr(c, "X")(gate[2])
            elif gate[1] == "y":
                getattr(c, "Y")(gate[2])
            elif gate[1] == "z":
                getattr(c, "Z")(gate[2])
            elif gate[1] == "s":
                getattr(c, "PHASE")(gate[2], theta=np.pi / 2)
            elif gate[1] == "t":
                getattr(c, "PHASE")(gate[2], theta=np.pi / 4)
            elif gate[1] == "x_1_2":
                getattr(c, "RX")(gate[2], theta=np.pi / 2)
            elif gate[1] == "y_1_2":
                getattr(c, "RY")(gate[2], theta=np.pi / 2)
            elif gate[1] == "z_1_2":
                getattr(c, "RZ")(gate[2], theta=np.pi / 2)
            elif gate[1] == "w_1_2":
                getattr(c, "U")(gate[2], theta=np.pi / 2, phi=-np.pi / 4, lbd=np.pi / 4)
            elif gate[1] == "hz_1_2":
                getattr(c, "WROOT")(gate[2])
            elif gate[1] == "cnot":
                getattr(c, "CNOT")(gate[2], gate[3])
            elif gate[1] == "cx":
                getattr(c, "CX")(gate[2], gate[3])
            elif gate[1] == "cy":
                getattr(c, "CY")(gate[2], gate[3])
            elif gate[1] == "cz":
                getattr(c, "CZ")(gate[2], gate[3])
            elif gate[1] == "is" or gate[1] == "iswap":
                getattr(c, "ISWAP")(gate[2], gate[3])
            elif gate[1] == "rx":
                getattr(c, "RX")(gate[2], theta=gate[3])
            elif gate[1] == "ry":
                getattr(c, "RY")(gate[2], theta=gate[3])
            elif gate[1] == "rz":
                getattr(c, "RZ")(gate[2], theta=gate[3])
            elif gate[1] == "fs" or gate[1] == "fsim":
                i, j, theta, phi = gate[2:]
                getattr(c, "ISWAP")(i, j, theta=-theta)  # type: ignore
                getattr(c, "CPHASE")(i, j, theta=-phi)  # type: ignore
            else:
                raise NotImplementedError
        return c

    @classmethod
    def from_json(
        cls, jsonstr: str, circuit_params: Optional[Dict[str, Any]] = None
    ) -> "AbstractCircuit":
        """
        load json str as a Circuit

        :param jsonstr: _description_
        :type jsonstr: str
        :param circuit_params: Extra circuit parameters in the format of ``__init__``,
            defaults to None
        :type circuit_params: Optional[Dict[str, Any]], optional
        :return: _description_
        :rtype: AbstractCircuit
        """
        from .translation import json2qir

        if isinstance(jsonstr, str):
            jsonstr = json.loads(jsonstr)
        qir = json2qir(jsonstr)  # type: ignore
        return cls.from_qir(qir, circuit_params)

    @classmethod
    def from_json_file(
        cls, file: str, circuit_params: Optional[Dict[str, Any]] = None
    ) -> "AbstractCircuit":
        """
        load json file and convert it to a circuit

        :param file: filename
        :type file: str
        :param circuit_params: _description_, defaults to None
        :type circuit_params: Optional[Dict[str, Any]], optional
        :return: _description_
        :rtype: AbstractCircuit
        """
        with open(file, "r") as f:
            jsonstr = json.load(f)
        return cls.from_json(jsonstr, circuit_params)

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
            [np.array([[1.0, 0], [0, 0]]), np.array([[0, 0], [0, 1]])],
            index,
            name="measure",
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

    def append(
        self, c: "AbstractCircuit", indices: Optional[List[int]] = None
    ) -> "AbstractCircuit":
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
        :param indices: the qubit indices to which ``c`` is appended on.
            Defaults to None, which means plain concatenation.
        :type indices: Optional[List[int]], optional
        :return: The composed circuit
        :rtype: BaseCircuit
        """
        qir1 = self.to_qir()
        qir2 = c.to_qir()
        if indices is not None:
            qir2_old = qir2
            qir2 = []
            for d in qir2_old:
                # avoid modifying the original circuit
                d = d.copy()
                d["index"] = [indices[i] for i in d["index"]]
                qir2.append(d)
        newc = type(self).from_qir(qir1 + qir2, self.circuit_param)
        self.__dict__.update(newc.__dict__)
        return self

    def copy(self) -> "AbstractCircuit":
        qir = self.to_qir()
        c = type(self).from_qir(qir, self.circuit_param)
        return c

    def expectation(
        self,
        *ops: Tuple[tn.Node, List[int]],
        reuse: bool = True,
        noise_conf: Optional[Any] = None,
        nmc: int = 1000,
        status: Optional[Tensor] = None,
        **kws: Any,
    ) -> Tensor:
        raise NotImplementedError

    def expectation_ps(
        self,
        x: Optional[Sequence[int]] = None,
        y: Optional[Sequence[int]] = None,
        z: Optional[Sequence[int]] = None,
        ps: Optional[Sequence[int]] = None,
        reuse: bool = True,
        noise_conf: Optional[Any] = None,
        nmc: int = 1000,
        status: Optional[Tensor] = None,
        **kws: Any,
    ) -> Tensor:
        """
        Shortcut for Pauli string expectation.
        x, y, z list are for X, Y, Z positions

        :Example:

        >>> c = tc.Circuit(2)
        >>> c.X(0)
        >>> c.H(1)
        >>> c.expectation_ps(x=[1], z=[0])
        array(-0.99999994+0.j, dtype=complex64)

        >>> c = tc.Circuit(2)
        >>> c.cnot(0, 1)
        >>> c.rx(0, theta=0.4)
        >>> c.rx(1, theta=0.8)
        >>> c.h(0)
        >>> c.h(1)
        >>> error1 = tc.channels.generaldepolarizingchannel(0.1, 1)
        >>> error2 = tc.channels.generaldepolarizingchannel(0.06, 2)
        >>> noise_conf = NoiseConf()
        >>> noise_conf.add_noise("rx", error1)
        >>> noise_conf.add_noise("cnot", [error2], [[0, 1]])
        >>> c.expectation_ps(x=[0], noise_conf=noise_conf, nmc=10000)
        (0.46274087-3.764033e-09j)

        :param x: sites to apply X gate, defaults to None
        :type x: Optional[Sequence[int]], optional
        :param y: sites to apply Y gate, defaults to None
        :type y: Optional[Sequence[int]], optional
        :param z: sites to apply Z gate, defaults to None
        :type z: Optional[Sequence[int]], optional
        :param ps: or one can apply a ps structures instead of ``x``, ``y``, ``z``,
            e.g. [0, 1, 3, 0, 2, 2] for X_1Z_2Y_4Y_5
            defaults to None, ``ps`` can overwrite ``x``, ``y`` and ``z``
        :type ps: Optional[Sequence[int]], optional
        :param reuse: whether to cache and reuse the wavefunction, defaults to True
        :type reuse: bool, optional
        :param noise_conf: Noise Configuration, defaults to None
        :type noise_conf: Optional[NoiseConf], optional
        :param nmc: repetition time for Monte Carlo sampling for noisfy calculation, defaults to 1000
        :type nmc: int, optional
        :param status: external randomness given by tensor uniformly from [0, 1], defaults to None,
            used for noisfy circuit sampling
        :type status: Optional[Tensor], optional
        :return: Expectation value
        :rtype: Tensor
        """
        obs = []
        if ps is not None:
            from .quantum import ps2xyz

            d = ps2xyz(ps)  # type: ignore
            x = d.get("x", None)
            y = d.get("y", None)
            z = d.get("z", None)

        if x is not None:
            for i in x:
                obs.append([gates.x(), [i]])  # type: ignore
        if y is not None:
            for i in y:
                obs.append([gates.y(), [i]])  # type: ignore
        if z is not None:
            for i in z:
                obs.append([gates.z(), [i]])  # type: ignore
        return self.expectation(
            *obs, reuse=reuse, noise_conf=noise_conf, nmc=nmc, status=status, **kws  # type: ignore
        )
