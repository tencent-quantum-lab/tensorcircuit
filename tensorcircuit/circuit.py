"""
Quantum circuit: the state simulator
"""
# pylint: disable=invalid-name

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from functools import reduce
from operator import add

import graphviz
import numpy as np
import tensornetwork as tn

from . import gates
from .cons import backend, contractor, dtypestr, rdtypestr, npdtype
from .quantum import QuVector, QuOperator, identity
from .simplify import _split_two_qubit_gate
from .vis import qir2tex

Gate = gates.Gate
Tensor = Any


class Circuit:
    """
    ``Circuit`` class.
    Simple usage demo below.

    .. code-block:: python

        c = tc.Circuit(3)
        c.H(1)
        c.CNOT(0, 1)
        c.RX(2, theta=tc.num_to_tensor(1.))
        c.expectation([tc.gates.z(), (2, )]) # 0.54

    """

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

    gate_alias_list = [
        ["cnot", "cx"],
        ["fredkin", "cswap"],
        ["toffoli", "ccnot"],
        ["any", "unitary"],
    ]

    def __init__(
        self,
        nqubits: int,
        inputs: Optional[Tensor] = None,
        mps_inputs: Optional[QuOperator] = None,
        split: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Circuit object based on state simulator.

        :param nqubits: The number of qubits in the circuit.
        :type nqubits: int
        :param inputs: If not None, the initial state of the circuit is taken as ``inputs``
            instead of :math:`\\vert 0\\rangle^n` qubits, defaults to None.
        :type inputs: Optional[Tensor], optional
        :param mps_inputs: (Nodes, dangling Edges) for a MPS like initial wavefunction.
        :type inputs: Optional[Tuple[Sequence[Gate], Sequence[Edge]]], optional
        :param split: dict if two qubit gate is ready for split, including parameters for at least one of
            ``max_singular_values`` and ``max_truncation_err``.
        :type split: Optional[Dict[str, Any]]
        """
        _prefix = "qb-"
        if inputs is not None:
            self.has_inputs = True
        else:
            self.has_inputs = False
        self.split = split
        if (inputs is None) and (mps_inputs is None):
            nodes = [
                tn.Node(
                    np.array(
                        [1.0, 0.0],
                        dtype=npdtype,
                    ),
                    name=_prefix + str(x + 1),
                )
                for x in range(nqubits)
            ]
            self._front = [n.get_edge(0) for n in nodes]
        elif inputs is not None:  # provide input function
            inputs = backend.convert_to_tensor(inputs)
            inputs = backend.cast(inputs, dtype=dtypestr)
            inputs = backend.reshape(inputs, [-1])
            N = inputs.shape[0]
            n = int(np.log(N) / np.log(2))
            assert n == nqubits or n == 2 * nqubits
            inputs = backend.reshape(inputs, [2 for _ in range(n)])
            inputs = Gate(inputs)
            nodes = [inputs]
            self._front = [inputs.get_edge(i) for i in range(n)]
        else:  # mps_inputs is not None
            mps_nodes = list(mps_inputs.nodes)  # type: ignore
            for i, n in enumerate(mps_nodes):
                mps_nodes[i].tensor = backend.cast(n.tensor, dtypestr)  # type: ignore
            mps_edges = mps_inputs.out_edges + mps_inputs.in_edges  # type: ignore
            ndict, edict = tn.copy(mps_nodes)
            new_nodes = []
            for n in mps_nodes:
                new_nodes.append(ndict[n])
            new_front = []
            for e in mps_edges:
                new_front.append(edict[e])
            nodes = new_nodes
            self._front = new_front

        self._nqubits = nqubits
        self._nodes = nodes
        self._start_index = len(nodes)
        # self._start = nodes
        # self._meta_apply()

        # self._qcode = ""  # deprecated
        # self._qcode += str(self._nqubits) + "\n"
        self._qir: List[Dict[str, Any]] = []

    def replace_inputs(self, inputs: Tensor) -> None:
        """
        Replace the input state with the circuit structure unchanged.

        :param inputs: Input wavefunction.
        :type inputs: Tensor
        """
        assert self.has_inputs is True
        inputs = backend.reshape(inputs, [-1])
        N = inputs.shape[0]
        n = int(np.log(N) / np.log(2))
        assert n == self._nqubits
        inputs = backend.reshape(inputs, [2 for _ in range(n)])
        self._nodes[0].tensor = inputs

    def replace_mps_inputs(self, mps_inputs: QuOperator) -> None:
        """
        Replace the input state in MPS representation while keep the circuit structure unchanged.

        :Example:
        >>> c = tc.Circuit(2)
        >>> c.X(0)
        >>>
        >>> c2 = tc.Circuit(2, mps_inputs=c.quvector())
        >>> c2.X(0)
        >>> c2.wavefunction()
        array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], dtype=complex64)
        >>>
        >>> c3 = tc.Circuit(2)
        >>> c3.X(0)
        >>> c3.replace_mps_inputs(c.quvector())
        >>> c3.wavefunction()
        array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], dtype=complex64)

        :param mps_inputs: (Nodes, dangling Edges) for a MPS like initial wavefunction.
        :type mps_inputs: Tuple[Sequence[Gate], Sequence[Edge]]
        """
        mps_nodes = mps_inputs.nodes
        mps_edges = mps_inputs.out_edges + mps_inputs.in_edges
        ndict, edict = tn.copy(mps_nodes)
        new_nodes = []
        for n in mps_nodes:
            new_nodes.append(ndict[n])
        new_front = []
        for e in mps_edges:
            new_front.append(edict[e])
        old = set(id(n) for n in self._nodes[: self._start_index])
        j = -1
        for n in self._nodes[: self._start_index]:
            for e in n:
                if e.is_dangling():
                    j += 1
                    self._front[j] = new_front[j]
                else:
                    if (id(e.node1) in old) and (id(e.node2) in old):
                        pass
                    else:
                        j += 1
                        if id(e.node2) == id(n):
                            other = (e.node1, e.axis1)
                        else:  # id(e.node1) == id(n):
                            other = (e.node2, e.axis2)
                        e.disconnect()
                        new_front[j] ^ other[0][other[1]]
        j += 1
        self._front += new_front[j:]
        self._nodes = new_nodes + self._nodes[self._start_index :]
        self._start_index = len(new_nodes)

    @classmethod
    def _meta_apply(cls) -> None:

        for g in cls.sgates:
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
            docs = """
            Apply **%s** gate on the circuit.

            :param index: Qubit number that the gate applies on.
            :type index: int.
            """ % (
                g.upper()
            )
            if g in ["rs"]:
                getattr(cls, g).__doc__ = docs
                getattr(cls, g.upper()).__doc__ = docs

            else:
                getattr(cls, g).__doc__ = doc
                getattr(cls, g.upper()).__doc__ = doc

        for g in cls.vgates:
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

        for g in cls.mpogates:
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

            :param index: Qubit number that the gate applies on.
            :type index: int.
            :param vars: Parameters for the gate.
            :type vars: float.
            """ % (
                g
            )
            getattr(cls, g).__doc__ = doc
            getattr(cls, g.upper()).__doc__ = doc

        for gate_alias in cls.gate_alias_list:
            present_gate = gate_alias[0]
            for alias_gate in gate_alias[1:]:
                setattr(cls, alias_gate, getattr(cls, present_gate))

    # @classmethod
    # def from_qcode(
    #     cls, qcode: str
    # ) -> "Circuit":  # forward reference, see https://github.com/python/mypy/issues/3661
    #     """
    #     [WIP], make circuit object from non universal simple assembly quantum language

    #     :param qcode:
    #     :type qcode: str
    #     :return: :py:class:`Circuit` object
    #     """
    #     # TODO(@refraction-ray): change to OpenQASM IO
    #     lines = [s for s in qcode.split("\n") if s.strip()]
    #     nqubits = int(lines[0])
    #     c = cls(nqubits)
    #     for l in lines[1:]:
    #         ls = [s for s in l.split(" ") if s.strip()]
    #         g = ls[0]
    #         index = []
    #         errloc = 0
    #         for i, s in enumerate(ls[1:]):
    #             try:
    #                 si = int(s)
    #                 index.append(si)
    #             except ValueError:
    #                 errloc = i + 1
    #                 break
    #         kwdict = {}
    #         if errloc > 0:
    #             for j, s in enumerate(ls[errloc::2]):
    #                 kwdict[s] = float(ls[2 * j + 1 + errloc])
    #         getattr(c, g)(*index, **kwdict)
    #     return c

    # def to_qcode(self) -> str:
    #     """
    #     [WIP]

    #     :return: qcode str of corresponding circuit
    #     :rtype: str
    #     """
    #     return self._qcode

    def apply_single_gate(self, gate: Gate, index: int) -> None:
        """
        Apply the gate to the bit with the given index.

        :Example:

        >>> gate = tc.gates.Gate(np.arange(4).reshape(2, 2).astype(np.complex64))
        >>> qc = tc.Circuit(2)
        >>> qc.apply_single_gate(gate, 0)
        >>> qc.wavefunction()
        array([0.+0.j, 0.+0.j, 2.+0.j, 0.+0.j], dtype=complex64)

        :param gate: The Gate applied on the bit.
        :type gate: Gate
        :param index: The index of the bit to apply the Gate.
        :type index: int
        """

        gate.get_edge(1) ^ self._front[index]  # pay attention on the rank index here
        self._front[index] = gate.get_edge(0)
        self._nodes.append(gate)

    def apply_double_gate(self, gate: Gate, index1: int, index2: int) -> None:
        """
        Apply the gate to two bits with given indexes.

        :Example:

        >>> gate = tc.gates.cr_gate()
        >>> qc = tc.Circuit(2)
        >>> qc.apply_double_gate(gate, 0, 1)
        >>> qc.wavefunction()
        array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], dtype=complex64)

        :param gate: The Gate applied on bits.
        :type gate: Gate
        :param index1: The index of the bit to apply the Gate.
        :type index1: int
        :param index2: The index of the bit to apply the Gate.
        :type index2: int
        """
        assert index1 != index2
        gate.get_edge(2) ^ self._front[index1]
        gate.get_edge(3) ^ self._front[index2]
        self._front[index1] = gate.get_edge(0)
        self._front[index2] = gate.get_edge(1)
        self._nodes.append(gate)

        # actually apply single and double gate never directly used in the Circuit class
        # and don't use, directly use general gate function as it is more diverse in feature

    def apply_general_gate(
        self,
        gate: Gate,
        *index: int,
        name: Optional[str] = None,
        split: Optional[Dict[str, Any]] = None,
        mpo: bool = False,
        ir_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        gate_dict = {
            "gate": gate,
            "index": index,
            "name": name,
            "split": split,
            "mpo": mpo,
        }
        if ir_dict is not None:
            ir_dict.update(gate_dict)
        else:
            ir_dict = gate_dict
        self._qir.append(ir_dict)
        assert len(index) == len(set(index))
        noe = len(index)
        applied = False
        split_conf = None
        if split is not None:
            split_conf = split
        elif self.split is not None:
            split_conf = self.split

        if not mpo:
            if (split_conf is not None) and noe == 2:
                results = _split_two_qubit_gate(gate, **split_conf)
                # max_err cannot be jax jitted
                if results is not None:
                    n1, n2, is_swap = results

                    if is_swap is False:
                        n1[1] ^ self._front[index[0]]
                        n2[2] ^ self._front[index[1]]
                        self._nodes.append(n1)
                        self._nodes.append(n2)
                        self._front[index[0]] = n1[0]
                        self._front[index[1]] = n2[1]
                    else:
                        n2[2] ^ self._front[index[0]]
                        n1[1] ^ self._front[index[1]]
                        self._nodes.append(n1)
                        self._nodes.append(n2)
                        self._front[index[0]] = n1[0]
                        self._front[index[1]] = n2[1]
                    applied = True

            if applied is False:
                for i, ind in enumerate(index):
                    gate.get_edge(i + noe) ^ self._front[ind]
                    self._front[ind] = gate.get_edge(i)
                self._nodes.append(gate)

        else:  # gate in MPO format
            gatec = gate.copy()
            self._nodes += gatec.nodes
            for i, ind in enumerate(index):
                gatec.in_edges[i] ^ self._front[ind]
                self._front[ind] = gatec.out_edges[i]

        self.state_tensor = None  # refresh the state cache
        # if name:
        #     # if no name is specified, then the corresponding op wont be recorded in qcode
        #     self._qcode += name + " "
        #     for i in index:
        #         self._qcode += str(i) + " "
        #     self._qcode = self._qcode[:-1] + "\n"

    apply = apply_general_gate

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
            self: "Circuit",
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
                gate, *index, name=localname, split=split, mpo=mpo, ir_dict=gate_dict
            )

        return apply

    @staticmethod
    def apply_general_variable_gate_delayed(
        gatef: Callable[..., Gate],
        name: Optional[str] = None,
        mpo: bool = False,
    ) -> Callable[..., None]:
        if name is None:
            name = getattr(gatef, "n")

        def apply(self: "Circuit", *index: int, **vars: Any) -> None:
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
                gate, *index, name=localname, split=split, mpo=mpo, ir_dict=gate_dict
            )  # type: ignore
            # self._qcode = self._qcode[:-1] + " "  # rip off the final "\n"
            # for k, v in vars.items():
            #     self._qcode += k + " " + str(v) + " "
            # self._qcode = self._qcode[:-1] + "\n"

        return apply

    def get_quvector(self) -> QuVector:
        """
        Get the representation of the output state in the form of ``QuVector``
        while maintaining the circuit uncomputed

        :return: ``QuVector`` representation of the output state from the circuit
        :rtype: QuVector
        """
        _, edges = self._copy()
        return QuVector(edges)

    quvector = get_quvector

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

    # TODO(@refraction-ray): add noise support in IR
    # TODO(@refraction-ray): IR for density matrix simulator?

    @staticmethod
    def _apply_qir(c: "Circuit", qir: List[Dict[str, Any]]) -> "Circuit":
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

    @classmethod
    def from_qir(
        cls, qir: List[Dict[str, Any]], circuit_params: Optional[Dict[str, Any]] = None
    ) -> "Circuit":
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

        c = cls(**circuit_params)
        c = cls._apply_qir(c, qir)
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

    def mid_measurement(self, index: int, keep: int = 0) -> Tensor:
        """
        Middle measurement in z-basis on the circuit, note the wavefunction output is not normalized
        with ``mid_measurement`` involved, one should normalize the state manually if needed.
        This is a post-selection method as keep is provided as a prior.

        :param index: The index of qubit that the Z direction postselection applied on.
        :type index: int
        :param keep: 0 for spin up, 1 for spin down, defaults to be 0.
        :type keep: int, optional
        """
        # normalization not guaranteed
        # assert keep in [0, 1]
        if keep < 0.5:
            gate = np.array(
                [
                    [1.0],
                    [0.0],
                ],
                dtype=npdtype,
            )
        else:
            gate = np.array(
                [
                    [0.0],
                    [1.0],
                ],
                dtype=npdtype,
            )

        mg1 = tn.Node(gate)
        mg2 = tn.Node(gate)
        mg1.get_edge(0) ^ self._front[index]
        mg1.get_edge(1) ^ mg2.get_edge(1)
        self._front[index] = mg2.get_edge(0)
        self._nodes.append(mg1)
        self._nodes.append(mg2)
        r = backend.convert_to_tensor(keep)
        r = backend.cast(r, "int32")
        return r

    mid_measure = mid_measurement
    post_select = mid_measurement
    post_selection = mid_measurement

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

        :param index: the qubit for the z-basis measurement
        :type index: int
        :return: 0 or 1 for z measurement on up and down freedom
        :rtype: Tensor
        """
        return self.general_kraus(
            [np.array([[1.0, 0], [0, 0]]), np.array([[0, 0], [0, 1]])], index  # type: ignore
        )

    cond_measure = cond_measurement

    def prepend(self, c: "Circuit") -> "Circuit":
        self.replace_mps_inputs(c.quvector())
        self._qir = c._qir + self._qir
        return self

    def append(self, c: "Circuit") -> "Circuit":
        c.replace_mps_inputs(self.quvector())
        c._qir = self._qir + c._qir
        self.__dict__ = c.__dict__
        return self

    def depolarizing2(
        self,
        index: int,
        *,
        px: float,
        py: float,
        pz: float,
        status: Optional[float] = None,
    ) -> float:
        if status is None:
            status = backend.implicit_randu()[0]
        g = backend.cond(
            status < px,
            lambda: gates.x().tensor,  # type: ignore
            lambda: backend.cond(
                status < px + py,
                lambda: gates.y().tensor,  # type: ignore
                lambda: backend.cond(
                    status < px + py + pz,
                    lambda: gates.z().tensor,  # type: ignore
                    lambda: gates.i().tensor,  # type: ignore
                ),
            ),
        )
        # after implementing this, I realized that plain if is enough here for jit
        # the failure for previous implementation is because we use self.X(i) inside ``if``,
        # which has list append and incur bug in tensorflow jit
        # in terms of jax jit, the only choice is jax.lax.cond, since ``if tensor``` paradigm
        # is not supported in jax jit at all. (``Concrete Tensor Error``)
        self.any(index, unitary=g)  # type: ignore
        return 0.0
        # roughly benchmark shows that performance of two depolarizing in terms of
        # building time and running time are similar

    def depolarizing(
        self,
        index: int,
        *,
        px: float,
        py: float,
        pz: float,
        status: Optional[float] = None,
    ) -> Tensor:
        """
        Apply depolarizing channel in a Monte Carlo way,
        i.e. for each call of this method, one of gates from
        X, Y, Z, I are applied on the circuit based on the probability
        indicated by ``px``, ``py``, ``pz``.

        :param index: The qubit that depolarizing channel is on
        :type index: int
        :param px: probability for X noise
        :type px: float
        :param py: probability for Y noise
        :type py: float
        :param pz: probability for Z noise
        :type pz: float
        :param status: random seed uniformly from 0 to 1, defaults to None (generated implicitly)
        :type status: Optional[float], optional
        :return: int Tensor, the element lookup: [0: x, 1: y, 2: z, 3: I]
        :rtype: Tensor
        """

        # px/y/z here not support differentiation for now
        # jit compatible for now
        # assert px + py + pz < 1 and px >= 0 and py >= 0 and pz >= 0

        def step_function(x: Tensor) -> Tensor:
            r = (
                backend.sign(x - px)
                + backend.sign(x - px - py)
                + backend.sign(x - px - py - pz)
            )
            r = backend.cast(r / 2 + 1.5, dtype="int32")
            # [0: x, 1: y, 2: z, 3: I]

            return r

        if status is None:
            status = backend.implicit_randu()[0]
        r = step_function(status)
        rv = backend.onehot(r, 4)
        rv = backend.cast(rv, dtype=dtypestr)
        g = (
            rv[0] * gates._x_matrix
            + rv[1] * gates._y_matrix
            + rv[2] * gates._z_matrix
            + rv[3] * gates._i_matrix
        )
        self.any(index, unitary=g)  # type: ignore
        return r

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

    def unitary_kraus2(
        self,
        kraus: Sequence[Gate],
        *index: int,
        prob: Optional[Sequence[float]] = None,
        status: Optional[float] = None,
    ) -> Tensor:
        # general impl from Monte Carlo trajectory depolarizing above
        # still jittable
        # speed is similar to ``unitary_kraus``
        def index2gate2(r: Tensor, kraus: Sequence[Tensor]) -> Tensor:
            # r is int type Tensor of shape []
            return backend.switch(r, [lambda _=k: _ for k in kraus])

        return self._unitary_kraus_template(
            kraus, *index, prob=prob, status=status, get_gate_from_index=index2gate2
        )

    def unitary_kraus(
        self,
        kraus: Sequence[Gate],
        *index: int,
        prob: Optional[Sequence[float]] = None,
        status: Optional[float] = None,
    ) -> Tensor:
        """
        Apply unitary gates in ``kraus`` randomly based on corresponding ``prob``.
        If ``prob`` is ``None``, this is reduced to kraus channel language.

        :param kraus: List of ``tc.gates.Gate`` or just Tensors
        :type kraus: Sequence[Gate]
        :param prob: prob list with the same size as ``kraus``, defaults to None
        :type prob: Optional[Sequence[float]], optional
        :param status: random seed between 0 to 1, defaults to None
        :type status: Optional[float], optional
        :return: shape [] int dtype tensor indicates which kraus gate is actually applied
        :rtype: Tensor
        """
        # general impl from Monte Carlo trajectory depolarizing above
        # still jittable

        def index2gate(r: Tensor, kraus: Sequence[Tensor]) -> Tensor:
            # r is int type Tensor of shape []
            l = len(kraus)
            r = backend.onehot(r, l)
            r = backend.cast(r, dtype=dtypestr)
            return reduce(add, [r[i] * kraus[i] for i in range(l)])

        return self._unitary_kraus_template(
            kraus, *index, prob=prob, status=status, get_gate_from_index=index2gate
        )

    def _unitary_kraus_template(
        self,
        kraus: Sequence[Gate],
        *index: int,
        prob: Optional[Sequence[float]] = None,
        status: Optional[float] = None,
        get_gate_from_index: Optional[
            Callable[[Tensor, Sequence[Tensor]], Tensor]
        ] = None,
    ) -> Tensor:  # DRY
        sites = len(index)
        kraus = [k.tensor if isinstance(k, tn.Node) else k for k in kraus]
        kraus = [gates.array_to_tensor(k) for k in kraus]
        if prob is None:
            prob = [
                backend.real(backend.trace(backend.adjoint(k) @ k) / k.shape[0])
                for k in kraus
            ]
            kraus = [
                k / backend.cast(backend.sqrt(p), dtypestr) for k, p in zip(kraus, prob)
            ]
        if not backend.is_tensor(prob):
            prob = backend.convert_to_tensor(prob)
        prob_cumsum = backend.cumsum(prob)
        l = int(prob.shape[0])  # type: ignore

        def step_function(x: Tensor) -> Tensor:
            r = backend.sum(
                backend.stack([backend.sign(x - prob_cumsum[i]) for i in range(l - 1)])
            )
            r = backend.cast(r / 2.0 + (l - 1) / 2.0, dtype="int32")
            # [0: kraus[0], 1: kraus[1]...]
            return r

        if status is None:
            status = backend.implicit_randu()[0]
        status = backend.real(status)
        prob_cumsum = backend.cast(prob_cumsum, dtype=status.dtype)  # type: ignore
        r = step_function(status)
        if get_gate_from_index is None:
            raise ValueError("no `get_gate_from_index` implementation is provided")
        g = get_gate_from_index(r, kraus)
        g = backend.reshape(g, [2 for _ in range(sites * 2)])
        self.any(*index, unitary=g)  # type: ignore
        return r

    def _general_kraus_tf(
        self,
        kraus: Sequence[Gate],
        *index: int,
        status: Optional[float] = None,
    ) -> float:
        # the graph building time is frustratingly slow, several minutes
        # though running time is in terms of ms
        sites = len(index)
        kraus_tensor = [k.tensor for k in kraus]
        kraus_tensor_f = [lambda _=k: _ for k in kraus_tensor]
        # must return tensor instead of ``tn.Node`` for switch`

        def calculate_kraus_p(i: Tensor) -> Tensor:
            # i: Tensor as int of shape []
            newnodes, newfront = self._copy()  # TODO(@refraction-ray): support reuse?
            # simply reuse=True is wrong, as the circuit is contracting at building
            # self._copy seems slower than self._copy_state, but anyway the building time is unacceptable
            lnewnodes, lnewfront = self._copy(conj=True)
            kraus_i = backend.switch(i, kraus_tensor_f)
            k = gates.Gate(kraus_i)
            kc = gates.Gate(backend.conj(kraus_i))
            # begin connect
            for ind, j in enumerate(index):
                newfront[j] ^ k[ind + sites]
                k[ind] ^ kc[ind]
                kc[ind + sites] ^ lnewfront[j]
            for j in range(self._nqubits):
                if j not in index:
                    newfront[j] ^ lnewfront[j]
            norm_square = contractor(newnodes + lnewnodes + [k, kc]).tensor
            return backend.real(norm_square)

        if status is None:
            status = backend.implicit_randu()[0]

        import tensorflow as tf  # tf only implementation

        weight = 1.0
        fallback_weight = 0.0
        fallback_weight_i = 0
        len_kraus = len(kraus)
        for i in tf.range(len_kraus):  # breaks backend agnostic
            # nested for and if, if tensor inner must come with for in tensor outer, s.t. autograph works
            weight = calculate_kraus_p(i)
            if weight > fallback_weight:
                fallback_weight_i = i
                fallback_weight = weight
            status -= weight
            if status < 0:
                # concern here, correctness not sure in tf jit, fail anyway in jax jit
                break
        # placing a Tensor-dependent break, continue or return inside a Python loop
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/common_errors.md

        if (
            status >= 0 or weight == 0
        ):  # the same concern, but this simple if is easy to convert to ``backend.cond``
            # Floating point error resulted in a malformed sample.
            # Fall back to the most likely case.
            # inspired from cirq implementation (Apcache 2).
            weight = fallback_weight
            i = fallback_weight_i
        kraus_i = backend.switch(i, kraus_tensor_f)
        newgate = kraus_i / backend.cast(backend.sqrt(weight), dtypestr)
        self.any(*index, unitary=newgate)  # type: ignore
        return 0.0

    def _general_kraus_2(
        self,
        kraus: Sequence[Gate],
        *index: int,
        status: Optional[float] = None,
    ) -> Tensor:
        # the graph building time is frustratingly slow, several minutes
        # though running time is in terms of ms
        # raw running time in terms of s
        # note jax gpu building time is fast, in the order of 10s.!!
        # the typical scenario we are talking: 10 qubits, 3 layers of entangle gates and 3 layers of noise
        # building for jax+GPU ~100s 12 qubit * 5 layers
        # 370s 14 qubit * 7 layers, 0.35s running on vT4
        # vmap, grad, vvag are all fine for this function
        # layerwise jit technique can greatly boost the staging time, see in /examples/mcnoise_boost.py
        sites = len(index)
        kraus_tensor = [k.tensor if isinstance(k, tn.Node) else k for k in kraus]
        kraus_tensor = [gates.array_to_tensor(k) for k in kraus_tensor]

        # tn with hole
        newnodes, newfront = self._copy()
        lnewnodes, lnewfront = self._copy(conj=True)
        des = [newfront[j] for j in index] + [lnewfront[j] for j in index]
        for j in range(self._nqubits):
            if j not in index:
                newfront[j] ^ lnewfront[j]
        ns = contractor(newnodes + lnewnodes, output_edge_order=des)
        ntensor = ns.tensor
        # ns, des

        def calculate_kraus_p(i: int) -> Tensor:
            # i: Tensor as int of shape []
            # kraus_i = backend.switch(i, kraus_tensor_f)
            kraus_i = kraus_tensor[i]
            dm = gates.Gate(ntensor)
            k = gates.Gate(kraus_i)
            kc = gates.Gate(backend.conj(kraus_i))
            # begin connect
            for ind in range(sites):
                dm[ind] ^ k[ind + sites]
                k[ind] ^ kc[ind]
                kc[ind + sites] ^ dm[ind + sites]
            norm_square = contractor([dm, k, kc]).tensor
            return backend.real(norm_square)

        prob = [calculate_kraus_p(i) for i in range(len(kraus))]
        new_kraus = [
            k / backend.cast(backend.sqrt(w), dtypestr)
            for w, k in zip(prob, kraus_tensor)
        ]

        return self.unitary_kraus2(new_kraus, *index, prob=prob, status=status)

    def general_kraus(
        self,
        kraus: Sequence[Gate],
        *index: int,
        status: Optional[float] = None,
    ) -> Tensor:
        """
        Monte Carlo trajectory simulation of general Kraus channel whose Kraus operators cannot be
        amplified to unitary operators. For unitary operators composed Kraus channel, :py:meth:`unitary_kraus`
        is much faster.

        This function is jittable in theory. But only jax+GPU combination is recommended for jit
        since the graph building time is too long for other backend options; though the running
        time of the function is very fast for every case.

        :param kraus: A list of ``tn.Node`` for Kraus operators.
        :type kraus: Sequence[Gate]
        :param index: The qubits index that Kraus channel is applied on.
        :type index: int
        :param status: Random tensor uniformly between 0 or 1, defaults to be None,
            when the random number will be generated automatically
        :type status: Optional[float], optional
        """
        return self._general_kraus_2(kraus, *index, status=status)

    apply_general_kraus = general_kraus

    def is_valid(self) -> bool:
        """
        [WIP], check whether the circuit is legal.

        :return: The bool indicating whether the circuit is legal
        :rtype: bool
        """
        try:
            assert len(self._front) == self._nqubits
            for n in self._nodes:
                for e in n.get_all_dangling():
                    assert e in self._front
            return True
        except AssertionError:
            return False

    def _copy(self, conj: bool = False) -> Tuple[List[tn.Node], List[tn.Edge]]:
        """
        Copy all nodes and dangling edges correspondingly.

        :param conj: Bool indicating whether the tensors for nodes should be conjugated.
        :type conj: bool
        :return: New copy of nodes and dangling edges for the circuit.
        :rtype: Tuple[List[tn.Node], List[tn.Edge]]
        """
        ndict, edict = tn.copy(self._nodes, conjugate=conj)
        newnodes = []
        for n in self._nodes:
            newnodes.append(ndict[n])
        newfront = []
        for e in self._front:
            newfront.append(edict[e])
        return newnodes, newfront

    def wavefunction(self, form: str = "default") -> tn.Node.tensor:
        """
        Compute the output wavefunction from the circuit.

        :param form: The str indicating the form of the output wavefunction.
            "default": [-1], "ket": [-1, 1], "bra": [1, -1]
        :type form: str, optional
        :return: Tensor with the corresponding shape.
        :rtype: Tensor
        """
        nodes, d_edges = self._copy()
        t = contractor(nodes, output_edge_order=d_edges)
        if form == "default":
            shape = [-1]
        elif form == "ket":
            shape = [-1, 1]
        elif form == "bra":  # no conj here
            shape = [1, -1]
        return backend.reshape(t.tensor, shape=shape)

    def _copy_state_tensor(
        self, conj: bool = False, reuse: bool = True
    ) -> Tuple[List[tn.Node], List[tn.Edge]]:
        if reuse:
            t = getattr(self, "state_tensor", None)
            if t is None:
                nodes, d_edges = self._copy()
                t = contractor(nodes, output_edge_order=d_edges)
                setattr(self, "state_tensor", t)
            ndict, edict = tn.copy([t], conjugate=conj)
            newnodes = []
            newnodes.append(ndict[t])
            newfront = []
            for e in t.edges:
                newfront.append(edict[e])
        else:
            return self._copy(conj)
        return newnodes, newfront

    state = wavefunction

    def get_quoperator(self) -> QuOperator:
        """
        Get the ``QuOperator`` MPO like representation of the circuit unitary without contraction.

        :return: ``QuOperator`` object for the circuit unitary (open indices for the input state)
        :rtype: QuOperator
        """
        mps = identity([2 for _ in range(self._nqubits)])
        c = Circuit(self._nqubits)
        ns, es = self._copy()
        c._nodes = ns
        c._front = es
        c.replace_mps_inputs(mps)
        return QuOperator(c._front[: self._nqubits], c._front[self._nqubits :])

    quoperator = get_quoperator

    def matrix(self) -> Tensor:
        """
        Get the unitary matrix for the circuit irrespective with the circuit input state.

        :return: The circuit unitary matrix
        :rtype: Tensor
        """
        mps = identity([2 for _ in range(self._nqubits)])
        c = Circuit(self._nqubits)
        ns, es = self._copy()
        c._nodes = ns
        c._front = es
        c.replace_mps_inputs(mps)
        return backend.reshapem(c.state())

    def amplitude(self, l: str) -> tn.Node.tensor:
        """
        Returns the amplitude of the circuit given the bitstring l.

        :Example:

        >>> c = tc.Circuit(2)
        >>> c.X(0)
        >>> c.amplitude("10")
        array(1.+0.j, dtype=complex64)
        >>> c.CNOT(0, 1)
        >>> c.amplitude("11")
        array(1.+0.j, dtype=complex64)

        :param l: The bitstring of 0 and 1s.
        :type l: string
        :return: The amplitude of the circuit.
        :rtype: tn.Node.tensor
        """
        assert len(l) == self._nqubits
        no, d_edges = self._copy()
        ms = []
        for i, s in enumerate(l):
            if s == "1":
                ms.append(
                    tn.Node(np.array([0, 1], dtype=npdtype), name=str(i) + "-measure")
                )
            elif s == "0":
                ms.append(
                    tn.Node(np.array([1, 0], dtype=npdtype), name=str(i) + "-measure")
                )
        for i, _ in enumerate(l):
            d_edges[i] ^ ms[i].get_edge(0)

        no.extend(ms)
        return contractor(no).tensor

    def measure_reference(
        self, *index: int, with_prob: bool = False
    ) -> Tuple[str, float]:
        """
        Take measurement on the given quantum lines by ``index``.

        :Example:

        >>> c = tc.Circuit(3)
        >>> c.H(0)
        >>> c.h(1)
        >>> c.toffoli(0, 1, 2)
        >>> c.measure(2)
        ('1', -1.0)
        >>> # Another possible output: ('0', -1.0)
        >>> c.measure(2, with_prob=True)
        ('1', (0.25000011920928955+0j))
        >>> # Another possible output: ('0', (0.7499998807907104+0j))

        :param index: Measure on which quantum line.
        :param with_prob: If true, theoretical probability is also returned.
        :return: The sample output and probability (optional) of the quantum line.
        :rtype: Tuple[str, float]
        """
        # not jit compatible due to random number generations!
        sample = ""
        p = 1.0
        for j in index:
            nodes1, edge1 = self._copy()
            nodes2, edge2 = self._copy(conj=True)
            for i, e in enumerate(edge1):
                if i != j:
                    e ^ edge2[i]
            for i in range(len(sample)):
                if sample[i] == "0":
                    m = np.array([1, 0], dtype=npdtype)
                else:
                    m = np.array([0, 1], dtype=npdtype)
                nodes1.append(tn.Node(m))
                nodes1[-1].get_edge(0) ^ edge1[index[i]]
                nodes2.append(tn.Node(m))
                nodes2[-1].get_edge(0) ^ edge2[index[i]]
            nodes1.extend(nodes2)
            rho = (
                1
                / p
                * contractor(nodes1, output_edge_order=[edge1[j], edge2[j]]).tensor
            )
            pu = rho[0, 0]
            r = backend.random_uniform([])
            r = backend.real(backend.cast(r, dtypestr))
            if r < backend.real(pu):
                sample += "0"
                p = p * pu
            else:
                sample += "1"
                p = p * (1 - pu)
        if with_prob:
            return sample, p
        else:
            return sample, -1.0

    def measure_jit(
        self, *index: int, with_prob: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Take measurement to the given quantum lines.
        This method is jittable is and about 100 times faster than unjit version!

        :param index: Measure on which quantum line.
        :type index: int
        :param with_prob: If true, theoretical probability is also returned.
        :type with_prob: bool, optional
        :return: The sample output and probability (optional) of the quantum line.
        :rtype: Tuple[Tensor, Tensor]
        """
        # finally jit compatible ! and much faster than unjit version ! (100x)
        sample: List[Tensor] = []
        p = 1.0
        p = backend.convert_to_tensor(p)
        p = backend.cast(p, dtype=rdtypestr)
        for k, j in enumerate(index):
            nodes1, edge1 = self._copy()
            nodes2, edge2 = self._copy(conj=True)
            for i, e in enumerate(edge1):
                if i != j:
                    e ^ edge2[i]
            for i in range(k):
                m = (1 - sample[i]) * gates.array_to_tensor(np.array([1, 0])) + sample[
                    i
                ] * gates.array_to_tensor(np.array([0, 1]))
                nodes1.append(Gate(m))
                nodes1[-1].get_edge(0) ^ edge1[index[i]]
                nodes2.append(tn.Node(m))
                nodes2[-1].get_edge(0) ^ edge2[index[i]]
            nodes1.extend(nodes2)
            rho = (
                1
                / backend.cast(p, dtypestr)
                * contractor(nodes1, output_edge_order=[edge1[j], edge2[j]]).tensor
            )
            pu = backend.real(rho[0, 0])
            r = backend.implicit_randu()[0]
            r = backend.real(backend.cast(r, dtypestr))
            sign = backend.sign(r - pu) / 2 + 0.5
            sign = backend.convert_to_tensor(sign)
            sign = backend.cast(sign, dtype=rdtypestr)
            sign_complex = backend.cast(sign, dtypestr)
            sample.append(sign_complex)
            p = p * (pu * (-1) ** sign + sign)

        sample = backend.stack(sample)
        sample = backend.real(sample)
        if with_prob:
            return sample, p
        else:
            return sample, -1.0

    measure = measure_jit

    def perfect_sampling(self) -> Tuple[str, float]:
        """
        Sampling bistrings from the circuit output based on quantum amplitudes.
        Reference: arXiv:1201.3974.

        :return: Sampled bit string and the corresponding theoretical probability.
        :rtype: Tuple[str, float]
        """
        return self.measure_jit(*[i for i in range(self._nqubits)], with_prob=True)

    sample = perfect_sampling

    # TODO(@refraction-ray): more _before function like state_before? and better API?

    def expectation_before(
        self, *ops: Tuple[tn.Node, List[int]], reuse: bool = True
    ) -> List[tn.Node]:
        """
        Return the list of nodes that consititues the expectation value just before the contraction.

        :param reuse: whether contract the output state firstly, defaults to True
        :type reuse: bool, optional
        :return: The tensor network for the expectation
        :rtype: List[tn.Node]
        """
        nodes1, edge1 = self._copy_state_tensor(reuse=reuse)
        nodes2, edge2 = self._copy_state_tensor(conj=True, reuse=reuse)
        nodes1.extend(nodes2)  # left right op order for plain contractor

        occupied = set()
        for op, index in ops:
            if not isinstance(op, tn.Node):
                # op is only a matrix
                op = backend.reshape2(op)
                op = backend.cast(op, dtype=dtypestr)
                op = gates.Gate(op)
            if isinstance(index, int):
                index = [index]
            noe = len(index)
            for j, e in enumerate(index):
                if e in occupied:
                    raise ValueError("Cannot measure two operators in one index")
                edge1[e] ^ op.get_edge(j)
                edge2[e] ^ op.get_edge(j + noe)
                occupied.add(e)
            nodes1.append(op)
        for j in range(self._nqubits):
            if j not in occupied:  # edge1[j].is_dangling invalid here!
                edge1[j] ^ edge2[j]
        return nodes1

    def expectation(
        self, *ops: Tuple[tn.Node, List[int]], reuse: bool = True
    ) -> Tensor:
        """
        Compute the expectation of corresponding operators.

        :Example:

        >>> c = tc.Circuit(2)
        >>> c.H(0)
        >>> c.expectation((tc.gates.z(), [0]))
        array(0.+0.j, dtype=complex64)

        :param ops: Operator and its position on the circuit,
            eg. ``(tc.gates.z(), [1, ]), (tc.gates.x(), [2, ])`` is for operator :math:`Z_1X_2`.
        :type ops: Tuple[tn.Node, List[int]]
        :param reuse: If True, then the wavefunction tensor is cached for further expectation evaluation,
            defaults to be true.
        :type reuse: bool, optional
        :raises ValueError: "Cannot measure two operators in one index"
        :return: Tensor with one element
        :rtype: Tensor
        """
        # if not reuse:
        #     nodes1, edge1 = self._copy()
        #     nodes2, edge2 = self._copy(conj=True)
        # else:  # reuse

        # self._nodes = nodes1
        nodes1 = self.expectation_before(*ops, reuse=reuse)
        return contractor(nodes1).tensor

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
        self, qc: Any, n: Optional[int] = None, inputs: Optional[List[float]] = None
    ) -> "Circuit":
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

        return qiskit2tc(qc.data, n, inputs)  # type: ignore

    def vis_tex(self, **kws: Any) -> str:
        """
        Generate latex string based on quantikz latex package

        :return: Latex string that can be directly compiled via, e.g. latexit
        :rtype: str
        """
        if self.has_inputs:
            init = ["" for _ in range(self._nqubits)]
            init[self._nqubits // 2] = "\psi"
            okws = {"init": init}
        else:
            okws = {"init": None}  # type: ignore
        okws.update(kws)
        return qir2tex(self._qir, self._nqubits, **okws)  # type: ignore

    tex = vis_tex


def _expectation_ps(
    c: Circuit,
    x: Optional[Sequence[int]] = None,
    y: Optional[Sequence[int]] = None,
    z: Optional[Sequence[int]] = None,
    reuse: bool = True,
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

    :param x: _description_, defaults to None
    :type x: Optional[Sequence[int]], optional
    :param y: _description_, defaults to None
    :type y: Optional[Sequence[int]], optional
    :param z: _description_, defaults to None
    :type z: Optional[Sequence[int]], optional
    :param reuse: whether to cache and reuse the wavefunction, defaults to True
    :type reuse: bool, optional
    :return: Expectation value
    :rtype: Tensor
    """
    obs = []
    if x is not None:
        for i in x:
            obs.append([gates.x(), [i]])  # type: ignore
    if y is not None:
        for i in y:
            obs.append([gates.y(), [i]])  # type: ignore
    if z is not None:
        for i in z:
            obs.append([gates.z(), [i]])  # type: ignore
    return c.expectation(*obs, reuse=reuse)  # type: ignore


Circuit._meta_apply()
Circuit.expectation_ps = _expectation_ps  # type: ignore


def to_graphviz(
    c: Circuit,
    graph: graphviz.Graph = None,
    include_all_names: bool = False,
    engine: str = "neato",
) -> graphviz.Graph:
    """
    Not an ideal visualization for quantum circuit, but reserve here as a general approach to show the tensornetwork
    [Deprecated, use ``Circuit.vis_tex`` or ``Circuit.draw`` instead]
    """
    # Modified from tensornetwork codebase
    nodes = c._nodes
    if graph is None:
        # pylint: disable=no-member
        graph = graphviz.Graph("G", engine=engine)
    for node in nodes:
        if not node.name.startswith("__") or include_all_names:
            label = node.name
        else:
            label = ""
        graph.node(str(id(node)), label=label)
    seen_edges = set()
    for node in nodes:
        for i, edge in enumerate(node.edges):
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            if not edge.name.startswith("__") or include_all_names:
                edge_label = edge.name + ": " + str(edge.dimension)
            else:
                edge_label = ""
            if edge.is_dangling():
                # We need to create an invisible node for the dangling edge
                # to connect to.
                graph.node(
                    "{}_{}".format(str(id(node)), i),
                    label="",
                    _attributes={"style": "invis"},
                )
                graph.edge(
                    "{}_{}".format(str(id(node)), i),
                    str(id(node)),
                    label=edge_label,
                )
            else:
                graph.edge(
                    str(id(edge.node1)),
                    str(id(edge.node2)),
                    label=edge_label,
                )
    return graph


def expectation(
    *ops: Tuple[tn.Node, List[int]],
    ket: Tensor,
    bra: Optional[Tensor] = None,
    conj: bool = True,
    normalization: bool = False,
) -> Tensor:
    """
    Compute :math:`\\langle bra\\vert ops \\vert ket\\rangle`.

    Example 1 (:math:`bra` is same as :math:`ket`)

    >>> c = tc.Circuit(3)
    >>> c.H(0)
    >>> c.ry(1, theta=tc.num_to_tensor(0.8 + 0.7j))
    >>> c.cnot(1, 2)
    >>> state = c.wavefunction() # the state of this circuit
    >>> x1z2 = [(tc.gates.x(), [0]), (tc.gates.z(), [1])] # input qubits
    >>>
    >>> # Expection of this circuit / <state|*x1z2|state>
    >>> c.expectation(*x1z2)
    array(0.69670665+0.j, dtype=complex64)
    >>> tc.expectation(*x1z2, ket=state)
    (0.6967066526412964+0j)
    >>>
    >>> # Normalize(expection of Circuit) / Normalize(<state|*x1z2|state>)
    >>> c.expectation(*x1z2) / tc.backend.norm(state) ** 2
    (0.5550700389340034+0j)
    >>> tc.expectation(*x1z2, ket=state, normalization=True)
    (0.55507004+0j)

    Example 2 (:math:`bra` is different from :math:`ket`)

    >>> c = tc.Circuit(2)
    >>> c.X(1)
    >>> s1 = c.state()
    >>> c2 = tc.Circuit(2)
    >>> c2.X(0)
    >>> s2 = c2.state()
    >>> c3 = tc.Circuit(2)
    >>> c3.H(1)
    >>> s3 = c3.state()
    >>> x1x2 = [(tc.gates.x(), [0]), (tc.gates.x(), [1])]
    >>>
    >>> tc.expectation(*x1x2, ket=s1, bra=s2)
    (1+0j)
    >>> tc.expectation(*x1x2, ket=s3, bra=s2)
    (0.7071067690849304+0j) # 1/sqrt(2)

    :param ket: :math:`ket`. The state in tensor or ``QuVector`` format
    :type ket: Tensor
    :param bra: :math:`bra`, defaults to None, which is the same as ``ket``.
    :type bra: Optional[Tensor], optional
    :param conj: :math:`bra` changes to the adjoint matrix of :math:`bra`, defaults to True.
    :type conj: bool, optional
    :param normalization: Normalize the :math:`ket` and :math:`bra`, defaults to False.
    :type normalization: bool, optional
    :raises ValueError: "Cannot measure two operators in one index"
    :return: The result of :math:`\\langle bra\\vert ops \\vert ket\\rangle`.
    :rtype: Tensor
    """
    if bra is None:
        bra = ket
    if isinstance(ket, QuOperator):
        if conj is True:
            bra = bra.adjoint()
        # TODO(@refraction-ray) omit normalization arg for now
        n = len(ket.out_edges)
        occupied = set()
        nodes = list(ket.nodes) + list(bra.nodes)
        # TODO(@refraction-ray): is the order guaranteed or affect some types of contractor?
        for op, index in ops:
            if not isinstance(op, tn.Node):
                # op is only a matrix
                op = backend.reshape2(op)
                op = gates.Gate(op)
            if isinstance(index, int):
                index = [index]
            noe = len(index)
            for j, e in enumerate(index):
                if e in occupied:
                    raise ValueError("Cannot measure two operators in one index")
                bra.in_edges[e] ^ op.get_edge(j)
                ket.out_edges[e] ^ op.get_edge(j + noe)
                occupied.add(e)
            nodes.append(op)
        for j in range(n):
            if j not in occupied:  # edge1[j].is_dangling invalid here!
                ket.out_edges[j] ^ bra.in_edges[j]
        # self._nodes = nodes1
        num = contractor(nodes).tensor
        return num

    else:
        # ket is the tensor
        if conj is True:
            bra = backend.conj(bra)
        ket = backend.reshape(ket, [-1])
        ket = backend.reshape2(ket)
        bra = backend.reshape2(bra)
        n = len(backend.shape_tuple(ket))
        ket = Gate(ket)
        bra = Gate(bra)
        occupied = set()
        nodes = [ket, bra]
        if normalization is True:
            normket = backend.norm(ket.tensor)
            normbra = backend.norm(bra.tensor)
        for op, index in ops:
            if not isinstance(op, tn.Node):
                # op is only a matrix
                op = backend.reshape2(op)
                op = gates.Gate(op)
            if isinstance(index, int):
                index = [index]
            noe = len(index)
            for j, e in enumerate(index):
                if e in occupied:
                    raise ValueError("Cannot measure two operators in one index")
                bra[e] ^ op.get_edge(j)
                ket[e] ^ op.get_edge(j + noe)
                occupied.add(e)
            nodes.append(op)
        for j in range(n):
            if j not in occupied:  # edge1[j].is_dangling invalid here!
                ket[j] ^ bra[j]
        # self._nodes = nodes1
        num = contractor(nodes).tensor
        if normalization is True:
            den = normket * normbra
        else:
            den = 1.0
        return num / den
