"""
quantum circuit: state simulator
"""

from typing import Tuple, List, Callable, Optional, Any, Sequence
from functools import reduce
from operator import add

import graphviz
import numpy as np
import tensornetwork as tn

from . import gates
from .cons import backend, contractor, dtypestr, npdtype
from .quantum import QuVector, QuOperator

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
        c.rx(2, theta=tc.num_to_tensor(1.))
        c.expectation([tc.gates.z(), (2, )]) # 0.54

    """

    sgates = (
        ["i", "x", "y", "z", "h", "t", "s", "rs", "wroot"]
        + ["cnot", "cz", "swap", "cy"]
        + ["toffoli"]
    )
    vgates = ["r", "cr", "rx", "ry", "rz", "any", "exp", "exp1"]

    def __init__(
        self,
        nqubits: int,
        inputs: Optional[Tensor] = None,
        mps_inputs: Optional[QuOperator] = None,
    ) -> None:
        """
        Circuit object based on state simulator.

        :param nqubits: The number of qubits in the circuit.
        :type nqubits: int
        :param inputs: If not None, the initial state of the circuit is taken as ``inputs``
            instead of :math:`\\vert 0\\rangle^n` qubits, defaults to None
        :type inputs: Optional[Tensor], optional
        :param mps_inputs: (Nodes, dangling Edges) for a MPS like initial wavefunction
        :type inputs: Optional[Tuple[Sequence[Gate], Sequence[Edge]]], optional
        """
        _prefix = "qb-"
        if inputs is not None:
            self.has_inputs = True
        else:
            self.has_inputs = False
        # Get nodes on the interior
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
            inputs = backend.reshape(inputs, [-1])
            N = inputs.shape[0]
            n = int(np.log(N) / np.log(2))
            assert n == nqubits or n == 2 * nqubits
            inputs = backend.reshape(inputs, [2 for _ in range(n)])
            inputs = Gate(inputs)
            nodes = [inputs]
            self._front = [inputs.get_edge(i) for i in range(n)]
        else:  # mps_inputs is not None
            mps_nodes = mps_inputs.nodes  # type: ignore
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
        self._qcode = ""
        self._qcode += str(self._nqubits) + "\n"

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

        :param mps_inputs: (Nodes, dangling Edges) for a MPS like initial wavefunction
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
        self._nodes = new_nodes + self._nodes[self._start_index :]

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
            Apply %s gate on the circuit.

            :param index: Qubit number than the gate applies on.
                The matrix for the gate is
                
                .. math::

                      %s
                
            :type index: int.
            """ % (
                g,
                matrix,
            )
            docs = """
            Apply %s gate on the circuit.

            :param index: Qubit number than the gate applies on.
            :type index: int.
            """ % (
                g
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
            Apply %s gate with parameters on the circuit.

            :param index: Qubit number than the gate applies on.
            :type index: int.
            :param vars: Parameters for the gate
            :type vars: float.
            """ % (
                g
            )
            getattr(cls, g).__doc__ = doc
            getattr(cls, g.upper()).__doc__ = doc

    @classmethod
    def from_qcode(
        cls, qcode: str
    ) -> "Circuit":  # forward reference, see https://github.com/python/mypy/issues/3661
        """
        [WIP], make circuit object from non universal simple assembly quantum language

        :param qcode:
        :type qcode: str
        :return: :py:class:`Circuit` object
        """
        # TODO(@refraction-ray): change to OpenQASM IO
        lines = [s for s in qcode.split("\n") if s.strip()]
        nqubits = int(lines[0])
        c = cls(nqubits)
        for l in lines[1:]:
            ls = [s for s in l.split(" ") if s.strip()]
            g = ls[0]
            index = []
            errloc = 0
            for i, s in enumerate(ls[1:]):
                try:
                    si = int(s)
                    index.append(si)
                except ValueError:
                    errloc = i + 1
                    break
            kwdict = {}
            if errloc > 0:
                for j, s in enumerate(ls[errloc::2]):
                    kwdict[s] = float(ls[2 * j + 1 + errloc])
            getattr(c, g)(*index, **kwdict)
        return c

    def to_qcode(self) -> str:
        """
        [WIP]

        :return: qcode str of corresponding circuit
        :rtype: str
        """
        return self._qcode

    def apply_single_gate(self, gate: Gate, index: int) -> None:
        gate.get_edge(1) ^ self._front[index]  # pay attention on the rank index here
        self._front[index] = gate.get_edge(0)
        self._nodes.append(gate)

    def apply_double_gate(self, gate: Gate, index1: int, index2: int) -> None:
        assert index1 != index2
        gate.get_edge(2) ^ self._front[index1]
        gate.get_edge(3) ^ self._front[index2]
        self._front[index1] = gate.get_edge(0)
        self._front[index2] = gate.get_edge(1)
        self._nodes.append(gate)

    def apply_general_gate(
        self, gate: Gate, *index: int, name: Optional[str] = None
    ) -> None:
        assert len(index) == len(set(index))
        noe = len(index)
        for i, ind in enumerate(index):
            gate.get_edge(i + noe) ^ self._front[ind]
            self._front[ind] = gate.get_edge(i)
        self._nodes.append(gate)
        if (
            name
        ):  # if no name is specified, then the corresponding op wont be recorded in qcode
            self._qcode += name + " "
            for i in index:
                self._qcode += str(i) + " "
            self._qcode = self._qcode[:-1] + "\n"

    apply = apply_general_gate

    @staticmethod
    def apply_general_gate_delayed(
        gatef: Callable[[], Gate], name: Optional[str] = None
    ) -> Callable[..., None]:
        # nested function must be utilized, functools.partial doesn't work for method register on class
        # see https://re-ra.xyz/Python-中实例方法动态绑定的几组最小对立/
        def apply(self: "Circuit", *index: int) -> None:
            gate = gatef()
            self.apply_general_gate(gate, *index, name=name)

        return apply

    @staticmethod
    def apply_general_variable_gate_delayed(
        gatef: Callable[..., Gate],
        name: Optional[str] = None,
    ) -> Callable[..., None]:
        def apply(self: "Circuit", *index: int, **vars: float) -> None:
            gate = gatef(**vars)
            self.apply_general_gate(gate, *index, name=name)
            self._qcode = self._qcode[:-1] + " "  # rip off the final "\n"
            for k, v in vars.items():
                self._qcode += k + " " + str(v) + " "
            self._qcode = self._qcode[:-1] + "\n"

        return apply

    def get_quvector(self) -> QuVector:
        _, edges = self._copy()
        return QuVector(edges)

    quvector = get_quvector

    def mid_measurement(self, index: int, keep: int = 0) -> None:
        """
        middle measurement in z basis on the circuit, note the wavefunction output is not normalized
        with ``mid_measurement`` involved, one should normalized the state manually if needed.

        :param index: the index of qubit that the Z direction postselection applied on
        :type index: int
        :param keep: 0 for spin up, 1 for spin down, defaults to 0
        :type keep: int, optional
        """
        # normalization not guaranteed
        assert keep in [0, 1]
        if keep == 0:
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
    ) -> float:
        # px/y/z here not support differentiation for now
        # jit compatible for now
        assert px + py + pz < 1 and px >= 0 and py >= 0 and pz >= 0

        def step_function(x: Tensor) -> Tensor:
            r = (
                backend.sign(x - px)
                + backend.sign(x - px - py)
                + backend.sign(x - px - py - pz)
            )
            r = backend.cast(r / 2 + 1.5, dtype="int32")
            # [0: x, 1: y, 2: z, 3: I]
            r = backend.onehot(r, 4)
            r = backend.cast(r, dtype=dtypestr)
            return r

        if status is None:
            status = backend.implicit_randu()[0]
        r = step_function(status)
        g = (
            r[0] * gates._x_matrix
            + r[1] * gates._y_matrix
            + r[2] * gates._z_matrix
            + r[3] * gates._i_matrix
        )
        self.any(index, unitary=g)  # type: ignore
        return 0.0

    def unitary_kraus2(
        self,
        kraus: Sequence[Gate],
        *index: int,
        prob: Optional[Sequence[float]] = None,
        status: Optional[float] = None,
    ) -> float:
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
    ) -> float:
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
    ) -> float:  # DRY
        sites = len(index)
        if isinstance(kraus[0], tn.Node):
            kraus = [k.tensor for k in kraus]
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
        r = step_function(status)
        if get_gate_from_index is None:
            raise ValueError("no `get_gate_from_index` implementation is provided")
        g = get_gate_from_index(r, kraus)
        g = backend.reshape(g, [2 for _ in range(sites * 2)])
        self.any(*index, unitary=g)  # type: ignore
        return 0.0

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
            newnodes, newfront = self._copy()  # TODO(@refraction-ray): support reuse
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
            # nested for and if, if tensor inner must come with for in tensor outter, s.t. autograph works
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
    ) -> float:
        # the graph building time is frustratingly slow, several minutes
        # though running time is in terms of ms
        # raw running time in terms of s
        # note jax gpu building time is fast, in the order of 10s.!!
        # the typical scenario we are talking: 10 qubits, 3 layers of entangle gates and 3 layers of noise
        # building for jax+GPU ~100s 12 qubit * 5 layers
        # 370s 14 qubit * 7 layers, 0.35s running on vT4
        sites = len(index)
        kraus_tensor = [k.tensor for k in kraus]

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

        self.unitary_kraus2(new_kraus, *index, prob=prob, status=status)
        return 0.0

    def general_kraus(
        self,
        kraus: Sequence[Gate],
        *index: int,
        status: Optional[float] = None,
    ) -> float:
        """
        Monte Carlo trajectory simulation of general Kraus channel whose Kraus operators cannot be
        amplified to unitary operators. For unitary operators composed Kraus channel, :py:meth:`unitary_kraus`
        is much faster.

        This function is jittable in theory. But only jax+GPU combination is recommended for jit,
        since the graph building time is too long for other backend options, though the running
        time of the function is very fast for every case.

        :param kraus: list of ``tn.Node`` for Kraus operators
        :type kraus: Sequence[Gate]
        :param index: the qubits index that Kraus channel is applied on
        :type index: int
        :param status: random tensor between 0 or 1, defaults to None,
            the random number will generated automatically
        :type status: Optional[float], optional
        """
        return self._general_kraus_2(kraus, *index, status=status)

    apply_general_kraus = general_kraus

    def is_valid(self) -> bool:
        """
        [WIP], check whether the circuit is legal

        :return:
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
        copy all nodes and dangling edges correspondingly

        :param conj: bool indicating whether the tensors for nodes should be conjugated
        :type conj: bool
        :return: new copy of nodes and dangling edges for the circuit
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

    def wavefunction(self) -> tn.Node.tensor:
        """
        compute the output wavefunction from the circuit

        :return: Tensor with shape [-1, 1]
        :rtype: Tensor
        """
        nodes, d_edges = self._copy()
        t = contractor(nodes, output_edge_order=d_edges)
        return backend.reshape(t.tensor, shape=[1, -1])

    def _copy_state_tensor(
        self, conj: bool = False, reuse: bool = True
    ) -> Tuple[List[tn.Node], List[tn.Edge]]:
        if reuse:
            t = getattr(self, "state_tensor", None)
        else:
            t = None
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
        return newnodes, newfront

    state = wavefunction

    def amplitude(self, l: str) -> tn.Node.tensor:
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

    def measure(self, *index: int, with_prob: bool = False) -> Tuple[str, float]:
        """
        :param index: measure on which quantum line
        :param with_prob: if true, theoretical probability is also returned
        :return:
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

        :param index: measure on which quantum line
        :param with_prob: if true, theoretical probability is also returned
        :return:
        """
        # finally jit compatible ! and much faster than unjit version ! (100x)
        sample: List[Tensor] = []
        p = 1.0
        p = backend.convert_to_tensor(p)
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
            sign_complex = backend.cast(sign, dtypestr)
            sample.append(sign_complex)
            p = p * (pu * (-1) ** sign + sign)

        sample = backend.stack(sample)
        sample = backend.real(sample)
        if with_prob:
            return sample, p
        else:
            return sample, -1.0

    def perfect_sampling(self) -> Tuple[str, float]:
        """
        reference: arXiv:1201.3974.

        :return: sampled bit string and the corresponding theoretical probability
        """
        return self.measure(*[i for i in range(self._nqubits)], with_prob=True)

    def expectation(
        self, *ops: Tuple[tn.Node, List[int]], reuse: bool = True
    ) -> tn.Node.tensor:
        """
        compute expectation of corresponding operators

        :param ops: operator and its position on the circuit,
            eg. ``(gates.Z(), [1]), (gates.X(), [2])`` is for operator :math:`Z_1X_2`
        :type ops: Tuple[tn.Node, List[int]]
        :param reuse: if True, then the wavefunction tensor is cached for further expectation evaluation,
            defaults to True
        :type reuse: bool, optional
        :raises ValueError: [description]
        :return: Tensor with one element
        :rtype: Tensor
        """
        # if not reuse:
        #     nodes1, edge1 = self._copy()
        #     nodes2, edge2 = self._copy(conj=True)
        # else:  # reuse
        nodes1, edge1 = self._copy_state_tensor(reuse=reuse)
        nodes2, edge2 = self._copy_state_tensor(conj=True, reuse=reuse)
        occupied = set()
        for op, index in ops:
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
        nodes1.extend(nodes2)
        # self._nodes = nodes1
        return contractor(nodes1).tensor

    def to_graphviz(
        self,
        graph: graphviz.Graph = None,
        include_all_names: bool = False,
        engine: str = "neato",
    ) -> graphviz.Graph:
        """
        Not an ideal visualization for quantum circuit, but reserve here as a general approch to show tensornetwork

        :param graph:
        :param include_all_names:
        :param engine:
        :return:
        """
        # Modified from tensornetwork codebase
        nodes = self._nodes
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


Circuit._meta_apply()


def expectation(
    *ops: Tuple[tn.Node, List[int]],
    ket: Tensor,
    bra: Optional[Tensor] = None,
    conj: bool = True,
    normalization: bool = False,
) -> Tensor:
    """
    [deprecated] direct manipulate on  ``QuOperator`` is suggested
    compute :math:`\\langle bra\\vert ops \\vert ket\\rangle`

    :param ket: [description]
    :type ket: Tensor
    :param bra: [description], defaults to None, which is the same as ``ket``
    :type bra: Optional[Tensor], optional
    :param conj: [description], defaults to True
    :type conj: bool, optional
    :param normalization: [description], defaults to False
    :type normalization: bool, optional
    :raises ValueError: [description]
    :return: [description]
    :rtype: Tensor
    """
    if bra is None:
        bra = ket
    if conj is True:
        bra = backend.conj(bra)
    ket = backend.reshape(ket, [-1])
    N = ket.shape[0]
    n = int(np.log(N) / np.log(2))
    ket = backend.reshape(ket, [2 for _ in range(n)])
    bra = backend.reshape(bra, [2 for _ in range(n)])
    ket = Gate(ket)
    bra = Gate(bra)
    occupied = set()
    nodes = [ket, bra]
    if normalization is True:
        normket = backend.norm(ket.tensor)
        normbra = backend.norm(bra.tensor)
    for op, index in ops:
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
