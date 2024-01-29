"""
Quantum circuit: the state simulator
"""

# pylint: disable=invalid-name

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from functools import reduce
from operator import add

import numpy as np
import tensornetwork as tn

from . import gates
from . import channels
from .cons import backend, contractor, dtypestr, npdtype
from .quantum import QuOperator, identity
from .simplify import _full_light_cone_cancel
from .basecircuit import BaseCircuit

Gate = gates.Gate
Tensor = Any


class Circuit(BaseCircuit):
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

    is_dm = False

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
        :param mps_inputs: QuVector for a MPS like initial wavefunction.
        :type mps_inputs: Optional[QuOperator]
        :param split: dict if two qubit gate is ready for split, including parameters for at least one of
            ``max_singular_values`` and ``max_truncation_err``.
        :type split: Optional[Dict[str, Any]]
        """
        self.inputs = inputs
        self.mps_inputs = mps_inputs
        self.split = split
        self._nqubits = nqubits

        self.circuit_param = {
            "nqubits": nqubits,
            "inputs": inputs,
            "mps_inputs": mps_inputs,
            "split": split,
        }
        if (inputs is None) and (mps_inputs is None):
            nodes = self.all_zero_nodes(nqubits)
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

        self.coloring_nodes(nodes)
        self._nodes = nodes

        self._start_index = len(nodes)
        # self._start = nodes
        # self._meta_apply()

        # self._qcode = ""  # deprecated
        # self._qcode += str(self._nqubits) + "\n"
        self._qir: List[Dict[str, Any]] = []
        self._extra_qir: List[Dict[str, Any]] = []

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
        self.coloring_nodes(new_nodes)
        self._nodes = new_nodes + self._nodes[self._start_index :]
        self._start_index = len(new_nodes)

    # TODO(@refraction-ray): add noise support in IR
    # TODO(@refraction-ray): unify mid measure to basecircuit

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
        # mg1.flag = "post-select"
        # mg1.is_dagger = False
        # mg1.id = id(mg1)
        # mg2.flag = "post-select"
        # mg2.is_dagger = False
        # mg2.id = id(mg2)
        self.coloring_nodes([mg1, mg2], flag="post-select")
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
                status < px + py,  # type: ignore
                lambda: gates.y().tensor,  # type: ignore
                lambda: backend.cond(
                    status < px + py + pz,  # type: ignore
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

    # overwritten now, deprecated
    def depolarizing_reference(
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
            rv[0] * gates.x().tensor  # type: ignore
            + rv[1] * gates.y().tensor  # type: ignore
            + rv[2] * gates.z().tensor  # type: ignore
            + rv[3] * gates.i().tensor  # type: ignore
        )
        self.any(index, unitary=g)  # type: ignore
        return r

    def unitary_kraus2(
        self,
        kraus: Sequence[Gate],
        *index: int,
        prob: Optional[Sequence[float]] = None,
        status: Optional[float] = None,
        name: Optional[str] = None,
    ) -> Tensor:
        # dont use, has issue conflicting with vmap, concurrent access lock emerged
        # potential issue raised from switch
        # general impl from Monte Carlo trajectory depolarizing above
        # still jittable
        # speed is similar to ``unitary_kraus``
        def index2gate2(r: Tensor, kraus: Sequence[Tensor]) -> Tensor:
            # r is int type Tensor of shape []
            return backend.switch(r, [lambda _=k: _ for k in kraus])  # type: ignore

        return self._unitary_kraus_template(
            kraus,
            *index,
            prob=prob,
            status=status,
            get_gate_from_index=index2gate2,
            name=name,
        )

    def unitary_kraus(
        self,
        kraus: Sequence[Gate],
        *index: int,
        prob: Optional[Sequence[float]] = None,
        status: Optional[float] = None,
        name: Optional[str] = None,
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
            kraus,
            *index,
            prob=prob,
            status=status,
            get_gate_from_index=index2gate,
            name=name,
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
        name: Optional[str] = None,
    ) -> Tensor:  # DRY
        sites = len(index)
        kraus = [k.tensor if isinstance(k, tn.Node) else k for k in kraus]
        kraus = [gates.array_to_tensor(k) for k in kraus]
        kraus = [backend.reshapem(k) for k in kraus]
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
            if l == 1:
                r = backend.convert_to_tensor(0.0)
            else:
                r = backend.sum(
                    backend.stack(
                        [backend.sign(x - prob_cumsum[i]) for i in range(l - 1)]
                    )
                )
            r = backend.cast(r / 2.0 + (l - 1) / 2.0, dtype="int32")
            # [0: kraus[0], 1: kraus[1]...]
            return r

        if status is None:
            status = backend.implicit_randu()[0]
        status = backend.convert_to_tensor(status)
        status = backend.real(status)
        prob_cumsum = backend.cast(prob_cumsum, dtype=status.dtype)  # type: ignore
        r = step_function(status)
        if get_gate_from_index is None:
            raise ValueError("no `get_gate_from_index` implementation is provided")
        g = get_gate_from_index(r, kraus)
        g = backend.reshape(g, [2 for _ in range(sites * 2)])
        self.any(*index, unitary=g, name=name)  # type: ignore
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
        with_prob: bool = False,
        name: Optional[str] = None,
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
        eps = 1e-10
        new_kraus = [
            k / backend.cast(backend.sqrt(w) + eps, dtypestr)
            for w, k in zip(prob, kraus_tensor)
        ]
        pick = self.unitary_kraus(
            new_kraus, *index, prob=prob, status=status, name=name
        )
        if with_prob is False:
            return pick
        else:
            return pick, prob

    def general_kraus(
        self,
        kraus: Sequence[Gate],
        *index: int,
        status: Optional[float] = None,
        with_prob: bool = False,
        name: Optional[str] = None,
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
        return self._general_kraus_2(
            kraus, *index, status=status, with_prob=with_prob, name=name
        )

    apply_general_kraus = general_kraus

    @staticmethod
    def apply_general_kraus_delayed(
        krausf: Callable[..., Sequence[Gate]], is_unitary: bool = False
    ) -> Callable[..., None]:
        def apply(
            self: "Circuit",
            *index: int,
            status: Optional[float] = None,
            name: Optional[str] = None,
            **vars: float,
        ) -> None:
            kraus = krausf(**vars)
            if not is_unitary:
                self.apply_general_kraus(kraus, *index, status=status, name=name)
            else:
                self.unitary_kraus(kraus, *index, status=status, name=name)

        return apply

    @classmethod
    def _meta_apply_channels(cls) -> None:
        for k in channels.channels:
            if k in ["depolarizing", "generaldepolarizing"]:
                is_unitary = True
            else:
                is_unitary = False
            setattr(
                cls,
                k,
                cls.apply_general_kraus_delayed(
                    getattr(channels, k + "channel"), is_unitary=is_unitary
                ),
            )
            doc = """
            Apply %s quantum channel on the circuit.
            See :py:meth:`tensorcircuit.channels.%schannel`

            :param index: Qubit number that the gate applies on.
            :type index: int.
            :param status: uniform external random number between 0 and 1
            :type status: Tensor
            :param vars: Parameters for the channel.
            :type vars: float.
            """ % (
                k,
                k,
            )
            getattr(cls, k).__doc__ = doc

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
    # both are not good names, but for backward compatibility

    get_circuit_as_quoperator = get_quoperator
    get_state_as_quvector = BaseCircuit.quvector

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

    # TODO(@refraction-ray): more _before function like state_before? and better API?

    def expectation(
        self,
        *ops: Tuple[tn.Node, List[int]],
        reuse: bool = True,
        enable_lightcone: bool = False,
        noise_conf: Optional[Any] = None,
        nmc: int = 1000,
        status: Optional[Tensor] = None,
        **kws: Any,
    ) -> Tensor:
        """
        Compute the expectation of corresponding operators.

        :Example:

        >>> c = tc.Circuit(2)
        >>> c.H(0)
        >>> c.expectation((tc.gates.z(), [0]))
        array(0.+0.j, dtype=complex64)

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
        >>> c.expectation((tc.gates.x(), [0]), noise_conf=noise_conf, nmc=10000)
        (0.46274087-3.764033e-09j)

        :param ops: Operator and its position on the circuit,
            eg. ``(tc.gates.z(), [1, ]), (tc.gates.x(), [2, ])`` is for operator :math:`Z_1X_2`.
        :type ops: Tuple[tn.Node, List[int]]
        :param reuse: If True, then the wavefunction tensor is cached for further expectation evaluation,
            defaults to be true.
        :type reuse: bool, optional
        :param enable_lightcone: whether enable light cone simplification, defaults to False
        :type enable_lightcone: bool, optional
        :param noise_conf: Noise Configuration, defaults to None
        :type noise_conf: Optional[NoiseConf], optional
        :param nmc: repetition time for Monte Carlo sampling for noisfy calculation, defaults to 1000
        :type nmc: int, optional
        :param status: external randomness given by tensor uniformly from [0, 1], defaults to None,
            used for noisfy circuit sampling
        :type status: Optional[Tensor], optional
        :raises ValueError: "Cannot measure two operators in one index"
        :return: Tensor with one element
        :rtype: Tensor
        """
        from .noisemodel import expectation_noisfy

        if noise_conf is None:
            # if not reuse:
            #     nodes1, edge1 = self._copy()
            #     nodes2, edge2 = self._copy(conj=True)
            # else:  # reuse

            # self._nodes = nodes1
            if enable_lightcone:
                reuse = False
            nodes1 = self.expectation_before(*ops, reuse=reuse)
            if enable_lightcone:
                nodes1 = _full_light_cone_cancel(nodes1)
            return contractor(nodes1).tensor
        else:
            return expectation_noisfy(
                self,
                *ops,
                noise_conf=noise_conf,
                nmc=nmc,
                status=status,
                **kws,
            )


Circuit._meta_apply()
Circuit._meta_apply_channels()


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
