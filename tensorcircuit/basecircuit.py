"""
Quantum circuit: common methods for all circuit classes as MixIn
"""
# pylint: disable=invalid-name

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from functools import partial

import numpy as np
import graphviz
import tensornetwork as tn

from . import gates
from .quantum import (
    QuOperator,
    QuVector,
    correlation_from_samples,
    correlation_from_counts,
    measurement_counts,
    sample_int2bin,
    sample_bin2int,
    sample2all,
)
from .abstractcircuit import AbstractCircuit
from .cons import npdtype, backend, dtypestr, contractor, rdtypestr
from .simplify import _split_two_qubit_gate
from .utils import arg_alias

Gate = gates.Gate
Tensor = Any


class BaseCircuit(AbstractCircuit):
    _nodes: List[tn.Node]
    _front: List[tn.Edge]
    is_dm: bool
    split: Optional[Dict[str, Any]]

    is_mps = False

    @staticmethod
    def all_zero_nodes(n: int, d: int = 2, prefix: str = "qb-") -> List[tn.Node]:
        l = [0.0 for _ in range(d)]
        l[0] = 1.0
        nodes = [
            tn.Node(
                np.array(
                    l,
                    dtype=npdtype,
                ),
                name=prefix + str(x),
            )
            for x in range(n)
        ]
        return nodes

    @staticmethod
    def front_from_nodes(nodes: List[tn.Node]) -> List[tn.Edge]:
        return [n.get_edge(0) for n in nodes]

    @staticmethod
    def coloring_nodes(
        nodes: Sequence[tn.Node], is_dagger: bool = False, flag: str = "inputs"
    ) -> None:
        for node in nodes:
            node.is_dagger = is_dagger
            node.flag = flag
            node.id = id(node)

    @staticmethod
    def coloring_copied_nodes(
        nodes: Sequence[tn.Node],
        nodes0: Sequence[tn.Node],
        is_dagger: bool = True,
        flag: str = "inputs",
    ) -> None:
        for node, n0 in zip(nodes, nodes0):
            node.is_dagger = is_dagger
            node.flag = flag
            node.id = getattr(n0, "id", id(n0))

    @staticmethod
    def copy(
        nodes: Sequence[tn.Node],
        dangling: Optional[Sequence[tn.Edge]] = None,
        conj: Optional[bool] = False,
    ) -> Tuple[List[tn.Node], List[tn.Edge]]:
        """
        copy all nodes and dangling edges correspondingly

        :return:
        """
        ndict, edict = tn.copy(nodes, conjugate=conj)
        newnodes = []
        for n in nodes:
            newn = ndict[n]
            newn.is_dagger = conj
            newn.flag = getattr(n, "flag", "") + "copy"
            newn.id = getattr(n, "id", id(n))
            newnodes.append(newn)
        newfront = []
        if not dangling:
            dangling = []
            for n in nodes:
                dangling.extend([e for e in n])
        for e in dangling:
            newfront.append(edict[e])
        return newnodes, newfront

    def _copy(
        self, conj: Optional[bool] = False
    ) -> Tuple[List[tn.Node], List[tn.Edge]]:
        return self.copy(self._nodes, self._front, conj)

    def apply_general_gate(
        self,
        gate: Union[Gate, QuOperator],
        *index: int,
        name: Optional[str] = None,
        split: Optional[Dict[str, Any]] = None,
        mpo: bool = False,
        ir_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        if name is None:
            name = ""
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
        index = tuple([i if i >= 0 else self._nqubits + i for i in index])  # type: ignore
        noe = len(index)
        nq = self._nqubits
        applied = False
        split_conf = None
        if split is not None:
            split_conf = split
        elif self.split is not None:
            split_conf = self.split

        if not mpo:
            assert isinstance(gate, tn.Node)
            if (split_conf is not None) and noe == 2:
                results = _split_two_qubit_gate(gate, **split_conf)
                # max_err cannot be jax jitted
                if results is not None:
                    n1, n2, is_swap = results
                    self.coloring_nodes([n1, n2], flag="gate")
                    # n1.flag = "gate"
                    # n1.is_dagger = False
                    n1.name = name
                    # n1.id = id(n1)
                    # n2.flag = "gate"
                    # n2.is_dagger = False
                    # n2.id = id(n2)
                    n2.name = name
                    if is_swap is False:
                        n1[1] ^ self._front[index[0]]
                        n2[2] ^ self._front[index[1]]
                        self._nodes.append(n1)
                        self._nodes.append(n2)
                        self._front[index[0]] = n1[0]
                        self._front[index[1]] = n2[1]
                        if self.is_dm:
                            [n1l, n2l], _ = self.copy([n1, n2], conj=True)
                            n1l[1] ^ self._front[index[0] + nq]
                            n2l[2] ^ self._front[index[1] + nq]
                            self._nodes.append(n1l)
                            self._nodes.append(n2l)
                            self._front[index[0] + nq] = n1l[0]
                            self._front[index[1] + nq] = n2l[1]
                    else:
                        n2[2] ^ self._front[index[0]]
                        n1[1] ^ self._front[index[1]]
                        self._nodes.append(n1)
                        self._nodes.append(n2)
                        self._front[index[0]] = n1[0]
                        self._front[index[1]] = n2[1]
                        if self.is_dm:
                            [n1l, n2l], _ = self.copy([n1, n2], conj=True)
                            n2l[1] ^ self._front[index[0] + nq]
                            n1l[2] ^ self._front[index[1] + nq]
                            self._nodes.append(n1l)
                            self._nodes.append(n2l)
                            self._front[index[0] + nq] = n1l[0]
                            self._front[index[1] + nq] = n2l[1]
                    applied = True

            if applied is False:
                gate.name = name
                self.coloring_nodes([gate])
                # gate.flag = "gate"
                # gate.is_dagger = False
                # gate.id = id(gate)
                self._nodes.append(gate)
                if self.is_dm:
                    lgates, _ = self.copy([gate], conj=True)
                    lgate = lgates[0]
                    self._nodes.append(lgate)
                for i, ind in enumerate(index):
                    gate.get_edge(i + noe) ^ self._front[ind]
                    self._front[ind] = gate.get_edge(i)
                    if self.is_dm:
                        lgate.get_edge(i + noe) ^ self._front[ind + nq]
                        self._front[ind + nq] = lgate.get_edge(i)

        else:  # gate in MPO format
            assert isinstance(gate, QuOperator)
            gatec = gate.copy()
            for n in gatec.nodes:
                n.flag = "gate"
                n.is_dagger = False
                n.id = id(gate)
                n.name = name
            self._nodes += gatec.nodes
            if self.is_dm:
                gateconj = gate.adjoint()
                for n0, n in zip(gatec.nodes, gateconj.nodes):
                    n.flag = "gate"
                    n.is_dagger = True
                    n.id = id(n0)
                    n.name = name
                self._nodes += gateconj.nodes

            for i, ind in enumerate(index):
                gatec.in_edges[i] ^ self._front[ind]
                self._front[ind] = gatec.out_edges[i]
                if self.is_dm:
                    gateconj.out_edges[i] ^ self._front[ind + nq]
                    self._front[ind + nq] = gateconj.in_edges[i]

        self.state_tensor = None  # refresh the state cache

    apply = apply_general_gate

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
            return newnodes, newfront
        return self._copy(conj)  # type: ignore

    def expectation_before(
        self,
        *ops: Tuple[tn.Node, List[int]],
        reuse: bool = True,
        **kws: Any,
    ) -> List[tn.Node]:
        """
        Get the tensor network in the form of a list of nodes
        for the expectation calculation before the real contraction

        :param reuse: _description_, defaults to True
        :type reuse: bool, optional
        :raises ValueError: _description_
        :return: _description_
        :rtype: List[tn.Node]
        """
        nq = self._nqubits
        if self.is_dm is True:
            nodes, newdang = self._copy_state_tensor(reuse=reuse)
        else:
            nodes1, edge1 = self._copy_state_tensor(reuse=reuse)
            nodes2, edge2 = self._copy_state_tensor(conj=True, reuse=reuse)
            nodes = nodes1 + nodes2
            newdang = edge1 + edge2
        occupied = set()
        for op, index in ops:
            if not isinstance(op, tn.Node):
                # op is only a matrix
                op = backend.reshape2(op)
                op = backend.cast(op, dtype=dtypestr)
                op = gates.Gate(op)
            else:
                op.tensor = backend.cast(op.tensor, dtype=dtypestr)
            if isinstance(index, int):
                index = [index]
            index = tuple([i if i >= 0 else self._nqubits + i for i in index])  # type: ignore
            noe = len(index)

            for j, e in enumerate(index):
                if e in occupied:
                    raise ValueError("Cannot measure two operators in one index")
                newdang[e + nq] ^ op.get_edge(j)
                newdang[e] ^ op.get_edge(j + noe)
                occupied.add(e)
            self.coloring_nodes([op], flag="operator")
            # op.flag = "operator"
            # op.is_dagger = False
            # op.id = id(op)
            nodes.append(op)
        for j in range(nq):
            if j not in occupied:  # edge1[j].is_dangling invalid here!
                newdang[j] ^ newdang[j + nq]
        return nodes  # type: ignore

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

    def perfect_sampling(self, status: Optional[Tensor] = None) -> Tuple[str, float]:
        """
        Sampling bistrings from the circuit output based on quantum amplitudes.
        Reference: arXiv:1201.3974.

        :param status: external randomness, with shape [nqubits], defaults to None
        :type status: Optional[Tensor]
        :return: Sampled bit string and the corresponding theoretical probability.
        :rtype: Tuple[str, float]
        """
        return self.measure_jit(*range(self._nqubits), with_prob=True, status=status)

    def measure_jit(
        self, *index: int, with_prob: bool = False, status: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Take measurement to the given quantum lines.
        This method is jittable is and about 100 times faster than unjit version!

        :param index: Measure on which quantum line.
        :type index: int
        :param with_prob: If true, theoretical probability is also returned.
        :type with_prob: bool, optional
        :param status: external randomness, with shape [index], defaults to None
        :type status: Optional[Tensor]
        :return: The sample output and probability (optional) of the quantum line.
        :rtype: Tuple[Tensor, Tensor]
        """
        # finally jit compatible ! and much faster than unjit version ! (100x)
        sample: List[Tensor] = []
        p = 1.0
        p = backend.convert_to_tensor(p)
        p = backend.cast(p, dtype=rdtypestr)
        for k, j in enumerate(index):
            if self.is_dm is False:
                nodes1, edge1 = self._copy()
                nodes2, edge2 = self._copy(conj=True)
                newnodes = nodes1 + nodes2
            else:
                newnodes, newfront = self._copy()
                nfront = len(newfront) // 2
                edge2 = newfront[nfront:]
                edge1 = newfront[:nfront]
            for i, e in enumerate(edge1):
                if i != j:
                    e ^ edge2[i]
            for i in range(k):
                m = (1 - sample[i]) * gates.array_to_tensor(np.array([1, 0])) + sample[
                    i
                ] * gates.array_to_tensor(np.array([0, 1]))
                newnodes.append(Gate(m))
                newnodes[-1].id = id(newnodes[-1])
                newnodes[-1].is_dagger = False
                newnodes[-1].flag = "measurement"
                newnodes[-1].get_edge(0) ^ edge1[index[i]]
                newnodes.append(Gate(m))
                newnodes[-1].id = id(newnodes[-1])
                newnodes[-1].is_dagger = True
                newnodes[-1].flag = "measurement"
                newnodes[-1].get_edge(0) ^ edge2[index[i]]
            rho = (
                1
                / backend.cast(p, dtypestr)
                * contractor(newnodes, output_edge_order=[edge1[j], edge2[j]]).tensor
            )
            pu = backend.real(rho[0, 0])
            if status is None:
                r = backend.implicit_randu()[0]
            else:
                r = status[k]
            r = backend.real(backend.cast(r, dtypestr))
            eps = 0.31415926 * 1e-12
            sign = backend.sign(r - pu + eps) / 2 + 0.5  # in case status is exactly 0.5
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

    def amplitude(self, l: Union[str, Tensor]) -> Tensor:
        r"""
        Returns the amplitude of the circuit given the bitstring l.
        For state simulator, it computes :math:`\langle l\vert \psi\rangle`,
        for density matrix simulator, it computes :math:`Tr(\rho \vert l\rangle \langle 1\vert)`
        Note how these two are different up to a square operation.

        :Example:

        >>> c = tc.Circuit(2)
        >>> c.X(0)
        >>> c.amplitude("10")
        array(1.+0.j, dtype=complex64)
        >>> c.CNOT(0, 1)
        >>> c.amplitude("11")
        array(1.+0.j, dtype=complex64)

        :param l: The bitstring of 0 and 1s.
        :type l: Union[str, Tensor]
        :return: The amplitude of the circuit.
        :rtype: tn.Node.tensor
        """
        no, d_edges = self._copy()
        ms = []
        if self.is_dm:
            msconj = []
        if isinstance(l, str):
            for s in l:
                if s == "1":
                    endn = np.array([0, 1], dtype=npdtype)
                elif s == "0":
                    endn = np.array([1, 0], dtype=npdtype)
                ms.append(tn.Node(endn))
                if self.is_dm:
                    msconj.append(tn.Node(endn))
        else:  # l is Tensor
            l = backend.cast(l, dtype=dtypestr)
            for i in range(self._nqubits):
                endn = l[i] * gates.array_to_tensor(np.array([0, 1])) + (
                    1 - l[i]
                ) * gates.array_to_tensor(np.array([1, 0]))
                ms.append(tn.Node(endn))
                if self.is_dm:
                    msconj.append(tn.Node(endn))

        for i in range(self._nqubits):
            d_edges[i] ^ ms[i].get_edge(0)
            if self.is_dm:
                d_edges[i + self._nqubits] ^ msconj[i].get_edge(0)
        for n in ms:
            n.flag = "measurement"
            n.is_dagger = False
            n.id = id(n)
            if self.is_dm:
                for n0, n in zip(ms, msconj):
                    n.flag = "measurement"
                    n.is_dagger = True
                    n.id = id(n0)
        no.extend(ms)
        if self.is_dm:
            no.extend(msconj)
        return contractor(no).tensor

    @partial(arg_alias, alias_dict={"format": ["format_"]})
    def sample(
        self,
        batch: Optional[int] = None,
        allow_state: bool = False,
        format: Optional[str] = None,
        random_generator: Optional[Any] = None,
    ) -> Any:
        """
        batched sampling from state or circuit tensor network directly

        :param batch: number of samples, defaults to None
        :type batch: Optional[int], optional
        :param allow_state: if true, we sample from the final state
            if memory allsows, True is prefered, defaults to False
        :type allow_state: bool, optional
        :param format: sample format, defaults to None as backward compatibility
            check the doc in :py:meth:`tensorcircuit.quantum.measurement_results`
        :type format: Optional[str]
        :param random_generator: random generator,  defaults to None
        :type random_generator: Optional[Any], optional
        :return: List (if batch) of tuple (binary configuration tensor and correponding probability)
            if the format is None, and consitent with format when given
        :rtype: Any
        """
        # allow_state = False is compatibility issue
        if not allow_state:
            if random_generator is None:
                random_generator = backend.get_random_state()

            if batch is None:
                seed = backend.stateful_randu(random_generator, shape=[self._nqubits])
                r = self.perfect_sampling(seed)
                if format is None:  # batch=None, format=None, backward compatibility
                    return r
                r = [r]  # type: ignore
            else:

                @backend.jit  # type: ignore
                def perfect_sampling(key: Any) -> Any:
                    backend.set_random_state(key)
                    return self.perfect_sampling()

                r = []  # type: ignore

                subkey = random_generator
                for _ in range(batch):
                    key, subkey = backend.random_split(subkey)
                    r.append(perfect_sampling(key))  # type: ignore
            if format is None:
                return r
            r = backend.stack([ri[0] for ri in r])  # type: ignore
            r = backend.cast(r, "int32")
            ch = sample_bin2int(r, self._nqubits)
        else:  # allow_state
            if batch is None:
                nbatch = 1
            else:
                nbatch = batch
            s = self.state()  # type: ignore
            if self.is_dm is False:
                p = backend.abs(s) ** 2
            else:
                p = backend.abs(backend.diagonal(s))
            a_range = backend.arange(2**self._nqubits)
            if random_generator is None:
                ch = backend.implicit_randc(a=a_range, shape=[nbatch], p=p)
            else:
                ch = backend.stateful_randc(
                    random_generator, a=a_range, shape=[nbatch], p=p
                )
            # confg = backend.mod(
            #     backend.right_shift(
            #         ch[..., None], backend.reverse(backend.arange(self._nqubits))
            #     ),
            #     2,
            # )
            if format is None:
                confg = sample_int2bin(ch, self._nqubits)
                prob = backend.gather1d(p, ch)
                r = list(zip(confg, prob))  # type: ignore
                if batch is None:
                    r = r[0]  # type: ignore
                return r
        return sample2all(sample=ch, n=self._nqubits, format=format, jittable=True)

    def sample_expectation_ps(
        self,
        x: Optional[Sequence[int]] = None,
        y: Optional[Sequence[int]] = None,
        z: Optional[Sequence[int]] = None,
        shots: Optional[int] = None,
        random_generator: Optional[Any] = None,
        **kws: Any,
    ) -> Tensor:
        """
        Compute the expectation with given Pauli string with measurement shots numbers

        :Example:

        >>> c = tc.Circuit(2)
        >>> c.H(0)
        >>> c.rx(1, theta=np.pi/2)
        >>> c.sample_expectation_ps(x=[0], y=[1])
        -0.99999976

        :param x: index for Pauli X, defaults to None
        :type x: Optional[Sequence[int]], optional
        :param y: index for Pauli Y, defaults to None
        :type y: Optional[Sequence[int]], optional
        :param z: index for Pauli Z, defaults to None
        :type z: Optional[Sequence[int]], optional
        :param shots: number of measurement shots, defaults to None, indicating analytical result
        :type shots: Optional[int], optional
        :param random_generator: random_generator, defaults to None
        :type random_general: Optional[Any]
        :return: [description]
        :rtype: Tensor
        """
        if self.is_dm is False:
            c = type(self)(self._nqubits, mps_inputs=self.quvector())  # type: ignore
        else:
            c = type(self)(self._nqubits, mpo_dminputs=self.get_dm_as_quoperator())  # type: ignore
        if x is None:
            x = []
        if y is None:
            y = []
        if z is None:
            z = []
        for i in x:
            c.H(i)  # type: ignore
        for i in y:
            c.rx(i, theta=np.pi / 2)  # type: ignore
        s = c.state()  # type: ignore
        if self.is_dm is False:
            p = backend.abs(s) ** 2
        else:
            p = backend.abs(backend.diagonal(s))
        # readout error can be processed here later
        x = list(x)
        y = list(y)
        z = list(z)
        if shots is None:
            mc = measurement_counts(
                p,
                counts=shots,
                format="count_vector",
                random_generator=random_generator,
                jittable=True,
                is_prob=True,
            )
            r = correlation_from_counts(x + y + z, mc)
        else:
            mc = measurement_counts(
                p,
                counts=shots,
                format="sample_bin",
                random_generator=random_generator,
                jittable=True,
                is_prob=True,
            )
            r = correlation_from_samples(x + y + z, mc, self._nqubits)
        # TODO(@refraction-ray): analytical standard deviation
        return r

    sexpps = sample_expectation_ps

    def replace_inputs(self, inputs: Tensor) -> None:
        """
        Replace the input state with the circuit structure unchanged.

        :param inputs: Input wavefunction.
        :type inputs: Tensor
        """
        inputs = backend.reshape(inputs, [-1])
        N = inputs.shape[0]
        n = int(np.log(N) / np.log(2))
        assert n == self._nqubits
        inputs = backend.reshape(inputs, [2 for _ in range(n)])
        if self.inputs is not None:
            self._nodes[0].tensor = inputs
            if self.is_dm:
                self._nodes[1].tensor = backend.conj(inputs)
        else:  # TODO(@refraction-ray) replace several start as inputs
            raise NotImplementedError("not support replace with no inputs")

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

    def to_graphviz(
        self,
        graph: graphviz.Graph = None,
        include_all_names: bool = False,
        engine: str = "neato",
    ) -> graphviz.Graph:
        """
        Not an ideal visualization for quantum circuit, but reserve here as a general approach to show the tensornetwork
        [Deprecated, use ``Circuit.vis_tex`` or ``Circuit.draw`` instead]
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
