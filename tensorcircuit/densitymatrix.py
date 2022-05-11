"""
Quantum circuit class but with density matrix simulator
"""
# pylint: disable=invalid-name

from functools import reduce
from operator import add
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import tensornetwork as tn

from . import gates
from . import channels
from .circuit import Circuit, _expectation_ps
from .cons import backend, contractor, npdtype, dtypestr, rdtypestr

Gate = gates.Gate
Tensor = Any


class DMCircuit:
    def __init__(
        self,
        nqubits: int,
        empty: bool = False,
        inputs: Optional[Tensor] = None,
        dminputs: Optional[Tensor] = None,
    ) -> None:
        """
        The density matrix simulator based on tensornetwork engine.

        :param nqubits: Number of qubits
        :type nqubits: int
        :param empty: if True, nothing initialized, only for internal use, defaults to False
        :type empty: bool, optional
        :param inputs: the state input for the circuit, defaults to None
        :type inputs: Optional[Tensor], optional
        :param dminputs: the density matrix input for the circuit, defaults to None
        :type dminputs: Optional[Tensor], optional
        """
        if not empty:
            _prefix = "qb-"
            if (inputs is None) and (dminputs is None):
                # Get nodes on the interior
                nodes = [
                    tn.Node(
                        np.array(
                            [
                                1.0,
                                0.0,
                            ],
                            dtype=npdtype,
                        ),
                        name=_prefix + str(x + 1),
                    )
                    for x in range(nqubits)
                ]
                self._rfront = [n.get_edge(0) for n in nodes]

                lnodes, self._lfront = self._copy(nodes, self._rfront, conj=True)
                lnodes.extend(nodes)
                self._nodes = lnodes
            elif inputs is not None:
                inputs = backend.convert_to_tensor(inputs)
                inputs = backend.cast(inputs, dtype=dtypestr)
                inputs = backend.reshape(inputs, [-1])
                N = inputs.shape[0]
                n = int(np.log(N) / np.log(2))
                assert n == nqubits
                inputs = backend.reshape(inputs, [2 for _ in range(n)])
                inputs = Gate(inputs)
                nodes = [inputs]
                self._rfront = [inputs.get_edge(i) for i in range(n)]

                lnodes, self._lfront = self._copy(nodes, self._rfront, conj=True)
                lnodes.extend(nodes)
                self._nodes = lnodes
            else:  # dminputs is not None
                dminputs = backend.convert_to_tensor(dminputs)
                dminputs = backend.cast(dminputs, dtype=dtypestr)
                dminputs = backend.reshape(dminputs, [2 for _ in range(2 * nqubits)])
                dminputs = Gate(dminputs)
                nodes = [dminputs]
                self._rfront = [dminputs.get_edge(i) for i in range(nqubits)]
                self._lfront = [dminputs.get_edge(i + nqubits) for i in range(nqubits)]
                self._nodes = nodes

            self._nqubits = nqubits

    @classmethod
    def _meta_apply(cls) -> None:
        for g in Circuit.sgates:
            setattr(cls, g, cls.apply_general_gate_delayed(getattr(gates, g), name=g))
            setattr(
                cls,
                g.upper(),
                cls.apply_general_gate_delayed(getattr(gates, g), name=g),
            )
            getattr(cls, g).__doc__ = getattr(Circuit, g).__doc__
            getattr(cls, g.upper()).__doc__ = getattr(Circuit, g).__doc__

        for g in Circuit.vgates:
            setattr(
                cls,
                g,
                cls.apply_general_variable_gate_delayed(getattr(gates, g), name=g),
            )
            setattr(
                cls,
                g.upper(),
                cls.apply_general_variable_gate_delayed(getattr(gates, g), name=g),
            )
            getattr(cls, g).__doc__ = getattr(Circuit, g).__doc__
            getattr(cls, g.upper()).__doc__ = getattr(Circuit, g).__doc__

        for k in channels.channels:
            setattr(
                cls,
                k,
                cls.apply_general_kraus_delayed(getattr(channels, k + "channel")),
            )
            doc = """
            Apply %s quantum channel on the circuit.
            See :py:meth:`tensorcircuit.channels.%schannel`

            :param index: Qubit number that the gate applies on.
            :type index: int.
            :param vars: Parameters for the channel.
            :type vars: float.
            """ % (
                k,
                k,
            )
            getattr(cls, k).__doc__ = doc

        for gate_alias in Circuit.gate_alias_list:
            present_gate = gate_alias[0]
            for alias_gate in gate_alias[1:]:
                setattr(cls, alias_gate, getattr(cls, present_gate))

    def _copy(
        self,
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
            newnodes.append(ndict[n])
        newfront = []
        if not dangling:
            dangling = []
            for n in nodes:
                dangling.extend([e for e in n])
        for e in dangling:
            newfront.append(edict[e])
        return newnodes, newfront

    def _copy_DMCircuit(self) -> "DMCircuit":
        newnodes, newfront = self._copy(self._nodes, self._lfront + self._rfront)
        newDMCircuit = DMCircuit(self._nqubits, empty=True)
        newDMCircuit._nqubits = self._nqubits
        newDMCircuit._lfront = newfront[: self._nqubits]
        newDMCircuit._rfront = newfront[self._nqubits :]
        newDMCircuit._nodes = newnodes
        return newDMCircuit

    def _copy_dm_tensor(
        self, conj: bool = False, reuse: bool = True
    ) -> Tuple[List[tn.Node], List[tn.Edge]]:
        if reuse:
            t = getattr(self, "state_tensor", None)
        else:
            t = None
        if t is None:
            nodes, d_edges = self._copy(
                self._nodes, self._rfront + self._lfront, conj=conj
            )
            t = contractor(nodes, output_edge_order=d_edges)
            setattr(self, "state_tensor", t)
        ndict, edict = tn.copy([t], conjugate=conj)
        newnodes = []
        newnodes.append(ndict[t])
        newfront = []
        for e in t.edges:
            newfront.append(edict[e])
        return newnodes, newfront

    def _contract(self) -> None:
        t = contractor(self._nodes, output_edge_order=self._rfront + self._lfront)
        self._nodes = [t]

    def apply_general_gate(
        self, gate: Gate, *index: int, name: Optional[str] = None
    ) -> None:
        assert len(index) == len(set(index))
        noe = len(index)
        lgated, _ = self._copy([gate], conj=True)
        lgate = lgated[0]
        for i, ind in enumerate(index):
            gate.get_edge(i + noe) ^ self._rfront[ind]
            self._rfront[ind] = gate.get_edge(i)
            lgate.get_edge(i + noe) ^ self._lfront[ind]
            self._lfront[ind] = lgate.get_edge(i)
        self._nodes.append(gate)
        self._nodes.append(lgate)
        setattr(self, "state_tensor", None)

    @staticmethod
    def apply_general_gate_delayed(
        gatef: Callable[[], Gate], name: Optional[str] = None
    ) -> Callable[..., None]:
        def apply(self: "DMCircuit", *index: int) -> None:
            gate = gatef()
            self.apply_general_gate(gate, *index, name=name)

        return apply

    @staticmethod
    def apply_general_variable_gate_delayed(
        gatef: Callable[..., Gate],
        name: Optional[str] = None,
    ) -> Callable[..., None]:
        def apply(self: "DMCircuit", *index: int, **vars: float) -> None:
            gate = gatef(**vars)
            self.apply_general_gate(gate, *index, name=name)

        return apply

    @staticmethod
    def check_kraus(kraus: Sequence[Gate]) -> bool:  # TODO(@refraction-ray)
        return True

    def apply_general_kraus(
        self, kraus: Sequence[Gate], index: Sequence[Tuple[int, ...]]
    ) -> None:
        # note the API difference for index arg between DM and DM2
        self.check_kraus(kraus)
        assert len(kraus) == len(index) or len(index) == 1
        if len(index) == 1:
            index = [index[0] for _ in range(len(kraus))]
        self._contract()
        circuits = []
        for k, i in zip(kraus, index):
            dmc = self._copy_DMCircuit()
            dmc.apply_general_gate(k, *i)
            dd = dmc.densitymatrix()
            circuits.append(dd)
        tensor = reduce(add, circuits)
        tensor = backend.reshape(tensor, [2 for _ in range(2 * self._nqubits)])
        self._nodes = [Gate(tensor)]
        dangling = [e for e in self._nodes[0]]
        self._rfront = dangling[: self._nqubits]
        self._lfront = dangling[self._nqubits :]
        setattr(self, "state_tensor", None)

    general_kraus = apply_general_kraus

    @staticmethod
    def apply_general_kraus_delayed(
        krausf: Callable[..., Sequence[Gate]]
    ) -> Callable[..., None]:
        def apply(self: "DMCircuit", *index: int, **vars: float) -> None:
            kraus = krausf(**vars)
            self.apply_general_kraus(kraus, [index])

        return apply

    def densitymatrix(self, check: bool = False, reuse: bool = True) -> Tensor:
        """
        Return the output density matrix of the circuit.

        :param check: check whether the final return is a legal density matrix, defaults to False
        :type check: bool, optional
        :param reuse: whether to reuse previous results, defaults to True
        :type reuse: bool, optional
        :return: The output densitymatrix in 2D shape tensor form
        :rtype: Tensor
        """
        nodes, _ = self._copy_dm_tensor(conj=False, reuse=reuse)
        # t = contractor(nodes, output_edge_order=d_edges)
        dm = backend.reshape(
            nodes[0].tensor, shape=[2**self._nqubits, 2**self._nqubits]
        )
        if check:
            self.check_density_matrix(dm)
        return dm

    state = densitymatrix

    def expectation(
        self, *ops: Tuple[tn.Node, List[int]], **kws: Any
    ) -> tn.Node.tensor:
        """
        Compute the expectation of corresponding operators.

        :param ops: Operator and its position on the circuit,
            eg. ``(tc.gates.z(), [1, ]), (tc.gates.x(), [2, ])`` is for operator :math:`Z_1X_2`.
        :type ops: Tuple[tn.Node, List[int]]
        :return: Tensor with one element
        :rtype: Tensor
        """
        # kws is reserved for unsupported feature such as reuse arg
        newdm, newdang = self._copy(self._nodes, self._rfront + self._lfront)
        occupied = set()
        nodes = newdm
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
                newdang[e + self._nqubits] ^ op.get_edge(j)
                newdang[e] ^ op.get_edge(j + noe)
                occupied.add(e)
            nodes.append(op)
        for j in range(self._nqubits):
            if j not in occupied:  # edge1[j].is_dangling invalid here!
                newdang[j] ^ newdang[j + self._nqubits]
        return contractor(nodes).tensor

    @staticmethod
    def check_density_matrix(dm: Tensor) -> None:
        assert np.allclose(backend.trace(dm), 1.0, atol=1e-5)

    def measure_jit(
        self, *index: int, with_prob: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Take measurement to the given quantum lines.

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
            newnodes, newfront = self._copy(self._nodes, self._lfront + self._rfront)
            nfront = len(newfront) // 2
            edge1 = newfront[nfront:]
            edge2 = newfront[:nfront]
            # _lfront is edge2
            for i, e in enumerate(edge1):
                if i != j:
                    e ^ edge2[i]
            for i in range(k):
                m = (1 - sample[i]) * gates.array_to_tensor(np.array([1, 0])) + sample[
                    i
                ] * gates.array_to_tensor(np.array([0, 1]))
                newnodes.append(Gate(m))
                newnodes[-1].get_edge(0) ^ edge1[index[i]]
                newnodes.append(tn.Node(m))
                newnodes[-1].get_edge(0) ^ edge2[index[i]]
            rho = (
                1
                / backend.cast(p, dtypestr)
                * contractor(newnodes, output_edge_order=[edge1[j], edge2[j]]).tensor
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

        :return: Sampled bit string and the corresponding theoretical probability.
        :rtype: Tuple[str, float]
        """
        return self.measure_jit(*[i for i in range(self._nqubits)], with_prob=True)

    sample = perfect_sampling


DMCircuit._meta_apply()
DMCircuit.expectation_ps = _expectation_ps  # type: ignore
