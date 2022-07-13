"""
Quantum circuit class but with density matrix simulator
"""
# pylint: disable=invalid-name

from functools import reduce
from operator import add
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensornetwork as tn

from . import gates
from . import channels
from .channels import kraus_to_super_gate
from .circuit import Circuit
from .cons import backend, contractor, npdtype, dtypestr, rdtypestr
from .commons import (
    copy,
    copy_circuit,
    copy_state,
    sgates,
    vgates,
    mpogates,
    gate_aliases,
    expectation_ps,
    apply_general_gate,
    apply_general_gate_delayed,
    apply_general_variable_gate_delayed,
    expectation_before,
)

Gate = gates.Gate
Tensor = Any


class DMCircuit:
    def __init__(
        self,
        nqubits: int,
        empty: bool = False,
        inputs: Optional[Tensor] = None,
        dminputs: Optional[Tensor] = None,
        split: Optional[Dict[str, Any]] = None,
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
                self._front = [n.get_edge(0) for n in nodes]

                lnodes, lfront = copy(nodes, self._front, conj=True)
                self._front.extend(lfront)
                nodes.extend(lnodes)
                self._nodes = nodes
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
                self._front = [inputs.get_edge(i) for i in range(n)]

                lnodes, lfront = copy(nodes, self._front, conj=True)
                self._front.extend(lfront)
                nodes.extend(lnodes)
                self._nodes = nodes
            else:  # dminputs is not None
                dminputs = backend.convert_to_tensor(dminputs)
                dminputs = backend.cast(dminputs, dtype=dtypestr)
                dminputs = backend.reshape(dminputs, [2 for _ in range(2 * nqubits)])
                dminputs = Gate(dminputs)
                nodes = [dminputs]
                self._front = [dminputs.get_edge(i) for i in range(2 * nqubits)]
                self._nodes = nodes

            self._nqubits = nqubits
        self._qir: List[Dict[str, Any]] = []
        self.split = split

    @classmethod
    def _meta_apply(cls) -> None:
        for g in sgates:
            setattr(
                cls,
                g,
                apply_general_gate_delayed(getattr(gates, g), name=g, is_dm=True),
            )
            setattr(
                cls,
                g.upper(),
                apply_general_gate_delayed(getattr(gates, g), name=g, is_dm=True),
            )
            getattr(cls, g).__doc__ = getattr(Circuit, g).__doc__
            getattr(cls, g.upper()).__doc__ = getattr(Circuit, g).__doc__

        for g in vgates:
            setattr(
                cls,
                g,
                apply_general_variable_gate_delayed(
                    getattr(gates, g), name=g, is_dm=True
                ),
            )
            setattr(
                cls,
                g.upper(),
                apply_general_variable_gate_delayed(
                    getattr(gates, g), name=g, is_dm=True
                ),
            )
            getattr(cls, g).__doc__ = getattr(Circuit, g).__doc__
            getattr(cls, g.upper()).__doc__ = getattr(Circuit, g).__doc__

        for g in mpogates:
            setattr(
                cls,
                g,
                apply_general_variable_gate_delayed(
                    getattr(gates, g), name=g, mpo=True, is_dm=True
                ),
            )
            setattr(
                cls,
                g.upper(),
                apply_general_variable_gate_delayed(
                    getattr(gates, g), name=g, mpo=True, is_dm=True
                ),
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

        for gate_alias in gate_aliases:
            present_gate = gate_alias[0]
            for alias_gate in gate_alias[1:]:
                setattr(cls, alias_gate, getattr(cls, present_gate))

    _copy = copy_circuit

    def _copy_DMCircuit(self) -> "DMCircuit":
        newnodes, newfront = self._copy()
        newDMCircuit = type(self)(self._nqubits, empty=True)
        newDMCircuit._nqubits = self._nqubits
        newDMCircuit._front = newfront
        newDMCircuit._nodes = newnodes
        return newDMCircuit

    _copy_dm_tensor = copy_state

    _copy_state_tensor = _copy_dm_tensor

    def _contract(self) -> None:
        t = contractor(self._nodes, output_edge_order=self._front)
        self._nodes = [t]

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
            apply_general_gate(dmc, k, *i, is_dm=True)
            dd = dmc.densitymatrix()
            circuits.append(dd)
        tensor = reduce(add, circuits)
        tensor = backend.reshape(tensor, [2 for _ in range(2 * self._nqubits)])
        self._nodes = [Gate(tensor)]
        dangling = [e for e in self._nodes[0]]
        self._front = dangling
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
        self, *ops: Tuple[tn.Node, List[int]], reuse: bool = True, **kws: Any
    ) -> tn.Node.tensor:
        """
        Compute the expectation of corresponding operators.

        :param ops: Operator and its position on the circuit,
            eg. ``(tc.gates.z(), [1, ]), (tc.gates.x(), [2, ])`` is for operator :math:`Z_1X_2`.
        :type ops: Tuple[tn.Node, List[int]]
        :param reuse: whether contract the density matrix in advance, defaults to True
        :type reuse: bool
        :return: Tensor with one element
        :rtype: Tensor
        """
        nodes = expectation_before(self, *ops, reuse=reuse, is_dm=True)
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
            newnodes, newfront = self._copy()
            nfront = len(newfront) // 2
            edge2 = newfront[nfront:]
            edge1 = newfront[:nfront]
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

    def to_qir(self) -> List[Dict[str, Any]]:
        """
        Return the quantum intermediate representation of the circuit.

        :return: The quantum intermediate representation of the circuit.
        :rtype: List[Dict[str, Any]]
        """
        return self._qir

    def to_circuit(self, circuit_params: Optional[Dict[str, Any]] = None) -> Circuit:
        """
        convert into state simulator
        (current implementation ignores all noise channels)

        :param circuit_params: kws to initialize circuit object,
            defaults to None
        :type circuit_params: Optional[Dict[str, Any]], optional
        :return: _description_
        :rtype: Circuit
        """
        qir = self.to_qir()
        c = Circuit.from_qir(qir, circuit_params)
        return c


# TODO(@refraction-ray): new sampling API as Circuit

DMCircuit._meta_apply()
DMCircuit.expectation_ps = expectation_ps  # type: ignore


class DMCircuit2(DMCircuit):
    def apply_general_kraus(self, kraus: Sequence[Gate], *index: int) -> None:  # type: ignore
        # incompatible API for now
        kraus = [
            k
            if isinstance(k, tn.Node)
            else Gate(backend.cast(backend.convert_to_tensor(k), dtypestr))
            for k in kraus
        ]
        self.check_kraus(kraus)
        if not isinstance(
            index[0], int
        ):  # try best to be compatible with DMCircuit interface
            index = index[0][0]
        # assert len(kraus) == len(index) or len(index) == 1
        # if len(index) == 1:
        #     index = [index[0] for _ in range(len(kraus))]
        super_op = kraus_to_super_gate(kraus)
        nlegs = 4 * len(index)
        super_op = backend.reshape(super_op, [2 for _ in range(nlegs)])
        super_op = Gate(super_op)
        o2i = int(nlegs / 2)
        r2l = int(nlegs / 4)
        for i, ind in enumerate(index):
            super_op.get_edge(i + r2l + o2i) ^ self._front[ind + self._nqubits]
            self._front[ind + self._nqubits] = super_op.get_edge(i + r2l)
            super_op.get_edge(i + o2i) ^ self._front[ind]
            self._front[ind] = super_op.get_edge(i)
        self._nodes.append(super_op)
        setattr(self, "state_tensor", None)

    general_kraus = apply_general_kraus  # type: ignore

    @staticmethod
    def apply_general_kraus_delayed(
        krausf: Callable[..., Sequence[Gate]]
    ) -> Callable[..., None]:
        def apply(self: "DMCircuit2", *index: int, **vars: float) -> None:
            kraus = krausf(**vars)
            self.apply_general_kraus(kraus, *index)

        return apply


DMCircuit2._meta_apply()
