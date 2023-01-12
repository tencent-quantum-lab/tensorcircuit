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
from .cons import backend, contractor, dtypestr
from .basecircuit import BaseCircuit
from .quantum import QuOperator

Gate = gates.Gate
Tensor = Any


class DMCircuit(BaseCircuit):
    is_dm = True

    def __init__(
        self,
        nqubits: int,
        empty: bool = False,
        inputs: Optional[Tensor] = None,
        mps_inputs: Optional[QuOperator] = None,
        dminputs: Optional[Tensor] = None,
        mpo_dminputs: Optional[QuOperator] = None,
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
        :param mps_inputs: QuVector for a MPS like initial pure state.
        :type mps_inputs: Optional[QuOperator]
        :param dminputs: the density matrix input for the circuit, defaults to None
        :type dminputs: Optional[Tensor], optional
        :param mpo_dminputs: QuOperator for a MPO like initial density matrix.
        :type mpo_dminputs: Optional[QuOperator]
        :param split: dict if two qubit gate is ready for split, including parameters for at least one of
            ``max_singular_values`` and ``max_truncation_err``.
        :type split: Optional[Dict[str, Any]]
        """
        if not empty:
            if (
                (inputs is None)
                and (dminputs is None)
                and (mps_inputs is None)
                and (mpo_dminputs is None)
            ):
                # Get nodes on the interior
                self._nodes = self.all_zero_nodes(nqubits)
                self._front = [n.get_edge(0) for n in self._nodes]
                self.coloring_nodes(self._nodes)
                self._double_nodes_front()

            elif inputs is not None:
                inputs = backend.convert_to_tensor(inputs)
                inputs = backend.cast(inputs, dtype=dtypestr)
                inputs = backend.reshape(inputs, [-1])
                N = inputs.shape[0]
                n = int(np.log(N) / np.log(2))
                assert n == nqubits
                inputs = backend.reshape(inputs, [2 for _ in range(n)])
                inputs_gate = Gate(inputs)
                self._nodes = [inputs_gate]
                self.coloring_nodes(self._nodes)
                self._front = [inputs_gate.get_edge(i) for i in range(n)]
                self._double_nodes_front()

            elif mps_inputs is not None:
                mps_nodes = list(mps_inputs.nodes)
                for i, n in enumerate(mps_nodes):
                    mps_nodes[i].tensor = backend.cast(n.tensor, dtypestr)  # type: ignore
                mps_edges = mps_inputs.out_edges + mps_inputs.in_edges
                self._nodes, self._front = self.copy(mps_nodes, mps_edges)
                self.coloring_nodes(self._nodes)
                self._double_nodes_front()

            elif dminputs is not None:
                dminputs = backend.convert_to_tensor(dminputs)
                dminputs = backend.cast(dminputs, dtype=dtypestr)
                dminputs = backend.reshape(dminputs, [2 for _ in range(2 * nqubits)])
                dminputs_gate = Gate(dminputs)
                nodes = [dminputs_gate]
                self._front = [dminputs_gate.get_edge(i) for i in range(2 * nqubits)]
                self._nodes = nodes
                self.coloring_nodes(self._nodes)

            else:  # mpo_dminputs is not None
                mpo_nodes = list(mpo_dminputs.nodes)  # type: ignore
                for i, n in enumerate(mpo_nodes):
                    mpo_nodes[i].tensor = backend.cast(n.tensor, dtypestr)  # type: ignore
                mpo_edges = mpo_dminputs.out_edges + mpo_dminputs.in_edges  # type: ignore
                self._nodes = mpo_nodes
                self._front = mpo_edges
                self.coloring_nodes(self._nodes)

            self._start_index = len(self._nodes)

        self._nqubits = nqubits
        self.inputs = inputs
        self.dminputs = dminputs
        self.mps_inputs = mps_inputs
        self.mpo_dminputs = mpo_dminputs
        self.split = split

        self.circuit_param = {
            "nqubits": nqubits,
            "inputs": inputs,
            "mps_inputs": mps_inputs,
            "dminputs": dminputs,
            "mpo_dminputs": mpo_dminputs,
            "split": split,
        }

        self._qir: List[Dict[str, Any]] = []
        self._extra_qir: List[Dict[str, Any]] = []

    def _double_nodes_front(self) -> None:
        lnodes, lfront = self.copy(self._nodes, self._front, conj=True)
        self._front.extend(lfront)
        self._nodes.extend(lnodes)

    @classmethod
    def _meta_apply_channels(cls) -> None:
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

    def _copy_DMCircuit(self) -> "DMCircuit":
        newnodes, newfront = self._copy()
        newDMCircuit = type(self)(self._nqubits, empty=True)
        newDMCircuit._nqubits = self._nqubits
        newDMCircuit._front = newfront
        newDMCircuit._nodes = newnodes
        return newDMCircuit

    _copy_dm_tensor = BaseCircuit._copy_state_tensor

    def _contract(self) -> None:
        t = contractor(self._nodes, output_edge_order=self._front)
        self._nodes = [t]

    @staticmethod
    def check_kraus(kraus: Sequence[Gate]) -> bool:  # TODO(@refraction-ray)
        return True

    def apply_general_kraus(
        self, kraus: Sequence[Gate], index: Sequence[Tuple[int, ...]], **kws: Any
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
        self._front = dangling
        setattr(self, "state_tensor", None)

    general_kraus = apply_general_kraus

    @staticmethod
    def apply_general_kraus_delayed(
        krausf: Callable[..., Sequence[Gate]]
    ) -> Callable[..., None]:
        def apply(self: "DMCircuit", *index: int, **vars: float) -> None:
            for key in ["status", "name"]:
                if key in vars:
                    del vars[key]
            # compatibility with circuit API
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
        nodes, d_edges = self._copy_dm_tensor(conj=False, reuse=reuse)
        if len(nodes) > 1:
            t = contractor(nodes, output_edge_order=d_edges)
        else:
            t = nodes[0]
        dm = backend.reshape(t.tensor, shape=[2**self._nqubits, 2**self._nqubits])
        if check:
            self.check_density_matrix(dm)
        return dm

    state = densitymatrix

    def wavefunction(self) -> Tensor:
        """
        get the wavefunction of outputs,
        raise error if the final state is not purified
        [Experimental: the phase factor is not fixed for different backend]

        :return: wavefunction vector
        :rtype: Tensor
        """
        dm = self.densitymatrix()
        e, v = backend.eigh(dm)
        np.testing.assert_allclose(
            e[:-1], backend.zeros([2**self._nqubits - 1]), atol=1e-5
        )
        return v[:, -1]

    get_dm_as_quvector = BaseCircuit.quvector

    def get_dm_as_quoperator(self) -> QuOperator:
        """
        Get the representation of the output state in the form of ``QuOperator``
        while maintaining the circuit uncomputed

        :return: ``QuOperator`` representation of the output state from the circuit
        :rtype: QuOperator
        """
        _, edges = self._copy()
        return QuOperator(edges[: self._nqubits], edges[self._nqubits :])

    def expectation(
        self,
        *ops: Tuple[tn.Node, List[int]],
        reuse: bool = True,
        noise_conf: Optional[Any] = None,
        status: Optional[Tensor] = None,
        **kws: Any
    ) -> tn.Node.tensor:
        """
        Compute the expectation of corresponding operators.

        :param ops: Operator and its position on the circuit,
            eg. ``(tc.gates.z(), [1, ]), (tc.gates.x(), [2, ])`` is for operator :math:`Z_1X_2`.
        :type ops: Tuple[tn.Node, List[int]]
        :param reuse: whether contract the density matrix in advance, defaults to True
        :type reuse: bool
        :param noise_conf: Noise Configuration, defaults to None
        :type noise_conf: Optional[NoiseConf], optional
        :param status: external randomness given by tensor uniformly from [0, 1], defaults to None,
            used for noisfy circuit sampling
        :type status: Optional[Tensor], optional
        :return: Tensor with one element
        :rtype: Tensor
        """
        from .noisemodel import expectation_noisfy

        if noise_conf is None:
            nodes = self.expectation_before(*ops, reuse=reuse)
            return contractor(nodes).tensor
        else:
            return expectation_noisfy(
                self,
                *ops,
                noise_conf=noise_conf,
                status=status,
                **kws,
            )

    @staticmethod
    def check_density_matrix(dm: Tensor) -> None:
        assert np.allclose(backend.trace(dm), 1.0, atol=1e-5)

    def to_circuit(self, circuit_params: Optional[Dict[str, Any]] = None) -> Circuit:
        """
        convert into state simulator
        (current implementation ignores all noise channels)

        :param circuit_params: kws to initialize circuit object,
            defaults to None
        :type circuit_params: Optional[Dict[str, Any]], optional
        :return: Circuit with no noise
        :rtype: Circuit
        """
        qir = self.to_qir()
        c = Circuit.from_qir(qir, circuit_params)
        return c  # type: ignore


DMCircuit._meta_apply()
DMCircuit._meta_apply_channels()


class DMCircuit2(DMCircuit):
    def apply_general_kraus(
        self, kraus: Sequence[Gate], *index: int, **kws: Any
    ) -> None:
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

    general_kraus = apply_general_kraus

    @staticmethod
    def apply_general_kraus_delayed(
        krausf: Callable[..., Sequence[Gate]]
    ) -> Callable[..., None]:
        def apply(self: "DMCircuit2", *index: int, **vars: float) -> None:
            for key in ["status", "name"]:
                if key in vars:
                    del vars[key]
            kraus = krausf(**vars)
            self.apply_general_kraus(kraus, *index)

        return apply


DMCircuit2._meta_apply()
DMCircuit2._meta_apply_channels()
