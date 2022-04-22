"""
Quantum circuit: MPS state simulator
"""
# pylint: disable=invalid-name

from functools import reduce
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from . import gates
from .cons import backend, npdtype
from .mps_base import FiniteMPS

Gate = gates.Gate
Tensor = Any

# TODO(@refraction-ray): support Circuit IR for MPSCircuit


def split_tensor(
    tensor: Tensor,
    left: bool = True,
    max_singular_values: Optional[int] = None,
    max_truncation_err: Optional[float] = None,
    relative: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Split the tensor by SVD or QR depends on whether a truncation is required.

    :param tensor: The input tensor to split.
    :type tensor: Tensor
    :param left: Determine the orthogonal center is on the left tensor or the right tensor.
    :type left: bool, optional
    :param max_singular_values: The maximum number of singular values to keep.
    :type max_singular_values: int, optional
    :param max_truncation_err: The maximum allowed truncation error.
    :type max_truncation_err: float, optional
    :param relative: Multiply `max_truncation_err` with the largest singular value.
    :type relative: bool, optional
    :return: Two tensors after splitting
    :rtype: Tuple[Tensor, Tensor]
    """
    # The behavior is a little bit different from tn.split_node because it explicitly requires a center
    svd = (max_truncation_err is not None) or (max_singular_values is not None)
    if svd:
        U, S, VH, _ = backend.svd(
            tensor,
            max_singular_values=max_singular_values,
            max_truncation_error=max_truncation_err,
            relative=relative,
        )
        if left:
            return backend.matmul(U, backend.diagflat(S)), VH
        else:
            return U, backend.matmul(backend.diagflat(S), VH)
    else:
        if left:
            return backend.rq(tensor)  # type: ignore
        else:
            return backend.qr(tensor)  # type: ignore


class MPSCircuit:
    """
    ``MPSCircuit`` class.
    Simple usage demo below.

    .. code-block:: python

        mps = tc.MPSCircuit(3)
        mps.H(1)
        mps.CNOT(0, 1)
        mps.rx(2, theta=tc.num_to_tensor(1.))
        mps.expectation_single_gate(tc.gates.z(), 2)

    """

    sgates = ["i", "x", "y", "z", "h", "t", "s", "wroot"] + [
        "cnot",
        "cz",
        "swap",
        "cy",
    ]
    # gates on > 2 qubits like toffoli is not available
    # however they can be constructed from 1 and 2 qubit gates
    vgates = ["r", "cr", "rx", "ry", "rz", "any", "exp", "exp1"]
    # TODO(@refraction-ray): gate list update

    def __init__(
        self,
        nqubits: int,
        tensors: Optional[Sequence[Tensor]] = None,
        center_position: int = 0,
    ) -> None:
        """
        MPSCircuit object based on state simulator.

        :param nqubits: The number of qubits in the circuit.
        :type nqubits: int
        :param tensors: If not None, the initial state of the circuit is taken as ``tensors``
            instead of :math:`\\vert 0\\rangle^n` qubits, defaults to None
        :type tensors: Sequence[Tensor], optional
        :param center_position: The center position of MPS, default to 0
        :type center_position: int, optional
        """
        if tensors is None:
            tensors = [
                np.array([1.0, 0.0], dtype=npdtype)[None, :, None]
                for i in range(nqubits)
            ]
        else:
            assert len(tensors) == nqubits
        self._mps = FiniteMPS(
            tensors, canonicalize=True, center_position=center_position
        )

        self._nqubits = nqubits
        self._fidelity = 1.0
        self.set_truncation_rule()

    # `MPSCircuit` does not has `replace_inputs` like `Circuit`
    # because the gates are immediately absorted into the MPS when applied,
    # so it is impossible to remember the initial structure

    def set_truncation_rule(
        self,
        max_singular_values: Optional[int] = None,
        max_truncation_err: Optional[float] = None,
        relative: bool = False,
    ) -> None:
        """
        Set truncation rules when double qubit gates are applied.
        If nothing is specified, no truncation will take place and the bond dimension will keep growing.
        For more details, refer to `split_tensor`.

        :param max_singular_values: The maximum number of singular values to keep.
        :type max_singular_values: int, optional
        :param max_truncation_err: The maximum allowed truncation error.
        :type max_truncation_err: float, optional
        :param relative: Multiply `max_truncation_err` with the largest singular value.
        :type relative: bool, optional
        """
        self.max_singular_values = max_singular_values
        self.max_truncation_err = max_truncation_err
        self.relative = relative
        self.do_truncation = (self.max_singular_values is not None) or (
            self.max_truncation_err is not None
        )

    # TODO(@refraction-ray): unified split truncation API between Circuit and MPSCircuit

    def position(self, site: int) -> None:
        """
        Wrapper of tn.FiniteMPS.position.
        Set orthogonality center.

        :param site: The orthogonality center
        :type site: int
        """
        self._mps.position(site, normalize=False)

    @classmethod
    def _meta_apply(cls) -> None:

        for g in cls.sgates:
            setattr(cls, g, cls.apply_general_gate_delayed(gatef=getattr(gates, g)))
            setattr(
                cls,
                g.upper(),
                cls.apply_general_gate_delayed(gatef=getattr(gates, g)),
            )
            matrix = gates.matrix_for_gate(getattr(gates, g)())
            matrix = gates.bmatrix(matrix)
            doc = """
            Apply **%s** gate on the circuit.

            :param index: Qubit number that the gate applies on.
                The matrix for the gate is

                .. math::

                      %s

            :type index: int.
            """ % (
                g.upper(),
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
                cls.apply_general_variable_gate_delayed(gatef=getattr(gates, g)),
            )
            setattr(
                cls,
                g.upper(),
                cls.apply_general_variable_gate_delayed(gatef=getattr(gates, g)),
            )
            doc = """
            Apply %s gate with parameters on the circuit.

            :param index: Qubit number that the gate applies on.
            :type index: int.
            :param vars: Parameters for the gate
            :type vars: float.
            """ % (
                g
            )
            getattr(cls, g).__doc__ = doc
            getattr(cls, g.upper()).__doc__ = doc

    def apply_single_gate(self, gate: Gate, index: int) -> None:
        """
        Apply a single qubit gate on MPS, and the gate must be unitary; no truncation is needed.

        :param gate: gate to be applied
        :type gate: Gate
        :param index: Qubit index of the gate
        :type index: int
        """
        self._mps.apply_one_site_gate(gate.tensor, index)

    def apply_adjacent_double_gate(
        self,
        gate: Gate,
        index1: int,
        index2: int,
        center_position: Optional[int] = None,
    ) -> None:
        """
        Apply a double qubit gate on adjacent qubits of Matrix Product States (MPS).
        Truncation rule is specified by `set_truncation_rule`.

        :param gate: The Gate to be applied
        :type gate: Gate
        :param index1: The first qubit index of the gate
        :type index1: int
        :param index2: The second qubit index of the gate
        :type index2: int
        :param center_position: Center position of MPS, default is None
        :type center_position: Optional[int]
        """

        # The center position of MPS must be either `index1` for `index2` before applying a double gate
        # Choose the one closer to the current center
        assert index2 - index1 == 1
        diff1 = abs(index1 - self._mps.center_position)  # type: ignore
        diff2 = abs(index2 - self._mps.center_position)  # type: ignore
        if diff1 < diff2:
            self.position(index1)
        else:
            self.position(index2)
        err = self._mps.apply_two_site_gate(
            gate.tensor,
            index1,
            index2,
            center_position=center_position,
            max_singular_values=self.max_singular_values,
            max_truncation_err=self.max_truncation_err,
            relative=self.relative,
        )
        self._fidelity *= 1 - backend.real(backend.sum(err**2))

    def apply_double_gate(
        self,
        gate: Gate,
        index1: int,
        index2: int,
    ) -> None:
        """
        Apply a double qubit gate on MPS. Truncation rule is specified by `set_truncation_rule`.

        :param gate: The Gate to be applied
        :type gate: Gate
        :param index1: The first qubit index of the gate
        :type index1: int
        :param index2: The second qubit index of the gate
        :type index2: int
        """
        # Equivalent to apply N SWPA gates, the required gate, N SWAP gates sequentially on adjacent gates
        diff1 = abs(index1 - self._mps.center_position)  # type: ignore
        diff2 = abs(index2 - self._mps.center_position)  # type: ignore
        if diff1 < diff2:
            self.position(index1)
            for index in np.arange(index1, index2 - 1):
                self.apply_adjacent_double_gate(
                    gates.swap(), index, index + 1, center_position=index + 1  # type: ignore
                )
            self.apply_adjacent_double_gate(
                gate, index2 - 1, index2, center_position=index2 - 1
            )
            for index in np.arange(index1, index2 - 1)[::-1]:
                self.apply_adjacent_double_gate(
                    gates.swap(), index, index + 1, center_position=index  # type: ignore
                )
        else:
            self.position(index2)
            for index in np.arange(index1 + 1, index2)[::-1]:
                self.apply_adjacent_double_gate(
                    gates.swap(), index, index + 1, center_position=index  # type: ignore
                )
            self.apply_adjacent_double_gate(
                gate, index1, index1 + 1, center_position=index1 + 1
            )
            for index in np.arange(index1 + 1, index2):
                self.apply_adjacent_double_gate(
                    gates.swap(), index, index + 1, center_position=index + 1  # type: ignore
                )

    def apply_general_gate(self, gate: Gate, *index: int) -> None:
        """
        Apply a general qubit gate on MPS.

        :param gate: The Gate to be applied
        :type gate: Gate
        :raises ValueError: "MPS does not support application of gate on > 2 qubits."
        :param index: Qubit indices of the gate
        :type index: int
        """
        assert len(index) == len(set(index))
        noe = len(index)
        if noe == 1:
            self.apply_single_gate(gate, *index)
        elif noe == 2:
            self.apply_double_gate(gate, *index)

        else:
            raise ValueError("MPS does not support application of gate on > 2 qubits")

    apply = apply_general_gate

    @staticmethod
    def apply_general_gate_delayed(gatef: Callable[[], Gate]) -> Callable[..., None]:
        # nested function must be utilized, functools.partial doesn't work for method register on class
        # see https://re-ra.xyz/Python-中实例方法动态绑定的几组最小对立/
        def apply(self: "MPSCircuit", *index: int) -> None:
            gate = gatef()
            self.apply_general_gate(gate, *index)

        return apply

    @staticmethod
    def apply_general_variable_gate_delayed(
        gatef: Callable[..., Gate],
    ) -> Callable[..., None]:
        def apply(self: "MPSCircuit", *index: int, **vars: float) -> None:
            gate = gatef(**vars)
            self.apply_general_gate(gate, *index)

        return apply

    def mid_measurement(self, index: int, keep: int = 0) -> None:
        """
        Middle measurement in the z-basis on the circuit, note the wavefunction output is not normalized
        with ``mid_measurement`` involved, one should normalized the state manually if needed.

        :param index: The index of qubit that the Z direction postselection applied on
        :type index: int
        :param keep: 0 for spin up, 1 for spin down, defaults to 0
        :type keep: int, optional
        """
        # normalization not guaranteed
        assert keep in [0, 1]
        self.position(index)
        self._mps.tensors[index] = self._mps.tensors[index][:, keep, :]

    def is_valid(self) -> bool:
        """
        Check whether the circuit is legal.

        :return: Whether the circuit is legal.
        :rtype: bool
        """
        mps = self._mps
        if len(mps) != self._nqubits:
            return False
        for i in range(self._nqubits):
            if len(mps.tensors[i].shape) != 3:
                return False
        for i in range(self._nqubits - 1):
            if mps.tensors[i].shape[-1] != mps.tensors[i + 1].shape[0]:
                return False
        return True

    @staticmethod
    def from_wavefunction(
        wavefunction: Tensor,
        max_singular_values: Optional[int] = None,
        max_truncation_err: Optional[float] = None,
        relative: bool = True,
    ) -> "MPSCircuit":
        """
        Construct the MPS from a given wavefunction.

        :param wavefunction: The given wavefunction (any shape is OK)
        :type wavefunction: Tensor
        :param max_singular_values: The maximum number of singular values to keep.
        :type max_singular_values: int, optional
        :param max_truncation_err: The maximum allowed truncation error.
        :type max_truncation_err: float, optional
        :param relative: Multiply `max_truncation_err` with the largest singular value.
        :type relative: bool, optional
        :return: The constructed MPS
        :rtype: MPSCircuit
        """
        wavefunction = backend.reshape(wavefunction, (-1, 1))
        tensors: List[Tensor] = []
        while True:  # not jittable
            nright = wavefunction.shape[1]
            wavefunction = backend.reshape(wavefunction, (-1, nright * 2))
            wavefunction, Q = split_tensor(
                wavefunction,
                left=True,
                max_singular_values=max_singular_values,
                max_truncation_err=max_truncation_err,
                relative=relative,
            )
            tensors.insert(0, backend.reshape(Q, (-1, 2, nright)))
            if wavefunction.shape == (1, 1):
                break
        return MPSCircuit(len(tensors), tensors=tensors)

    def wavefunction(self, form: str = "default") -> Tensor:
        """
        Compute the output wavefunction from the circuit.

        :param form: the str indicating the form of the output wavefunction
        :type form: str, optional
        :return: Tensor with shape [1, -1]
        :rtype: Tensor
        """
        result = backend.ones((1, 1, 1), dtype=npdtype)
        for tensor in self._mps.tensors:
            result = backend.einsum("iaj,jbk->iabk", result, tensor)
            ni, na, nb, nk = result.shape
            result = backend.reshape(result, (ni, na * nb, nk))
        if form == "default":
            shape = [-1]
        elif form == "ket":
            shape = [-1, 1]
        elif form == "bra":  # no conj here
            shape = [1, -1]
        return backend.reshape(result, shape)

    state = wavefunction

    # TODO(@refraction-ray): mps form quvector

    def copy_without_tensor(self) -> "MPSCircuit":
        """
        Copy the current MPS without the tensors.

        :return: The constructed MPS
        :rtype: MPSCircuit
        """
        result: "MPSCircuit" = MPSCircuit.__new__(MPSCircuit)
        info = vars(self)
        for key in vars(self):
            if key == "_mps":
                continue
            setattr(result, key, info[key])
        return result

    def copy(self) -> "MPSCircuit":
        """
        Copy the current MPS.

        :return: The constructed MPS
        :rtype: MPSCircuit
        """
        result = self.copy_without_tensor()
        result._mps = self._mps.copy()
        return result

    def conj(self) -> "MPSCircuit":
        """
        Compute the conjugate of the current MPS.

        :return: The constructed MPS
        :rtype: MPSCircuit
        """
        result = self.copy_without_tensor()
        result._mps = self._mps.conj()
        return result

    def get_norm(self) -> Tensor:
        """
        Get the normalized Center Position.

        :return: Normalized Center Position.
        :rtype: Tensor
        """
        return self._mps.norm(self._mps.center_position)

    def normalize(self) -> None:
        """
        Normalize MPS Circuit according to the center position.
        """
        center = self._mps.center_position
        norm = self._mps.norm(center)
        self._mps.tensor[center] /= norm

    def amplitude(self, l: str) -> Tensor:
        assert len(l) == self._nqubits
        tensors = [self._mps.tensors[i][:, int(s), :] for i, s in enumerate(l)]
        return reduce(backend.matmul, tensors)[0, 0]

    def measure(self, *index: int, with_prob: bool = False) -> Tuple[str, float]:
        """
        :param index: integer indicating the measure on which quantum line
        :param with_prob: if true, theoretical probability is also returned
        :return:
        """
        n = len(index)
        if not np.all(np.diff(index) >= 0):
            argsort = np.argsort(index)
            invargsort = np.zeros((n,), dtype=int)
            invargsort[argsort] = np.arange(n)
            sample, prob = self.measure(*np.array(index)[argsort], with_prob=with_prob)
            return "".join(np.array(list(sample))[invargsort]), prob

        # Assume no equivalent indices
        assert np.all(np.diff(index) > 0)
        # Assume that the index is in correct order
        mpscircuit = self.copy()
        sample = ""
        p = 1.0
        # TODO@(SUSYUSTC): add the possibility to move from right to left
        for i in index:
            # Move the center position to each index from left to right
            mpscircuit.position(i)
            tensor = mpscircuit._mps.tensors[i]
            probs = backend.sum(backend.power(backend.abs(tensor), 2), axis=(0, 2))
            # TODO@(SUSYUSTC): normalize the tensor to avoid error accumulation
            probs /= backend.sum(probs)
            pu = probs[0]
            r = backend.implicit_randu([])
            if r < pu:
                choice = 0
            else:
                choice = 1
            sample += str(choice)
            p *= probs[choice]
            tensor = tensor[:, choice, :][:, None, :]
            mpscircuit._mps.tensors[i] = tensor
        if with_prob:
            return sample, p
        else:
            return sample, -1

    def proj_with_mps(self, other: "MPSCircuit") -> Tensor:
        """
        Compute the projection between `other` as bra and `self` as ket.

        :param other: ket of the other MPS, which will be converted to bra automatically
        :type other: MPSCircuit
        :return: The projection in form of tensor
        :rtype: Tensor
        """
        bra = other.conj().copy()
        ket = self.copy()
        assert bra._nqubits == ket._nqubits
        n = bra._nqubits

        while n > 1:
            # --bA---bB
            #   |    |
            #   |    |
            # --kA---kB
            bra_A, bra_B = bra._mps.tensors[-2:]
            ket_A, ket_B = ket._mps.tensors[-2:]
            proj_B = backend.einsum("iak,jak->ij", bra_B, ket_B)
            new_kA = backend.einsum("iak,jk->iaj", ket_A, proj_B)
            bra._mps.tensors = bra._mps.tensors[:-1]
            ket._mps.tensors = ket._mps.tensors[:-1]
            ket._mps.tensors[-1] = new_kA
            n -= 1
        bra_A = bra._mps.tensors[0]
        ket_A = ket._mps.tensors[0]
        result = backend.sum(bra_A * ket_A)
        return backend.convert_to_tensor(result)

    def general_expectation(self, *ops: Tuple[Gate, List[int]]) -> Tensor:
        """
        Compute the expectation of corresponding operators in the form of tensor.

        :param ops: Operator and its position on the circuit,
            eg. ``(gates.Z(), [1]), (gates.X(), [2])`` is for operator :math:`Z_1X_2`
        :type ops: Tuple[tn.Node, List[int]]
        :return: The expectation of corresponding operators
        :rtype: Tensor
        """
        # A better idea is to create a MPO class and have a function to transform gates to MPO
        mpscircuit = self.copy()
        for gate, index in ops:
            mpscircuit.apply_general_gate(gate, *index)
        value = mpscircuit.proj_with_mps(self)
        return backend.convert_to_tensor(value)

    def expectation_single_gate(
        self,
        gate: Gate,
        site: int,
    ) -> Tensor:
        """
        Compute the expectation of the corresponding single qubit gate in the form of tensor.

        :param gate: Gate to be applied
        :type gate: Gate
        :param site: Qubit index of the gate
        :type site: int
        :return: The expectation of the corresponding single qubit gate
        :rtype: Tensor
        """
        value = self._mps.measure_local_operator([gate.tensor], [site])[0]
        return backend.convert_to_tensor(value)

    def expectation_double_gates(
        self,
        gate: Gate,
        site1: int,
        site2: int,
    ) -> Tensor:
        # TODO@(SUSYUSTC): Could be more efficient by representing distant double gates as MPO
        """
        Compute the expectation of the corresponding double qubit gate.

        :param gate: gate to be applied
        :type gate: Gate
        :param site: qubit index of the gate
        :type site: int
        """
        mps = self.copy()
        # disable truncation
        mps.set_truncation_rule()
        mps.apply_double_gate(gate, site1, site2)
        return mps.proj_with_mps(self)

    def expectation_two_gates_product(
        self, gate1: Gate, gate2: Gate, site1: int, site2: int
    ) -> Tensor:
        """
        Compute the expectation of the direct product of the corresponding two gates.

        :param gate1: First gate to be applied
        :type gate1: Gate
        :param gate2: Second gate to be applied
        :type gate2: Gate
        :param site1: Qubit index of the first gate
        :type site1: int
        :param site2: Qubit index of the second gate
        :type site2: int
        :return: The correlation of the corresponding two qubit gates
        :rtype: Tensor
        """
        value = self._mps.measure_two_body_correlator(
            gate1.tensor, gate2.tensor, site1, [site2]
        )[0]
        return backend.convert_to_tensor(value)


MPSCircuit._meta_apply()
