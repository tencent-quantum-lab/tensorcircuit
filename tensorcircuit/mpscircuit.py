"""
Quantum circuit: MPS state simulator
"""
# pylint: disable=invalid-name

from functools import reduce
from posixpath import split
from tkinter import W
from typing import Any, List, Optional, Sequence, Tuple, Dict, Union
import wave

import numpy as np
import tensornetwork as tn
from tensorcircuit.quantum import QuOperator, QuVector

from . import gates
from .cons import backend, npdtype, contractor, rdtypestr, dtypestr
from .mps_base import FiniteMPS
from .abstractcircuit import AbstractCircuit

Gate = gates.Gate
Tensor = Any

# TODO(@refraction-ray): support Circuit IR for MPSCircuit

def truncation_rules(
    max_singular_values: Optional[int] = None, max_truncation_err: Optional[float] = None,
    relative: bool = False,
) -> Dict[str, Any]:
    '''
    Obtain the direcionary of truncation rules
    :param max_singular_values: The maximum number of singular values to keep.
    :type max_singular_values: int, optional
    :param max_truncation_err: The maximum allowed truncation error.
    :type max_truncation_err: float, optional
    :param relative: Multiply `max_truncation_err` with the largest singular value.
    :type relative: bool, optional
    '''
    rules = {}
    if max_singular_values is not None:
        rules['max_singular_values'] = max_singular_values
    if max_truncation_err is not None:
        rules['max_truncattion_err'] = max_truncation_err
    if relative is not None:
        rules['relative'] = relative
    return rules


def split_tensor(
    tensor: Tensor,
    center_left: bool = True,
    **rules,
) -> Tuple[Tensor, Tensor]:
    """
    Split the tensor by SVD or QR depends on whether a truncation is required.

    :param tensor: The input tensor to split.
    :type tensor: Tensor
    :param center_left: Determine the orthogonal center is on the left tensor or the right tensor.
    :type center_left: bool, optional
    :return: Two tensors after splitting
    :rtype: Tuple[Tensor, Tensor]
    """
    # The behavior is a little bit different from tn.split_node because it explicitly requires a center
    svd = len(rules) > 0
    if svd:
        U, S, VH, _ = backend.svd(tensor, **rules)
        if center_left:
            return backend.matmul(U, backend.diagflat(S)), VH
        else:
            return U, backend.matmul(backend.diagflat(S), VH)
    else:
        if center_left:
            return backend.rq(tensor)  # type: ignore
        else:
            return backend.qr(tensor)  # type: ignore


class MPSCircuit(AbstractCircuit):
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

    is_mps = True

    def __init__(
        self,
        nqubits: int,
        center_position: Optional[int] = None,
        tensors: Optional[Sequence[Tensor]] = None,
        wavefunction: Optional[Union[QuVector, Tensor]] = None,
        **rules: Dict[str, Any],
    ) -> None:
        """
        MPSCircuit object based on state simulator.

        :param nqubits: The number of qubits in the circuit.
        :type nqubits: int
        :param center_position: The center position of MPS, default to 0
        :type center_position: int, optional
        :param tensors: If not None, the initial state of the circuit is taken as ``tensors`` instead of :math:`\\vert 0\\rangle^n` qubits, defaults to None. 
        When ``tensors`` are specified, if ``center_position`` is None, then the tensors are canonicalized, otherwise it is assumed the tensors are already canonicalized at the ``center_position``
        :type tensors: Sequence[Tensor], optional
        :param wavefunction: If not None, it is transformed to the MPS form according to the truncation rules
        :type wavefunction: Tensor
        :param rules: Truncation rules
        :type rules: Dict
        """
        self.circuit_param = {
            "nqubits": nqubits,
            "center_position": center_position,
            "rules": rules,
            "tensors": tensors,
            "wavefunction": wavefunction,
        }
        self.rules = rules
        if wavefunction is not None:
            assert tensors is None, "tensors and wavefunction cannot be used at input simutaneously"
            if isinstance(wavefunction, QuVector):
                wavefunction = wavefunction.eval()
            tensors = self.wavefunction_to_tensors(wavefunction, **self.rules)
            assert len(tensors) == nqubits
            self._mps = FiniteMPS(
                tensors, canonicalize=False)
            self._mps.center_position = 0
            if center_position is not None:
                self.position(center_position)
        elif tensors is not None:
            if center_position is not None:
                self._mps = FiniteMPS(
                    tensors, canonicalize=False)
                self._mps.center_position = center_position
            else:
                self._mps = FiniteMPS(
                    tensors, canonicalize=True, center_position=0)
        else:
            tensors = [
                np.array([1.0, 0.0], dtype=npdtype)[None, :, None]
                for i in range(nqubits)
            ]
            self._mps = FiniteMPS(
                tensors, canonicalize=False)
            if center_position is not None:
                self._mps.center_position = center_position
            else:
                self._mps.center_position = 0

        self._nqubits = nqubits
        self._fidelity = 1.0
        self._qir: List[Dict[str, Any]] = []

    # `MPSCircuit` does not has `replace_inputs` like `Circuit`
    # because the gates are immediately absorted into the MPS when applied,
    # so it is impossible to remember the initial structure

    #@property
    def get_bond_dimensions(self):
        return self._mps.bond_dimensions

    #@property
    def get_tensors(self):
        return self._mps.tensors

    #@property
    def get_center_position(self):
        return self._mps.center_position
    
    def set_truncation_rule(self, **rules: Dict[str, Any]) -> None:
        """
        Set truncation rules when double qubit gates are applied.
        If nothing is specified, no truncation will take place and the bond dimension will keep growing.
        For more details, refer to `split_tensor`.

        :param rules: Truncation rules
        :type rules: Dict
        """
        self.rules = rules

    # TODO(@refraction-ray): unified split truncation API between Circuit and MPSCircuit

    def position(self, site: int) -> None:
        """
        Wrapper of tn.FiniteMPS.position.
        Set orthogonality center.

        :param site: The orthogonality center
        :type site: int
        """
        self._mps.position(site, normalize=False)

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
            **self.rules,
        )
        self._fidelity *= 1 - backend.real(backend.sum(err**2))

    def consecutive_swap(
        self,
        index_from: int,
        index_to: int,
    ) -> None:
        self.position(index_from)
        if index_from < index_to:
            for i in range(index_from, index_to):
                self.apply_adjacent_double_gate(gates.swap(), i, i+1, center_position=i+1)
        elif index_from > index_to:
            for i in range(index_from, index_to, -1):
                self.apply_adjacent_double_gate(gates.swap(), i-1, i, center_position=i-1)
        else:
            # index_from == index_to
            pass
        assert self._mps.center_position == index_to

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
        # apply N SWPA gates, the required gate, N SWAP gates sequentially on adjacent gates
        # start the swap from the side that the current center is closer to
        diff1 = abs(index1 - self._mps.center_position)  # type: ignore
        diff2 = abs(index2 - self._mps.center_position)  # type: ignore
        if diff1 < diff2:
            self.consecutive_swap(index1, index2 - 1)
            self.apply_adjacent_double_gate(
                gate, index2 - 1, index2, center_position=index2 - 1
            )
            self.consecutive_swap(index2 - 1, index1)
        else:
            self.consecutive_swap(index2, index1 + 1)
            self.apply_adjacent_double_gate(
                gate, index1, index1 + 1, center_position=index1 + 1
            )
            self.consecutive_swap(index1 + 1, index2)

    @classmethod
    def gate_to_MPO(
        cls,
        gate: Union[Gate, Tensor],
        *index: int,
        ) -> Tuple[Sequence[Tensor], int]:
        # should I put this function here?
        '''
        If sites are not adjacent, insert identities in the middle, i.e.
          |       |             |   |   |
        --A---x---B--   ->    --A---I---B--
          |       |             |   |   |
        where
             a
             |
        --i--I--j-- = \delta_{i,j} \delta_{a,b}
             |
             b
        '''
        index_left = np.min(index)
        if isinstance(gate, tn.Node):
            gate = backend.copy(gate.tensor)
        index = np.array(index) - index_left
        nindex = len(index)
        # transform gate from (in1, in2, ..., out1, out2 ...) to
        order = tuple(np.arange(2*nindex).reshape((2, nindex)).T.flatten())
        shape = (4, ) * nindex
        gate = backend.reshape(backend.transpose(gate, order), shape)
        # (in1, out1, in2, out2, ...)
        argsort = np.argsort(index)
        # reorder the gate according to the site positions
        gate = backend.transpose(gate, tuple(argsort))
        index = index[argsort]
        length = index[-1] + 1
        # split the gate into tensors assuming they are adjacent
        main_tensors = cls.wavefunction_to_tensors(gate, dim_phys=4, norm=False)
        # each tensor is in shape of (i, a, b, j)
        tensors = []
        previous_i = None
        for i, main_tensor in zip(index, main_tensors):
            # insert identites in the middle
            if previous_i is not None:
                for middle in range(previous_i + 1, i):
                    bond_dim = tensors[-1].shape[-1]
                    I = np.eye(bond_dim * 2).reshape((bond_dim, 2, bond_dim, 2)).transpose((0, 1, 3, 2)).astype(dtypestr)
                    tensors.append(backend.convert_to_tensor(I))
            nleft, _, nright = main_tensor.shape
            tensor = backend.reshape(main_tensor, (nleft, 2, 2, nright))
            tensors.append(tensor)
            previous_i = i
        return tensors, index_left
        
    @classmethod
    def MPO_to_gate(
        cls,
        tensors: Sequence[Tensor],
    ) -> Gate:
        '''
             1
             |
        --0--A--3--
             |
             2
        '''
        nodes = [tn.Node(tensor) for tensor in tensors]
        length = len(nodes)
        output_edges = [nodes[0].get_edge(0)]
        for i in range(length-1):
            nodes[i].get_edge(3) ^ nodes[i+1].get_edge(0)
        for i in range(length):
            output_edges.append(nodes[i].get_edge(1))
        for i in range(length):
            output_edges.append(nodes[i].get_edge(2))
        output_edges.append(nodes[length-1].get_edge(3))
        gate = contractor(nodes, output_edge_order=output_edges)
        gate = Gate(gate.tensor[0, ..., 0])
        return gate
        
    @classmethod
    def reduce_tensor_dimension(
        cls,
        tensor_left: Tensor, 
        tensor_right: Tensor, 
        center_left: bool=True,
        **rules: Dict[str, Any],
        ):
        ni = tensor_left.shape[0]
        nk = tensor_right.shape[-1]
        T = backend.einsum("iaj,jbk->iabk", tensor_left, tensor_right)
        T = backend.reshape(T, (ni * 2, nk * 2))
        new_tensor_left, new_tensor_right = split_tensor(T, center_left=center_left, **rules)
        new_tensor_left = backend.reshape(new_tensor_left, (ni, 2, -1))
        new_tensor_right = backend.reshape(new_tensor_right, (-1, 2, nk))
        return new_tensor_left, new_tensor_right

    def reduce_dimension(
        self, 
        index_left: int, 
        index_right: int, 
        center_left: bool=True,
        ):
        if index_left > index_right:
            self.reduce_dimension(index_right, index_left, center_left=center_left)
            return
        assert index_left + 1 == index_right
        assert self._mps.center_position in [index_left, index_right]
        tensor_left = self._mps.tensors[index_left]
        tensor_right = self._mps.tensors[index_right]
        new_tensor_left, new_tensor_right = self.reduce_tensor_dimension(tensor_left, tensor_right, center_left=center_left, **self.rules)
        self._mps.tensors[index_left] = new_tensor_left
        self._mps.tensors[index_right] = new_tensor_right
        if center_left:
            self._mps.center_position = index_left
        else:
            self._mps.center_position = index_right

    def apply_MPO(self, 
        tensors: Sequence[Tensor],
        index_left: int,
        center_left: bool = True,
        ) -> None:
        '''
        O means MPO operator, T means MPS tensor
        step 1:
            contract tensor
                  a
                  |
            i-----O-----j            a
                  |        ->        |
                  b             ik---X---jl
                  |   
            k-----T-----l
        step 2:
            canonicalize the tensors one by one from end1 to end2
        setp 3:
            reduce the bond dimension one by one from end2 to end1
        '''
        nindex = len(tensors)
        index_right = index_left + nindex - 1
        if center_left:
            end1 = index_left
            end2 = index_right
            step = 1
        else:
            end1 = index_right
            end2 = index_left
            step = -1
        idx_list = np.arange(index_left, index_right + 1)[::step]
        i_list = np.arange(nindex)[::step]

        self.position(end1)
        # contract
        for i, idx in zip(i_list, idx_list):
            O = tensors[i]
            T = self._mps.tensors[idx]
            ni, _, _, nj = O.shape
            nk, _, nl = T.shape
            OT = backend.einsum("iabj,kbl->ikajl", O, T)
            OT = backend.reshape(OT, (ni*nk, 2, nj*nl))
            self._mps.tensors[idx] = OT

        # canonicalize
        # FiniteMPS.position applies QR sequentially from index_left to index_right
        self.position(end2)

        # reduce bond dimension
        for i in idx_list[::-1][:-1]:
            self.reduce_dimension(i, i - step, center_left=center_left)
        assert self._mps.center_position == end1

    def apply_nqubit_gate(
        self,
        gate: Gate,
        *index: int,
    ) -> None:
        MPO, index_left = self.gate_to_MPO(gate, *index)
        index_right = index_left + len(MPO) - 1
        # start the MPO from the side that the current center is closer to
        diff_left = abs(index_left - self._mps.center_position)  # type: ignore
        diff_right = abs(index_right - self._mps.center_position)  # type: ignore
        self.apply_MPO(MPO, index_left, center_left=diff_left < diff_right)
        #self.apply_MPO(MPO, index_left, center_left=False)

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
        Apply a general qubit gate on MPS.

        :param gate: The Gate to be applied
        :type gate: Gate
        :raises ValueError: "MPS does not support application of gate on > 2 qubits."
        :param index: Qubit indices of the gate
        :type index: int
        """
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
        assert split is None, "MPS does not support gate split"
        assert mpo is False, "MPO not implemented for MPS"
        assert isinstance(gate, tn.Node)
        noe = len(index)
        if noe == 1:
            self.apply_single_gate(gate, *index)
        elif noe == 2:
            self.apply_double_gate(gate, *index)
        else:
            self.apply_nqubit_gate(gate, *index)
    apply = apply_general_gate

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
        self._mps.tensors[index] = self._mps.tensors[index][:, keep, :][:, None, :]

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
    def wavefunction_to_tensors(
        wavefunction: Tensor,
        dim_phys: int = 2,
        norm: bool = True,
        **rules: Dict[str, Any],
    ) -> List[Tensor]:
        """
        Construct the MPS tensors from a given wavefunction.

        :param wavefunction: The given wavefunction (any shape is OK)
        :type wavefunction: Tensor
        :param rules: Truncation rules
        :type rules: Dict
        :param dim_phys: Physical dimension, 2 for MPS and 4 for MPO
        :type dim_phys: int
        :param norm: Whether to normalize the wavefunction
        :type norm: bool
        :return: The tensors
        :rtype: List[Tensor]
        """
        wavefunction = backend.reshape(wavefunction, (-1, 1))
        tensors: List[Tensor] = []
        while True:  # not jittable
            nright = wavefunction.shape[1]
            wavefunction = backend.reshape(wavefunction, (-1, nright * dim_phys))
            wavefunction, Q = split_tensor(
                wavefunction,
                center_left=True,
                **rules,
            )
            tensors.insert(0, backend.reshape(Q, (-1, dim_phys, nright)))
            if wavefunction.shape == (1, 1):
                break
        if not norm:
            tensors[-1] *= wavefunction[0, 0]
        return tensors

    def wavefunction(self, form: str = "default") -> Tensor:
        """
        Compute the output wavefunction from the circuit.

        :param form: the str indicating the form of the output wavefunction
        :type form: str, optional
        :return: Tensor with shape [1, -1]
        :rtype: Tensor
           a  b           ab
           |  |           ||
        i--A--B--j  -> i--XX--j
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
        from copy import copy
        result: "MPSCircuit" = MPSCircuit.__new__(MPSCircuit)
        info = vars(self)
        for key in vars(self):
            if key == "_mps":
                continue
            if backend.is_tensor(info[key]):
                copied_value = backend.copy(info[key])
            else:
                copied_value = copy(info[key])
            setattr(result, key, copied_value)
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
        return backend.norm(self._mps.tensors[self._mps.center_position])

    def normalize(self) -> None:
        """
        Normalize MPS Circuit according to the center position.
        """
        norm = self.get_norm()
        self._mps.tensors[self._mps.center_position] /= norm

    def amplitude(self, l: str) -> Tensor:
        assert len(l) == self._nqubits
        tensors = [self._mps.tensors[i][:, int(s), :] for i, s in enumerate(l)]
        return reduce(backend.matmul, tensors)[0, 0]

    def proj_with_mps(self, other: "MPSCircuit", conj=True) -> Tensor:
        """
        Compute the projection between `other` as bra and `self` as ket.

        :param other: ket of the other MPS, which will be converted to bra automatically
        :type other: MPSCircuit
        :return: The projection in form of tensor
        :rtype: Tensor
        """
        if conj:
            bra = other.conj().copy()
        else:
            bra = other.copy()
        ket = self.copy()
        assert bra._nqubits == ket._nqubits
        n = bra._nqubits

        while n > 1:
            '''
             i--bA--k--bB---m
                 |      |   |
                 a      b   |
                 |      |   |
             j--kA--l--kB---m
            '''
            bra_A, bra_B = bra._mps.tensors[-2:]
            ket_A, ket_B = ket._mps.tensors[-2:]
            proj_B = backend.einsum("kbm,lbm->kl", bra_B, ket_B)
            new_kA = backend.einsum("jal,kl->jak", ket_A, proj_B)
            bra._mps.tensors = bra._mps.tensors[:-1]
            ket._mps.tensors = ket._mps.tensors[:-1]
            ket._mps.tensors[-1] = new_kA
            n -= 1
        bra_A = bra._mps.tensors[0]
        ket_A = ket._mps.tensors[0]
        result = backend.sum(bra_A * ket_A)
        return backend.convert_to_tensor(result)

    def slice(self, begin: int, end: int) -> "MPSCircuit":
        nqubits = end - begin + 1
        tensors = [backend.copy(t) for t in self._mps.tensors[begin:end+1]]
        mps = self.__class__(nqubits, tensors=tensors, center_position=self._mps.center_position, **self.rules)
        return mps
        
    def expectation(
        self, 
        *ops: Tuple[Gate, List[int]], 
        other: Optional["MPSCircuit"] = None, 
        conj=True,
        normalize=False,
        **rules: Dict[str, Any],
        ) -> Tensor:
        """
        Compute the expectation of corresponding operators in the form of tensor.

        :param ops: Operator and its position on the circuit,
            eg. ``(gates.Z(), [1]), (gates.X(), [2])`` is for operator :math:`Z_1X_2`
        :type ops: Tuple[tn.Node, List[int]]
        :return: The expectation of corresponding operators
        :rtype: Tensor
        """
        # If the bra is ket itself, the environments outside the operators can be viewed as identities so does not need to contract
        ops = [list(op) for op in ops] # turn to list for modification
        for op in ops:
            if isinstance(op[1], int):
                op[1] = [op[1]]
        all_sites = np.concatenate([op[1] for op in ops])

        # move position inside the operator range
        if other is None:
            site_begin = np.min(all_sites)
            site_end = np.max(all_sites)
            if self._mps.center_position < site_begin:
                self.position(site_begin)
            elif self._mps.center_position > site_end:
                self.position(site_end)
        else:
            assert isinstance(other, MPSCircuit), "the bra has to be a MPSCircuit"

        # apply the gate
        mps = self.copy()
        # when the truncation rules is None (which is the default), it is guaranteed that the result is a real number since the calculation is exact, otherwise there's no guarantee
        mps.set_truncation_rule(**rules)
        for gate, index in ops:
            mps.apply(gate, *index)

        if other is None:
            assert (self._mps.center_position >= site_begin) and (self._mps.center_position <= site_end)
            ket = mps.slice(site_begin, site_end)
            bra = self.slice(site_begin, site_end)
        else:
            ket = mps
            bra = other
        value = ket.proj_with_mps(bra, conj=conj)
        value = backend.convert_to_tensor(value)

        if normalize:
            norm1 = self.get_norm()
            if other is None:
                norm2 = norm1
            else:
                norm2 = other.get_norm()
            norm = backend.sqrt(norm1 * norm2)
            value /= norm
        return value

    def get_quvector(self) -> QuVector:
        return QuVector.from_tensor(backend.reshape2(self.wavefunction()))
    
    def measure(
        self,
        *index: int,
        with_prob: bool = False,
        status: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Take measurement to the given quantum lines.

        :param index: Measure on which quantum line.
        :type index: int
        :param with_prob: If true, theoretical probability is also returned.
        :type with_prob: bool, optional
        :param status: external randomness, with shape [index], defaults to None
        :type status: Optional[Tensor]
        :return: The sample output and probability (optional) of the quantum line.
        :rtype: Tuple[Tensor, Tensor]
        """
        is_sorted = np.all(np.sort(index) == np.array(index))
        if not is_sorted:
            order = backend.convert_to_tensor(np.argsort(index).tolist())
            sample, p = self.measure(*np.sort(index), with_prob=with_prob, status=status)
            return backend.convert_to_tensor([sample[i] for i in order]), p
        # set the center to the left side, then gradually move to the right and do measurement at sites
        mps = self.copy()
        up = backend.convert_to_tensor(np.array([1, 0]).astype(dtypestr))
        down = backend.convert_to_tensor(np.array([0, 1]).astype(dtypestr))

        p = 1.0
        p = backend.convert_to_tensor(p)
        p = backend.cast(p, dtype=rdtypestr)
        sample = []
        for k, site in enumerate(index):
            mps.position(site)
            # do measurement
            tensor = mps._mps.tensors[site]
            ps = backend.real(backend.einsum("iaj,iaj->a", tensor, backend.conj(tensor)))
            ps /= backend.sum(ps)
            pu = ps[0]
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
            m = (1 - sign_complex) * up + sign_complex * down
            mps._mps.tensors[site] = backend.einsum("iaj,a->ij", tensor, m)[:, None, :]
        sample = backend.stack(sample)
        sample = backend.real(sample)
        if with_prob:
            return sample, p
        else:
            return sample, -1.0


MPSCircuit._meta_apply()
