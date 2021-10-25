"""
quantum circuit class but with density matrix simulator: v2
"""

from functools import partial, reduce
from operator import add
from typing import Tuple, List, Callable, Union, Optional, Sequence, Any

import numpy as np
import tensornetwork as tn
import graphviz

from . import gates
from . import channels
from .cons import dtypestr, npdtype, backend, contractor
from .circuit import Circuit
from .channels import kraus_to_super_gate
from .densitymatrix import DMCircuit

Gate = gates.Gate
Tensor = Any


class DMCircuit2(DMCircuit):
    def _copy_DMCircuit(self) -> "DMCircuit2":
        newnodes, newfront = self._copy(self._nodes, self._lfront + self._rfront)
        newDMCircuit = DMCircuit2(self._nqubits, empty=True)
        newDMCircuit._nqubits = self._nqubits
        newDMCircuit._lfront = newfront[: self._nqubits]
        newDMCircuit._rfront = newfront[self._nqubits :]
        newDMCircuit._nodes = newnodes
        return newDMCircuit

    def apply_general_kraus(self, kraus: Sequence[Gate], *index: int) -> None:  # type: ignore
        # incompatible API for now
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
            super_op.get_edge(i + r2l + o2i) ^ self._lfront[ind]
            self._lfront[ind] = super_op.get_edge(i + r2l)
            super_op.get_edge(i + o2i) ^ self._rfront[ind]
            self._rfront[ind] = super_op.get_edge(i)
        self._nodes.append(super_op)
        setattr(self, "state_tensor", None)

    @staticmethod
    def apply_general_kraus_delayed(
        krausf: Callable[..., Sequence[Gate]]
    ) -> Callable[..., None]:
        def apply(self: "DMCircuit2", *index: int, **vars: float) -> None:
            kraus = krausf(**vars)
            self.apply_general_kraus(kraus, *index)

        return apply


DMCircuit2._meta_apply()
