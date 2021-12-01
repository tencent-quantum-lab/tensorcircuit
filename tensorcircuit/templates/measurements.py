"""
shortcuts for measurement patterns on circuit
"""

from typing import Any

from ..circuit import Circuit
from ..cons import backend, dtypestr
from .. import gates

Tensor = Any


def any_measurements(c: Circuit, structures: Tensor, onehot: bool = False) -> Tensor:
    """
    This measurements pattern is specifically suitable for vmap. Parameterize the Pauli string
    to be measured

    :param c: [description]
    :type c: Circuit
    :param structures: parameter tensors determines what Pauli string to be measured,
        shape is [nwires, 4] if onehot is False.
    :type structures: Tensor
    :param onehot: [description], defaults to False
    :type onehot: bool, optional
    :return: [description]
    :rtype: Tensor
    """
    if onehot is True:
        structuresc = backend.cast(structures, dtype="int32")
        structuresc = backend.onehot(structuresc, num=4)
        structuresc = backend.cast(structuresc, dtype=dtypestr)
    else:
        structuresc = structures
    nwires = c._nqubits
    obs = []
    for i in range(nwires):
        obs.append(
            [
                gates.Gate(
                    sum(
                        [
                            structuresc[i, k] * g.tensor
                            for k, g in enumerate(gates.pauli_gates)
                        ]
                    )
                ),
                (i,),
            ]
        )
    loss = c.expectation(*obs, reuse=False)  # type: ignore
    # TODO(@refraction-ray): is reuse=True in this setup has user case?
    return backend.real(loss)
