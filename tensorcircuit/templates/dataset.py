"""
Quantum machine learning related data preprocessing and embedding
"""

from typing import Any, Optional, Sequence

from ..cons import backend
from ..gates import array_to_tensor

Tensor = Any


def amplitude_encoding(
    fig: Tensor, nqubits: int, index: Optional[Sequence[int]] = None
) -> Tensor:
    # non-batch version
    fig = backend.reshape(fig, shape=[-1])
    norm = backend.norm(fig)
    fig = fig / norm
    if backend.shape_tuple(fig)[0] < 2 ** nqubits:
        fig = backend.concat(
            [
                fig,
                backend.zeros(
                    [2 ** nqubits - backend.shape_tuple(fig)[0]], dtype=fig.dtype
                ),
            ],
        )
    if index is not None:
        index = array_to_tensor(index, dtype="int32")
        fig = backend.gather1d(fig, index)
    return fig


batched_amplitude_encoding = backend.vmap(amplitude_encoding, vectorized_argnums=0)
