"""
PyTorch nn Module wrapper for quantum function
"""

from typing import Any, Callable, Sequence, Tuple, Union

import torch

from .cons import backend
from .interfaces import torch_interface, is_sequence

Tensor = Any


class QuantumNet(torch.nn.Module):  # type: ignore
    def __init__(
        self,
        f: Callable[..., Any],
        weights_shape: Sequence[Tuple[int, ...]],
        initializer: Union[Any, Sequence[Any]] = None,
        use_vmap: bool = True,
        use_interface: bool = True,
        use_jit: bool = True,
    ):
        super().__init__()
        if use_vmap:
            f = backend.vmap(f, vectorized_argnums=0)
        if use_interface:
            f = torch_interface(f, jit=use_jit)
        self.f = f
        self.q_weights = []
        if isinstance(weights_shape[0], int):
            weights_shape = [weights_shape]
        if not is_sequence(initializer):
            initializer = [initializer]
        for ws, initf in zip(weights_shape, initializer):
            if initf is None:
                initf = torch.randn
            self.q_weights.append(torch.nn.Parameter(initf(ws)))  # type: ignore

    def forward(self, inputs: Tensor) -> Tensor:
        ypred = self.f(inputs, *self.q_weights)
        return ypred
