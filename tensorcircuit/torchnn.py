"""
PyTorch nn Module wrapper for quantum function
"""

from typing import Any, Callable, Sequence, Tuple, Union

import torch

from .cons import backend
from .interfaces import torch_interface
from .utils import is_sequence

Tensor = Any


class QuantumNet(torch.nn.Module):
    def __init__(
        self,
        f: Callable[..., Any],
        weights_shape: Sequence[Tuple[int, ...]],
        initializer: Union[Any, Sequence[Any]] = None,
        use_vmap: bool = True,
        vectorized_argnums: Union[int, Sequence[int]] = 0,
        use_interface: bool = True,
        use_jit: bool = True,
        enable_dlpack: bool = False,
    ):
        """
        PyTorch nn Module wrapper on quantum function ``f``.

        :Example:

        .. code-block:: python

            K = tc.set_backend("tensorflow")

            n = 6
            nlayers = 2
            batch = 2

            def qpred(x, weights):
                c = tc.Circuit(n)
                for i in range(n):
                    c.rx(i, theta=x[i])
                for j in range(nlayers):
                    for i in range(n - 1):
                        c.cnot(i, i + 1)
                    for i in range(n):
                        c.rx(i, theta=weights[2 * j, i])
                        c.ry(i, theta=weights[2 * j + 1, i])
                ypred = K.stack([c.expectation_ps(x=[i]) for i in range(n)])
                ypred = K.real(ypred)
                return ypred

            ql = tc.torchnn.QuantumNet(qpred, weights_shape=[2*nlayers, n])

            ql(torch.ones([batch, n]))


        :param f: Quantum function with tensor in (input and weights) and tensor out.
        :type f: Callable[..., Any]
        :param weights_shape: list of shape tuple for different weights as the non-first parameters for ``f``
        :type weights_shape: Sequence[Tuple[int, ...]]
        :param initializer: function that gives the shape tuple returns torch tensor, defaults to None
        :type initializer: Union[Any, Sequence[Any]], optional
        :param use_vmap: whether apply vmap (batch input) on ``f``, defaults to True
        :type use_vmap: bool, optional
        :param vectorized_argnums: which position of input should be batched, need to be customized when
            multiple inputs for the torch model, defaults to be 0.
        :type vectorized_argnums: Union[int, Sequence[int]]
        :param use_interface: whether transform ``f`` with torch interface, defaults to True
        :type use_interface: bool, optional
        :param use_jit: whether jit ``f``, defaults to True
        :type use_jit: bool, optional
        :param enable_dlpack: whether enbale dlpack in interfaces, defaults to False
        :type enable_dlpack: bool, optional
        """
        super().__init__()
        if use_vmap:
            f = backend.vmap(f, vectorized_argnums=vectorized_argnums)
        if use_interface:
            f = torch_interface(f, jit=use_jit, enable_dlpack=enable_dlpack)
        self.f = f
        self.q_weights = torch.nn.ParameterList()
        if isinstance(weights_shape[0], int):
            weights_shape = [weights_shape]
        if not is_sequence(initializer):
            initializer = [initializer]
        for ws, initf in zip(weights_shape, initializer):
            if initf is None:
                initf = torch.randn
            self.q_weights.append(torch.nn.Parameter(initf(ws)))

    def forward(self, *inputs: Tensor) -> Tensor:
        ypred = self.f(*inputs, *self.q_weights)
        return ypred


TorchLayer = QuantumNet


class HardwareNet(QuantumNet):
    """
    PyTorch Layer wrapping quantum function with cloud qpu access
    (using :py:mod:`tensorcircuit.cloud` module)
    """

    def __init__(
        self,
        f: Callable[..., Any],
        weights_shape: Sequence[Tuple[int, ...]],
        initializer: Union[Any, Sequence[Any]] = None,
        use_vmap: bool = True,
    ):
        super().__init__(
            f,
            weights_shape,
            initializer,
            use_vmap=False,
            use_interface=False,
            use_jit=False,
        )
        self.batch_support = use_vmap

    def forward(self, *inputs: Tensor) -> Tensor:
        if self.batch_support:
            ypred = []
            batch = inputs[0].shape[0]
            for i in range(batch):
                inp = tuple([a[i] for a in inputs])
                ypred.append(self.f(*inp, *self.q_weights))
            ypred = torch.stack(ypred)  # type: ignore
        else:
            ypred = self.f(*inputs, *self.q_weights)
        return ypred


TorchHardwareLayer = HardwareNet
