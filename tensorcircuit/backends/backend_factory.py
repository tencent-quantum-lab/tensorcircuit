"""
Backend register
"""

from typing import Any, Dict, Text, Union

import tensornetwork as tn

try:  # old version tn compatiblity
    from tensornetwork.backends import base_backend

    tnbackend = base_backend.BaseBackend

except ImportError:
    from tensornetwork.backends import abstract_backend

    tnbackend = abstract_backend.AbstractBackend

from .numpy_backend import NumpyBackend
from .jax_backend import JaxBackend
from .tensorflow_backend import TensorFlowBackend
from .pytorch_backend import PyTorchBackend
from .cupy_backend import CuPyBackend

bk = Any  # tnbackend

_BACKENDS = {
    "numpy": NumpyBackend,
    "jax": JaxBackend,
    "tensorflow": TensorFlowBackend,
    "pytorch": PyTorchBackend,  # no intention to fully maintain this one
    "cupy": CuPyBackend,  # no intention to fully maintain this one
}

tn.backends.backend_factory._BACKENDS["cupy"] = CuPyBackend

_INSTANTIATED_BACKENDS: Dict[str, bk] = dict()


def get_backend(backend: Union[Text, bk]) -> bk:
    """
    Get the `tc.backend` object.

    :param backend: "numpy", "tensorflow", "jax", "pytorch"
    :type backend: Union[Text, tnbackend]
    :raises ValueError: Backend doesn't exist for `backend` argument.
    :return: The `tc.backend` object that with all registered universal functions.
    :rtype: backend object
    """
    if isinstance(backend, tnbackend):
        return backend
    backend = backend.lower()
    if backend not in _BACKENDS:
        raise ValueError("Backend '{}' does not exist".format(backend))
    if backend in _INSTANTIATED_BACKENDS:
        return _INSTANTIATED_BACKENDS[backend]
    _INSTANTIATED_BACKENDS[backend] = _BACKENDS[backend]()

    return _INSTANTIATED_BACKENDS[backend]
