from typing import Union, Text, Any, Optional
from scipy.linalg import expm

from tensornetwork.backends.tensorflow import tensorflow_backend
from tensornetwork.backends.numpy import numpy_backend
from tensornetwork.backends.jax import jax_backend
from tensornetwork.backends.shell import shell_backend
from tensornetwork.backends.pytorch import pytorch_backend
from tensornetwork.backends import base_backend

Tensor = Any


class NumpyBackend(numpy_backend.NumPyBackend):  # type: ignore
    def expm(self, a: Tensor) -> Tensor:
        return expm(a)

    def sin(self, a: Tensor) -> Tensor:
        return self.np.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return self.np.cos(a)


class JaxBackend(NumpyBackend, jax_backend.JaxBackend):  # type: ignore
    def __init__(self) -> None:
        super(JaxBackend, self).__init__()
        try:
            import jax
        except ImportError:
            raise ImportError(
                "Jax not installed, please switch to a different "
                "backend or install Jax."
            )
        self.jax = jax
        self.np = self.jax.numpy
        self.sp = self.jax.scipy
        self.name = "jax"

    # it is already child of numpy backend, and self.np = self.jax.np

    def expm(self, a: Tensor) -> Tensor:
        return self.sp.linalg.expm(a)


class TensorFlowBackend(tensorflow_backend.TensorFlowBackend):  # type: ignore
    def expm(self, a: Tensor) -> Tensor:
        return self.tf.linalg.expm(a)

    def sin(self, a: Tensor) -> Tensor:
        return self.tf.math.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return self.tf.math.cos(a)


_BACKENDS = {
    "tensorflow": TensorFlowBackend,
    "numpy": NumpyBackend,
    "jax": JaxBackend,
    "shell": shell_backend.ShellBackend,  # no intention to maintain this one
    "pytorch": pytorch_backend.PyTorchBackend,  # no intention to maintain this one
}


def get_backend(
    backend: Union[Text, base_backend.BaseBackend]
) -> base_backend.BaseBackend:
    if isinstance(backend, base_backend.BaseBackend):
        return backend
    if backend not in _BACKENDS:
        raise ValueError("Backend '{}' does not exist".format(backend))
    return _BACKENDS[backend]()
