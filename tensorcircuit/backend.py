"""
backend magic inherited from tensornetwork
"""

from typing import Union, Text, Any, Optional, Callable
from scipy.linalg import expm
import numpy as np

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

    def i(self, dtype: Any = None) -> Tensor:
        if not dtype:
            dtype = npdtype  # type: ignore
        return self.np.array(1j, dtype=dtype)

    def is_tensor(self, a: Any) -> bool:
        if isinstance(a, self.np.ndarray):
            return True
        return False

    def real(self, a: Tensor) -> Tensor:
        return self.np.real(a)

    def grad(self, f: Callable[..., Any]) -> Callable[..., Any]:
        raise NotImplementedError("numpy backend doesn't support AD")


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

    def convert_to_tensor(self, tensor: Tensor) -> Tensor:
        return tensor

    # it is already child of numpy backend, and self.np = self.jax.np

    def expm(self, a: Tensor) -> Tensor:
        return self.sp.linalg.expm(a)
        # currently expm in jax doesn't support AD, it will raise an AssertError, see https://github.com/google/jax/issues/2645

    def is_tensor(self, a: Any) -> bool:
        if not isinstance(a, self.np.ndarray):
            return False
        # isinstance(np.eye(1), jax.numpy.ndarray) = True!
        if getattr(a, "_value", None):
            return True
        return False

    def grad(self, f: Callable[..., Any]) -> Any:
        # TODO
        return self.jax.grad(f)


class TensorFlowBackend(tensorflow_backend.TensorFlowBackend):  # type: ignore
    def __init__(self) -> None:
        super(TensorFlowBackend, self).__init__()
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "Tensorflow not installed, please switch to a "
                "different backend or install Tensorflow."
            )
        self.tf = tf
        self.name = "tensorflow"
        self.np = np

    def expm(self, a: Tensor) -> Tensor:
        return self.tf.linalg.expm(a)

    def sin(self, a: Tensor) -> Tensor:
        return self.tf.math.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return self.tf.math.cos(a)

    def i(self, dtype: Any = None) -> Tensor:
        if not dtype:
            dtype = tfdtype  # type: ignore
        return self.tf.constant(1j, dtype=dtype)

    def is_tensor(self, a: Any) -> bool:
        if isinstance(a, self.tf.Tensor):
            return True
        return False

    def real(self, a: Tensor) -> Tensor:
        return self.tf.math.real(a)

    def grad(self, f: Callable[..., Any]) -> Callable[..., Any]:
        # experimental attempt [WIP]
        # TODO
        def wrapper(*args: Any, **kws: Any) -> Any:
            with self.tf.GradientTape() as t:
                t.watch(*args)
                y = f(*args, **kws)
                g = t.gradient(y, args)
            return g

        return wrapper


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
