"""
backend magic inherited from tensornetwork
"""

from typing import Union, Text, Any, Optional, Callable, Sequence
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

    def cast(self, a: Tensor, dtype: str) -> Tensor:
        return a.astype(getattr(self.np, dtype))

    def grad(self, f: Callable[..., Any]) -> Callable[..., Any]:
        raise NotImplementedError("numpy backend doesn't support AD")

    def jit(self, f: Callable[..., Any]) -> Callable[..., Any]:
        raise NotImplementedError("numpy backend doesn't support jit compiling")


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
        # currently expm in jax doesn't support AD, it will raise an AssertError, see https://github.com/google/jax/issues/2645

    def is_tensor(self, a: Any) -> bool:
        if not isinstance(a, self.np.ndarray):
            return False
        # isinstance(np.eye(1), jax.numpy.ndarray) = True!
        if getattr(a, "_value", None):
            return True
        return False

    def grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Any:
        # TODO
        return self.jax.grad(f, argnums=argnums)

    def jit(self, f: Callable[..., Any]) -> Any:
        return self.jax.jit(f)


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
        if isinstance(a, self.tf.Tensor) or isinstance(a, self.tf.Variable):
            return True
        return False

    def real(self, a: Tensor) -> Tensor:
        return self.tf.math.real(a)

    def cast(self, a: Tensor, dtype: str) -> Tensor:
        return self.tf.cast(a, dtype=getattr(self.tf, dtype))

    def grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Any]:
        # experimental attempt
        # Note: tensorflow grad is gradient while jax grad is derivative, they are different with a conjugate!
        def wrapper(*args: Any, **kws: Any) -> Any:
            with self.tf.GradientTape() as t:
                t.watch(args)
                y = f(*args, **kws)
                if isinstance(argnums, int):
                    x = args[argnums]
                else:
                    x = [args[i] for i in argnums]
                g = t.gradient(y, x)
            return g

        return wrapper

    def jit(self, f: Callable[..., Any]) -> Any:
        return self.tf.function(f)


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
