"""
backend magic inherited from tensornetwork
"""

from typing import Union, Text, Any, Optional, Callable, Sequence
from functools import partial
from scipy.linalg import expm
import numpy as np
import warnings

from tensornetwork.backends.tensorflow import tensorflow_backend
from tensornetwork.backends.numpy import numpy_backend
from tensornetwork.backends.jax import jax_backend

# from tensornetwork.backends.shell import shell_backend
from tensornetwork.backends.pytorch import pytorch_backend

try:
    from tensornetwork.backends import base_backend

    tnbackend = base_backend.BaseBackend

except ImportError:
    from tensornetwork.backends import abstract_backend

    tnbackend = abstract_backend.AbstractBackend


Tensor = Any


libjax: Any
jnp: Any
jsp: Any
torchlib: Any
tf: Any


class NumpyBackend(numpy_backend.NumPyBackend):  # type: ignore
    def expm(self, a: Tensor) -> Tensor:
        return expm(a)

    def abs(self, a: Tensor) -> Tensor:
        return np.abs(a)

    def sin(self, a: Tensor) -> Tensor:
        return np.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return np.cos(a)

    def i(self, dtype: Any = None) -> Tensor:
        if not dtype:
            dtype = npdtype  # type: ignore
        if isinstance(dtype, str):
            dtype = getattr(np, dtype)
        return np.array(1j, dtype=dtype)

    def is_tensor(self, a: Any) -> bool:
        if isinstance(a, np.ndarray):
            return True
        return False

    def real(self, a: Tensor) -> Tensor:
        return np.real(a)

    def cast(self, a: Tensor, dtype: str) -> Tensor:
        return a.astype(getattr(np, dtype))

    def grad(self, f: Callable[..., Any]) -> Callable[..., Any]:
        raise NotImplementedError("numpy backend doesn't support AD")

    def jit(self, f: Callable[..., Any]) -> Callable[..., Any]:
        warnings.warn("numpy backend has no parallel as jit, just do nothing")
        return f
        # raise NotImplementedError("numpy backend doesn't support jit compiling")

    def vmap(self, f: Callable[..., Any]) -> Any:
        warnings.warn(
            "numpy backend has no intrinsic vmap like interface"
            ", use vectorize instead (plain for loop)"
        )
        return np.vectorize(f)


class JaxBackend(jax_backend.JaxBackend):  # type: ignore
    # Jax doesn't support 64bit dtype, unless claim
    # from jax.config import config
    # config.update("jax_enable_x64", True)
    # at very beginning, i.e. before import tensorcircuit
    def __init__(self) -> None:
        global libjax  # Jax module
        global jnp  # jax.numpy module
        global jsp  # jax.scipy module
        super(JaxBackend, self).__init__()
        try:
            import jax
        except ImportError:
            raise ImportError(
                "Jax not installed, please switch to a different "
                "backend or install Jax."
            )
        libjax = jax
        jnp = libjax.numpy
        jsp = libjax.scipy
        self.name = "jax"

    # it is already child of numpy backend, and self.np = self.jax.np

    def convert_to_tensor(self, tensor: Tensor) -> Tensor:
        result = jnp.asarray(tensor)
        return result

    def abs(self, a: Tensor) -> Tensor:
        return jnp.abs(a)

    def sin(self, a: Tensor) -> Tensor:
        return jnp.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return jnp.cos(a)

    def i(self, dtype: Any = None) -> Tensor:
        if not dtype:
            dtype = npdtype  # type: ignore
        if isinstance(dtype, str):
            dtype = getattr(jnp, dtype)
        return np.array(1j, dtype=dtype)

    def real(self, a: Tensor) -> Tensor:
        return jnp.real(a)

    def cast(self, a: Tensor, dtype: str) -> Tensor:
        return a.astype(getattr(jnp, dtype))

    def expm(self, a: Tensor) -> Tensor:
        return jsp.linalg.expm(a)
        # currently expm in jax doesn't support AD, it will raise an AssertError, see https://github.com/google/jax/issues/2645

    def is_tensor(self, a: Any) -> bool:
        if not isinstance(a, jnp.ndarray):
            return False
        # isinstance(np.eye(1), jax.numpy.ndarray) = True!
        if getattr(a, "_value", None) is not None:
            return True
        return False

    def grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Any:
        # TODO
        return libjax.grad(f, argnums=argnums)

    def jit(self, f: Callable[..., Any]) -> Any:
        return libjax.jit(f)

    def vmap(self, f: Callable[..., Any]) -> Any:
        return libjax.vmap(f)
        # since tf doesn't support in&out axes options, we don't support them in universal backend


class TensorFlowBackend(tensorflow_backend.TensorFlowBackend):  # type: ignore
    def __init__(self) -> None:
        global tf
        super(TensorFlowBackend, self).__init__()
        try:
            import tensorflow
        except ImportError:
            raise ImportError(
                "Tensorflow not installed, please switch to a "
                "different backend or install Tensorflow."
            )
        tf = tensorflow
        self.name = "tensorflow"

    def expm(self, a: Tensor) -> Tensor:
        return tf.linalg.expm(a)

    def sin(self, a: Tensor) -> Tensor:
        return tf.math.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return tf.math.cos(a)

    def i(self, dtype: Any = None) -> Tensor:
        if not dtype:
            dtype = getattr(tf, dtypestr)  # type: ignore
        if isinstance(dtype, str):
            dtype = getattr(tf, dtype)
        return tf.constant(1j, dtype=dtype)

    def is_tensor(self, a: Any) -> bool:
        if isinstance(a, tf.Tensor) or isinstance(a, tf.Variable):
            return True
        return False

    def abs(self, a: Tensor) -> Tensor:
        return tf.math.abs(a)

    def real(self, a: Tensor) -> Tensor:
        return tf.math.real(a)

    def cast(self, a: Tensor, dtype: str) -> Tensor:
        return tf.cast(a, dtype=getattr(tf, dtype))

    def grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Any]:
        # experimental attempt
        # Note: tensorflow grad is gradient while jax grad is derivative, they are different with a conjugate!
        def wrapper(*args: Any, **kws: Any) -> Any:
            with tf.GradientTape() as t:
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
        return tf.function(f)

    def vmap(self, f: Callable[..., Any]) -> Any:
        def wrapper(f: Callable[..., Any], args: Sequence[Any]) -> Any:
            return f(*args)

        wrapper = partial(wrapper, f)

        def own_vectorized_map(f: Callable[..., Any], *args: Any) -> Any:
            return tf.vectorized_map(f, args)

        return partial(own_vectorized_map, wrapper)


class PyTorchBackend(pytorch_backend.PyTorchBackend):  # type: ignore
    def __init__(self) -> None:
        super(PyTorchBackend, self).__init__()
        global torchlib
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch not installed, please switch to a different "
                "backend or install PyTorch."
            )
        torchlib = torch
        self.name = "pytorch"

    def expm(self, a: Tensor) -> Tensor:
        raise NotImplementedError("pytorch backend doesn't support expm")
        # in 2020, torch has no expm, hmmm. but that's ok, it doesn't support complex numbers which is more severe issue.
        # see https://github.com/pytorch/pytorch/issues/9983

    def sin(self, a: Tensor) -> Tensor:
        return torchlib.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return torchlib.cos(a)

    def i(self, dtype: Any = None) -> Tensor:
        raise NotImplementedError(
            "pytorch backend doesn't support imaginary numbers at all!"
        )

    def real(self, a: Tensor) -> Tensor:
        return a
        # hmm, in torch, everyone is real.

    def is_tensor(self, a: Any) -> bool:
        if isinstance(a, torchlib.Tensor):
            return True
        return False

    def cast(self, a: Tensor, dtype: str) -> Tensor:
        return a.type(getattr(torchlib, dtype))

    def grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Any]:
        def wrapper(*args: Any, **kws: Any) -> Any:
            x = []
            if isinstance(argnums, int):
                argnumsl = [argnums]
                # if you also call lhs as argnums, something weird may happen
                # the reason is that python then take it as local vars
            else:
                argnumsl = argnums  # type: ignore
            for i, arg in enumerate(args):
                if i in argnumsl:
                    x.append(arg.requires_grad_(True))
                else:
                    x.append(arg)
            y = f(*x, **kws)
            y.backward()
            gs = [x[i].grad for i in argnumsl]
            if len(gs) == 1:
                gs = gs[0]
            return gs

        return wrapper

    def vmap(self, f: Callable[..., Any]) -> Any:
        warnings.warn(
            "pytorch backend has no intrinsic vmap like interface"
            ", use plain for loop for compatibility"
        )
        # the vmap support is vey limited, f must return one tensor
        # nested list of tensor as return is not supported
        def vmapf(*args: Tensor, **kws: Any) -> Tensor:
            r = []
            for i in range(args[0].shape[0]):
                nargs = [arg[i] for arg in args]
                r.append(f(*nargs, **kws))
            return torchlib.stack(r)

        return vmapf
        # raise NotImplementedError("pytorch backend doesn't support vmap")
        # There seems to be no map like architecture in pytorch for now
        # see https://discuss.pytorch.org/t/fast-way-to-use-map-in-pytorch/70814

    def jit(self, f: Callable[..., Any]) -> Any:
        return f  # do nothing here until I figure out what torch.jit is for and how does it work
        # see https://github.com/pytorch/pytorch/issues/36910


_BACKENDS = {
    "tensorflow": TensorFlowBackend,
    "numpy": NumpyBackend,
    "jax": JaxBackend,
    # "shell": shell_backend.ShellBackend,  # no intention to maintain this one
    "pytorch": PyTorchBackend,  # no intention to fully maintain this one
}


def get_backend(backend: Union[Text, tnbackend]) -> Any:  # type: ignore
    if isinstance(backend, tnbackend):
        return backend
    if backend not in _BACKENDS:
        raise ValueError("Backend '{}' does not exist".format(backend))
    return _BACKENDS[backend]()
