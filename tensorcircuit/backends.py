"""
backend magic inherited from tensornetwork
"""

import inspect
import warnings
from functools import partial
from typing import Union, Text, Any, Optional, Callable, Sequence, Tuple

import numpy as np
import tensornetwork
from scipy.linalg import expm
from tensornetwork.backends.tensorflow import tensorflow_backend
from tensornetwork.backends.numpy import numpy_backend
from tensornetwork.backends.jax import jax_backend
from tensornetwork.backends.pytorch import pytorch_backend

try:  # old version
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


def _doc_string_for_backend(tnbackend: Any) -> None:
    def expm(self: Any, a: Tensor) -> Tensor:
        """
        Return expm of ``a``, matrix exponential.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: matrix exponential of matrix ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `expm`.".format(self.name)
        )

    def sin(self: Any, a: Tensor) -> Tensor:
        """
        Return sin of ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: sin of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `sin`.".format(self.name)
        )

    def cos(self: Any, a: Tensor) -> Tensor:
        """
        Return cos of ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: cos of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `cos`.".format(self.name)
        )

    def abs(self: Any, a: Tensor) -> Tensor:
        """
        Return elementwise abs value of ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: abs of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `abs`.".format(self.name)
        )

    # TODO(@refraction-ray): abs docstring doesn't get registered in the doc

    def kron(self: Any, a: Tensor, b: Tensor) -> Tensor:
        """
        Return kronecker product of two matrix ``a`` and ``b``.

        :param a: tensor in matrix form
        :type a: Tensor
        :param b: tensor in matrix form
        :type b: Tensor
        :return: kronecker product of ``a`` and ``b``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `kron`.".format(self.name)
        )

    def size(self: Any, a: Tensor) -> Tensor:
        """
        Return the total number of elements in ``a``.

        :param a: tensor
        :type a: Tensor
        :return: the total number of elements in ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `size`.".format(self.name)
        )

    def numpy(self: Any, a: Tensor) -> Tensor:
        """
        Return numpy array of tensor ``a``, may not work in jitted function.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: numpy array of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `numpy`.".format(self.name)
        )

    def real(self: Any, a: Tensor) -> Tensor:
        """
        Return elementwise real value of ``a``.

        :param a: tensor
        :type a: Tensor
        :return: real value of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `real`.".format(self.name)
        )

    def i(self: Any, dtype: str) -> Tensor:
        """
        Return 1.j in as tensor comoatible with backend.

        :param dtype: "complex64" or "complex128"
        :type dtype: str
        :return: 1.j tensor
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `i`.".format(self.name)
        )

    def is_tensor(self: Any, a: Tensor) -> bool:
        """
        Return boolean on whether ``a`` is a tensor in backend package.

        :param a: a tensor to be determined
        :type a: Tensor
        :return: whether ``a`` is a tensor
        :rtype: bool
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `is_tensor`.".format(self.name)
        )

    def cast(self: Any, a: Tensor, dtype: str) -> Tensor:
        """
        Cast the tensor dtype of a ``a``.

        :param a: tensor
        :type a: Tensor
        :param dtype: "float32", "float64", "complex64", "complex128"
        :type dtype: str
        :return: ``a`` of new dtype
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `cast`.".format(self.name)
        )

    def grad(
        self: Any, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Any]:
        """
        Return function which is the grad function of input ``f``

        .. code-block:: python

            f = lambda x,y: x**2+2*y
            g = tc.backend.grad(f)
            g(tc.num_to_tensor(1),tc.num_to_tensor(2)) # return 2
            g = tc.backend.grad(f, argnums=(0,1))
            g(tc.num_to_tensor(1),tc.num_to_tensor(2)) # return [2, 2]


        :param f: function to be differentiated
        :type f: Callable[..., Any]
        :param argnums: the position of args in ``f`` that are to be differentiated, defaults to 0
        :type argnums: Union[int, Sequence[int]], optional
        :return: the grad fuction of ``f`` with the same set of arguments as ``f``
        :rtype: Callable[..., Any]
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `grad`.".format(self.name)
        )

    def value_and_grad(
        self: Any, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Tuple[Any, Any]]:
        """
        Return function which returns the value and grad of ``f``

        .. code-block:: python

            f = lambda x,y: x**2+2*y
            g = tc.backend.value_and_grad(f)
            g(tc.num_to_tensor(1),tc.num_to_tensor(2)) # return 5, 2
            g = tc.backend.value_and_grad(f, argnums=(0,1))
            g(tc.num_to_tensor(1),tc.num_to_tensor(2)) # return 5, [2, 2]


        :param f: function to be differentiated
        :type f: Callable[..., Any]
        :param argnums: the position of args in ``f`` that are to be differentiated, defaults to 0
        :type argnums: Union[int, Sequence[int]], optional
        :return: the value and grad fuction of ``f`` with the same set of arguments as ``f``
        :rtype: Callable[..., Tuple[Any, Any]]
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `value_and_grad`.".format(self.name)
        )

    def jit(self: Any, f: Callable[..., Any]) -> Callable[..., Any]:
        """
        Return jitted version function of ``f``

        :param f: function to be jitted
        :type f: Callable[..., Any]
        :return: jitted ``f``
        :rtype: Callable[..., Any]
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `jit`.".format(self.name)
        )

    def vmap(self: Any, f: Callable[..., Any]) -> Any:
        """
        Return vectorized map or batched version of ``f`` on the first extra axis,
        the general interface only support ``f`` with one argument and broadcast in the fist dimension.

        :param f: function to be broadcasted.
        :type f: Callable[..., Any]
        :return: vmap version of ``f``
        :rtype: Any
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `vmap`.".format(self.name)
        )

    def vectorized_value_and_grad(
        self: Any, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Tuple[Any, Any]]:
        """
        Return vvag function of ``f``. the inputs for ``f`` is (args[0], args[1], args[2], ...),
        and the output of ``f`` is a scalar. Suppose vvag(f) is a function with inputs in the form
        (vargs[0], args[1], args[2], ...), where vagrs[0] has one extra dimension than args[0] in the first axis
        and consistent with args[0] in shape for remaining dimensions, i.e. shape(vargs[0]) = [batch] + shape(args[0]).
        vvag(f) returns a tuple as a value tensor with shape [batch, 1] and a gradient tuple with shape:
        ([batch]+shape(args[argnum]) for argnum in argnums). The gradient for argnums=k is defined as

        .. math::

            g^k = \\frac{\\partial \\sum_{i\\in batch} f(vargs[0][i], args[1], ...)}{\\partial args[k]}

        Therefore, if argnums=0, the gradient is reduced to

        .. math::

            g^0_i = \\frac{\\partial f(vargs[0][i])}{\\partial vargs[0][i]}

        , which is specifically suitable for batched VQE optimization, where args[0] is the circuit parameters.

        And if argnums=1, the gradient is like

        .. math::
            g^1_i = \\frac{\\partial \sum_j f(vargs[0][j], args[1])}{\\partial args[1][i]}

        , which is suitable for quantum machine learning scenarios, where ``f`` is the loss function,
        args[0] corresponds the input data and args[1] corresponds to the weights in the QML model.

        :param f: [description]
        :type f: Callable[..., Any]
        :param argnums: [description], defaults to 0
        :type argnums: Union[int, Sequence[int]], optional
        :return: [description]
        :rtype: Callable[..., Tuple[Any, Any]]
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `vectorized_value_and_grad`.".format(
                self.name
            )
        )

    r = inspect.stack()
    d = r[0].frame.f_locals
    for k, v in d.items():
        if k != "r":
            setattr(tnbackend, k, v)


_doc_string_for_backend(tnbackend)


class NumpyBackend(numpy_backend.NumPyBackend):  # type: ignore
    def expm(self, a: Tensor) -> Tensor:
        return expm(a)

    def abs(self, a: Tensor) -> Tensor:
        return np.abs(a)

    def sin(self, a: Tensor) -> Tensor:
        return np.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return np.cos(a)

    def size(self, a: Tensor) -> Tensor:
        return a.size

    def kron(self, a: Tensor, b: Tensor) -> Tensor:
        return np.kron(a, b)

    def numpy(self, a: Tensor) -> Tensor:
        return a

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

    def grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Any]:
        raise NotImplementedError("numpy backend doesn't support AD")

    def value_and_grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Tuple[Any, Any]]:
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

    def vectorized_value_and_grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Tuple[Any, Any]]:
        raise NotImplementedError("numpy backend doesn't support AD")

    vvag = vectorized_value_and_grad


def _convert_to_tensor_jax(self: Any, tensor: Tensor) -> Tensor:
    if not isinstance(tensor, (np.ndarray, jnp.ndarray)) and not jnp.isscalar(tensor):
        raise TypeError(
            ("Expected a `jnp.array`, `np.array` or scalar. " f"Got {type(tensor)}")
        )
    result = jnp.asarray(tensor)
    return result


tensornetwork.backends.jax.jax_backend.JaxBackend.convert_to_tensor = (
    _convert_to_tensor_jax
)


class JaxBackend(jax_backend.JaxBackend):  # type: ignore
    # Jax doesn't support 64bit dtype, unless claim
    # ``from jax.config import config```
    # ``config.update("jax_enable_x64", True)``
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

    def size(self, a: Tensor) -> Tensor:
        return jnp.size(a)

    def kron(self, a: Tensor, b: Tensor) -> Tensor:
        return jnp.kron(a, b)

    def numpy(self, a: Tensor) -> Tensor:
        return np.array(a)

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

    def value_and_grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Tuple[Any, Any]]:
        return libjax.value_and_grad(f, argnums=argnums)  # type: ignore

    def jit(self, f: Callable[..., Any]) -> Any:
        return libjax.jit(f)

    def vmap(self, f: Callable[..., Any]) -> Any:
        return libjax.vmap(f)
        # since tf doesn't support in&out axes options, we don't support them in universal backend

    def vectorized_value_and_grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Tuple[Any, Any]]:
        @libjax.jit  # type: ignore
        def wrapper(
            *args: Any, **kws: Any
        ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
            jf = self.value_and_grad(f, argnums=argnums)
            jf = libjax.vmap(jf, (0, None), 0)
            jf = self.jit(jf)
            vs, gs = jf(*args, **kws)
            if argnums == 0:
                pass
            else:
                if isinstance(argnums, int):
                    argnums_list = [argnums]
                else:
                    argnums_list = argnums  # type: ignore
                gs = list(gs)
                for i, (j, g) in enumerate(zip(argnums_list, gs)):
                    if j != 0:
                        gs[i] = jnp.sum(g, axis=0)
                gs = tuple(gs)
            return vs, gs

        return wrapper  # type: ignore

        # f = self.value_and_grad(f, argnums=argnums)
        # f = libjax.vmap(f, (0, None), 0)
        # f = self.jit(f)
        # return f

    vvag = vectorized_value_and_grad


def _tensordot_tf(
    self: Any, a: Tensor, b: Tensor, axes: Union[int, Sequence[Sequence[int]]]
) -> Tensor:
    return tf.tensordot(a, b, axes)


def _outer_product_tf(self: Any, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tf.tensordot(tensor1, tensor2, 0)


# temporary hot replace until new version of tensorflow is released, see issue: https://github.com/google/TensorNetwork/issues/940
# avoid buggy tensordot2 in tensornetwork

tensornetwork.backends.tensorflow.tensorflow_backend.TensorFlowBackend.tensordot = (
    _tensordot_tf
)
tensornetwork.backends.tensorflow.tensorflow_backend.TensorFlowBackend.outer_product = (
    _outer_product_tf
)


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

    def size(self, a: Tensor) -> Tensor:
        return tf.size(a)

    def kron(self, a: Tensor, b: Tensor) -> Tensor:
        # array more than 2d consistency is not guranteed for different backends
        return tf.reshape(
            tf.reshape(a, [a.shape[0], 1, a.shape[1], 1])
            * tf.reshape(b, [1, b.shape[0], 1, b.shape[1]]),
            [a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]],
        )

    def numpy(self, a: Tensor) -> Tensor:
        return a.numpy()  # only valid in eager mode

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

    def value_and_grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Tuple[Any, Any]]:
        def wrapper(*args: Any, **kws: Any) -> Any:
            with tf.GradientTape() as t:
                if isinstance(argnums, int):
                    x = args[argnums]
                else:
                    x = [args[i] for i in argnums]
                t.watch(x)
                y = f(*args, **kws)
                g = t.gradient(y, x)
            return y, g

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

    def vectorized_value_and_grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Tuple[Any, Any]]:
        # note how tf only works in this order, due to the bug reported as: https://github.com/google/TensorNetwork/issues/940
        @tf.function  # type: ignore
        def wrapper(
            *args: Any, **kws: Any
        ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
            with tf.GradientTape() as tape:
                if isinstance(argnums, int):
                    x = args[argnums]
                else:
                    x = [args[i] for i in argnums]
                tape.watch(x)

                def pg(inp: Tensor) -> Tensor:
                    return f(inp, *(args[1:]), **kws)

                vs = tf.vectorized_map(pg, args[0])
            grad = tape.gradient(vs, x)
            return vs, grad

        return wrapper  # type: ignore

        # f = self.vmap(f)
        # f = self.value_and_grad(f, argnums=argnums)
        # f = self.jit(f)
        # return f

    vvag = vectorized_value_and_grad


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

    def size(self, a: Tensor) -> Tensor:
        return a.size()

    def kron(self, a: Tensor, b: Tensor) -> Tensor:
        return torchlib.kron(a, b)

    def numpy(self, a: Tensor) -> Tensor:
        return a.numpy()

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

    def value_and_grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Tuple[Any, Any]]:
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
            return y, gs

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

    def vectorized_value_and_grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Tuple[Any, Any]]:
        f = self.value_and_grad(f, argnums=argnums)
        f = self.vmap(f)
        f = self.jit(f)
        return f

    vvag = vectorized_value_and_grad


_BACKENDS = {
    "tensorflow": TensorFlowBackend,
    "numpy": NumpyBackend,
    "jax": JaxBackend,
    # "shell": shell_backend.ShellBackend,  # no intention to maintain this one
    "pytorch": PyTorchBackend,  # no intention to fully maintain this one
}


def get_backend(backend: Union[Text, tnbackend]) -> Any:  # type: ignore
    """
    get the `tc.backend` object

    :param backend: "numpy", "tensorflow", "jax", "pytorch"
    :type backend: Union[Text, tnbackend]
    :raises ValueError: Backend doesn't exist for `backend` argument
    :return: the `tc.backend` object that with all registered universal functions
    :rtype: Any
    """
    if isinstance(backend, tnbackend):
        return backend
    if backend not in _BACKENDS:
        raise ValueError("Backend '{}' does not exist".format(backend))
    return _BACKENDS[backend]()
