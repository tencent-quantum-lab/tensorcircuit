"""
backend magic inherited from tensornetwork
"""

import inspect
import warnings
from functools import partial, reduce
from operator import mul
from typing import Any, Callable, Optional, Sequence, Text, Tuple, Union

import numpy as np
import tensornetwork
from scipy.linalg import expm
from scipy.special import softmax
from tensornetwork.backends.tensorflow import tensorflow_backend
from tensornetwork.backends.numpy import numpy_backend
from tensornetwork.backends.jax import jax_backend
from tensornetwork.backends.pytorch import pytorch_backend

try:  # old version tn compatiblity
    from tensornetwork.backends import base_backend

    tnbackend = base_backend.BaseBackend

except ImportError:
    from tensornetwork.backends import abstract_backend

    tnbackend = abstract_backend.AbstractBackend


Tensor = Any
PRNGKeyArray = Any  # libjax.random.PRNGKeyArray
RGenerator = Any  # tf.random.Generator

libjax: Any
jnp: Any
jsp: Any
torchlib: Any
tf: Any

# TODO(@refraction-ray): refactor backends as a dir


def _more_methods_for_backend(tnbackend: Any) -> None:
    """
    Add tensorcircuit specific backend methods, especially with their docstrings
    """

    def expm(self: Any, a: Tensor) -> Tensor:  # pylint: disable=unused-variable
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

    def sqrtmh(self: Any, a: Tensor) -> Tensor:  # pylint: disable=unused-variable
        """
        Return sqrtm of Hermitian matrix ``a``

        :param a: tensor in matrix form
        :type a: Tensor
        :return: sqrtm of ``a``
        :rtype: Tensor
        """
        # maybe friendly for AD and also cosidering that several backend has no support for native sqrtm
        e, v = self.eigh(a)
        e = self.sqrt(e)
        return v @ self.diagflat(e) @ self.adjoint(v)

    def sin(self: Any, a: Tensor) -> Tensor:  # pylint: disable=unused-variable
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

    def cos(self: Any, a: Tensor) -> Tensor:  # pylint: disable=unused-variable
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

    def abs(self: Any, a: Tensor) -> Tensor:  # pylint: disable=unused-variable
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

    def kron(  # pylint: disable=unused-variable
        self: Any, a: Tensor, b: Tensor
    ) -> Tensor:
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

    def size(self: Any, a: Tensor) -> Tensor:  # pylint: disable=unused-variable
        """
        Return the total number of elements in ``a`` in tensor form.

        :param a: tensor
        :type a: Tensor
        :return: the total number of elements in ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `size`.".format(self.name)
        )

    def sizen(self: Any, a: Tensor) -> int:  # pylint: disable=unused-variable
        """
        Return the total number of elements in ``a``, but in int form

        :param a: tensor
        :type a: Tensor
        :return: [description]
        :rtype: int
        """
        return reduce(mul, a.shape)  # type: ignore

    def numpy(self: Any, a: Tensor) -> Tensor:  # pylint: disable=unused-variable
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

    def real(self: Any, a: Tensor) -> Tensor:  # pylint: disable=unused-variable
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

    def adjoint(self: Any, a: Tensor) -> Tensor:  # pylint: disable=unused-variable
        """
        conjugate and transpose of the tensor ``a``

        :param a: Input tensor
        :type a: Tensor
        :return: adjoint tensor of ``a``
        :rtype: Tensor
        """
        return self.conj(self.transpose(a))

    def i(self: Any, dtype: str) -> Tensor:  # pylint: disable=unused-variable
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

    def stack(  # pylint: disable=unused-variable
        self: Any, a: Sequence[Tensor], axis: int = 0
    ) -> Tensor:
        """
        Concatenates a sequence of tensors ``a`` along a new dimension ``axis``.

        :param a: List of tensors in the same shape
        :type a: Sequence[Tensor]
        :param axis: the stack axis, defaults to 0
        :type axis: int, optional
        :return: concatenated tensor
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `stack`.".format(self.name)
        )

    def tile(  # pylint: disable=unused-variable
        self: Any, a: Tensor, rep: Tensor
    ) -> Tensor:
        """
        Constructs a tensor by tiling a given tensor.

        :param a: [description]
        :type a: Tensor
        :param rep: 1d tensor with length the same as the rank of ``a``
        :type rep: Tensor
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `tile`.".format(self.name)
        )

    def min(  # pylint: disable=unused-variable
        self: Any, a: Tensor, axis: Optional[int] = None
    ) -> Tensor:
        """
        Return the minimum of an array or minimum along an axis.

        :param a: [description]
        :type a: Tensor
        :param axis: [description], defaults to None
        :type axis: Optional[int], optional
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `min`.".format(self.name)
        )

    def max(  # pylint: disable=unused-variable
        self: Any, a: Tensor, axis: Optional[int] = None
    ) -> Tensor:
        """
        Return the maximum of an array or maximum along an axis.

        :param a: [description]
        :type a: Tensor
        :param axis: [description], defaults to None
        :type axis: Optional[int], optional
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `max`.".format(self.name)
        )

    def unique_with_counts(  # pylint: disable=unused-variable
        self: Any, a: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Find the unique elements and their corresponding counts of the given tensor ``a``.

        :param a: [description]
        :type a: Tensor
        :return: Unique elements, corresponding counts
        :rtype: Tuple[Tensor, Tensor]
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `unique_with_counts`.".format(self.name)
        )

    def relu(self: Any, a: Tensor) -> Tensor:  # pylint: disable=unused-variable
        """
        Rectified linear unit activation function.
        Computes the element-wise function:

        .. math ::

            \\mathrm{relu}(x)=\\max(x,0)


        :param a: Input tensor
        :type a: Tensor
        :return: Tensor after relu
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `relu`.".format(self.name)
        )

    def softmax(  # pylint: disable=unused-variable
        self: Any, a: Sequence[Tensor], axis: Optional[int] = None
    ) -> Tensor:
        """
        Softmax function.
        Computes the function which rescales elements to the range [0,1] such that the elements along axis sum to 1.

        .. math ::

            \\mathrm{softmax}(x) = \\frac{\exp(x_i)}{\\sum_j \\exp(x_j)}


        :param a: Tensor
        :type a: Sequence[Tensor]
        :param axis: A dimension along which Softmax will be computed , defaults to None for all axis sum.
        :type axis: int, optional
        :return: concatenated tensor
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `softmax`.".format(self.name)
        )

    def onehot(  # pylint: disable=unused-variable
        self: Any, a: Tensor, num: int
    ) -> Tensor:
        """
        One-hot encodes the given ``a``.
        Each index in the input ``a`` is encoded as a vector of zeros of length ``num``
        with the element at index set to one:

        :param a: input tensor
        :type a: Tensor
        :param num: number of features in onehot dimension
        :type num: int
        :return: onehot tensor with the last extra dimension
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `onehot`.".format(self.name)
        )

    # one_hot = onehot doesn't work
    def one_hot(  # pylint: disable=unused-variable
        self: Any, a: Tensor, num: int
    ) -> Tensor:
        """
        See doc for :py:meth:`onehot`
        """
        return self.onehot(a, num)

    def cumsum(  # pylint: disable=unused-variable
        self: Any, a: Tensor, axis: Optional[int] = None
    ) -> Tensor:
        """
        Return the cumulative sum of the elements along a given axis.

        :param a: [description]
        :type a: Tensor
        :param axis: The default behavior is the same as numpy, different from tf/torch
            as cumsum of the flattern 1D array, defaults to None
        :type axis: Optional[int], optional
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `cumsum`.".format(self.name)
        )

    def is_tensor(self: Any, a: Tensor) -> bool:  # pylint: disable=unused-variable
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

    def cast(  # pylint: disable=unused-variable
        self: Any, a: Tensor, dtype: str
    ) -> Tensor:
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

    def tree_map(  # pylint: disable=unused-variable
        self: Any, f: Callable[..., Any], *pytrees: Any
    ) -> Any:
        """
        tree map with multiple arg function ``f`` through pytrees

        :param f: The function
        :type f: Callable[..., Any]
        :param pytrees: inputs as any python structure
        :type pytrees: Any
        :raises NotImplementedError: raise when neither tensorflow or jax is installed
        :return: [description]
        :rtype: Any
        """
        global libjax
        global tf
        if libjax is None and tf is None:
            try:
                import jax

                libjax = jax
            except ImportError:
                try:
                    import tensorflow

                    tf = tensorflow
                except ImportError:
                    pass
        if libjax is not None:
            r = libjax.tree_map(f, *pytrees)
        elif tf is not None:
            r = tf.nest.map_structure(f, *pytrees)
        else:
            raise NotImplementedError("Only tensorflow and jax support `tree_map`")

        return r

    def set_random_state(  # pylint: disable=unused-variable
        self: Any, seed: Optional[int] = None
    ) -> None:
        """
        set random state attached in the backend

        :param seed: int, defaults to None
        :type seed: Optional[int], optional
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `set_random_state`.".format(self.name)
        )

    def implicit_randn(  # pylint: disable=unused-variable
        self: Any,
        shape: Union[int, Sequence[int]] = 1,
        mean: float = 0,
        stddev: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        """
        call random normal function with the random state management behind the scene

        :param shape: [description], defaults to 1
        :type shape: Union[int, Sequence[int]], optional
        :param mean: [description], defaults to 0
        :type mean: float, optional
        :param stddev: [description], defaults to 1
        :type stddev: float, optional
        :param dtype: [description], defaults to "32"
        :type dtype: str, optional
        :return: [description]
        :rtype: Tensor
        """
        g = getattr(self, "g", None)
        if g is None:
            self.set_random_state()
            g = getattr(self, "g", None)
        r = self.stateful_randn(g, shape, mean, stddev, dtype)
        return r

    def stateful_randn(  # pylint: disable=unused-variable
        self: Any,
        g: Any,
        shape: Union[int, Sequence[int]] = 1,
        mean: float = 0,
        stddev: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        """
        [summary]

        :param self: [description]
        :type self: Any
        :param g: stateful register for each package
        :type g: Any
        :param shape: shape of output sampling tensor
        :type shape: Union[int, Sequence[int]]
        :param mean: [description], defaults to 0
        :type mean: float, optional
        :param stddev: [description], defaults to 1
        :type stddev: float, optional
        :param dtype: only real data type is supported, "32" or "64", defaults to "32"
        :type dtype: str, optional
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `stateful_randn`.".format(self.name)
        )

    def implicit_randu(  # pylint: disable=unused-variable
        self: Any,
        shape: Union[int, Sequence[int]] = 1,
        low: float = 0,
        high: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        """
        call random normal function with the random state management behind the scene

        :param shape: [description], defaults to 1
        :type shape: Union[int, Sequence[int]], optional
        :param mean: [description], defaults to 0
        :type mean: float, optional
        :param stddev: [description], defaults to 1
        :type stddev: float, optional
        :param dtype: [description], defaults to "32"
        :type dtype: str, optional
        :return: [description]
        :rtype: Tensor
        """
        g = getattr(self, "g", None)
        if g is None:
            self.set_random_state()
            g = getattr(self, "g", None)
        r = self.stateful_randu(g, shape, low, high, dtype)
        return r

    def stateful_randu(  # pylint: disable=unused-variable
        self: Any,
        g: Any,
        shape: Union[int, Sequence[int]] = 1,
        low: float = 0,
        high: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        """
        uniform random sampler for ``low`` to ``high``

        :param g: stateful register for each package
        :type g: Any
        :param shape: shape of output sampling tensor, defaults to 1
        :type shape: Union[int, Sequence[int]], optional
        :param low: [description], defaults to 0
        :type low: float, optional
        :param high: [description], defaults to 1
        :type high: float, optional
        :param dtype: only real data type is supported, "32" or "64", defaults to "32"
        :type dtype: str, optional
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `stateful_randu`.".format(self.name)
        )

    def implicit_randc(  # pylint: disable=unused-variable
        self: Any,
        a: Union[int, Sequence[int], Tensor],
        shape: Union[int, Sequence[int]],
        p: Optional[Union[Sequence[float], Tensor]] = None,
    ) -> Tensor:
        """
        [summary]

        :param g: [description]
        :type g: Any
        :param a: The possible options
        :type a: Union[int, Sequence[int], Tensor]
        :param shape: Sampling output shape
        :type shape: Union[int, Sequence[int]]
        :param p: probability for each option in a, defaults to None, as equal probability distribution
        :type p: Optional[Union[Sequence[float], Tensor]], optional
        :return: [description]
        :rtype: Tensor
        """
        g = getattr(self, "g", None)
        if g is None:
            self.set_random_state()
            g = getattr(self, "g", None)
        r = self.stateful_randc(g, a, shape, p)
        return r

    def stateful_randc(  # pylint: disable=unused-variable
        self: Any,
        g: Any,
        a: Union[int, Sequence[int], Tensor],
        shape: Union[int, Sequence[int]],
        p: Optional[Union[Sequence[float], Tensor]] = None,
    ) -> Tensor:
        """
        [summary]

        :param g: [description]
        :type g: Any
        :param a: The possible options
        :type a: Union[int, Sequence[int], Tensor]
        :param shape: Sampling output shape
        :type shape: Union[int, Sequence[int]]
        :param p: probability for each option in a, defaults to None, as equal probability distribution
        :type p: Optional[Union[Sequence[float], Tensor]], optional
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `stateful_randc`.".format(self.name)
        )

    def scatter(  # pylint: disable=unused-variable
        self: Any, operand: Tensor, indices: Tensor, updates: Tensor
    ) -> Tensor:
        """
        Roughly equivalent to operand[indices] = updates, indices only support shape with rank 2 for now

        :param operand: [description]
        :type operand: Tensor
        :param indices: [description]
        :type indices: Tensor
        :param updates: [description]
        :type updates: Tensor
        :return: [description]
        :rtype: Tensor
        """
        # only tested on scalar update for now, as the general XLA scatter syntax is too complicated
        # https://www.tensorflow.org/xla/operation_semantics?hl=en#scatter
        # TODO(@refraction-ray): implement more general and consistent scatter syntax for different backends.
        raise NotImplementedError(
            "Backend '{}' has not implemented `scatter`.".format(self.name)
        )

    def cond(  # pylint: disable=unused-variable
        self: Any,
        pred: bool,
        true_fun: Callable[[], Tensor],
        false_fun: Callable[[], Tensor],
    ) -> Tensor:
        """
        the native cond for XLA compiling, wrapper for ``tf.cond`` and limited functionality of ``jax.lax.cond``

        :param pred: [description]
        :type pred: bool
        :param true_fun: [description]
        :type true_fun: Callable[[], Tensor]
        :param false_fun: [description]
        :type false_fun: Callable[[], Tensor]
        :return: [description]
        :rtype: Tensor
        """
        # possibly the most weird thing introduced in the backend :(
        raise NotImplementedError(
            "Backend '{}' has not implemented `cond`.".format(self.name)
        )

    def switch(  # pylint: disable=unused-variable
        self: Any, index: Tensor, branches: Sequence[Callable[[], Tensor]]
    ) -> Tensor:
        """
        ``branches[index]()``

        :param index: [description]
        :type index: Tensor
        :param branches: [description]
        :type branches: Sequence[Callable[[], Tensor]]
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `switch`.".format(self.name)
        )

    def grad(  # pylint: disable=unused-variable
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

    def value_and_grad(  # pylint: disable=unused-variable
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

    def vjp(  # pylint: disable=unused-variable
        self: Any,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        """
        Function that computes the dot product between a vector v and the Jacobian
        of the given function at the point given by the inputs. (reverse mode AD relevant)
        Strictly speaking, this function is value_and_vjp.


        :param f: The function to carry out vjp calculation
        :type f: Callable[..., Any]
        :param inputs: input for ``f``
        :type inputs: Union[Tensor, Sequence[Tensor]]
        :param v: value vector or gradient from downstream in reserse mode AD
            the same shape as return of function ``f``
        :type v: Union[Tensor, Sequence[Tensor]]
        :return: (``f(*inputs)``, vjp_tensor), where vjp_tensor is the same shape as inputs
        :rtype: Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `vjp`.".format(self.name)
        )

    def jit(  # pylint: disable=unused-variable
        self: Any,
        f: Callable[..., Any],
        static_argnums: Optional[Union[int, Sequence[int]]] = None,
        jit_compile: Optional[bool] = None,
    ) -> Callable[..., Any]:
        """
        Return jitted version function of ``f``

        :param f: function to be jitted
        :type f: Callable[..., Any]
        :param static_argnums: index of args that doesn't regarded as tensor,
            only work for jax backend
        :type static_argnums: Optional[Union[int, Sequence[int]]], defaults to None
        :param jit_compile: whether open XLA compliation, only works for tensorflow backend,
            defaults False since several ops has no XLA correspondence
        :type jit_compile: bool
        :return: jitted ``f``
        :rtype: Callable[..., Any]
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `jit`.".format(self.name)
        )

    def vmap(  # pylint: disable=unused-variable
        self: Any,
        f: Callable[..., Any],
        vectorized_argnums: Union[int, Sequence[int]] = 0,
    ) -> Any:
        """
        Return vectorized map or batched version of ``f`` on the first extra axis,
        the general interface support ``f`` with multiple arguments and broadcast in the fist dimension.

        :param f: function to be broadcasted.
        :type f: Callable[..., Any]
        :param vectorized_argnums: the args to be vectorized,
            these arguments should share the same batch shape in the fist dimension
        :type vectorized_argnums: Union[int, Sequence[int]], defaults to 0
        :return: vmap version of ``f``
        :rtype: Any
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `vmap`.".format(self.name)
        )

    def vectorized_value_and_grad(  # pylint: disable=unused-variable
        self: Any,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        vectorized_argnums: Union[int, Sequence[int]] = 0,
    ) -> Callable[..., Tuple[Any, Any]]:
        """
        Return vvag function of ``f``. the inputs for ``f`` is (args[0], args[1], args[2], ...),
        and the output of ``f`` is a scalar. Suppose vvag(f) is a function with inputs in the form
        (vargs[0], args[1], args[2], ...), where vagrs[0] has one extra dimension than args[0] in the first axis
        and consistent with args[0] in shape for remaining dimensions, i.e. shape(vargs[0]) = [batch] + shape(args[0]).
        (We only cover case where ``vectorized_argnums`` defaults to 0 here for demonstration).
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
        :param vectorized_argnums: the args to be vectorized, these arguments should share the same batch shape
            in the fist dimension
        :type vectorized_argnums: Union[int, Sequence[int]], defaults to 0
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


_more_methods_for_backend(tnbackend)


def _sum_numpy(
    self: Any, a: Tensor, axis: Optional[Sequence[int]] = None, keepdims: bool = False
) -> Tensor:
    return np.sum(a, axis=axis, keepdims=keepdims)
    # see https://github.com/google/TensorNetwork/issues/952


tensornetwork.backends.numpy.numpy_backend.NumPyBackend.sum = _sum_numpy


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

    def stack(self, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return np.stack(a, axis=axis)

    def tile(self, a: Tensor, rep: Tensor) -> Tensor:
        return np.tile(a, rep)

    def unique_with_counts(self, a: Tensor) -> Tuple[Tensor, Tensor]:
        return np.unique(a, return_counts=True)  # type: ignore

    def min(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return np.min(a, axis=axis)

    def max(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return np.max(a, axis=axis)

    def relu(self, a: Tensor) -> Tensor:
        return (abs(a) + a) / 2
        # this impl seems to be the fastest
        # see https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy

    def softmax(self, a: Sequence[Tensor], axis: Optional[int] = None) -> Tensor:
        return softmax(a, axis=axis)

    def onehot(self, a: Tensor, num: int) -> Tensor:
        res = np.eye(num)[a.reshape([-1])]
        return res.reshape(list(a.shape) + [num])
        # https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy

    def cumsum(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return np.cumsum(a, axis)

    def is_tensor(self, a: Any) -> bool:
        if isinstance(a, np.ndarray):
            return True
        return False

    def real(self, a: Tensor) -> Tensor:
        return np.real(a)

    def cast(self, a: Tensor, dtype: str) -> Tensor:
        if isinstance(dtype, str):
            return a.astype(getattr(np, dtype))
        return a.astype(dtype)

    def set_random_state(self, seed: Optional[int] = None) -> None:
        g = np.random.default_rng(seed)  # None auto supported
        self.g = g

    def stateful_randn(
        self,
        g: np.random.Generator,
        shape: Union[int, Sequence[int]] = 1,
        mean: float = 0,
        stddev: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        if isinstance(dtype, str):
            dtype = dtype[-2:]
        if isinstance(shape, int):
            shape = (shape,)
        r = g.normal(loc=mean, scale=stddev, size=shape)
        if dtype == "32":
            r = r.astype(np.float32)
        elif dtype == "64":
            r = r.astype(np.float64)
        elif not isinstance(dtype, str):
            r = r.astype(dtype)
        else:
            raise ValueError("unspported `dtype` %s" % dtype)
        return r

    def stateful_randu(
        self,
        g: np.random.Generator,
        shape: Union[int, Sequence[int]] = 1,
        low: float = 0,
        high: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        if isinstance(dtype, str):
            dtype = dtype[-2:]
        if isinstance(shape, int):
            shape = (shape,)
        r = g.uniform(low=low, high=high, size=shape)
        if dtype == "32":
            r = r.astype(np.float32)
        elif dtype == "64":
            r = r.astype(np.float64)
        elif not isinstance(dtype, str):
            r = r.astype(dtype)
        else:
            raise ValueError("unspported `dtype` %s" % dtype)
        return r

    def stateful_randc(
        self,
        g: np.random.Generator,
        a: Union[int, Sequence[int], Tensor],
        shape: Union[int, Sequence[int]],
        p: Optional[Union[Sequence[float], Tensor]] = None,
    ) -> Tensor:
        if isinstance(shape, int):
            shape = (shape,)
        return g.choice(a, size=shape, replace=True, p=p)

    def scatter(self, operand: Tensor, indices: Tensor, updates: Tensor) -> Tensor:
        operand_new = np.copy(operand)
        operand_new[[indices[:, i] for i in range(indices.shape[1])]] = updates
        return operand_new

    def cond(
        self,
        pred: bool,
        true_fun: Callable[[], Tensor],
        false_fun: Callable[[], Tensor],
    ) -> Tensor:
        if pred:
            return true_fun()
        return false_fun()

    def switch(self, index: Tensor, branches: Sequence[Callable[[], Tensor]]) -> Tensor:
        return branches[index]()

    def grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Any]:
        raise NotImplementedError("numpy backend doesn't support AD")

    def value_and_grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Tuple[Any, Any]]:
        raise NotImplementedError("numpy backend doesn't support AD")

    def jit(
        self,
        f: Callable[..., Any],
        static_argnums: Optional[Union[int, Sequence[int]]] = None,
        jit_compile: Optional[bool] = None,
    ) -> Callable[..., Any]:
        warnings.warn("numpy backend has no parallel as jit, just do nothing")
        return f
        # raise NotImplementedError("numpy backend doesn't support jit compiling")

    def vmap(
        self, f: Callable[..., Any], vectorized_argnums: Union[int, Sequence[int]] = 0
    ) -> Any:
        warnings.warn(
            "numpy backend has no intrinsic vmap like interface"
            ", use vectorize instead (plain for loop)"
        )
        if isinstance(vectorized_argnums, int):
            vectorized_argnums = (vectorized_argnums,)

        def wrapper(*args: Any, **kws: Any) -> Tensor:
            results = []
            for barg in zip(*[args[i] for i in vectorized_argnums]):  # type: ignore
                narg = []
                j = 0
                for k in range(len(args)):
                    if k in vectorized_argnums:  # type: ignore
                        narg.append(barg[j])
                        j += 1
                    else:
                        narg.append(args[k])
                results.append(f(*narg, **kws))
            return np.array(results)

        return wrapper

    def vectorized_value_and_grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        vectorized_argnums: Union[int, Sequence[int]] = 0,
    ) -> Callable[..., Tuple[Any, Any]]:
        raise NotImplementedError("numpy backend doesn't support AD")

    vvag = vectorized_value_and_grad

    def vjp(
        self,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        raise NotImplementedError("numpy backend doesn't support AD")


def _convert_to_tensor_jax(self: Any, tensor: Tensor) -> Tensor:
    if not isinstance(tensor, (np.ndarray, jnp.ndarray)) and not jnp.isscalar(tensor):
        raise TypeError(
            ("Expected a `jnp.array`, `np.array` or scalar. " f"Got {type(tensor)}")
        )
    result = jnp.asarray(tensor)
    return result


def _svd_jax(
    self: Any,
    tensor: Tensor,
    pivot_axis: int = -1,
    max_singular_values: Optional[int] = None,
    max_truncation_error: Optional[float] = None,
    relative: Optional[bool] = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    from .ops import adaware_svd

    left_dims = tensor.shape[:pivot_axis]
    right_dims = tensor.shape[pivot_axis:]

    tensor = jnp.reshape(tensor, [np.prod(left_dims), np.prod(right_dims)])
    u, s, vh = adaware_svd(tensor)

    if max_singular_values is None:
        max_singular_values = jnp.size(s)

    if max_truncation_error is not None:
        # Cumulative norms of singular values in ascending order.
        trunc_errs = jnp.sqrt(jnp.cumsum(jnp.square(s[::-1])))
        # If relative is true, rescale max_truncation error with the largest
        # singular value to yield the absolute maximal truncation error.
        if relative:
            abs_max_truncation_error = max_truncation_error * s[0]
        else:
            abs_max_truncation_error = max_truncation_error
        # We must keep at least this many singular values to ensure the
        # truncation error is <= abs_max_truncation_error.
        num_sing_vals_err = jnp.count_nonzero(
            (trunc_errs > abs_max_truncation_error).astype(jnp.int32)
        )
    else:
        num_sing_vals_err = max_singular_values

    num_sing_vals_keep = min(max_singular_values, num_sing_vals_err)

    s = s.astype(tensor.dtype)

    s_rest = s[num_sing_vals_keep:]
    s = s[:num_sing_vals_keep]
    u = u[:, :num_sing_vals_keep]
    vh = vh[:num_sing_vals_keep, :]

    dim_s = s.shape[0]
    u = jnp.reshape(u, list(left_dims) + [dim_s])
    vh = jnp.reshape(vh, [dim_s] + list(right_dims))

    return u, s, vh, s_rest


tensornetwork.backends.jax.jax_backend.JaxBackend.convert_to_tensor = (
    _convert_to_tensor_jax
)
tensornetwork.backends.jax.jax_backend.JaxBackend.svd = _svd_jax


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
        if isinstance(dtype, str):
            return a.astype(getattr(jnp, dtype))
        return a.astype(dtype)

    def expm(self, a: Tensor) -> Tensor:
        return jsp.linalg.expm(a)
        # currently expm in jax doesn't support AD, it will raise an AssertError,
        # see https://github.com/google/jax/issues/2645

    def stack(self: Any, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return jnp.stack(a, axis=axis)

    def tile(self: Any, a: Tensor, rep: Tensor) -> Tensor:
        return jnp.tile(a, rep)

    def min(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return jnp.min(a, axis=axis)

    def max(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return jnp.max(a, axis=axis)

    def unique_with_counts(self, a: Tensor) -> Tuple[Tensor, Tensor]:
        return jnp.unique(a, return_counts=True)  # type: ignore

    def relu(self, a: Tensor) -> Tensor:
        return libjax.nn.relu(a)

    def softmax(self, a: Sequence[Tensor], axis: Optional[int] = None) -> Tensor:
        return libjax.nn.softmax(a, axis=axis)

    def onehot(self, a: Tensor, num: int) -> Tensor:
        return libjax.nn.one_hot(a, num)

    def cumsum(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return jnp.cumsum(a, axis)

    def is_tensor(self, a: Any) -> bool:
        if not isinstance(a, jnp.ndarray):
            return False
        # isinstance(np.eye(1), jax.numpy.ndarray) = True!
        if getattr(a, "_value", None) is not None:
            return True
        return False

    def set_random_state(self, seed: Optional[Union[int, PRNGKeyArray]] = None) -> None:
        if seed is None:
            seed = np.random.randint(42)
        if isinstance(seed, int):
            g = libjax.random.PRNGKey(seed)
        else:
            g = seed
        self.g = g

    def implicit_randn(
        self,
        shape: Union[int, Sequence[int]] = 1,
        mean: float = 0,
        stddev: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        g = getattr(self, "g", None)
        if g is None:
            self.set_random_state()
            g = getattr(self, "g", None)
        key, subkey = libjax.random.split(g)
        r = self.stateful_randn(subkey, shape, mean, stddev, dtype)
        self.g = key
        return r

    def implicit_randu(
        self,
        shape: Union[int, Sequence[int]] = 1,
        low: float = 0,
        high: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        g = getattr(self, "g", None)
        if g is None:
            self.set_random_state()
            g = getattr(self, "g", None)
        key, subkey = libjax.random.split(g)
        r = self.stateful_randu(subkey, shape, low, high, dtype)
        self.g = key
        return r

    def implicit_randc(
        self,
        a: Union[int, Sequence[int], Tensor],
        shape: Union[int, Sequence[int]],
        p: Optional[Union[Sequence[float], Tensor]] = None,
    ) -> Tensor:
        g = getattr(self, "g", None)
        if g is None:
            self.set_random_state()
            g = getattr(self, "g", None)
        key, subkey = libjax.random.split(g)
        r = self.stateful_randc(subkey, a, shape, p)
        self.g = key
        return r

    def stateful_randn(
        self,
        g: PRNGKeyArray,
        shape: Union[int, Sequence[int]] = 1,
        mean: float = 0,
        stddev: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        if isinstance(dtype, str):
            dtype = dtype[-2:]
        if isinstance(shape, int):
            shape = (shape,)
        if dtype == "32":
            dtyper = jnp.float32
        elif dtype == "64":
            dtyper = jnp.float64
        elif not isinstance(dtype, str):
            dtyper = dtype
        r = libjax.random.normal(g, shape=shape, dtype=dtyper) * stddev + mean
        return r

    def stateful_randu(
        self,
        g: PRNGKeyArray,
        shape: Union[int, Sequence[int]] = 1,
        low: float = 0,
        high: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        if isinstance(dtype, str):
            dtype = dtype[-2:]
        if isinstance(shape, int):
            shape = (shape,)
        if dtype == "32":
            dtyper = jnp.float32
        elif dtype == "64":
            dtyper = jnp.float64
        elif not isinstance(dtype, str):
            dtyper = dtype
        r = libjax.random.uniform(g, shape=shape, dtype=dtyper, minval=low, maxval=high)
        return r

    def stateful_randc(
        self,
        g: PRNGKeyArray,
        a: Union[int, Sequence[int], Tensor],
        shape: Union[int, Sequence[int]],
        p: Optional[Union[Sequence[float], Tensor]] = None,
    ) -> Tensor:
        if isinstance(shape, int):
            shape = (shape,)
        if not self.is_tensor(a):
            a = jnp.array(a)
        if p is not None:
            if not self.is_tensor(p):
                p = jnp.array(p)
        return libjax.random.choice(g, a, shape=shape, replace=True, p=p)

    def cond(
        self,
        pred: bool,
        true_fun: Callable[[], Tensor],
        false_fun: Callable[[], Tensor],
    ) -> Tensor:
        return libjax.lax.cond(pred, lambda _: true_fun(), lambda _: false_fun(), None)

    def switch(self, index: Tensor, branches: Sequence[Callable[[], Tensor]]) -> Tensor:
        # branches_null = [lambda _: b() for b in branches]
        # see https://stackoverflow.com/a/34021333 for weird behavior of lambda with list comprehension
        branches_null = [lambda _, f=b: f() for b in branches]
        return libjax.lax.switch(index, branches_null, None)

    def scatter(self, operand: Tensor, indices: Tensor, updates: Tensor) -> Tensor:
        rank = len(operand.shape)
        dnums = libjax.lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=tuple([i for i in range(rank)]),
            scatter_dims_to_operand_dims=tuple([i for i in range(rank)]),
        )
        r = libjax.lax.scatter(operand, indices, updates, dnums)
        return r

    def grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Any:
        return libjax.grad(f, argnums=argnums)

    def value_and_grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Tuple[Any, Any]]:
        return libjax.value_and_grad(f, argnums=argnums)  # type: ignore

    def vjp(
        self,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        if not (isinstance(inputs, list) or isinstance(inputs, tuple)):
            # one input tensor
            inputs = [inputs]
            one_input = True
        else:
            one_input = False
        value, vjpf = libjax.vjp(f, *inputs)
        if isinstance(v, list):
            v = tuple(v)
        vjpv = vjpf(v)
        if one_input:
            vjpv = vjpv[0]
        return value, vjpv

    def jit(
        self,
        f: Callable[..., Any],
        static_argnums: Optional[Union[int, Sequence[int]]] = None,
        jit_compile: Optional[bool] = None,
    ) -> Any:
        return libjax.jit(f, static_argnums=static_argnums)

    def vmap(
        self, f: Callable[..., Any], vectorized_argnums: Union[int, Sequence[int]] = 0
    ) -> Any:
        if isinstance(vectorized_argnums, int):
            vectorized_argnums = (vectorized_argnums,)
        # if vectorized_argnums == (0,): # fast shortcuts
        #     return libjax.vmap(f)

        def wrapper(*args: Any, **kws: Any) -> Tensor:
            in_axes = [0 if i in vectorized_argnums else None for i in range(len(args))]  # type: ignore
            return libjax.vmap(f, in_axes, 0)(*args, **kws)

        return wrapper

        # since tf doesn't support general in&out axes options, we don't support them in universal backend

    def vectorized_value_and_grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        vectorized_argnums: Union[int, Sequence[int]] = 0,
    ) -> Callable[..., Tuple[Any, Any]]:
        if isinstance(vectorized_argnums, int):
            vectorized_argnums = (vectorized_argnums,)

        def wrapper(
            *args: Any, **kws: Any
        ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
            jf = self.value_and_grad(f, argnums=argnums)
            in_axes = [0 if i in vectorized_argnums else None for i in range(len(args))]  # type: ignore
            jf = libjax.vmap(jf, in_axes, 0)
            # jf = self.jit(jf)
            vs, gs = jf(*args, **kws)

            if isinstance(argnums, int):
                argnums_list = [argnums]
                gs = [gs]
            else:
                argnums_list = argnums  # type: ignore
                gs = list(gs)
            for i, (j, g) in enumerate(zip(argnums_list, gs)):
                if j not in vectorized_argnums:  # type: ignore
                    gs[i] = libjax.tree_map(partial(jnp.sum, axis=0), g)
            if isinstance(argnums, int):
                gs = gs[0]
            else:
                gs = tuple(gs)

            return vs, gs

        return wrapper

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


def _random_choice_tf(
    g: RGenerator,
    a: Union[int, Sequence[int], Tensor],
    shape: Union[int, Sequence[int]],
    p: Optional[Union[Sequence[float], Tensor]] = None,
) -> Tensor:
    # only replace=True support, replace=False is not implemented
    # for stateless random module, tf has corresponding categorical function similar to choice
    # however, such utility is not implemented with ``tf.random.Generator``
    # part of the code below is inspired by corresponding implementation in jax (Apache 2.0)
    if isinstance(a, int):
        assert a > 0
        a = tf.range(a)
    if not (isinstance(a, tf.Tensor) or isinstance(a, tf.Variable)):
        a = tf.constant(a)
    assert len(a.shape) == 1
    if isinstance(shape, int):
        shape = (shape,)
    if p is None:
        dtype = tf.float32
        p = tf.ones_like(a)
        p = tf.cast(p, dtype=dtype)
        p /= tf.reduce_sum(p)
    else:
        if not (isinstance(p, tf.Tensor) or isinstance(p, tf.Variable)):
            p = tf.constant(p)
        dtype = p.dtype
    shape1 = reduce(mul, shape)
    p_cuml = tf.cumsum(p)
    r = p_cuml[-1] * (1 - g.uniform([shape1], dtype=dtype))
    ind = tf.searchsorted(p_cuml, r)
    res = tf.gather(a, ind)
    return tf.reshape(res, shape)


# temporary hot replace until new version of tensorflow is released,
# see issue: https://github.com/google/TensorNetwork/issues/940
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
        self.minor = int(tf.__version__.split(".")[1])
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

    def min(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return tf.reduce_min(a, axis=axis)

    def max(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return tf.reduce_max(a, axis=axis)

    def unique_with_counts(self, a: Tensor) -> Tuple[Tensor, Tensor]:
        r = tf.unique_with_counts(a)
        order = tf.argsort(r.y)
        return tf.gather(r.y, order), tf.gather(r.count, order)

    def stack(self, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return tf.stack(a, axis=axis)

    def tile(self, a: Tensor, rep: Tensor) -> Tensor:
        return tf.tile(a, rep)

    def relu(self, a: Tensor) -> Tensor:
        return tf.nn.relu(a)

    def onehot(self, a: Tensor, num: int) -> Tensor:
        return tf.one_hot(a, num)

    def softmax(self, a: Sequence[Tensor], axis: Optional[int] = None) -> Tensor:
        if axis is None:  # make the default behavior consistent
            r = tf.keras.activations.softmax(tf.reshape(a, [1, -1]), axis=axis)
            return tf.reshape(r, a.shape)  # type: ignore
        return tf.keras.activations.softmax(a, axis=axis)

    def cumsum(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        if axis is None:
            a = tf.reshape(a, [-1])
            return tf.cumsum(a)
        else:
            return tf.cumsum(a, axis)

    def is_tensor(self, a: Any) -> bool:
        if isinstance(a, tf.Tensor) or isinstance(a, tf.Variable):
            return True
        return False

    def abs(self, a: Tensor) -> Tensor:
        return tf.math.abs(a)

    def real(self, a: Tensor) -> Tensor:
        return tf.math.real(a)

    def cast(self, a: Tensor, dtype: str) -> Tensor:
        if isinstance(dtype, str):
            return tf.cast(a, dtype=getattr(tf, dtype))
        return tf.cast(a, dtype=dtype)

    def set_random_state(self, seed: Optional[Union[int, RGenerator]] = None) -> None:
        if seed is None:
            g = tf.random.Generator.from_non_deterministic_state()
        elif isinstance(seed, int):
            g = tf.random.Generator.from_seed(seed)
        else:
            g = seed
        self.g = g

    def stateful_randn(
        self,
        g: RGenerator,
        shape: Union[int, Sequence[int]] = 1,
        mean: float = 0,
        stddev: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        if isinstance(dtype, str):
            dtype = dtype[-2:]
        if isinstance(shape, int):
            shape = (shape,)
        if dtype == "32":
            dtyper = tf.float32
        elif dtype == "64":
            dtyper = tf.float64
        elif not isinstance(dtype, str):
            dtyper = dtype
        return g.normal(shape, mean, stddev, dtype=dtyper)

    def stateful_randu(
        self,
        g: RGenerator,
        shape: Union[int, Sequence[int]] = 1,
        low: float = 0,
        high: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        if isinstance(dtype, str):
            dtype = dtype[-2:]
        if isinstance(shape, int):
            shape = (shape,)
        if dtype == "32":
            dtyper = tf.float32
        elif dtype == "64":
            dtyper = tf.float64
        elif not isinstance(dtype, str):
            dtyper = dtype
        return g.uniform(shape, minval=low, maxval=high, dtype=dtyper)

    def stateful_randc(
        self,
        g: RGenerator,
        a: Union[int, Sequence[int], Tensor],
        shape: Union[int, Sequence[int]],
        p: Optional[Union[Sequence[float], Tensor]] = None,
    ) -> Tensor:
        return _random_choice_tf(g, a, shape, p)

    def scatter(self, operand: Tensor, indices: Tensor, updates: Tensor) -> Tensor:
        return tf.tensor_scatter_nd_update(operand, indices, updates)

    def cond(
        self,
        pred: bool,
        true_fun: Callable[[], Tensor],
        false_fun: Callable[[], Tensor],
    ) -> Tensor:
        return tf.cond(pred, true_fun, false_fun)

    def switch(
        self: Any, index: Tensor, branches: Sequence[Callable[[], Tensor]]
    ) -> Tensor:
        return tf.switch_case(index, branches)

    def grad(
        self, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Callable[..., Any]:
        # experimental attempt
        # Note: tensorflow grad is gradient while jax grad is derivative, they are different with a conjugate!
        # And we DONT make them consitent by mannually set conjugate of the returns.
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

    def vjp(
        self,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        if not (isinstance(inputs, list) or isinstance(inputs, tuple)):
            # one input tensor
            one_input = True
            inputs = [inputs]
        else:
            one_input = False
        with tf.GradientTape() as t:
            t.watch(inputs)
            y = f(*inputs)
        g = t.gradient(y, inputs, v)
        if one_input:
            g = g[0]
        return y, g

    def jit(
        self,
        f: Callable[..., Any],
        static_argnums: Optional[Union[int, Sequence[int]]] = None,
        jit_compile: Optional[bool] = None,
    ) -> Any:
        # static_argnums not supported in tf case, this is only for a consistent interface
        # for more on static_argnums in tf.function, see issue: https://github.com/tensorflow/tensorflow/issues/52193
        if self.minor < 5:
            return tf.function(f, experimental_compile=jit_compile)
        else:
            return tf.function(f, jit_compile=jit_compile)

    # old vmap impl before I know pytrees support in tf
    # def vmap(
    #     self, f: Callable[..., Any], vectorized_argnums: Union[int, Sequence[int]] = 0
    # ) -> Any:
    #     if isinstance(vectorized_argnums, int):
    #         vectorized_argnums = (vectorized_argnums,)
    #     if vectorized_argnums == (0,):  # fast shortcut

    #         def wrapper(*args: Any, **kws: Any) -> Tensor:
    #             def pf(x: Tensor) -> Tensor:
    #                 return f(x, *args[1:], **kws)

    #             return tf.vectorized_map(pf, args[0])

    #     else:

    #         @self.jit
    #         def wrapper(*args: Any, **kws: Any) -> Tensor:
    #             shapes = []
    #             l = len(args)
    #             seps = [0]
    #             batch = args[vectorized_argnums[0]].shape[0]  # type: ignore
    #             nargs = []
    #             for i, arg in enumerate(args):
    #                 if i in vectorized_argnums:  # type: ignore
    #                     shapes.append(arg.shape[1:])
    #                     assert (
    #                         arg.shape[0] == batch
    #                     ), "different tensors has different batch dimensions!"
    #                     arg = tf.reshape(arg, [batch, -1])
    #                     nargs.append(arg)
    #                     seps.append(seps[-1] + arg.shape[-1])
    #             sargs = tf.concat(nargs, 1)

    #             def sf(sarg: Any) -> Any:
    #                 vargs = []
    #                 for i in range(len(shapes)):
    #                     arg = sarg[seps[i] : seps[i + 1]]
    #                     arg = tf.reshape(arg, shapes[i])
    #                     vargs.append(arg)
    #                 vvargs = []
    #                 j = 0
    #                 for i in range(l):
    #                     if i in vectorized_argnums:  # type: ignore
    #                         vvargs.append(vargs[j])
    #                         j += 1
    #                     else:
    #                         vvargs.append(args[i])
    #                 return f(*vvargs, **kws)

    #             return tf.vectorized_map(sf, sargs)

    #     return wrapper

    # def wrapper(f: Callable[..., Any], args: Sequence[Any]) -> Any:
    #     return f(*args)

    # wrapper = partial(wrapper, f)

    # def own_vectorized_map(f: Callable[..., Any], *args: Any) -> Any:
    #     return tf.vectorized_map(f, args)

    # return partial(own_vectorized_map, wrapper)

    def vmap(
        self, f: Callable[..., Any], vectorized_argnums: Union[int, Sequence[int]] = 0
    ) -> Any:
        if isinstance(vectorized_argnums, int):
            vectorized_argnums = (vectorized_argnums,)
        if vectorized_argnums == (0,):  # fast shortcut

            def wrapper(*args: Any, **kws: Any) -> Tensor:
                def pf(x: Tensor) -> Tensor:
                    return f(x, *args[1:], **kws)

                return tf.vectorized_map(pf, args[0])

        else:

            @self.jit  # otherwise, vectorized_map claim on retracing
            def wrapper(*args: Any, **kws: Any) -> Tensor:
                def sf(sarg: Any) -> Any:
                    vvargs = []
                    j = 0
                    for i in range(len(args)):
                        if i in vectorized_argnums:  # type: ignore
                            vvargs.append(sarg[j])
                            j += 1
                        else:
                            vvargs.append(args[i])
                    return f(*vvargs, **kws)

                sarg = []
                for i in vectorized_argnums:  # type: ignore
                    sarg.append(args[i])
                return tf.vectorized_map(sf, sarg)

        return wrapper

    def vectorized_value_and_grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        vectorized_argnums: Union[int, Sequence[int]] = 0,
    ) -> Callable[..., Tuple[Any, Any]]:
        # note how tf only works in this order, due to the bug reported as:
        # https://github.com/google/TensorNetwork/issues/940
        vf = self.vmap(f, vectorized_argnums=vectorized_argnums)

        def wrapper(
            *args: Any, **kws: Any
        ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
            with tf.GradientTape() as tape:
                if isinstance(argnums, int):
                    x = args[argnums]
                else:
                    x = [args[i] for i in argnums]
                tape.watch(x)
                vs = vf(*args, **kws)
            grad = tape.gradient(vs, x)
            return vs, grad

        return wrapper

        # f = self.vmap(f)
        # f = self.value_and_grad(f, argnums=argnums)
        # f = self.jit(f)
        # return f

    vvag = vectorized_value_and_grad


# TODO(@refraction-ray): lack stateful random methods implementation for now
# TODO(@refraction-ray): lack scatter impl for now
# To be added once pytorch backend is ready
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
        # in 2020, torch has no expm, hmmm. but that's ok,
        # it doesn't support complex numbers which is more severe issue.
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

    def stack(self: Any, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return torchlib.stack(a, dim=axis)

    def tile(self, a: Tensor, rep: Tensor) -> Tensor:
        return torchlib.tile(a, rep)

    def min(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        if axis is None:
            return torchlib.min(a)
        return torchlib.min(a, dim=axis).values

    def max(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        if axis is None:
            return torchlib.max(a)
        return torchlib.max(a, dim=axis).values

    def unique_with_counts(self, a: Tensor) -> Tuple[Tensor, Tensor]:
        return torchlib.unique(a, return_counts=True)  # type: ignore

    def relu(self, a: Tensor) -> Tensor:
        return torchlib.nn.ReLU(a)

    def softmax(self, a: Sequence[Tensor], axis: Optional[int] = None) -> Tensor:
        return torchlib.nn.Softmax(a, dim=axis)

    def onehot(self, a: Tensor, num: int) -> Tensor:
        return torchlib.nn.functional.one_hot(a, num)

    def cumsum(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        if axis is None:
            a = self.reshape(a, [-1])
            return torchlib.cumsum(a, dim=0)
        else:
            return torchlib.cumsum(a, dim=axis)

    def is_tensor(self, a: Any) -> bool:
        if isinstance(a, torchlib.Tensor):
            return True
        return False

    def cast(self, a: Tensor, dtype: str) -> Tensor:
        if isinstance(dtype, str):
            return a.type(getattr(torchlib, dtype))
        return a.type(dtype)

    def cond(
        self,
        pred: bool,
        true_fun: Callable[[], Tensor],
        false_fun: Callable[[], Tensor],
    ) -> Tensor:
        if pred:
            return true_fun()
        return false_fun()

    def switch(self, index: Tensor, branches: Sequence[Callable[[], Tensor]]) -> Tensor:
        return branches[index]()

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

    def vjp(
        self,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        if isinstance(v, list):
            v = tuple(v)
        return torchlib.autograd.functional.vjp(f, inputs, v)  # type: ignore

    def vmap(
        self,
        f: Callable[..., Any],
        vectorized_argnums: Optional[Union[int, Sequence[int]]] = None,
    ) -> Any:
        warnings.warn(
            "pytorch backend has no intrinsic vmap like interface"
            ", use plain for loop for compatibility"
        )
        # the vmap support is vey limited, f must return one tensor
        # nested list of tensor as return is not supported
        if isinstance(vectorized_argnums, int):
            vectorized_argnums = (vectorized_argnums,)

        def wrapper(*args: Any, **kws: Any) -> Tensor:
            results = []
            for barg in zip(*[args[i] for i in vectorized_argnums]):  # type: ignore
                narg = []
                j = 0
                for k in range(len(args)):
                    if k in vectorized_argnums:  # type: ignore
                        narg.append(barg[j])
                        j += 1
                    else:
                        narg.append(args[k])
                results.append(f(*narg, **kws))
            return torchlib.stack(results)

        return wrapper

        # def vmapf(*args: Tensor, **kws: Any) -> Tensor:
        #     r = []
        #     for i in range(args[0].shape[0]):
        #         nargs = [arg[i] for arg in args]
        #         r.append(f(*nargs, **kws))
        #     return torchlib.stack(r)

        # return vmapf

        # raise NotImplementedError("pytorch backend doesn't support vmap")
        # There seems to be no map like architecture in pytorch for now
        # see https://discuss.pytorch.org/t/fast-way-to-use-map-in-pytorch/70814

    def jit(
        self,
        f: Callable[..., Any],
        static_argnums: Optional[Union[int, Sequence[int]]] = None,
        jit_compile: Optional[bool] = None,
    ) -> Any:
        return f  # do nothing here until I figure out what torch.jit is for and how does it work
        # see https://github.com/pytorch/pytorch/issues/36910

    def vectorized_value_and_grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        vectorized_argnums: Union[int, Sequence[int]] = 0,
    ) -> Callable[..., Tuple[Any, Any]]:
        # [WIP], not a consistent impl compared to tf and jax backend, but pytorch backend is not fully supported anyway
        f = self.value_and_grad(f, argnums=argnums)
        f = self.vmap(f, vectorized_argnums=vectorized_argnums)
        # f = self.jit(f)
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
