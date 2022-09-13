"""
Backend magic inherited from tensornetwork: abstract backend
"""
# pylint: disable=invalid-name
# pylint: disable=unused-variable

from functools import reduce, partial
from operator import mul
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from ..utils import return_partial

Tensor = Any


class ExtendedBackend:
    """
    Add tensorcircuit specific backend methods, especially with their docstrings.
    """

    def copy(self: Any, a: Tensor) -> Tensor:
        """
        Return the copy of ``a``, matrix exponential.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: matrix exponential of matrix ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `copy`.".format(self.name)
        )

    def expm(self: Any, a: Tensor) -> Tensor:
        """
        Return the expm of tensor ''a''.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: matrix exponential of matrix ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `expm`.".format(self.name)
        )

    def sqrtmh(self: Any, a: Tensor) -> Tensor:
        """
        Return the sqrtm of a Hermitian matrix ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: sqrtm of ``a``
        :rtype: Tensor
        """
        # maybe friendly for AD and also cosidering that several backend has no support for native sqrtm
        e, v = self.eigh(a)
        e = self.sqrt(e)
        return v @ self.diagflat(e) @ self.adjoint(v)

    def eigvalsh(self: Any, a: Tensor) -> Tensor:
        """
        Get the eigenvalues of matrix ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: eigenvalues of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `eigvalsh`.".format(self.name)
        )

    def sin(self: Any, a: Tensor) -> Tensor:
        """
        Return the  elementwise sine of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: sine of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `sin`.".format(self.name)
        )

    def cos(self: Any, a: Tensor) -> Tensor:
        """
        Return the cosine of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: cosine of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `cos`.".format(self.name)
        )

    def acos(self: Any, a: Tensor) -> Tensor:
        """
        Return the acos of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: acos of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `acos`.".format(self.name)
        )

    def acosh(self: Any, a: Tensor) -> Tensor:
        """
        Return the acosh of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: acosh of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `acosh`.".format(self.name)
        )

    def asin(self: Any, a: Tensor) -> Tensor:
        """
        Return the acos of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: asin of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `asin`.".format(self.name)
        )

    def asinh(self: Any, a: Tensor) -> Tensor:
        """
        Return the asinh of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: asinh of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `asinh`.".format(self.name)
        )

    def atan(self: Any, a: Tensor) -> Tensor:
        """
        Return the atan of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: atan of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `atan`.".format(self.name)
        )

    def atan2(self: Any, y: Tensor, x: Tensor) -> Tensor:
        """
        Return the atan of a tensor ``y``/``x``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: atan2 of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `atan2`.".format(self.name)
        )

    def atanh(self: Any, a: Tensor) -> Tensor:
        """
        Return the atanh of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: atanh of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `atanh`.".format(self.name)
        )

    def cosh(self: Any, a: Tensor) -> Tensor:
        """
        Return the cosh of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: cosh of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `cosh`.".format(self.name)
        )

    def tan(self: Any, a: Tensor) -> Tensor:
        """
        Return the tan of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: tan of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `tan`.".format(self.name)
        )

    def tanh(self: Any, a: Tensor) -> Tensor:
        """
        Return the tanh of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: tanh of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `tanh`.".format(self.name)
        )

    def sinh(self: Any, a: Tensor) -> Tensor:
        """
        Return the sinh of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: sinh of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `sinh`.".format(self.name)
        )

    # acos acosh asin asinh atan atan2 atanh cosh (cos) tan tanh (sin) sinh

    def abs(self: Any, a: Tensor) -> Tensor:
        """
        Return the elementwise abs value of a matrix ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: abs of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `abs`.".format(self.name)
        )

    def kron(self: Any, a: Tensor, b: Tensor) -> Tensor:
        """
        Return the kronecker product of two matrices ``a`` and ``b``.

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
        Return the total number of elements in ``a`` in tensor form.

        :param a: tensor
        :type a: Tensor
        :return: the total number of elements in ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `size`.".format(self.name)
        )

    def sizen(self: Any, a: Tensor) -> int:
        """
        Return the total number of elements in tensor ``a``, but in integer form.

        :param a: tensor
        :type a: Tensor
        :return: the total number of elements in tensor ``a``
        :rtype: int
        """
        return reduce(mul, list(a.shape) + [1])  # type: ignore

    def numpy(self: Any, a: Tensor) -> Tensor:
        """
        Return the numpy array of a tensor ``a``, but may not work in a jitted function.

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
        Return the elementwise real value of a tensor ``a``.

        :param a: tensor
        :type a: Tensor
        :return: real value of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `real`.".format(self.name)
        )

    def imag(self: Any, a: Tensor) -> Tensor:
        """
        Return the elementwise imaginary value of a tensor ``a``.

        :param a: tensor
        :type a: Tensor
        :return: imaginary value of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `imag`.".format(self.name)
        )

    def adjoint(self: Any, a: Tensor) -> Tensor:
        """
        Return the conjugate and transpose of a tensor ``a``

        :param a: Input tensor
        :type a: Tensor
        :return: adjoint tensor of ``a``
        :rtype: Tensor
        """
        return self.conj(self.transpose(a))

    def i(self: Any, dtype: str) -> Tensor:
        """
        Return 1.j in as a tensor compatible with the backend.

        :param dtype: "complex64" or "complex128"
        :type dtype: str
        :return: 1.j tensor
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `i`.".format(self.name)
        )

    def reshape2(self: Any, a: Tensor) -> Tensor:
        """
        Reshape a tensor to the [2, 2, ...] shape.

        :param a: Input tensor
        :type a: Tensor
        :return: the reshaped tensor
        :rtype: Tensor
        """
        nleg = int(np.log2(self.sizen(a)))
        a = self.reshape(a, [2 for _ in range(nleg)])
        return a

    def reshapem(self: Any, a: Tensor) -> Tensor:
        """
        Reshape a tensor to the [l, l] shape.

        :param a: Input tensor
        :type a: Tensor
        :return: the reshaped tensor
        :rtype: Tensor
        """
        l = int(np.sqrt(self.sizen(a)))
        a = self.reshape(a, [l, l])
        return a

    def dtype(self: Any, a: Tensor) -> str:
        """
        Obtain dtype string for tensor ``a``

        :param a: The tensor
        :type a: Tensor
        :return: dtype str, such as "complex64"
        :rtype: str
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `dtype`.".format(self.name)
        )

    def stack(self: Any, a: Sequence[Tensor], axis: int = 0) -> Tensor:
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

    def concat(self: Any, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        """
        Join a sequence of arrays along an existing axis.

        :param a: [description]
        :type a: Sequence[Tensor]
        :param axis: [description], defaults to 0
        :type axis: int, optional
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `concat`.".format(self.name)
        )

    def tile(self: Any, a: Tensor, rep: Tensor) -> Tensor:
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

    def mean(
        self: Any,
        a: Tensor,
        axis: Optional[Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        """
        Compute the arithmetic mean for ``a`` along the specified ``axis``.

        :param a: tensor to take average
        :type a: Tensor
        :param axis: the axis to take mean, defaults to None indicating sum over flatten array
        :type axis: Optional[Sequence[int]], optional
        :param keepdims: _description_, defaults to False
        :type keepdims: bool, optional
        :return: _description_
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `mean`.".format(self.name)
        )

    def min(self: Any, a: Tensor, axis: Optional[int] = None) -> Tensor:
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

    def max(self: Any, a: Tensor, axis: Optional[int] = None) -> Tensor:
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

    def argmax(self: Any, a: Tensor, axis: int = 0) -> Tensor:
        """
        Return the index of maximum of an array an axis.

        :param a: [description]
        :type a: Tensor
        :param axis: [description], defaults to 0, different behavior from numpy defaults!
        :type axis: int
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `argmax`.".format(self.name)
        )

    def argmin(self: Any, a: Tensor, axis: int = 0) -> Tensor:
        """
        Return the index of minimum of an array an axis.

        :param a: [description]
        :type a: Tensor
        :param axis: [description], defaults to 0, different behavior from numpy defaults!
        :type axis: int
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `argmin`.".format(self.name)
        )

    def unique_with_counts(self: Any, a: Tensor, **kws: Any) -> Tuple[Tensor, Tensor]:
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

    def sigmoid(self: Any, a: Tensor) -> Tensor:
        """
        Compute sigmoid of input ``a``

        :param a: [description]
        :type a: Tensor
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `sigmoid`.".format(self.name)
        )

    def relu(self: Any, a: Tensor) -> Tensor:
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

    def softmax(self: Any, a: Sequence[Tensor], axis: Optional[int] = None) -> Tensor:
        """
        Softmax function.
        Computes the function which rescales elements to the range [0,1] such that the elements along axis sum to 1.

        .. math ::

            \\mathrm{softmax}(x) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}


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

    def onehot(self: Any, a: Tensor, num: int) -> Tensor:
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
    def one_hot(self: Any, a: Tensor, num: int) -> Tensor:
        """
        See doc for :py:meth:`onehot`
        """
        return self.onehot(a, num)

    def cumsum(self: Any, a: Tensor, axis: Optional[int] = None) -> Tensor:
        """
        Return the cumulative sum of the elements along a given axis.

        :param a: [description]
        :type a: Tensor
        :param axis: The default behavior is the same as numpy, different from tf/torch
            as cumsum of the flatten 1D array, defaults to None
        :type axis: Optional[int], optional
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `cumsum`.".format(self.name)
        )

    def is_tensor(self: Any, a: Tensor) -> bool:
        """
        Return a boolean on whether ``a`` is a tensor in backend package.

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

    def mod(self: Any, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute y-mod of x (negative number behavior is not guaranteed to be consistent)

        :param x: input values
        :type x: Tensor
        :param y: mod ``y``
        :type y: Tensor
        :return: results
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `mod`.".format(self.name)
        )

    def reverse(self: Any, a: Tensor) -> Tensor:
        """
        return ``a[::-1]``, only 1D tensor is guaranteed for consistent behavior

        :param a: 1D tensor
        :type a: Tensor
        :return: 1D tensor in reverse order
        :rtype: Tensor
        """
        return a[::-1]

    def right_shift(self: Any, x: Tensor, y: Tensor) -> Tensor:
        """
        Shift the bits of an integer x to the right y bits.

        :param x: input values
        :type x: Tensor
        :param y: Number of bits shift to ``x``
        :type y: Tensor
        :return: result with the same shape as ``x``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `right_shift`.".format(self.name)
        )

    def left_shift(self: Any, x: Tensor, y: Tensor) -> Tensor:
        """
        Shift the bits of an integer x to the left y bits.

        :param x: input values
        :type x: Tensor
        :param y: Number of bits shift to ``x``
        :type y: Tensor
        :return: result with the same shape as ``x``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `left_shift`.".format(self.name)
        )

    def arange(
        self: Any, start: int, stop: Optional[int] = None, step: int = 1
    ) -> Tensor:
        """
        Values are generated within the half-open interval [start, stop)

        :param start: start index
        :type start: int
        :param stop: end index, defaults to None
        :type stop: Optional[int], optional
        :param step: steps, defaults to 1
        :type step: Optional[int], optional
        :return: _description_
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `arange`.".format(self.name)
        )

    def solve(self: Any, A: Tensor, b: Tensor, **kws: Any) -> Tensor:
        """
        Solve the linear system Ax=b and return the solution x.

        :param A: The multiplied matrix.
        :type A: Tensor
        :param b: The resulted matrix.
        :type b: Tensor
        :return: The solution of the linear system.
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `solve`.".format(self.name)
        )

    def tree_map(self: Any, f: Callable[..., Any], *pytrees: Any) -> Any:
        """
        Return the new tree map with multiple arg function ``f`` through pytrees.

        :param f: The function
        :type f: Callable[..., Any]
        :param pytrees: inputs as any python structure
        :type pytrees: Any
        :raises NotImplementedError: raise when neither tensorflow or jax is installed.
        :return: The new tree map with the same structure but different values.
        :rtype: Any
        """
        try:
            import tensorflow as tf

        except ImportError:
            raise NotImplementedError("No installed ML backend for `tree_map`")

        return tf.nest.map_structure(f, *pytrees)

    def tree_flatten(self: Any, pytree: Any) -> Tuple[Any, Any]:
        """
        Flatten python structure to 1D list

        :param pytree: python structure to be flattened
        :type pytree: Any
        :return: The 1D list of flattened structure and treedef
            which can be used for later unflatten
        :rtype: Tuple[Any, Any]
        """
        try:
            import tensorflow as tf

        except ImportError:
            raise NotImplementedError("No installed ML backend for `tree_flatten`")

        leaves = tf.nest.flatten(pytree)
        treedef = pytree

        return leaves, treedef

    def tree_unflatten(self: Any, treedef: Any, leaves: Any) -> Any:
        """
        Pack 1D list to pytree defined via ``treedef``

        :param treedef: Def of pytree structure, the second return from ``tree_flatten``
        :type treedef: Any
        :param leaves: the 1D list of flattened data structure
        :type leaves: Any
        :return: Packed pytree
        :rtype: Any
        """
        try:
            import tensorflow as tf

        except ImportError:
            raise NotImplementedError("No installed ML backend for `tree_unflatten`")

        return tf.nest.pack_sequence_as(treedef, leaves)

    def to_dlpack(self: Any, a: Tensor) -> Any:
        """
        Transform the tensor ``a`` as a dlpack capsule

        :param a: _description_
        :type a: Tensor
        :return: _description_
        :rtype: Any
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `to_dlpack`.".format(self.name)
        )

    def from_dlpack(self: Any, a: Any) -> Tensor:
        """
        Transform a dlpack capsule to a tensor

        :param a: the dlpack capsule
        :type a: Any
        :return: _description_
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `from_dlpack`.".format(self.name)
        )

    # TODO(@refraction-ray): omit numpy implementation of dlpack for now
    # their convention is not consistent with other ML libs and requires a rather high version
    # https://github.com/numpy/numpy/issues/19013
    # complex dlpack may still have issues across various backends

    def set_random_state(
        self: Any, seed: Optional[int] = None, get_only: bool = False
    ) -> Any:
        """
        Set the random state attached to the backend.

        :param seed: the random seed, defaults to be None
        :type seed: Optional[int], optional
        :param get_only: If set to be true, only get the random state in return
            instead of setting the state on the backend
        :type get_only: bool, defaults to be False
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `set_random_state`.".format(self.name)
        )

    def get_random_state(self: Any, seed: Optional[int] = None) -> Any:
        """
        Get the backend specific random state object.

        :param seed: [description], defaults to be None
        :type seed: Optional[int], optional
        :return:the backend specific random state object
        :rtype: Any
        """
        return self.set_random_state(seed, True)

    def random_split(self: Any, key: Any) -> Tuple[Any, Any]:
        """
        A jax like split API, but it doesn't split the key generator for other backends.
        It is just for a consistent interface of random code;
        make sure you know what the function actually does.
        This function is mainly a utility to write backend agnostic code instead of doing magic things.

        :param key: [description]
        :type key: Any
        :return: [description]
        :rtype: Tuple[Any, Any]
        """
        return key, key

    # Though try hard, the current random API abstraction may not be perfect when with nested jit or vmap
    # so keep every random function a direct status parameters in case for development.

    def implicit_randn(
        self: Any,
        shape: Union[int, Sequence[int]] = 1,
        mean: float = 0,
        stddev: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        """
        Call the random normal function with the random state management behind the scene.

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

    def stateful_randn(
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

    def implicit_randu(
        self: Any,
        shape: Union[int, Sequence[int]] = 1,
        low: float = 0,
        high: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        """
        Call the random normal function with the random state management behind the scene.

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

    def stateful_randu(
        self: Any,
        g: Any,
        shape: Union[int, Sequence[int]] = 1,
        low: float = 0,
        high: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        """
        Uniform random sampler from ``low`` to ``high``.

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

    def implicit_randc(
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

    def stateful_randc(
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

    def gather1d(self: Any, operand: Tensor, indices: Tensor) -> Tensor:
        """
        Return ``operand[indices]``, both ``operand`` and ``indices`` are rank-1 tensor.

        :param operand: rank-1 tensor
        :type operand: Tensor
        :param indices: rank-1 tensor with int dtype
        :type indices: Tensor
        :return: ``operand[indices]``
        :rtype: Tensor
        """
        return operand[indices]

    def scatter(self: Any, operand: Tensor, indices: Tensor, updates: Tensor) -> Tensor:
        """
        Roughly equivalent to operand[indices] = updates, indices only support shape with rank 2 for now.

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

    def coo_sparse_matrix(
        self: Any, indices: Tensor, values: Tensor, shape: Tensor
    ) -> Tensor:
        """
        Generate the coo format sparse matrix from indices and values,
        which is the only sparse format supported in different ML backends.

        :param indices: shape [n, 2] for n non zero values in the returned matrix
        :type indices: Tensor
        :param values: shape [n]
        :type values: Tensor
        :param shape: Tuple[int, ...]
        :type shape: Tensor
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `coo`.".format(self.name)
        )

    def coo_sparse_matrix_from_numpy(self: Any, a: Tensor) -> Tensor:
        """
        Generate the coo format sparse matrix from scipy coo sparse matrix.

        :param a: Scipy coo format sparse matrix
        :type a: Tensor
        :return: SparseTensor in backend format
        :rtype: Tensor
        """
        return self.coo_sparse_matrix(
            indices=np.array([a.row, a.col]).T,
            values=a.data,
            shape=a.shape,
        )

    def sparse_dense_matmul(
        self: Any,
        sp_a: Tensor,
        b: Tensor,
    ) -> Tensor:
        """
        A sparse matrix multiplies a dense matrix.

        :param sp_a: a sparse matrix
        :type sp_a: Tensor
        :param b: a dense matrix
        :type b: Tensor
        :return: dense matrix
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `sparse_dense_matmul`.".format(self.name)
        )

    def to_dense(self: Any, sp_a: Tensor) -> Tensor:
        """
        Convert a sparse matrix to dense tensor.

        :param sp_a: a sparse matrix
        :type sp_a: Tensor
        :return: the resulted dense matrix
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `to_dense`.".format(self.name)
        )

    def is_sparse(self: Any, a: Tensor) -> bool:
        """
        Determine whether the type of input ``a`` is  ``sparse``.

        :param a: input matrix ``a``
        :type a: Tensor
        :return: a bool indicating whether the matrix ``a`` is sparse
        :rtype: bool
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `is_sparse`.".format(self.name)
        )

    def device(self: Any, a: Tensor) -> str:
        """
        get the universal device str for the tensor, in the format of tf

        :param a: the tensor
        :type a: Tensor
        :return: device str where the tensor lives on
        :rtype: str
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `device`.".format(self.name)
        )

    def device_move(self: Any, a: Tensor, dev: Any) -> Tensor:
        """
        move tensor ``a`` to device ``dev``

        :param a: the tensor
        :type a: Tensor
        :param dev: device str or device obj in corresponding backend
        :type dev: Any
        :return: the tensor on new device
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `device_move`.".format(self.name)
        )

    def _dev2str(self: Any, dev: Any) -> str:
        """
        device object to universal dev str

        :param dev: device object
        :type dev: Any
        :return: dev str
        :rtype: str
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `_dev2str`.".format(self.name)
        )

    def _str2dev(self: Any, str_: str) -> Any:
        """
        device object to universal dev str

        :param str_: dev str
        :type str_: str
        :return: device object
        :rtype: Any
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `_str2dev`.".format(self.name)
        )

    def cond(
        self: Any,
        pred: bool,
        true_fun: Callable[[], Tensor],
        false_fun: Callable[[], Tensor],
    ) -> Tensor:
        """
        The native cond for XLA compiling, wrapper for ``tf.cond`` and limited functionality of ``jax.lax.cond``.

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

    def switch(
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

    def stop_gradient(self: Any, a: Tensor) -> Tensor:
        """
        Stop backpropagation from ``a``.

        :param a: [description]
        :type a: Tensor
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `stop_gradient`.".format(self.name)
        )

    def grad(
        self: Any,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Any]:
        """
        Return the function which is the grad function of input ``f``.

        :Example:

        >>> f = lambda x,y: x**2+2*y
        >>> g = tc.backend.grad(f)
        >>> g(tc.num_to_tensor(1),tc.num_to_tensor(2))
        2
        >>> g = tc.backend.grad(f, argnums=(0,1))
        >>> g(tc.num_to_tensor(1),tc.num_to_tensor(2))
        [2, 2]

        :param f: the function to be differentiated
        :type f: Callable[..., Any]
        :param argnums: the position of args in ``f`` that are to be differentiated, defaults to be 0
        :type argnums: Union[int, Sequence[int]], optional
        :return: the grad function of ``f`` with the same set of arguments as ``f``
        :rtype: Callable[..., Any]
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `grad`.".format(self.name)
        )

    def value_and_grad(
        self: Any,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        hax_aux: bool = False,
    ) -> Callable[..., Tuple[Any, Any]]:
        """
        Return the function which returns the value and grad of ``f``.

        :Example:

        >>> f = lambda x,y: x**2+2*y
        >>> g = tc.backend.value_and_grad(f)
        >>> g(tc.num_to_tensor(1),tc.num_to_tensor(2))
        5, 2
        >>> g = tc.backend.value_and_grad(f, argnums=(0,1))
        >>> g(tc.num_to_tensor(1),tc.num_to_tensor(2))
        5, [2, 2]

        :param f: the function to be differentiated
        :type f: Callable[..., Any]
        :param argnums: the position of args in ``f`` that are to be differentiated, defaults to be 0
        :type argnums: Union[int, Sequence[int]], optional
        :return: the value and grad function of ``f`` with the same set of arguments as ``f``
        :rtype: Callable[..., Tuple[Any, Any]]
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `value_and_grad`.".format(self.name)
        )

    def jvp(
        self: Any,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        """
        Function that computes a (forward-mode) Jacobian-vector product of ``f``.
        Strictly speaking, this function is value_and_jvp.

        :param f: The function to compute jvp
        :type f: Callable[..., Any]
        :param inputs: input for ``f``
        :type inputs: Union[Tensor, Sequence[Tensor]]
        :param v: tangents
        :type v: Union[Tensor, Sequence[Tensor]]
        :return: (``f(*inputs)``, jvp_tensor), where jvp_tensor is the same shape as the output of ``f``
        :rtype: Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `jvp`.".format(self.name)
        )

    def vjp(
        self: Any,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        """
        Function that computes the dot product between a vector v and the Jacobian
        of the given function at the point given by the inputs. (reverse mode AD relevant)
        Strictly speaking, this function is value_and_vjp.

        :param f: the function to carry out vjp calculation
        :type f: Callable[..., Any]
        :param inputs: input for ``f``
        :type inputs: Union[Tensor, Sequence[Tensor]]
        :param v: value vector or gradient from downstream in reverse mode AD
            the same shape as return of function ``f``
        :type v: Union[Tensor, Sequence[Tensor]]
        :return: (``f(*inputs)``, vjp_tensor), where vjp_tensor is the same shape as inputs
        :rtype: Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `vjp`.".format(self.name)
        )

    def jacfwd(
        self: Any, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Tensor:
        """
        Compute the Jacobian of ``f`` using the forward mode AD.

        :param f: the function whose Jacobian is required
        :type f: Callable[..., Any]
        :param argnums: the position of the arg as Jacobian input, defaults to 0
        :type argnums: Union[int, Sequence[int]], optional
        :return: outer tuple for input args, inner tuple for outputs
        :rtype: Tensor
        """
        # TODO(@refraction-ray): support pytree for input and output
        jvp1 = return_partial(self.jvp, return_argnums=1)
        if isinstance(argnums, int):
            argnums = (argnums,)

        def _transform(t: Tensor, input_shape: List[int]) -> Tensor:
            output_shape = list(self.shape_tuple(t))[1:]
            t = self.reshape(t, [t.shape[0], -1])
            t = self.transpose(t)
            t = self.reshape(t, output_shape + input_shape)
            return t

        def wrapper(*args: Any, **kws: Any) -> Any:
            pf = partial(f, **kws)
            jjs = []
            for argnum in argnums:  # type: ignore
                jj = self.vmap(jvp1, vectorized_argnums=2)(
                    pf,
                    args,
                    tuple(
                        [
                            self.reshape(
                                self.eye(self.sizen(arg), dtype=arg.dtype),
                                [-1] + list(self.shape_tuple(arg)),
                            )
                            if i == argnum
                            else self.reshape(
                                self.zeros(
                                    [self.sizen(arg), self.sizen(arg)], dtype=arg.dtype
                                ),
                                [-1] + list(self.shape_tuple(arg)),
                            )
                            for i, arg in enumerate(args)
                        ]
                    ),
                )
                jj = self.tree_map(
                    partial(
                        _transform, input_shape=list(self.shape_tuple(args[argnum]))
                    ),
                    jj,
                )
                jjs.append(jj)
            if len(jjs) == 1:
                return jjs[0]
            return tuple(jjs)

        return wrapper

    def jacrev(
        self: Any, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Tensor:
        """
        Compute the Jacobian of ``f`` using reverse mode AD.

        :param f: The function whose Jacobian is required
        :type f: Callable[..., Any]
        :param argnums: the position of the arg as Jacobian input, defaults to 0
        :type argnums: Union[int, Sequence[int]], optional
        :return: outer tuple for output, inner tuple for input args
        :rtype: Tensor
        """
        # not that stable API
        vjp1 = return_partial(self.vjp, return_argnums=1)
        if isinstance(argnums, int):
            argnums = (argnums,)

        def wrapper(*args: Any, **kws: Any) -> Any:
            pf = partial(f, **kws)
            values = f(*args, **kws)  # TODO(@refraction-ray): any way to save this?
            collect = tuple
            if isinstance(values, list):
                values = tuple(values)
                collect = list  # type: ignore
            elif not isinstance(values, tuple):
                values = tuple([values])

                def _first(x: Sequence[Any]) -> Any:
                    return x[0]

                collect = _first  # type: ignore

            jjs = []
            for k in range(len(values)):  # type: ignore
                jj = self.vmap(vjp1, vectorized_argnums=2)(
                    pf,
                    args,
                    collect(
                        [
                            self.reshape(
                                self.eye(self.sizen(v), dtype=v.dtype),
                                [-1] + list(self.shape_tuple(v)),
                            )
                            if i == k
                            else self.reshape(
                                self.zeros(
                                    [self.sizen(v), self.sizen(v)], dtype=v.dtype
                                ),
                                [-1] + list(self.shape_tuple(v)),
                            )
                            for i, v in enumerate(values)
                        ]
                    ),
                )
                jj = self.tree_map(
                    lambda _: self.reshape(
                        _,
                        list(self.shape_tuple(values[k]))
                        + list(self.shape_tuple(_))[1:],
                    ),
                    jj,
                )
                if len(jj) == 1:
                    jj = jj[0]
                jjs.append(jj)
            if len(jjs) == 1:
                return jjs[0]
            return tuple(jjs)

        return wrapper

    jacbwd = jacrev

    def hessian(
        self: Any, f: Callable[..., Any], argnums: Union[int, Sequence[int]] = 0
    ) -> Tensor:
        # different by conjugate for tf and jax backend
        # tensorflow native hessian use naive for loop
        # idealy fwd over rev is the best choice, but this combination fails on tensorflow backend
        # (solved via commit abe771eff somehow)
        # original error with complains like: AttributeError: 'IndexedSlices' object has no attribute '_id'
        return self.jacfwd(self.jacrev(f, argnums=argnums), argnums=argnums)

    def jit(
        self: Any,
        f: Callable[..., Any],
        static_argnums: Optional[Union[int, Sequence[int]]] = None,
        jit_compile: Optional[bool] = None,
    ) -> Callable[..., Any]:
        """
        Return the jitted version of function ``f``.

        :param f: function to be jitted
        :type f: Callable[..., Any]
        :param static_argnums: index of args that doesn't regarded as tensor,
            only work for jax backend
        :type static_argnums: Optional[Union[int, Sequence[int]]], defaults to None
        :param jit_compile: whether open XLA compilation, only works for tensorflow backend,
            defaults False since several ops has no XLA correspondence
        :type jit_compile: bool
        :return: jitted version of ``f``
        :rtype: Callable[..., Any]
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `jit`.".format(self.name)
        )

    def vmap(
        self: Any,
        f: Callable[..., Any],
        vectorized_argnums: Union[int, Sequence[int]] = 0,
    ) -> Any:
        """
        Return the vectorized map or batched version of ``f`` on the first extra axis.
        The general interface supports ``f`` with multiple arguments and broadcast in the fist dimension.

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

    def vectorized_value_and_grad(
        self: Any,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        vectorized_argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Tuple[Any, Any]]:
        """
        Return the VVAG function of ``f``. The inputs for ``f`` is (args[0], args[1], args[2], ...),
        and the output of ``f`` is a scalar. Suppose VVAG(f) is a function with inputs in the form
        (vargs[0], args[1], args[2], ...), where vagrs[0] has one extra dimension than args[0] in the first axis
        and consistent with args[0] in shape for remaining dimensions, i.e. shape(vargs[0]) = [batch] + shape(args[0]).
        (We only cover cases where ``vectorized_argnums`` defaults to 0 here for demonstration).
        VVAG(f) returns a tuple as a value tensor with shape [batch, 1] and a gradient tuple with shape:
        ([batch]+shape(args[argnum]) for argnum in argnums). The gradient for argnums=k is defined as

        .. math::

            g^k = \\frac{\\partial \\sum_{i\\in batch} f(vargs[0][i], args[1], ...)}{\\partial args[k]}

        Therefore, if argnums=0, the gradient is reduced to

        .. math::

            g^0_i = \\frac{\\partial f(vargs[0][i])}{\\partial vargs[0][i]}

        , which is specifically suitable for batched VQE optimization, where args[0] is the circuit parameters.

        And if argnums=1, the gradient is like

        .. math::
            g^1_i = \\frac{\\partial \\sum_j f(vargs[0][j], args[1])}{\\partial args[1][i]}

        , which is suitable for quantum machine learning scenarios, where ``f`` is the loss function,
        args[0] corresponds to the input data and args[1] corresponds to the weights in the QML model.

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

    def __repr__(self: Any) -> str:
        return self.name + "_backend"  # type: ignore

    __str__ = __repr__
