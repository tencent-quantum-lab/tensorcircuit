"""
backend magic inherited from tensornetwork: abstract backend
"""

import inspect
from functools import reduce
from operator import mul
from typing import Any, Callable, Optional, Sequence, Tuple, Union

try:  # old version tn compatiblity
    from tensornetwork.backends import base_backend

    tnbackend = base_backend.BaseBackend

except ImportError:
    from tensornetwork.backends import abstract_backend

    tnbackend = abstract_backend.AbstractBackend


Tensor = Any


def _more_methods_for_backend(tnbackend: Any) -> None:
    """
    Add tensorcircuit specific backend methods, especially with their docstrings
    """

    def copy(self: Any, a: Tensor) -> Tensor:  # pylint: disable=unused-variable
        """
        Return expm of ``a``, matrix exponential.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: matrix exponential of matrix ``a``
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `copy`.".format(self.name)
        )

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
        try:
            import jax as libjax

            has_jax = True
        except ImportError:
            has_jax = False
            try:
                import tensorflow as tf

                has_tf = True
            except ImportError:
                has_tf = False

        if has_jax:
            r = libjax.tree_map(f, *pytrees)
        elif has_tf:
            r = tf.nest.map_structure(f, *pytrees)
        else:
            raise NotImplementedError("Only tensorflow and jax support `tree_map`")

        return r

    def set_random_state(  # pylint: disable=unused-variable
        self: Any, seed: Optional[int] = None, get_only: bool = False
    ) -> Any:
        """
        set random state attached in the backend

        :param seed: int, defaults to None
        :type seed: Optional[int], optional
        :param get_only:
        :type get_only: bool
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `set_random_state`.".format(self.name)
        )

    def get_random_state(  # pylint: disable=unused-variable
        self: Any, seed: Optional[int] = None
    ) -> Any:
        """
        get backend specific random state object

        :param seed: [description], defaults to None
        :type seed: Optional[int], optional
        :return: [description]
        :rtype: Any
        """
        return self.set_random_state(seed, True)

    def random_split(  # pylint: disable=unused-variable
        self: Any, key: Any
    ) -> Tuple[Any, Any]:
        """
        a jax like split API, but does't split the key generator for other backends.
        just for a consistent interface of random code, be careful that you know what the function actually does.

        :param key: [description]
        :type key: Any
        :return: [description]
        :rtype: Tuple[Any, Any]
        """
        return key, key

    # Though try hard, the current random API abstraction may not be perfect when with nested jit or vmap
    # so keep every random function a direct status parameters in case.

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

    def coo_sparse_matrix(  # pylint: disable=unused-variable
        self: Any, indices: Tensor, values: Tensor, shape: Tensor
    ) -> Tensor:
        """
        generate coo format sparse matrix from indices and values,
        the only sparse format supported in different ML backends

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

    def sparse_dense_matmul(  # pylint: disable=unused-variable
        self: Any,
        sp_a: Tensor,
        b: Tensor,
    ) -> Tensor:
        """
        sparse matrix times dense matrix

        :param sp_a: [description]
        :type sp_a: Tensor
        :param b: [description]
        :type b: Tensor
        :return: dense matrix
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `sparse_dense_matmul`.".format(self.name)
        )

    def to_dense(self: Any, sp_a: Tensor) -> Tensor:  # pylint: disable=unused-variable
        """
        convert sparse matrix to dense tensor

        :param sp_a: [description]
        :type sp_a: Tensor
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `to_dense`.".format(self.name)
        )

    def is_sparse(self: Any, a: Tensor) -> bool:  # pylint: disable=unused-variable
        """
        determine whether the type of input ``a`` is of sparse type

        :param a: [description]
        :type a: Tensor
        :return: [description]
        :rtype: bool
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `is_sparse`.".format(self.name)
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
        self: Any,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
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
        self: Any,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        hax_aux: bool = False,
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

    def jvp(  # pylint: disable=unused-variable
        self: Any,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        """
        Function that computes a (forward-mode) Jacobian-vector product of ``f``.
        Strictly speaking, this function is value_and_jvp

        :param f: The function to compute jvp
        :type f: Callable[..., Any]
        :param inputs: primals
        :type inputs: Union[Tensor, Sequence[Tensor]]
        :param v: tangents
        :type v: Union[Tensor, Sequence[Tensor]]
        :return: (``f(*inputs)``, jvp_tensor), where jvp_tensor is the same shape as output of ``f``
        :rtype: Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]
        """
        raise NotImplementedError(
            "Backend '{}' has not implemented `jvp`.".format(self.name)
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
            g^1_i = \\frac{\\partial \\sum_j f(vargs[0][j], args[1])}{\\partial args[1][i]}

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
