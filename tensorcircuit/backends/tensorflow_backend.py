"""
Backend magic inherited from tensornetwork: tensorflow backend
"""

# pylint: disable=invalid-name

import os
import re
from functools import reduce, partial
from operator import mul
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from scipy.sparse import coo_matrix
import tensornetwork
from tensornetwork.backends.tensorflow import tensorflow_backend
from .abstract_backend import ExtendedBackend

dtypestr: str
Tensor = Any
RGenerator = Any  # tf.random.Generator
pytree = Any

tf: Any


class keras_optimizer:
    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer
        self.is_initialized = False

    # def _apply_gradients(self, grads: Tensor, params: Tensor) -> None:
    #     self.optimizer.apply_gradients([(grads, params)])

    def update(self, grads: pytree, params: pytree) -> pytree:
        # if not self.is_initialized:
        #     l, treedef = TensorFlowBackend.tree_flatten(None, params)
        #     # https://github.com/tensorflow/tensorflow/issues/58973
        #     # still breaks tf2.11
        #     ol = [deepcopy(self.optimizer) for _ in l]
        #     self.optimizer = TensorFlowBackend.tree_unflatten(None, treedef, ol)
        #     self.is_initialized = True
        # params = TensorFlowBackend.tree_map(None, self._c2v, params)
        # don't do the () initialization since cache is in upper level of backend_factory
        grads_l, _ = TensorFlowBackend.tree_flatten(None, grads)
        params_l, params_def = TensorFlowBackend.tree_flatten(None, params)
        if not self.is_initialized:
            self.params_v = []
            self.is_variable = []
            for p in params_l:
                if not isinstance(p, tf.Variable):
                    self.params_v.append(tf.Variable(p))
                    self.is_variable.append(False)
                else:
                    self.params_v.append(p)
                    self.is_variable.append(True)
            self.is_initialized = True
        else:
            for i, p in enumerate(params_l):
                if not isinstance(p, tf.Variable):
                    self.params_v[i] = self.params_v[i].assign(p)
                else:
                    self.params_v[i] = self.params_v[i].assign(p.value())

        self.optimizer.apply_gradients(zip(grads_l, self.params_v))
        nparams_l = []
        for p, flag in zip(self.params_v, self.is_variable):
            if flag is True:
                nparams_l.append(p)
            else:
                nparams_l.append(p.value())
        params = TensorFlowBackend.tree_unflatten(None, params_def, nparams_l)
        return params


def _tensordot_tf(
    self: Any, a: Tensor, b: Tensor, axes: Union[int, Sequence[Sequence[int]]]
) -> Tensor:
    return tf.tensordot(a, b, axes)


def _outer_product_tf(self: Any, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tf.tensordot(tensor1, tensor2, 0)


def _matmul_tf(self: Any, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    if (len(tensor1.shape) <= 1) or (len(tensor2.shape) <= 1):
        raise ValueError("inputs to `matmul` have to be a tensors of order > 1,")

    return tf.matmul(tensor1, tensor2)


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


def _qr_tf(
    self: Any,
    tensor: Tensor,
    pivot_axis: int = -1,
    non_negative_diagonal: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Computes the QR decomposition of a tensor.
    The QR decomposition is performed by treating the tensor as a matrix,
    with an effective left (row) index resulting from combining the
    axes `tensor.shape[:pivot_axis]` and an effective right (column)
    index resulting from combining the axes `tensor.shape[pivot_axis:]`.

    :Example:

    If `tensor` had a shape (2, 3, 4, 5) and `pivot_axis` was 2,
    then `q` would have shape (2, 3, 6), and `r` would
    have shape (6, 4, 5).
    The output consists of two tensors `Q, R` such that:

    Q[i1,...,iN, j] * R[j, k1,...,kM] == tensor[i1,...,iN, k1,...,kM]

    Note that the output ordering matches numpy.linalg.svd rather than tf.svd.

    :param tensor: A tensor to be decomposed.
    :type tensor: Tensor
    :param pivot_axis: Where to split the tensor's axes before flattening into a matrix.
    :type pivot_axis: int, optional
    :param non_negative_diagonal: a bool indicating whether the tenor is diagonal non-negative matrix.
    :type non_negative_diagonal: bool, optional
    :returns: Q, the left tensor factor, and R, the right tensor factor.
    :rtype: Tuple[Tensor, Tensor]
    """
    from .tf_ops import tfqr

    left_dims = tf.shape(tensor)[:pivot_axis]
    right_dims = tf.shape(tensor)[pivot_axis:]

    tensor = tf.reshape(tensor, [tf.reduce_prod(left_dims), tf.reduce_prod(right_dims)])
    q, r = tfqr(tensor)
    if non_negative_diagonal:
        phases = tf.math.sign(tf.linalg.diag_part(r))
        q = q * phases
        r = phases[:, None] * r
    center_dim = tf.shape(q)[1]
    q = tf.reshape(q, tf.concat([left_dims, [center_dim]], axis=-1))
    r = tf.reshape(r, tf.concat([[center_dim], right_dims], axis=-1))
    return q, r


def _rq_tf(
    self: Any,
    tensor: Tensor,
    pivot_axis: int = 1,
    non_negative_diagonal: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Computes the RQ decomposition of a tensor.
    The QR decomposition is performed by treating the tensor as a matrix,
    with an effective left (row) index resulting from combining the axes
    `tensor.shape[:pivot_axis]` and an effective right (column) index
    resulting from combining the axes `tensor.shape[pivot_axis:]`.

    :Example:

    If `tensor` had a shape (2, 3, 4, 5) and `pivot_axis` was 2,
    then `r` would have shape (2, 3, 6), and `q` would
    have shape (6, 4, 5).
    The output consists of two tensors `Q, R` such that:

    Q[i1,...,iN, j] * R[j, k1,...,kM] == tensor[i1,...,iN, k1,...,kM]

    Note that the output ordering matches numpy.linalg.svd rather than tf.svd.

    :param tensor: A tensor to be decomposed.
    :type tensor: Tensor
    :param pivot_axis: Where to split the tensor's axes before flattening into a matrix.
    :type pivot_axis: int, optional
    :param non_negative_diagonal: a bool indicating whether the tenor is diagonal non-negative matrix.
    :type non_negative_diagonal: bool, optional
    :returns: Q, the left tensor factor, and R, the right tensor factor.
    :rtype: Tuple[Tensor, Tensor]
    """
    from .tf_ops import tfqr

    left_dims = tf.shape(tensor)[:pivot_axis]
    right_dims = tf.shape(tensor)[pivot_axis:]

    tensor = tf.reshape(tensor, [tf.reduce_prod(left_dims), tf.reduce_prod(right_dims)])
    q, r = tfqr(tf.math.conj(tf.transpose(tensor)))
    if non_negative_diagonal:
        phases = tf.math.sign(tf.linalg.diag_part(r))
        q = q * phases
        r = phases[:, None] * r
    r, q = (
        tf.math.conj(tf.transpose(r)),
        tf.math.conj(tf.transpose(q)),
    )  # M=r*q at this point
    center_dim = tf.shape(r)[1]
    r = tf.reshape(r, tf.concat([left_dims, [center_dim]], axis=-1))
    q = tf.reshape(q, tf.concat([[center_dim], right_dims], axis=-1))
    return r, q


def _svd_tf(
    self: Any,
    tensor: Tensor,
    pivot_axis: int = -1,
    max_singular_values: Optional[int] = None,
    max_truncation_error: Optional[float] = None,
    relative: Optional[bool] = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Computes the singular value decomposition (SVD) of a tensor.

    The SVD is performed by treating the tensor as a matrix, with an effective
    left (row) index resulting from combining the axes `tensor.shape[:pivot_axis]`
    and an effective right (column) index resulting from combining the axes
    `tensor.shape[pivot_axis:]`.

    For example, if `tensor` had a shape (2, 3, 4, 5) and `pivot_axis` was 2, then
    `u` would have shape (2, 3, 6), `s` would have shape (6), and `vh` would
    have shape (6, 4, 5).

    If `max_singular_values` is set to an integer, the SVD is truncated to keep
    at most this many singular values.

    If `max_truncation_error > 0`, as many singular values will be truncated as
    possible, so that the truncation error (the norm of discarded singular
    values) is at most `max_truncation_error`.
    If `relative` is set `True` then `max_truncation_err` is understood
    relative to the largest singular value.

    If both `max_singular_values` snd `max_truncation_error` are specified, the
    number of retained singular values will be
    `min(max_singular_values, nsv_auto_trunc)`, where `nsv_auto_trunc` is the
    number of singular values that must be kept to maintain a truncation error
    smaller than `max_truncation_error`.

    The output consists of three tensors `u, s, vh` such that:
    ```python
    u[i1,...,iN, j] * s[j] * vh[j, k1,...,kM] == tensor[i1,...,iN, k1,...,kM]
    ```
    Note that the output ordering matches numpy.linalg.svd rather than tf.svd.

    Args:
    tf: The tensorflow module.
    tensor: A tensor to be decomposed.
    pivot_axis: Where to split the tensor's axes before flattening into a
        matrix.
    max_singular_values: The number of singular values to keep, or `None` to
        keep them all.
    max_truncation_error: The maximum allowed truncation error or `None` to not
        do any truncation.
    relative: Multiply `max_truncation_err` with the largest singular value.

    Returns:
    u: Left tensor factor.
    s: Vector of ordered singular values from largest to smallest.
    vh: Right tensor factor.
    s_rest: Vector of discarded singular values (length zero if no
            truncation).
    """
    left_dims = tf.shape(tensor)[:pivot_axis]
    right_dims = tf.shape(tensor)[pivot_axis:]

    tensor = tf.reshape(tensor, [tf.reduce_prod(left_dims), tf.reduce_prod(right_dims)])

    eps = os.environ.get("TC_BACKENDS_TENSORFLOW_BACKEND__SVD_TF_EPS")
    if eps is not None:
        eps = 10 ** (-int(eps))
        tensor += eps * tf.ones(tensor.shape, dtype=tensor.dtype)
        # for numerical stability at least in tf+cpu
    s, u, v = tf.linalg.svd(tensor)

    if max_singular_values is None:
        max_singular_values = tf.size(s, out_type=tf.int64)
    else:
        max_singular_values = tf.constant(max_singular_values, dtype=tf.int64)

    if max_truncation_error is not None:
        # Cumulative norms of singular values in ascending order.
        trunc_errs = tf.sqrt(tf.cumsum(tf.square(s), reverse=True))
        # If relative is true, rescale max_truncation error with the largest
        # singular value to yield the absolute maximal truncation error.
        if relative:
            abs_max_truncation_error = max_truncation_error * s[0]
        else:
            abs_max_truncation_error = max_truncation_error
        # We must keep at least this many singular values to ensure the
        # truncation error is <= abs_max_truncation_error.
        num_sing_vals_err = tf.math.count_nonzero(
            tf.cast(trunc_errs > abs_max_truncation_error, dtype=tf.int32)
        )
    else:
        num_sing_vals_err = max_singular_values

    num_sing_vals_keep = tf.minimum(max_singular_values, num_sing_vals_err)

    # tf.svd() always returns the singular values as a vector of float{32,64}.
    # since tf.math_ops.real is automatically applied to s. This causes
    # s to possibly not be the same dtype as the original tensor, which can cause
    # issues for later contractions. To fix it, we recast to the original dtype.
    s = tf.cast(s, tensor.dtype)

    s_rest = s[num_sing_vals_keep:]
    s = s[:num_sing_vals_keep]
    u = u[:, :num_sing_vals_keep]
    v = v[:, :num_sing_vals_keep]

    vh = tf.linalg.adjoint(v)

    dim_s = tf.shape(s)[0]  # must use tf.shape (not s.shape) to compile
    u = tf.reshape(u, tf.concat([left_dims, [dim_s]], axis=-1))
    vh = tf.reshape(vh, tf.concat([[dim_s], right_dims], axis=-1))

    return u, s, vh, s_rest


# temporary hot replace until new version of tensorflow is released,
# see issue: https://github.com/google/TensorNetwork/issues/940
# avoid buggy tensordot2 in tensornetwork

tensornetwork.backends.tensorflow.tensorflow_backend.TensorFlowBackend.tensordot = (
    _tensordot_tf
)
tensornetwork.backends.tensorflow.tensorflow_backend.TensorFlowBackend.outer_product = (
    _outer_product_tf
)
tensornetwork.backends.tensorflow.tensorflow_backend.TensorFlowBackend.matmul = (
    _matmul_tf
)
tensornetwork.backends.tensorflow.tensorflow_backend.TensorFlowBackend.qr = _qr_tf
tensornetwork.backends.tensorflow.tensorflow_backend.TensorFlowBackend.rq = _rq_tf
tensornetwork.backends.tensorflow.tensorflow_backend.TensorFlowBackend.svd = _svd_tf


class TensorFlowBackend(tensorflow_backend.TensorFlowBackend, ExtendedBackend):  # type: ignore
    """
    See the original backend API at `tensorflow backend
    <https://github.com/google/TensorNetwork/blob/master/tensornetwork/backends/tensorflow/tensorflow_backend.py>`_
    """

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
        tf.sparse.SparseTensor.__add__ = tf.sparse.add
        self.minor = int(tf.__version__.split(".")[1])
        self.name = "tensorflow"
        logger = tf.get_logger()  # .setLevel('ERROR')
        logger.addFilter(lambda s: not re.match(".*You are casting.*", s.getMessage()))
        # ignore casting warning by logger

    def eye(
        self, N: int, dtype: Optional[str] = None, M: Optional[int] = None
    ) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        r = tf.eye(num_rows=N, num_columns=M)
        return self.cast(r, dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        r = tf.ones(shape=shape)
        return self.cast(r, dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        r = tf.zeros(shape=shape)
        return self.cast(r, dtype)

    def copy(self, a: Tensor) -> Tensor:
        return tf.identity(a)

    def expm(self, a: Tensor) -> Tensor:
        return tf.linalg.expm(a)

    def sin(self, a: Tensor) -> Tensor:
        return tf.math.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return tf.math.cos(a)

    def acos(self, a: Tensor) -> Tensor:
        return tf.math.acos(a)

    def acosh(self, a: Tensor) -> Tensor:
        return tf.math.acosh(a)

    def asin(self, a: Tensor) -> Tensor:
        return tf.math.asin(a)

    def asinh(self, a: Tensor) -> Tensor:
        return tf.math.asinh(a)

    def atan(self, a: Tensor) -> Tensor:
        return tf.math.atan(a)

    def atan2(self, y: Tensor, x: Tensor) -> Tensor:
        return tf.math.atan2(y, x)

    def atanh(self, a: Tensor) -> Tensor:
        return tf.math.atanh(a)

    def cosh(self, a: Tensor) -> Tensor:
        return tf.math.cosh(a)

    def tan(self, a: Tensor) -> Tensor:
        return tf.math.tan(a)

    def tanh(self, a: Tensor) -> Tensor:
        return tf.math.tanh(a)

    def sinh(self, a: Tensor) -> Tensor:
        return tf.math.sinh(a)

    def size(self, a: Tensor) -> Tensor:
        return tf.size(a)

    def eigvalsh(self, a: Tensor) -> Tensor:
        return tf.linalg.eigvalsh(a)

    def dtype(self, a: Tensor) -> str:
        return a.dtype.__repr__().split(".")[-1]  # type: ignore

    def kron(self, a: Tensor, b: Tensor) -> Tensor:
        # array more than 2d consistency is not guranteed for different backends
        return tf.reshape(
            tf.reshape(a, [a.shape[0], 1, a.shape[1], 1])
            * tf.reshape(b, [1, b.shape[0], 1, b.shape[1]]),
            [a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]],
        )

    def numpy(self, a: Tensor) -> Tensor:
        if self.is_sparse(a):
            return coo_matrix(
                (a.values, (a.indices[:, 0], a.indices[:, 1])), shape=a.get_shape()
            )

        return a.numpy()  # only valid in eager mode

    def i(self, dtype: Any = None) -> Tensor:
        if not dtype:
            dtype = getattr(tf, dtypestr)
        if isinstance(dtype, str):
            dtype = getattr(tf, dtype)
        return tf.constant(1j, dtype=dtype)

    def det(self, a: Tensor) -> Tensor:
        return tf.linalg.det(a)

    def min(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return tf.reduce_min(a, axis=axis)

    def max(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return tf.reduce_max(a, axis=axis)

    def argmax(self, a: Tensor, axis: int = 0) -> Tensor:
        return tf.math.argmax(a, axis=axis)

    def argmin(self, a: Tensor, axis: int = 0) -> Tensor:
        return tf.math.argmin(a, axis=axis)

    def unique_with_counts(self, a: Tensor, **kws: Any) -> Tuple[Tensor, Tensor]:
        r = tf.unique_with_counts(a)
        order = tf.argsort(r.y)
        return tf.gather(r.y, order), tf.gather(r.count, order)

    def stack(self, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return tf.stack(a, axis=axis)

    def concat(self, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return tf.concat(a, axis=axis)

    def tile(self, a: Tensor, rep: Tensor) -> Tensor:
        return tf.tile(a, rep)

    def mean(
        self,
        a: Tensor,
        axis: Optional[Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        return tf.math.reduce_mean(a, axis=axis, keepdims=keepdims)

    def std(
        self, a: Tensor, axis: Optional[Sequence[int]] = None, keepdims: bool = False
    ) -> Tensor:
        return tf.math.reduce_std(a, axis=axis, keepdims=keepdims)

    def sigmoid(self, a: Tensor) -> Tensor:
        return tf.nn.sigmoid(a)

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

    def imag(self, a: Tensor) -> Tensor:
        return tf.math.imag(a)

    def cast(self, a: Tensor, dtype: str) -> Tensor:
        if isinstance(dtype, str):
            return tf.cast(a, dtype=getattr(tf, dtype))
        return tf.cast(a, dtype=dtype)

    def arange(self, start: int, stop: Optional[int] = None, step: int = 1) -> Tensor:
        if stop is None:
            return tf.range(start=0, limit=start, delta=step)
        return tf.range(start=start, limit=stop, delta=step)

    def mod(self, x: Tensor, y: Tensor) -> Tensor:
        return tf.math.mod(x, y)

    def right_shift(self, x: Tensor, y: Tensor) -> Tensor:
        return tf.bitwise.right_shift(x, y)

    def left_shift(self, x: Tensor, y: Tensor) -> Tensor:
        return tf.bitwise.left_shift(x, y)

    def solve(self, A: Tensor, b: Tensor, **kws: Any) -> Tensor:
        if b.shape[-1] == A.shape[-1]:
            b = b[..., tf.newaxis]
            vector = True
        else:
            vector = False
        x = tf.linalg.solve(A, b)
        if vector:
            return self.reshape(x, x.shape[:-1])
        return x

    def searchsorted(self, a: Tensor, v: Tensor, side: str = "left") -> Tensor:
        return tf.searchsorted(a, v, side)

    def from_dlpack(self, a: Any) -> Tensor:
        return tf.experimental.dlpack.from_dlpack(a)

    def to_dlpack(self, a: Tensor) -> Any:
        return tf.experimental.dlpack.to_dlpack(a)

    # note complex tensor support for dlpack is only available for tf>=2.9

    def set_random_state(
        self, seed: Optional[Union[int, RGenerator]] = None, get_only: bool = False
    ) -> Any:
        if seed is None:
            g = tf.random.Generator.from_non_deterministic_state()
        elif isinstance(seed, int):
            g = tf.random.Generator.from_seed(seed)
        else:
            g = seed
        if get_only is False:
            self.g = g
        return g

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

    def gather1d(self, operand: Tensor, indices: Tensor) -> Tensor:
        return tf.gather(operand, indices)

    def scatter(self, operand: Tensor, indices: Tensor, updates: Tensor) -> Tensor:
        return tf.tensor_scatter_nd_update(operand, indices, updates)

    def coo_sparse_matrix(
        self, indices: Tensor, values: Tensor, shape: Tensor
    ) -> Tensor:
        return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)

    def sparse_dense_matmul(
        self,
        sp_a: Tensor,
        b: Tensor,
    ) -> Tensor:
        return tf.sparse.sparse_dense_matmul(sp_a, b)

    def _densify(self) -> Tensor:
        @partial(self.jit, jit_compile=True)
        def densify(sp_a: Tensor) -> Tensor:
            return tf.sparse.to_dense(sp_a)

        return densify

    def to_dense(self, sp_a: Tensor) -> Tensor:
        return self._densify()(sp_a)

    # very weirdly, at least for cpu, tf.sparse.to_dense only works within tf.function(jit_compile=True)
    # and will fail with tf.function or bare function
    # on the contrary, tf.sparse.sparse_dense_matmul only fails within tf.function(jit_compile=True)
    # and works well with tf.function or bare function...

    def is_sparse(self, a: Tensor) -> bool:
        return isinstance(a, tf.SparseTensor)

    def cond(
        self,
        pred: bool,
        true_fun: Callable[[], Tensor],
        false_fun: Callable[[], Tensor],
    ) -> Tensor:
        return tf.cond(pred, true_fun, false_fun)

    def switch(self, index: Tensor, branches: Sequence[Callable[[], Tensor]]) -> Tensor:
        return tf.switch_case(index, branches)

    def scan(
        self, f: Callable[[Tensor, Tensor], Tensor], xs: Tensor, init: Tensor
    ) -> Tensor:
        return tf.scan(f, xs, init)[-1]

    def device(self, a: Tensor) -> str:
        dev = a.device
        return self._dev2str(dev)

    def device_move(self, a: Tensor, dev: Any) -> Tensor:
        if isinstance(dev, str):
            dev = self._str2dev(dev)
        with tf.device(dev):
            a = tf.identity(a)
        return a

    def _dev2str(self, dev: Any) -> str:
        platform, id_ = dev.split(":")[-2:]
        if platform == "CPU":
            return "cpu"
        if platform == "GPU":
            return "gpu" + ":" + str(id_)
        raise ValueError("TensorFlowBackend don't support non-GPU/CPU device")

    def _str2dev(self, str_: str) -> Any:
        return str_

    def stop_gradient(self, a: Tensor) -> Tensor:
        return tf.stop_gradient(a)

    def grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Any]:
        # experimental attempt
        # Note: tensorflow grad is gradient while jax grad is derivative, they are different with a conjugate!
        # And we DONT make them consitent by mannually set conjugate of the returns.
        def wrapper(*args: Any, **kws: Any) -> Any:
            with tf.GradientTape() as t:
                if isinstance(argnums, int):
                    x = args[argnums]
                else:
                    args = tuple(
                        [
                            tf.identity(arg) if i in argnums else arg
                            for i, arg in enumerate(args)
                        ]
                    )
                    # in case wrong grad for f(x, x)
                    x = [args[i] for i in argnums]
                t.watch(x)
                y = f(*args, **kws)
                if has_aux:
                    g = t.gradient(y[0], x)
                else:
                    g = t.gradient(y, x)
            if has_aux:
                return (g, y[1:])
            return g

        return wrapper

    def value_and_grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Tuple[Any, Any]]:
        def wrapper(*args: Any, **kws: Any) -> Any:
            with tf.GradientTape() as t:
                if isinstance(argnums, int):
                    x = args[argnums]
                else:
                    args = tuple(
                        [
                            tf.identity(arg) if i in argnums else arg
                            for i, arg in enumerate(args)
                        ]
                    )
                    x = [args[i] for i in argnums]
                t.watch(x)
                y = f(*args, **kws)
                if has_aux:
                    g = t.gradient(y[0], x)
                else:
                    g = t.gradient(y, x)
            return y, g

        return wrapper

    def jvp(
        self,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        if not (isinstance(inputs, list) or isinstance(inputs, tuple)):
            # one input tensor
            inputs = [inputs]
        elif isinstance(inputs, list):
            inputs = [tf.identity(inp) for inp in inputs]
        else:  # inputs, tuple
            inputs = tuple([tf.identity(inp) for inp in inputs])
        if not (isinstance(v, list) or isinstance(v, tuple)):
            v = [v]
        elif isinstance(v, list):
            v = [tf.identity(vi) for vi in v]
        else:
            v = tuple([tf.identity(vi) for vi in v])
        with tf.autodiff.ForwardAccumulator(inputs, v) as t:
            y = f(*inputs)
        g = t.jvp(y)
        return y, g

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
        elif isinstance(inputs, list):
            inputs = [tf.identity(inp) for inp in inputs]
            one_input = False
        else:  # inputs tuple
            inputs = tuple([tf.identity(inp) for inp in inputs])
            one_input = False
        with tf.GradientTape() as t:
            t.watch(inputs)
            y = f(*inputs)
        g = t.gradient(y, inputs, v)
        g = list(g)
        for i, gi in enumerate(g):
            if gi is None:
                g[i] = tf.zeros_like(inputs[i])
            if isinstance(gi, tf.IndexedSlices):
                # gradient can return sth weird
                # TODO(@refraction-ray): check whether other AD tf methods have such issues
                # shape is still unkown, dense_shape attr doesn't work?
                g[i] = tf.convert_to_tensor(gi)
        g = tuple(g)
        if one_input:
            g = g[0]
        return y, g

    def jit(
        self,
        f: Callable[..., Any],
        static_argnums: Optional[Union[int, Sequence[int]]] = None,
        jit_compile: Optional[bool] = None,
        **kws: Any
    ) -> Any:
        # static_argnums not supported in tf case, this is only for a consistent interface
        # for more on static_argnums in tf.function, see issue: https://github.com/tensorflow/tensorflow/issues/52193
        # tf.function works with dict pytree but fails at list pytree, hmm...
        # no full jittable pytree support in tf
        # another difference from jax.jit
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
            # @self.jit  # otherwise, vectorized_map claim on retracing
            def wrapper(*args: Any, **kws: Any) -> Tensor:
                # @self.jit
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
        has_aux: bool = False,
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
            if has_aux:
                grad = tape.gradient(vs[0], x)
            else:
                grad = tape.gradient(vs, x)
            return vs, grad

        return wrapper

        # f = self.vmap(f)
        # f = self.value_and_grad(f, argnums=argnums)
        # f = self.jit(f)
        # return f

    vvag = vectorized_value_and_grad

    optimizer = keras_optimizer
