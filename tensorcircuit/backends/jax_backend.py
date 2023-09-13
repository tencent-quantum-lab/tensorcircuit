"""
Backend magic inherited from tensornetwork: jax backend
"""
# pylint: disable=invalid-name

from functools import partial
import logging
import warnings
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.sparse import coo_matrix
import tensornetwork
from tensornetwork.backends.jax import jax_backend
from .abstract_backend import ExtendedBackend

logger = logging.getLogger(__name__)


dtypestr: str
Tensor = Any
PRNGKeyArray = Any  # libjax.random.PRNGKeyArray
pytree = Any

libjax: Any
jnp: Any
jsp: Any
optax: Any


class optax_optimizer:
    # the behavior of this optimizer abstraction with jit is not guranteed
    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer
        self.state = None

    def update(self, grads: pytree, params: pytree) -> pytree:
        if self.state is None:
            self.state = self.optimizer.init(params)
        updates, self.state = self.optimizer.update(grads, self.state)
        params = optax.apply_updates(params, updates)
        return params


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
    from .jax_ops import adaware_svd_jit as adaware_svd

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


def _qr_jax(
    self: Any,
    tensor: Tensor,
    pivot_axis: int = -1,
    non_negative_diagonal: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Computes the QR decomposition of a tensor.
    See tensornetwork.backends.tensorflow.decompositions for details.
    """
    from .jax_ops import adaware_qr_jit as adaware_qr

    left_dims = tensor.shape[:pivot_axis]
    right_dims = tensor.shape[pivot_axis:]
    tensor = jnp.reshape(tensor, [np.prod(left_dims), np.prod(right_dims)])
    q, r = adaware_qr(tensor)
    if non_negative_diagonal:
        phases = jnp.sign(jnp.diagonal(r))
        q = q * phases
        r = phases.conj()[:, None] * r
    center_dim = q.shape[1]
    q = jnp.reshape(q, list(left_dims) + [center_dim])
    r = jnp.reshape(r, [center_dim] + list(right_dims))
    return q, r


def _rq_jax(
    self: Any,
    tensor: Tensor,
    pivot_axis: int = -1,
    non_negative_diagonal: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Computes the RQ (reversed QR) decomposition of a tensor.
    See tensornetwork.backends.tensorflow.decompositions for details.
    """
    from .jax_ops import adaware_qr_jit as adaware_qr

    left_dims = tensor.shape[:pivot_axis]
    right_dims = tensor.shape[pivot_axis:]
    tensor = jnp.reshape(tensor, [np.prod(left_dims), np.prod(right_dims)])
    q, r = adaware_qr(jnp.conj(jnp.transpose(tensor)))
    if non_negative_diagonal:
        phases = jnp.sign(jnp.diagonal(r))
        q = q * phases
        r = phases.conj()[:, None] * r
    r, q = jnp.conj(jnp.transpose(r)), jnp.conj(jnp.transpose(q))  # M=r*q at this point
    center_dim = r.shape[1]
    r = jnp.reshape(r, list(left_dims) + [center_dim])
    q = jnp.reshape(q, [center_dim] + list(right_dims))
    return r, q


def _eigh_jax(self: Any, tensor: Tensor) -> Tensor:
    from .jax_ops import adaware_eigh_jit as adaware_eigh

    return adaware_eigh(tensor)


tensornetwork.backends.jax.jax_backend.JaxBackend.convert_to_tensor = (
    _convert_to_tensor_jax
)
tensornetwork.backends.jax.jax_backend.JaxBackend.svd = _svd_jax
tensornetwork.backends.jax.jax_backend.JaxBackend.qr = _qr_jax
tensornetwork.backends.jax.jax_backend.JaxBackend.rq = _rq_jax
tensornetwork.backends.jax.jax_backend.JaxBackend.eigh = _eigh_jax


class JaxBackend(jax_backend.JaxBackend, ExtendedBackend):  # type: ignore
    """
    See the original backend API at `jax backend
    <https://github.com/google/TensorNetwork/blob/master/tensornetwork/backends/jax/jax_backend.py>`_
    """

    # Jax doesn't support 64bit dtype, unless claim
    # ``from jax.config import config```
    # ``config.update("jax_enable_x64", True)``
    # at very beginning, i.e. before import tensorcircuit
    def __init__(self) -> None:
        global libjax  # Jax module
        global jnp  # jax.numpy module
        global jsp  # jax.scipy module
        global sparse  # jax.experimental.sparse
        global optax  # optax
        super(JaxBackend, self).__init__()
        try:
            import jax
        except ImportError:
            raise ImportError(
                "Jax not installed, please switch to a different "
                "backend or install Jax."
            )
        from jax.experimental import sparse
        import jax.scipy

        try:
            import optax

        except ImportError:
            logger.warning(
                "optax not installed, `optimizer` from jax backend cannot work"
            )
        libjax = jax
        jnp = libjax.numpy
        jsp = libjax.scipy

        self.name = "jax"

    # it is already child of numpy backend, and self.np = self.jax.np
    def eye(
        self, N: int, dtype: Optional[str] = None, M: Optional[int] = None
    ) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        r = jnp.eye(N, M=M)
        return self.cast(r, dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        r = jnp.ones(shape)
        return self.cast(r, dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        r = jnp.zeros(shape)
        return self.cast(r, dtype)

    def copy(self, tensor: Tensor) -> Tensor:
        return jnp.array(tensor, copy=True)

    def convert_to_tensor(self, tensor: Tensor) -> Tensor:
        result = jnp.asarray(tensor)
        return result

    def abs(self, a: Tensor) -> Tensor:
        return jnp.abs(a)

    def sin(self, a: Tensor) -> Tensor:
        return jnp.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return jnp.cos(a)

    def acos(self, a: Tensor) -> Tensor:
        return jnp.arccos(a)

    def acosh(self, a: Tensor) -> Tensor:
        return jnp.arccosh(a)

    def asin(self, a: Tensor) -> Tensor:
        return jnp.arcsin(a)

    def asinh(self, a: Tensor) -> Tensor:
        return jnp.arcsinh(a)

    def atan(self, a: Tensor) -> Tensor:
        return jnp.arctan(a)

    def atan2(self, y: Tensor, x: Tensor) -> Tensor:
        return jnp.arctan2(y, x)

    def atanh(self, a: Tensor) -> Tensor:
        return jnp.arctanh(a)

    def cosh(self, a: Tensor) -> Tensor:
        return jnp.cosh(a)

    def tan(self, a: Tensor) -> Tensor:
        return jnp.tan(a)

    def tanh(self, a: Tensor) -> Tensor:
        return jnp.tanh(a)

    def sinh(self, a: Tensor) -> Tensor:
        return jnp.sinh(a)

    def size(self, a: Tensor) -> Tensor:
        return jnp.size(a)

    def eigvalsh(self, a: Tensor) -> Tensor:
        return jnp.linalg.eigvalsh(a)

    def kron(self, a: Tensor, b: Tensor) -> Tensor:
        return jnp.kron(a, b)

    def numpy(self, a: Tensor) -> Tensor:
        if self.is_sparse(a):
            return coo_matrix(
                (a.data, (a.indices[:, 0], a.indices[:, 1])), shape=a.shape
            )
        return np.array(a)

    def i(self, dtype: Any = None) -> Tensor:
        if not dtype:
            dtype = npdtype  # type: ignore
        if isinstance(dtype, str):
            dtype = getattr(jnp, dtype)
        return jnp.array(1j, dtype=dtype)

    def det(self, a: Tensor) -> Tensor:
        return jnp.linalg.det(a)

    def schur(self, a: Tensor, output: str = "real") -> Tuple[Tensor, Tensor]:
        return jsp.linalg.schur(a, output=output)  # type: ignore

    def real(self, a: Tensor) -> Tensor:
        return jnp.real(a)

    def imag(self, a: Tensor) -> Tensor:
        return jnp.imag(a)

    def dtype(self, a: Tensor) -> str:
        return a.dtype.__str__()  # type: ignore

    def cast(self, a: Tensor, dtype: str) -> Tensor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            if isinstance(dtype, str):
                return a.astype(getattr(jnp, dtype))
            return a.astype(dtype)

    def arange(self, start: int, stop: Optional[int] = None, step: int = 1) -> Tensor:
        if stop is None:
            return jnp.arange(start=0, stop=start, step=step)
        return jnp.arange(start=start, stop=stop, step=step)

    def mod(self, x: Tensor, y: Tensor) -> Tensor:
        return jnp.mod(x, y)

    def right_shift(self, x: Tensor, y: Tensor) -> Tensor:
        return jnp.right_shift(x, y)

    def left_shift(self, x: Tensor, y: Tensor) -> Tensor:
        return jnp.left_shift(x, y)

    def expm(self, a: Tensor) -> Tensor:
        return jsp.linalg.expm(a)
        # currently expm in jax doesn't support AD, it will raise an AssertError,
        # see https://github.com/google/jax/issues/2645

    def stack(self, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return jnp.stack(a, axis=axis)

    def concat(self, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return jnp.concatenate(a, axis=axis)

    def tile(self, a: Tensor, rep: Tensor) -> Tensor:
        return jnp.tile(a, rep)

    def mean(
        self,
        a: Tensor,
        axis: Optional[Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        return jnp.mean(a, axis=axis, keepdims=keepdims)

    def std(
        self, a: Tensor, axis: Optional[Sequence[int]] = None, keepdims: bool = False
    ) -> Tensor:
        return jnp.std(a, axis=axis, keepdims=keepdims)

    def min(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return jnp.min(a, axis=axis)

    def max(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return jnp.max(a, axis=axis)

    def argmax(self, a: Tensor, axis: int = 0) -> Tensor:
        return jnp.argmax(a, axis=axis)

    def argmin(self, a: Tensor, axis: int = 0) -> Tensor:
        return jnp.argmin(a, axis=axis)

    def unique_with_counts(  # type: ignore
        self, a: Tensor, *, size: Optional[int] = None, fill_value: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        return jnp.unique(a, return_counts=True, size=size, fill_value=fill_value)  # type: ignore

    def sigmoid(self, a: Tensor) -> Tensor:
        return libjax.nn.sigmoid(a)

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

    def solve(self, A: Tensor, b: Tensor, assume_a: str = "gen") -> Tensor:  # type: ignore
        return jsp.linalg.solve(A, b, assume_a=assume_a)

    def searchsorted(self, a: Tensor, v: Tensor, side: str = "left") -> Tensor:
        if not self.is_tensor(a):
            a = self.convert_to_tensor(a)
        if not self.is_tensor(v):
            v = self.convert_to_tensor(v)
        return jnp.searchsorted(a, v, side)

    def tree_map(self, f: Callable[..., Any], *pytrees: Any) -> Any:
        return libjax.tree_map(f, *pytrees)

    def tree_flatten(self, pytree: Any) -> Tuple[Any, Any]:
        return libjax.tree_util.tree_flatten(pytree)  # type: ignore

    def tree_unflatten(self, treedef: Any, leaves: Any) -> Any:
        return libjax.tree_util.tree_unflatten(treedef, leaves)

    def from_dlpack(self, a: Any) -> Tensor:
        import jax.dlpack

        return jax.dlpack.from_dlpack(a)

    def to_dlpack(self, a: Tensor) -> Any:
        import jax.dlpack

        return jax.dlpack.to_dlpack(a)

    def set_random_state(
        self, seed: Optional[Union[int, PRNGKeyArray]] = None, get_only: bool = False
    ) -> Any:
        if seed is None:
            seed = np.random.randint(42)
        if isinstance(seed, int):
            g = libjax.random.PRNGKey(seed)
        else:
            g = seed
        if get_only is False:
            self.g = g
        return g

    def random_split(self, key: Any) -> Tuple[Any, Any]:
        return libjax.random.split(key)  # type: ignore

    def implicit_randn(
        self,
        shape: Union[int, Sequence[int]] = 1,
        mean: float = 0,
        stddev: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        g = getattr(self, "g", None)
        if g is None:  # or getattr(g, "_trace", None) is not None:
            # avoid random state is set in a jitted function
            # which call outside jitted regime lead to UnexpectedTracerError
            # set with _trace is bad, since the function can itself in jit env
            self.set_random_state()
            g = getattr(self, "g", None)
        try:
            key, subkey = libjax.random.split(g)
        except libjax.errors.UnexpectedTracerError:
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
            # set with _trace is bad, since the function can itself in jit env
            self.set_random_state()
            g = getattr(self, "g", None)
        try:
            key, subkey = libjax.random.split(g)
        except libjax.errors.UnexpectedTracerError:
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
        try:
            key, subkey = libjax.random.split(g)
        except libjax.errors.UnexpectedTracerError:
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

    def scan(
        self, f: Callable[[Tensor, Tensor], Tensor], xs: Tensor, init: Tensor
    ) -> Tensor:
        def f_jax(*args: Any, **kws: Any) -> Any:
            r = f(*args, **kws)
            return r, None

        carry, _ = libjax.lax.scan(f_jax, init, xs)
        return carry

    def scatter(self, operand: Tensor, indices: Tensor, updates: Tensor) -> Tensor:
        rank = len(operand.shape)
        dnums = libjax.lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=tuple([i for i in range(rank)]),
            scatter_dims_to_operand_dims=tuple([i for i in range(rank)]),
        )
        r = libjax.lax.scatter(operand, indices, updates, dnums)
        return r

    def coo_sparse_matrix(
        self, indices: Tensor, values: Tensor, shape: Tensor
    ) -> Tensor:
        return sparse.BCOO((values, indices), shape=shape)  # type: ignore

    def sparse_dense_matmul(
        self,
        sp_a: Tensor,
        b: Tensor,
    ) -> Tensor:
        return sp_a @ b

    def to_dense(self, sp_a: Tensor) -> Tensor:
        return sp_a.todense()

    def is_sparse(self, a: Tensor) -> bool:
        return isinstance(a, sparse.BCOO)  # type: ignore

    def device(self, a: Tensor) -> str:
        dev = a.device()
        return self._dev2str(dev)

    def device_move(self, a: Tensor, dev: Any) -> Tensor:
        if isinstance(dev, str):
            dev = self._str2dev(dev)
        return libjax.device_put(a, dev)

    def _dev2str(self, dev: Any) -> str:
        if dev.platform == "cpu":
            return "cpu"
        if dev.platform == "gpu":
            return "gpu:" + str(dev.id)
        raise ValueError("JaxBackend don't support non-GPU/CPU device")

    def _str2dev(self, str_: str) -> Any:
        if str_ == "cpu":
            return libjax.devices("cpu")[0]
        if str_.startswith("gpu"):
            _id = int(str_.split(":")[-1])
            return libjax.devices("gpu")[_id]
        raise ValueError("JaxBackend don't support non-GPU/CPU device")

    def stop_gradient(self, a: Tensor) -> Tensor:
        return libjax.lax.stop_gradient(a)

    def grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Any:
        return libjax.grad(f, argnums=argnums, has_aux=has_aux)

    def value_and_grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Tuple[Any, Any]]:
        return libjax.value_and_grad(f, argnums=argnums, has_aux=has_aux)  # type: ignore

    def jvp(
        self,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)
        if not isinstance(v, (tuple, list)):
            v = (v,)
        value, jvpv = libjax.jvp(f, inputs, v)
        return value, jvpv

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
        **kws: Any,
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
        has_aux: bool = False,
    ) -> Callable[..., Tuple[Any, Any]]:
        if isinstance(vectorized_argnums, int):
            vectorized_argnums = (vectorized_argnums,)

        def wrapper(
            *args: Any, **kws: Any
        ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
            jf = self.value_and_grad(f, argnums=argnums, has_aux=has_aux)
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

    optimizer = optax_optimizer
