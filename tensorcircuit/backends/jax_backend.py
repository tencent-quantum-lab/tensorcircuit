"""
backend magic inherited from tensornetwork: jax backend
"""

from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import tensornetwork
from tensornetwork.backends.jax import jax_backend

try:  # old version tn compatiblity
    from tensornetwork.backends import base_backend

    tnbackend = base_backend.BaseBackend

except ImportError:
    from tensornetwork.backends import abstract_backend

    tnbackend = abstract_backend.AbstractBackend

dtypestr: str
Tensor = Any
PRNGKeyArray = Any  # libjax.random.PRNGKeyArray

libjax: Any
jnp: Any
jsp: Any


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
    from .ops import adaware_svd_jit as adaware_svd

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
        global sparse  # jax.experimental.sparse
        super(JaxBackend, self).__init__()
        try:
            import jax
        except ImportError:
            raise ImportError(
                "Jax not installed, please switch to a different "
                "backend or install Jax."
            )
        from jax.experimental import sparse

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
        if g is None or getattr(g, "_trace", None) is not None:
            # avoid random state is set in a jitted function
            # which call outside jitted regime lead to UnexpectedTracerError
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
        if g is None or getattr(g, "_trace", None) is not None:
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
        if g is None or getattr(g, "_trace", None) is not None:
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
