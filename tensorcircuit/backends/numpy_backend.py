"""
Backend magic inherited from tensornetwork: numpy backend
"""
# pylint: disable=invalid-name

import logging
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import tensornetwork
from scipy.linalg import expm, solve
from scipy.special import softmax, expit
from scipy.sparse import coo_matrix, issparse
from tensornetwork.backends.numpy import numpy_backend
from .abstract_backend import ExtendedBackend

logger = logging.getLogger(__name__)

dtypestr: str
Tensor = Any


def _sum_numpy(
    self: Any, a: Tensor, axis: Optional[Sequence[int]] = None, keepdims: bool = False
) -> Tensor:
    return np.sum(a, axis=axis, keepdims=keepdims)  # type: ignore
    # see https://github.com/google/TensorNetwork/issues/952


def _convert_to_tensor_numpy(self: Any, a: Tensor) -> Tensor:
    if not isinstance(a, np.ndarray) and not np.isscalar(a):
        a = np.array(a)
    a = np.asarray(a)
    return a


tensornetwork.backends.numpy.numpy_backend.NumPyBackend.sum = _sum_numpy
tensornetwork.backends.numpy.numpy_backend.NumPyBackend.convert_to_tensor = (
    _convert_to_tensor_numpy
)


class NumpyBackend(numpy_backend.NumPyBackend, ExtendedBackend):  # type: ignore
    """
    see the original backend API at `numpy backend
    <https://github.com/google/TensorNetwork/blob/master/tensornetwork/backends/numpy/numpy_backend.py>`_
    """

    def eye(
        self, N: int, dtype: Optional[str] = None, M: Optional[int] = None
    ) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        r = np.eye(N, M=M)
        return self.cast(r, dtype)

    def ones(self, shape: Sequence[int], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        r = np.ones(shape)
        return self.cast(r, dtype)

    def zeros(self, shape: Sequence[int], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        r = np.zeros(shape)
        return self.cast(r, dtype)

    def copy(self, a: Tensor) -> Tensor:
        return a.copy()

    def expm(self, a: Tensor) -> Tensor:
        return expm(a)

    def abs(self, a: Tensor) -> Tensor:
        return np.abs(a)

    def sin(self, a: Tensor) -> Tensor:
        return np.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return np.cos(a)

    # acos acosh asin asinh atan atan2 atanh cosh (cos) tan tanh sinh (sin)
    def acos(self, a: Tensor) -> Tensor:
        return np.arccos(a)

    def acosh(self, a: Tensor) -> Tensor:
        return np.arccosh(a)

    def asin(self, a: Tensor) -> Tensor:
        return np.arcsin(a)

    def asinh(self, a: Tensor) -> Tensor:
        return np.arcsinh(a)

    def atan(self, a: Tensor) -> Tensor:
        return np.arctan(a)

    def atan2(self, y: Tensor, x: Tensor) -> Tensor:
        return np.arctan2(y, x)

    def atanh(self, a: Tensor) -> Tensor:
        return np.arctanh(a)

    def cosh(self, a: Tensor) -> Tensor:
        return np.cosh(a)

    def tan(self, a: Tensor) -> Tensor:
        return np.tan(a)

    def tanh(self, a: Tensor) -> Tensor:
        return np.tanh(a)

    def sinh(self, a: Tensor) -> Tensor:
        return np.sinh(a)

    def size(self, a: Tensor) -> Tensor:
        return a.size

    def eigvalsh(self, a: Tensor) -> Tensor:
        return np.linalg.eigvalsh(a)

    def kron(self, a: Tensor, b: Tensor) -> Tensor:
        return np.kron(a, b)

    def dtype(self, a: Tensor) -> str:
        return a.dtype.__str__()  # type: ignore

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

    def concat(self, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return np.concatenate(a, axis=axis)

    def tile(self, a: Tensor, rep: Tensor) -> Tensor:
        return np.tile(a, rep)

    def mean(
        self,
        a: Tensor,
        axis: Optional[Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        return np.mean(a, axis=axis, keepdims=keepdims)

    def unique_with_counts(self, a: Tensor, **kws: Any) -> Tuple[Tensor, Tensor]:
        return np.unique(a, return_counts=True)  # type: ignore

    def min(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return np.min(a, axis=axis)

    def max(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return np.max(a, axis=axis)

    def argmax(self, a: Tensor, axis: int = 0) -> Tensor:
        return np.argmax(a, axis=axis)

    def argmin(self, a: Tensor, axis: int = 0) -> Tensor:
        return np.argmin(a, axis=axis)

    def sigmoid(self, a: Tensor) -> Tensor:
        return expit(a)

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

    def imag(self, a: Tensor) -> Tensor:
        return np.imag(a)

    def cast(self, a: Tensor, dtype: str) -> Tensor:
        if isinstance(dtype, str):
            return a.astype(getattr(np, dtype))
        return a.astype(dtype)

    def arange(self, start: int, stop: Optional[int] = None, step: int = 1) -> Tensor:
        if stop is None:
            return np.arange(start=0, stop=start, step=step)
        return np.arange(start=start, stop=stop, step=step)

    def mod(self, x: Tensor, y: Tensor) -> Tensor:
        return np.mod(x, y)

    def right_shift(self, x: Tensor, y: Tensor) -> Tensor:
        return np.right_shift(x, y)

    def left_shift(self, x: Tensor, y: Tensor) -> Tensor:
        return np.left_shift(x, y)

    def solve(self, A: Tensor, b: Tensor, assume_a: str = "gen") -> Tensor:  # type: ignore
        # gen, sym, her, pos
        # https://stackoverflow.com/questions/44672029/difference-between-numpy-linalg-solve-and-numpy-linalg-lu-solve/44710451
        return solve(A, b, assume_a)

    def set_random_state(
        self, seed: Optional[int] = None, get_only: bool = False
    ) -> Any:
        g = np.random.default_rng(seed)  # None auto supported
        if get_only is False:
            self.g = g
        return g

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
        operand_new[tuple([indices[:, i] for i in range(indices.shape[1])])] = updates
        return operand_new

    def coo_sparse_matrix(
        self, indices: Tensor, values: Tensor, shape: Tensor
    ) -> Tensor:
        return coo_matrix((values, (indices[:, 0], indices[:, 1])), shape=shape)

    def sparse_dense_matmul(
        self,
        sp_a: Tensor,
        b: Tensor,
    ) -> Tensor:
        return sp_a @ b

    def to_dense(self, sp_a: Tensor) -> Tensor:
        return sp_a.todense()

    def is_sparse(self, a: Tensor) -> bool:
        return issparse(a)  # type: ignore

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

    def device(self, a: Tensor) -> str:
        return "cpu"

    def device_move(self, a: Tensor, dev: Any) -> Tensor:
        if dev == "cpu":
            return a
        raise ValueError("NumPy backend only support CPU device")

    def _dev2str(self, dev: Any) -> str:
        if dev == "cpu":
            return "cpu"
        raise ValueError("NumPy backend only support CPU device")

    def _str2dev(self, str_: str) -> Any:
        if str_ == "cpu":
            return "cpu"
        raise ValueError("NumPy backend only support CPU device")

    def stop_gradient(self, a: Tensor) -> Tensor:
        raise NotImplementedError("numpy backend doesn't support AD")

    def grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Any]:
        raise NotImplementedError("numpy backend doesn't support AD")

    def value_and_grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Tuple[Any, Any]]:
        raise NotImplementedError("numpy backend doesn't support AD")

    def jit(
        self,
        f: Callable[..., Any],
        static_argnums: Optional[Union[int, Sequence[int]]] = None,
        jit_compile: Optional[bool] = None,
    ) -> Callable[..., Any]:
        logger.warning("numpy backend has no jit interface, just do nothing")
        return f
        # raise NotImplementedError("numpy backend doesn't support jit compiling")

    def vmap(
        self, f: Callable[..., Any], vectorized_argnums: Union[int, Sequence[int]] = 0
    ) -> Any:
        logger.warning(
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
        has_aux: bool = False,
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

    def jvp(
        self,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        raise NotImplementedError("numpy backend doesn't support AD")
