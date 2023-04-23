"""
CuPy backend. Not in the tensornetwork package and highly experimental.
"""
# pylint: disable=invalid-name

import logging
import warnings
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import scipy

try:  # old version tn compatiblity
    from tensornetwork.backends import base_backend

    tnbackend = base_backend.BaseBackend

except ImportError:
    from tensornetwork.backends import abstract_backend

    tnbackend = abstract_backend.AbstractBackend


from .abstract_backend import ExtendedBackend

logger = logging.getLogger(__name__)

dtypestr: str
Tensor = Any

cp: Any
cpx: Any


class CuPyBackend(tnbackend, ExtendedBackend):  # type: ignore
    def __init__(self) -> None:
        super().__init__()
        try:
            import cupy
            import cupyx
        except ImportError:
            raise ImportError(
                "CuPy not installed, please switch to a different "
                "backend or install CuPy."
            )
        global cp
        global cpx
        cp = cupy
        cpx = cupyx
        self.name = "cupy"

    def convert_to_tensor(self, a: Tensor) -> Tensor:
        if not isinstance(a, cp.ndarray) and not cp.isscalar(a):
            a = cp.array(a)
        a = cp.asarray(a)
        return a

    def sum(
        self: Any,
        a: Tensor,
        axis: Optional[Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        return cp.sum(a, axis=axis, keepdims=keepdims)

    def conj(self, tensor: Tensor) -> Tensor:
        return tensor.conj()

    def sign(self, tensor: Tensor) -> Tensor:
        return cp.sign(tensor)

    def multiply(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        return tensor1 * tensor2

    def norm(self, tensor: Tensor) -> Tensor:
        return cp.linalg.norm(tensor)

    def shape_tuple(self, tensor: Tensor) -> Tuple[int]:
        return tensor.shape  # type:ignore

    def tensordot(
        self, a: Tensor, b: Tensor, axes: Union[int, Sequence[Sequence[int]]]
    ) -> Tensor:
        return cp.tensordot(a, b, axes)

    def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        return cp.tensordot(tensor1, tensor2, 0)

    def transpose(self, tensor: Tensor, perm: Optional[Sequence[int]] = None) -> Tensor:
        return cp.transpose(tensor, perm)

    def reshape(self, tensor: Tensor, shape: Tensor) -> Tensor:
        return cp.reshape(tensor, np.asarray(shape).astype(np.int32))

    def eye(
        self, N: int, dtype: Optional[str] = None, M: Optional[int] = None
    ) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        return cp.eye(N, M=M, dtype=dtype)

    def ones(self, shape: Sequence[int], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        return cp.ones(shape, dtype=dtype)

    def zeros(self, shape: Sequence[int], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        return cp.zeros(shape, dtype=dtype)

    def copy(self, a: Tensor) -> Tensor:
        return a.copy()

    def expm(self, a: Tensor) -> Tensor:
        return self.convert_to_tensor(scipy.linalg.expm(self.numpy(a)))

    def abs(self, a: Tensor) -> Tensor:
        return cp.abs(a)

    def sin(self, a: Tensor) -> Tensor:
        return cp.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return cp.cos(a)

    # acos acosh asin asinh atan atan2 atanh cosh (cos) tan tanh sinh (sin)
    def acos(self, a: Tensor) -> Tensor:
        return cp.arccos(a)

    def acosh(self, a: Tensor) -> Tensor:
        return cp.arccosh(a)

    def asin(self, a: Tensor) -> Tensor:
        return cp.arcsin(a)

    def asinh(self, a: Tensor) -> Tensor:
        return cp.arcsinh(a)

    def atan(self, a: Tensor) -> Tensor:
        return cp.arctan(a)

    def atan2(self, y: Tensor, x: Tensor) -> Tensor:
        return cp.arctan2(y, x)

    def atanh(self, a: Tensor) -> Tensor:
        return cp.arctanh(a)

    def cosh(self, a: Tensor) -> Tensor:
        return cp.cosh(a)

    def tan(self, a: Tensor) -> Tensor:
        return cp.tan(a)

    def tanh(self, a: Tensor) -> Tensor:
        return cp.tanh(a)

    def sinh(self, a: Tensor) -> Tensor:
        return cp.sinh(a)

    def size(self, a: Tensor) -> Tensor:
        return a.size

    def eigvalsh(self, a: Tensor) -> Tensor:
        return cp.linalg.eigvalsh(a)

    def kron(self, a: Tensor, b: Tensor) -> Tensor:
        return cp.kron(a, b)

    def dtype(self, a: Tensor) -> str:
        return a.dtype.__str__()  # type: ignore

    def numpy(self, a: Tensor) -> Tensor:
        if isinstance(a, cp.ndarray):
            return a.get()
        else:
            return np.array(a)

    def i(self, dtype: Any = None) -> Tensor:
        if not dtype:
            dtype = npdtype  # type: ignore
        if isinstance(dtype, str):
            dtype = getattr(np, dtype)
        return cp.array(1j, dtype=dtype)

    def stack(self, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return cp.stack(a, axis=axis)

    def concat(self, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return cp.concatenate(a, axis=axis)

    def tile(self, a: Tensor, rep: Tensor) -> Tensor:
        return cp.tile(a, rep)

    def mean(
        self,
        a: Tensor,
        axis: Optional[Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        return cp.mean(a, axis=axis, keepdims=keepdims)

    def std(
        self, a: Tensor, axis: Optional[Sequence[int]] = None, keepdims: bool = False
    ) -> Tensor:
        return cp.std(a, axis=axis, keepdims=keepdims)

    def unique_with_counts(self, a: Tensor, **kws: Any) -> Tuple[Tensor, Tensor]:
        return cp.unique(a, return_counts=True)  # type: ignore

    def min(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return cp.min(a, axis=axis)

    def max(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return cp.max(a, axis=axis)

    def argmax(self, a: Tensor, axis: int = 0) -> Tensor:
        return cp.argmax(a, axis=axis)

    def argmin(self, a: Tensor, axis: int = 0) -> Tensor:
        return cp.argmin(a, axis=axis)

    def sigmoid(self, a: Tensor) -> Tensor:
        return cpx.scipy.special.expit(a)

    def relu(self, a: Tensor) -> Tensor:
        return (abs(a) + a) / 2
        # this impl seems to be the fastest
        # see https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy

    def softmax(self, a: Sequence[Tensor], axis: Optional[int] = None) -> Tensor:
        return cpx.scipy.special.softmax(a, axis=axis)

    def onehot(self, a: Tensor, num: int) -> Tensor:
        res = cp.eye(num)[a.reshape([-1])]
        return res.reshape(list(a.shape) + [num])
        # https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy

    def cumsum(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return cp.cumsum(a, axis)

    def is_tensor(self, a: Any) -> bool:
        if isinstance(a, cp.ndarray):
            return True
        return False

    def real(self, a: Tensor) -> Tensor:
        return cp.real(a)

    def imag(self, a: Tensor) -> Tensor:
        return cp.imag(a)

    def cast(self, a: Tensor, dtype: str) -> Tensor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            if isinstance(dtype, str):
                return a.astype(getattr(np, dtype))
            return a.astype(dtype)

    def arange(self, start: int, stop: Optional[int] = None, step: int = 1) -> Tensor:
        if stop is None:
            return cp.arange(start=0, stop=start, step=step)
        return cp.arange(start=start, stop=stop, step=step)

    def mod(self, x: Tensor, y: Tensor) -> Tensor:
        return cp.mod(x, y)

    def right_shift(self, x: Tensor, y: Tensor) -> Tensor:
        return cp.right_shift(x, y)

    def left_shift(self, x: Tensor, y: Tensor) -> Tensor:
        return cp.left_shift(x, y)

    def solve(self, A: Tensor, b: Tensor, assume_a: str = "gen") -> Tensor:  # type: ignore
        raise NotImplementedError

    def searchsorted(self, a: Tensor, v: Tensor, side: str = "left") -> Tensor:
        return cp.searchsorted(a, v, side=side)

    def set_random_state(
        self, seed: Optional[int] = None, get_only: bool = False
    ) -> Any:
        g = cp.random.default_rng(seed)  # None auto supported
        if get_only is False:
            self.g = g
        return g

    def stateful_randn(
        self,
        g: "cp.random.Generator",
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
        g: "cp.random.Generator",
        shape: Union[int, Sequence[int]] = 1,
        low: float = 0,
        high: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        if isinstance(dtype, str):
            dtype = dtype[-2:]
        if isinstance(shape, int):
            shape = (shape,)
        r = g.random(shape) * (high - low) + low
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
        g: "cp.random.Generator",
        a: Union[int, Sequence[int], Tensor],
        shape: Union[int, Sequence[int]],
        p: Optional[Union[Sequence[float], Tensor]] = None,
    ) -> Tensor:
        if isinstance(shape, int):
            shape = (shape,)
        return g.choice(a, size=shape, replace=True, p=p)

    def scatter(self, operand: Tensor, indices: Tensor, updates: Tensor) -> Tensor:
        operand_new = cp.copy(operand)
        operand_new[tuple([indices[:, i] for i in range(indices.shape[1])])] = updates
        return operand_new

    def coo_sparse_matrix(
        self, indices: Tensor, values: Tensor, shape: Tensor
    ) -> Tensor:
        values = self.convert_to_tensor(values)
        indices = self.convert_to_tensor(indices).T
        return cp.sparse.coo_matrix((values, indices), shape=shape)

    def sparse_dense_matmul(
        self,
        sp_a: Tensor,
        b: Tensor,
    ) -> Tensor:
        return sp_a @ b

    def to_dense(self, sp_a: Tensor) -> Tensor:
        return sp_a.todense()

    def is_sparse(self, a: Tensor) -> bool:
        return cpx.scipy.sparse.issparse(a)  # type: ignore

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
        return self._dev2str(a.device)

    def device_move(self, a: Tensor, dev: Any) -> Tensor:
        if isinstance(dev, str):
            dev = self._str2dev(dev)
        with dev:
            return cp.asarray(a)

    def _dev2str(self, dev: Any) -> str:
        return f"gpu:{dev.id}"

    def _str2dev(self, str_: str) -> Any:
        if str_ == "cpu":
            raise ValueError("CuPy backend only support GPU device")
        else:
            return cp.cuda.Device(int(str_.split(":")[-1]))

    def stop_gradient(self, a: Tensor) -> Tensor:
        raise NotImplementedError("CuPy backend doesn't support AD")

    def grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Any]:
        raise NotImplementedError("CuPy backend doesn't support AD")

    def value_and_grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Tuple[Any, Any]]:
        raise NotImplementedError("CuPy backend doesn't support AD")

    def jit(
        self,
        f: Callable[..., Any],
        static_argnums: Optional[Union[int, Sequence[int]]] = None,
        jit_compile: Optional[bool] = None,
        **kws: Any,
    ) -> Callable[..., Any]:
        logger.warning("CuPy backend has no jit interface, just do nothing")
        return f
        # raise NotImplementedError("numpy backend doesn't support jit compiling")

    def vmap(
        self, f: Callable[..., Any], vectorized_argnums: Union[int, Sequence[int]] = 0
    ) -> Any:
        logger.warning(
            "CuPy backend has no intrinsic vmap like interface"
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
            return cp.array(results)

        return wrapper

    def vectorized_value_and_grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        vectorized_argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Tuple[Any, Any]]:
        raise NotImplementedError("CuPy backend doesn't support AD")

    vvag = vectorized_value_and_grad

    def vjp(
        self,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        raise NotImplementedError("CuPy backend doesn't support AD")

    def jvp(
        self,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        raise NotImplementedError("CuPy backend doesn't support AD")
