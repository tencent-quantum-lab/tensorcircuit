"""
Backend magic inherited from tensornetwork: pytorch backend
"""

# pylint: disable=invalid-name

import logging
from typing import Any, Callable, Optional, Sequence, Tuple, Union
from operator import mul
from functools import reduce, partial

import tensornetwork
from tensornetwork.backends.pytorch import pytorch_backend
from .abstract_backend import ExtendedBackend

dtypestr: str
Tensor = Any
pytree = Any

torchlib: Any

logger = logging.getLogger(__name__)

# TODO(@refraction-ray): lack stateful random methods implementation for now
# TODO(@refraction-ray): lack scatter impl for now
# TODO(@refraction-ray): lack sparse relevant methods for now
# To be added once pytorch backend is ready


class torch_jit_func:
    """
    Delay the tracing of torch jit to the first run time:
    consistent with tf and jax mechanism
    """

    def __init__(self, f: Callable[..., Any]):
        self.compiled = False
        self.f = f

    def __call__(self, *args: Any, **kws: Any) -> Any:
        if self.compiled is False:
            self.f = torchlib.jit.trace(self.f, example_inputs=args)
            self.compiled = True

        return self.f(*args, **kws)


class torch_optimizer:
    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer
        self.is_init = False

    def update(self, grads: pytree, params: pytree) -> pytree:
        # flatten grad and param
        params, treedef = PyTorchBackend.tree_flatten(None, params)
        grads, _ = PyTorchBackend.tree_flatten(None, grads)
        if self.is_init is False:
            self.optimizer = self.optimizer(params)
            self.is_init = True
        with torchlib.no_grad():
            for g, p in zip(grads, params):
                p.grad = g
        self.optimizer.step()
        self.optimizer.zero_grad()
        # reorg the param
        params = PyTorchBackend.tree_unflatten(None, treedef, params)
        return params


def _conj_torch(self: Any, tensor: Tensor) -> Tensor:
    t = torchlib.conj(tensor)
    return t.resolve_conj()  # any side effect?


def _sum_torch(
    self: Any,
    tensor: Tensor,
    axis: Optional[Sequence[int]] = None,
    keepdims: bool = False,
) -> Tensor:
    if axis is None:
        axis = tuple([i for i in range(len(tensor.shape))])
    return torchlib.sum(tensor, dim=axis, keepdim=keepdims)


def _qr_torch(
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
    from .pytorch_ops import torchqr

    left_dims = list(tensor.shape[:pivot_axis])
    right_dims = list(tensor.shape[pivot_axis:])

    tensor = torchlib.reshape(tensor, [reduce(mul, left_dims), reduce(mul, right_dims)])
    q, r = torchqr.apply(tensor)
    if non_negative_diagonal:
        phases = torchlib.sign(torchlib.linalg.diagonal(r))
        q = q * phases
        r = phases[:, None] * r
    center_dim = q.shape[1]
    q = torchlib.reshape(q, left_dims + [center_dim])
    r = torchlib.reshape(r, [center_dim] + right_dims)
    return q, r


def _rq_torch(
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
    from .pytorch_ops import torchqr

    left_dims = list(tensor.shape[:pivot_axis])
    right_dims = list(tensor.shape[pivot_axis:])

    tensor = torchlib.reshape(tensor, [reduce(mul, left_dims), reduce(mul, right_dims)])
    q, r = torchqr.apply(tensor.adjoint())
    if non_negative_diagonal:
        phases = torchlib.sign(torchlib.linalg.diagonal(r))
        q = q * phases
        r = phases[:, None] * r
    r, q = r.adjoint(), q.adjoint()
    # M=r*q at this point
    center_dim = r.shape[1]
    r = torchlib.reshape(r, left_dims + [center_dim])
    q = torchlib.reshape(q, [center_dim] + right_dims)
    return r, q


tensornetwork.backends.pytorch.pytorch_backend.PyTorchBackend.sum = _sum_torch
tensornetwork.backends.pytorch.pytorch_backend.PyTorchBackend.conj = _conj_torch
tensornetwork.backends.pytorch.pytorch_backend.PyTorchBackend.qr = _qr_torch
tensornetwork.backends.pytorch.pytorch_backend.PyTorchBackend.rq = _rq_torch


class PyTorchBackend(pytorch_backend.PyTorchBackend, ExtendedBackend):  # type: ignore
    """
    See the original backend API at `pytorch backend
    <https://github.com/google/TensorNetwork/blob/master/tensornetwork/backends/pytorch/pytorch_backend.py>`_

    Note the functionality provided by pytorch backend is incomplete,
    it currenly lacks native efficicent jit and vmap support.
    """

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

    def eye(
        self, N: int, dtype: Optional[str] = None, M: Optional[int] = None
    ) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        if not M:
            M = N
        r = torchlib.eye(n=N, m=M)
        return self.cast(r, dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        r = torchlib.ones(shape)
        return self.cast(r, dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        r = torchlib.zeros(shape)
        return self.cast(r, dtype)

    def copy(self, a: Tensor) -> Tensor:
        return a.clone()

    def expm(self, a: Tensor) -> Tensor:
        raise NotImplementedError("pytorch backend doesn't support expm")
        # in 2020, torch has no expm, hmmm. but that's ok,
        # it doesn't support complex numbers which is more severe issue.
        # see https://github.com/pytorch/pytorch/issues/9983

    def sin(self, a: Tensor) -> Tensor:
        return torchlib.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return torchlib.cos(a)

    def acos(self, a: Tensor) -> Tensor:
        return torchlib.acos(a)

    def acosh(self, a: Tensor) -> Tensor:
        return torchlib.acosh(a)

    def asin(self, a: Tensor) -> Tensor:
        return torchlib.asin(a)

    def asinh(self, a: Tensor) -> Tensor:
        return torchlib.asinh(a)

    def atan(self, a: Tensor) -> Tensor:
        return torchlib.atan(a)

    def atan2(self, y: Tensor, x: Tensor) -> Tensor:
        return torchlib.atan2(y, x)

    def atanh(self, a: Tensor) -> Tensor:
        return torchlib.atanh(a)

    def cosh(self, a: Tensor) -> Tensor:
        return torchlib.cosh(a)

    def tan(self, a: Tensor) -> Tensor:
        return torchlib.tan(a)

    def tanh(self, a: Tensor) -> Tensor:
        return torchlib.tanh(a)

    def sinh(self, a: Tensor) -> Tensor:
        return torchlib.sinh(a)

    def size(self, a: Tensor) -> Tensor:
        return a.size()

    def eigvalsh(self, a: Tensor) -> Tensor:
        return torchlib.linalg.eigvalsh(a)

    def kron(self, a: Tensor, b: Tensor) -> Tensor:
        return torchlib.kron(a, b)

    def numpy(self, a: Tensor) -> Tensor:
        a = a.cpu()
        if a.is_conj():
            return a.resolve_conj().numpy()
        if a.requires_grad:
            return a.detach().numpy()
        return a.numpy()

    def i(self, dtype: Any = None) -> Tensor:
        if not dtype:
            dtype = getattr(torchlib, dtypestr)
        if isinstance(dtype, str):
            dtype = getattr(torchlib, dtype)
        return torchlib.tensor(1j, dtype=dtype)

    def det(self, a: Tensor) -> Tensor:
        return torchlib.linalg.det(a)

    def real(self, a: Tensor) -> Tensor:
        try:
            a = torchlib.real(a)
        except RuntimeError:
            pass
        return a

    def imag(self, a: Tensor) -> Tensor:
        try:
            a = torchlib.imag(a)
        except RuntimeError:
            pass
        return a

    def dtype(self, a: Tensor) -> str:
        return a.dtype.__str__().split(".")[-1]  # type: ignore

    def stack(self, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return torchlib.stack(a, dim=axis)

    def concat(self, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return torchlib.cat(a, dim=axis)

    def tile(self, a: Tensor, rep: Tensor) -> Tensor:
        return torchlib.tile(a, rep)

    def mean(
        self,
        a: Tensor,
        axis: Optional[Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        if axis is None:
            axis = tuple([i for i in range(len(a.shape))])
        return torchlib.mean(a, dim=axis, keepdim=keepdims)

    def std(
        self, a: Tensor, axis: Optional[Sequence[int]] = None, keepdims: bool = False
    ) -> Tensor:
        if axis is None:
            axis = tuple([i for i in range(len(a.shape))])
        return torchlib.std(a, dim=axis, unbiased=False, keepdim=keepdims)

    def min(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        if axis is None:
            return torchlib.min(a)
        return torchlib.min(a, dim=axis).values

    def max(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        if axis is None:
            return torchlib.max(a)
        return torchlib.max(a, dim=axis).values

    def argmax(self, a: Tensor, axis: int = 0) -> Tensor:
        return torchlib.argmax(a, dim=axis)

    def argmin(self, a: Tensor, axis: int = 0) -> Tensor:
        return torchlib.argmin(a, dim=axis)

    def unique_with_counts(self, a: Tensor, **kws: Any) -> Tuple[Tensor, Tensor]:
        return torchlib.unique(a, return_counts=True)  # type: ignore

    def sigmoid(self, a: Tensor) -> Tensor:
        return torchlib.sigmoid(a)

    def relu(self, a: Tensor) -> Tensor:
        return torchlib.relu(a)

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

    def arange(self, start: int, stop: Optional[int] = None, step: int = 1) -> Tensor:
        if stop is None:
            return torchlib.arange(start=0, end=start, step=step)
        return torchlib.arange(start=start, end=stop, step=step)

    def mod(self, x: Tensor, y: Tensor) -> Tensor:
        return torchlib.fmod(x, y)

    def right_shift(self, x: Tensor, y: Tensor) -> Tensor:
        return torchlib.bitwise_right_shift(x, y)

    def left_shift(self, x: Tensor, y: Tensor) -> Tensor:
        return torchlib.bitwise_left_shift(x, y)

    def solve(self, A: Tensor, b: Tensor, **kws: Any) -> Tensor:
        return torchlib.linalg.solve(A, b)

    def searchsorted(self, a: Tensor, v: Tensor, side: str = "left") -> Tensor:
        if not self.is_tensor(a):
            a = self.convert_to_tensor(a)
        if not self.is_tensor(v):
            v = self.convert_to_tensor(v)
        return torchlib.searchsorted(a, v, side=side)

    def reverse(self, a: Tensor) -> Tensor:
        return torchlib.flip(a, dims=(-1,))

    def tree_map(self, f: Callable[..., Any], *pytrees: Any) -> Any:
        # torch native tree_map not support multiple pytree args
        # return torchlib.utils._pytree.tree_map(f, *pytrees)
        args = []
        for pytree in pytrees:
            flat_args, spec = self.tree_flatten(pytree)
            args.append(flat_args)
        res = [
            f(*[args[i][k] for i in range(len(pytrees))]) for k in range(len(flat_args))
        ]
        return self.tree_unflatten(spec, res)

    def tree_flatten(self: Any, pytree: Any) -> Tuple[Any, Any]:
        return torchlib.utils._pytree.tree_flatten(pytree)  # type: ignore

    def tree_unflatten(self: Any, treedef: Any, leaves: Any) -> Any:
        return torchlib.utils._pytree.tree_unflatten(leaves, treedef)

    def from_dlpack(self, a: Any) -> Tensor:
        return torchlib.utils.dlpack.from_dlpack(a)

    def to_dlpack(self, a: Tensor) -> Any:
        return torchlib.utils.dlpack.to_dlpack(a)

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
        return branches[index.numpy()]()

    def device(self, a: Tensor) -> str:
        dev = a.device
        return self._dev2str(dev)

    def device_move(self, a: Tensor, dev: Any) -> Tensor:
        if not isinstance(dev, str):
            dev = self._dev2str(dev)
        if dev.startswith("gpu"):
            dev = "cuda:" + dev.split(":")[-1]
        return a.to(device=dev)

    def _dev2str(self, dev: Any) -> str:
        if dev.type == "cpu":
            return "cpu"
        if dev.type == "cuda":
            return "gpu:" + str(dev.index)
        raise ValueError("PyTorchBackend don't support non-GPU/CPU device")

    def _str2dev(self, str_: str) -> Any:
        if str_ == "cpu":
            return torchlib.device("cpu")
        if str_.startswith("gpu"):
            _id = int(str_.split(":")[-1])
            return torchlib.cuda.device(_id)
        raise ValueError("PyTorchBackend don't support non-GPU/CPU device")

    def stop_gradient(self, a: Tensor) -> Tensor:
        return a.detach()

    def grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Any]:
        def wrapper(*args: Any, **kws: Any) -> Any:
            y, gr = self.value_and_grad(f, argnums, has_aux)(*args, **kws)
            if has_aux:
                return gr, y[1:]
            return gr

        return wrapper

        # def wrapper(*args: Any, **kws: Any) -> Any:
        #     x = []
        #     if isinstance(argnums, int):
        #         argnumsl = [argnums]
        #         # if you also call lhs as argnums, something weird may happen
        #         # the reason is that python then take it as local vars
        #     else:
        #         argnumsl = argnums  # type: ignore
        #     for i, arg in enumerate(args):
        #         if i in argnumsl:
        #             x.append(arg.requires_grad_(True))
        #         else:
        #             x.append(arg)
        #     y = f(*x, **kws)
        #     y.backward()
        #     gs = [x[i].grad for i in argnumsl]
        #     if len(gs) == 1:
        #         gs = gs[0]
        #     return gs

        # return wrapper

    def value_and_grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Tuple[Any, Any]]:
        def wrapper(*args: Any, **kws: Any) -> Any:
            gavf = torchlib.func.grad_and_value(f, argnums=argnums, has_aux=has_aux)
            g, v = gavf(*args, **kws)
            return v, g

        return wrapper
        # def ask_require(t: Tensor) -> Any:
        #     t.requires_grad_(True)
        #     return t

        # def get_grad(t: Tensor) -> Tensor:
        #     return t.grad

        # def wrapper(*args: Any, **kws: Any) -> Any:
        #     # x = []
        #     if isinstance(argnums, int):
        #         argnumsl = [argnums]
        #         # if you also call lhs as argnums, something weird may happen
        #         # the reason is that python then take it as local vars
        #     else:
        #         argnumsl = argnums  # type: ignore
        #     args = list(args)
        #     for i, arg in enumerate(args):
        #         if i in argnumsl:
        #             args[i] = self.tree_map(ask_require, arg)
        #     args = tuple(args)
        #     y = f(*args, **kws)
        #     if has_aux:
        #         y[0].backward()
        #     else:
        #         y.backward()
        #     gs = [self.tree_map(get_grad, x[i]) for i in argnumsl]
        #     if len(gs) == 1:
        #         gs = gs[0]
        #     return y, gs

        # return wrapper

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

    def jvp(
        self,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        if isinstance(v, list):
            v = tuple(v)
        # for both tf and torch
        # behind the scene: https://j-towns.github.io/2017/06/12/A-new-trick.html
        # to be investigate whether the overhead issue remains as in
        # https://github.com/renmengye/tensorflow-forward-ad/issues/2
        return torchlib.autograd.functional.jvp(f, inputs, v)  # type: ignore

    def vmap(
        self,
        f: Callable[..., Any],
        vectorized_argnums: Union[int, Sequence[int]] = 0,
    ) -> Any:
        if isinstance(vectorized_argnums, int):
            vectorized_argnums = (vectorized_argnums,)

        def wrapper(*args: Any, **kws: Any) -> Tensor:
            in_axes = tuple([0 if i in vectorized_argnums else None for i in range(len(args))])  # type: ignore
            return torchlib.vmap(f, in_axes, 0)(*args, **kws)

        return wrapper
        # v3
        # logger.warning(
        #     "pytorch backend has no intrinsic vmap like interface"
        #     ", use plain for loop for compatibility"
        # )
        # # the vmap support is vey limited, f must return one tensor
        # # nested list of tensor as return is not supported
        # if isinstance(vectorized_argnums, int):
        #     vectorized_argnums = (vectorized_argnums,)

        # def wrapper(*args: Any, **kws: Any) -> Tensor:
        #     results = []
        #     for barg in zip(*[args[i] for i in vectorized_argnums]):  # type: ignore
        #         narg = []
        #         j = 0
        #         for k in range(len(args)):
        #             if k in vectorized_argnums:  # type: ignore
        #                 narg.append(barg[j])
        #                 j += 1
        #             else:
        #                 narg.append(args[k])
        #         results.append(f(*narg, **kws))
        #     return torchlib.stack(results)

        # return wrapper

        # v2
        # def vmapf(*args: Tensor, **kws: Any) -> Tensor:
        #     r = []
        #     for i in range(args[0].shape[0]):
        #         nargs = [arg[i] for arg in args]
        #         r.append(f(*nargs, **kws))
        #     return torchlib.stack(r)

        # return vmapf

        # v1
        # raise NotImplementedError("pytorch backend doesn't support vmap")
        # There seems to be no map like architecture in pytorch for now
        # see https://discuss.pytorch.org/t/fast-way-to-use-map-in-pytorch/70814

    def jit(
        self,
        f: Callable[..., Any],
        static_argnums: Optional[Union[int, Sequence[int]]] = None,
        jit_compile: Optional[bool] = None,
        **kws: Any
    ) -> Any:
        if jit_compile is True:
            # experimental feature reusing the jit_compile flag for tf
            return torch_jit_func(f)
        return f
        # return f  # do nothing here until I figure out what torch.jit is for and how does it work
        # see https://github.com/pytorch/pytorch/issues/36910

    def vectorized_value_and_grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        vectorized_argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Tuple[Any, Any]]:
        # [WIP], not a consistent impl compared to tf and jax backend, but pytorch backend is not fully supported anyway
        if isinstance(vectorized_argnums, int):
            vectorized_argnums = (vectorized_argnums,)

        def wrapper(
            *args: Any, **kws: Any
        ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
            jf = self.value_and_grad(f, argnums=argnums, has_aux=has_aux)
            jf = self.vmap(jf, vectorized_argnums=vectorized_argnums)
            vs, gs = jf(*args, **kws)

            if isinstance(argnums, int):
                argnums_list = [argnums]
                gs = [gs]
            else:
                argnums_list = argnums  # type: ignore
                gs = list(gs)
            for i, (j, g) in enumerate(zip(argnums_list, gs)):
                if j not in vectorized_argnums:  # type: ignore
                    gs[i] = self.tree_map(partial(torchlib.sum, dim=0), g)
            if isinstance(argnums, int):
                gs = gs[0]
            else:
                gs = tuple(gs)

            return vs, gs

        return wrapper

    vvag = vectorized_value_and_grad

    optimizer = torch_optimizer
