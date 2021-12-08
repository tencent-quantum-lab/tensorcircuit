"""
interfaces bridging different backends
"""

from typing import Any, Callable, Tuple

import numpy as np
import torch

from .cons import backend
from .backends import get_backend  # type: ignore

Tensor = Any
Array = Any

# this module is highly experimental! expect sharp edges and active API change!


def tensor_to_numpy(t: Tensor) -> Array:
    from jax import numpy as jnp
    import tensorflow as tf

    if isinstance(t, torch.Tensor):
        return t.numpy()
    if isinstance(t, tf.Tensor) or isinstance(t, tf.Variable):
        return t.numpy()
    if isinstance(t, jnp.ndarray):
        return np.array(t)
    return t


def general_args_to_numpy(args: Any, same_pytree: bool = True) -> Any:
    res = []
    alone = False
    if not (isinstance(args, tuple) or isinstance(args, list)):
        args = [args]
        alone = True
    for i in args:
        res.append(tensor_to_numpy(i))
    if not same_pytree:
        return res  # all list
    if isinstance(args, tuple):
        return tuple(res)
    if isinstance(args, list) and alone is True:
        return res[0]
    return res  # plain list


def numpy_args_to_backend(
    args: Any, same_pytree: bool = True, dtype: Any = None, target_backend: Any = None
) -> Any:
    # TODO(@refraction-ray): switch same_pytree default to True
    if target_backend is None:
        target_backend = backend
    else:
        target_backend = get_backend(target_backend)
    res = []
    alone = False
    if not (isinstance(args, tuple) or isinstance(args, list)):
        args = [args]
        alone = True
    if not (isinstance(dtype, list) or isinstance(dtype, tuple)):
        dtype = [dtype for _ in range(len(args))]
    for i, dt in zip(args, dtype):
        if dt is None:
            res.append(target_backend.convert_to_tensor(i))
        else:
            t = target_backend.convert_to_tensor(i)
            t = target_backend.cast(t, dtype=dt)
            res.append(t)
    if not same_pytree:
        return res  # all list
    if isinstance(args, tuple):
        return tuple(res)
    if isinstance(args, list) and alone is True:
        return res[0]
    return res  # plain list


def is_sequence(x: Any) -> bool:
    if isinstance(x, list) or isinstance(x, tuple):
        return True
    return False


def torch_interface(fun: Callable[..., Any], jit: bool = False) -> Callable[..., Any]:
    def vjp_fun(x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        return backend.vjp(fun, x, v)  # type: ignore

    if jit is True:
        fun = backend.jit(fun)
        vjp_fun = backend.jit(vjp_fun)

    class Fun(torch.autograd.Function):  # type: ignore
        @staticmethod
        def forward(ctx: Any, *x: Any) -> Any:  # type: ignore
            ctx.xdtype = [xi.dtype for xi in x]
            x = general_args_to_numpy(x)
            x = numpy_args_to_backend(x)
            y = fun(*x)
            if not is_sequence(y):
                ctx.ydtype = [y.dtype]
            else:
                ctx.ydtype = [yi.dtype for yi in y]
            if len(x) == 1:
                ctx.x = x[0]
            else:
                ctx.x = x
            y = numpy_args_to_backend(
                general_args_to_numpy(y),
                target_backend="pytorch",
            )
            return y

        @staticmethod
        def backward(ctx: Any, *grad_y: Any) -> Any:
            grad_y = general_args_to_numpy(grad_y)
            grad_y = numpy_args_to_backend(
                grad_y, dtype=[d for d in ctx.ydtype]
            )  # backend.dtype
            if len(grad_y) == 1:
                grad_y = grad_y[0]
            _, g = vjp_fun(ctx.x, grad_y)
            # a redundency due to current vjp API
            r = numpy_args_to_backend(
                general_args_to_numpy(g),
                dtype=[d for d in ctx.xdtype],  # torchdtype
                target_backend="pytorch",
            )
            if not is_sequence(r):
                return (r,)
            return r

    # currently, memory transparent dlpack in these ML framework has broken support on complex dtypes
    return Fun.apply  # type: ignore
