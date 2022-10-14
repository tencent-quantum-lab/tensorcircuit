"""
Interface wraps quantum function as a torch function
"""

from typing import Any, Callable, Tuple

from ..cons import backend
from ..utils import is_sequence
from .tensortrans import general_args_to_backend


Tensor = Any


def torch_interface(
    fun: Callable[..., Any], jit: bool = False, enable_dlpack: bool = False
) -> Callable[..., Any]:
    """
    Wrap a quantum function on different ML backend with a pytorch interface.

    :Example:

    .. code-block:: python

        import torch

        tc.set_backend("tensorflow")


        def f(params):
            c = tc.Circuit(1)
            c.rx(0, theta=params[0])
            c.ry(0, theta=params[1])
            return c.expectation([tc.gates.z(), [0]])


        f_torch = tc.interfaces.torch_interface(f, jit=True)

        a = torch.ones([2], requires_grad=True)
        b = f_torch(a)
        c = b ** 2
        c.backward()

        print(a.grad)

    :param fun: The quantum function with tensor in and tensor out
    :type fun: Callable[..., Any]
    :param jit: whether to jit ``fun``, defaults to False
    :type jit: bool, optional
    :param enable_dlpack: whether transform tensor backend via dlpack, defaults to False
    :type enable_dlpack: bool, optional
    :return: The same quantum function but now with torch tensor in and torch tensor out
        while AD is also supported
    :rtype: Callable[..., Any]
    """
    import torch

    def vjp_fun(x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        return backend.vjp(fun, x, v)

    if jit is True:
        fun = backend.jit(fun)
        vjp_fun = backend.jit(vjp_fun)

    class Fun(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *x: Any) -> Any:  # type: ignore
            # ctx.xdtype = [xi.dtype for xi in x]
            ctx.xdtype = backend.tree_map(lambda s: s.dtype, x)
            # (x, )
            if len(ctx.xdtype) == 1:
                ctx.xdtype = ctx.xdtype[0]
            ctx.device = (backend.tree_flatten(x)[0][0]).device
            x = general_args_to_backend(x, enable_dlpack=enable_dlpack)
            y = fun(*x)
            ctx.ydtype = backend.tree_map(lambda s: s.dtype, y)
            if len(x) == 1:
                x = x[0]
            ctx.x = x
            y = general_args_to_backend(
                y, target_backend="pytorch", enable_dlpack=enable_dlpack
            )
            y = backend.tree_map(lambda s: s.to(device=ctx.device), y)
            return y

        @staticmethod
        def backward(ctx: Any, *grad_y: Any) -> Any:
            if len(grad_y) == 1:
                grad_y = grad_y[0]
            grad_y = backend.tree_map(lambda s: s.contiguous(), grad_y)
            grad_y = general_args_to_backend(
                grad_y, dtype=ctx.ydtype, enable_dlpack=enable_dlpack
            )
            # grad_y = general_args_to_numpy(grad_y)
            # grad_y = numpy_args_to_backend(grad_y, dtype=ctx.ydtype)  # backend.dtype
            _, g = vjp_fun(ctx.x, grad_y)
            # a redundency due to current vjp API

            r = general_args_to_backend(
                g,
                dtype=ctx.xdtype,
                target_backend="pytorch",
                enable_dlpack=enable_dlpack,
            )
            r = backend.tree_map(lambda s: s.to(device=ctx.device), r)
            if not is_sequence(r):
                return (r,)
            return r

    # currently, memory transparent dlpack in these ML framework has broken support on complex dtypes
    return Fun.apply  # type: ignore


pytorch_interface = torch_interface
