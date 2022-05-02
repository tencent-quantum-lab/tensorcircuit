"""
Interfaces bridging different backends
"""

from typing import Any, Callable, Tuple, Optional

import numpy as np

from .cons import backend, dtypestr
from .backends import get_backend  # type: ignore

Tensor = Any
Array = Any

# this module is highly experimental! expect sharp edges and active API change!


def tensor_to_numpy(t: Tensor) -> Array:
    try:
        return t.numpy()
    except AttributeError:
        return np.array(t)


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
    :return: The same quantum function but now with torch tensor in and torch tensor out
        while AD is also supported
    :rtype: Callable[..., Any]
    """
    import torch

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


def scipy_optimize_interface(
    fun: Callable[..., Any],
    shape: Optional[Tuple[int, ...]] = None,
    jit: bool = True,
    gradient: bool = True,
) -> Callable[..., Any]:
    """
    Convert ``fun`` into a scipy optimize interface compatible version

    :Example:

    .. code-block:: python

        n = 3

        def f(param):
            c = tc.Circuit(n)
            for i in range(n):
                c.rx(i, theta=param[0, i])
                c.rz(i, theta=param[1, i])
            loss = c.expectation(
                [
                    tc.gates.y(),
                    [
                        0,
                    ],
                ]
            )
            return tc.backend.real(loss)

        # A gradient-based optimization interface

        f_scipy = tc.interfaces.scipy_optimize_interface(f, shape=[2, n])
        r = optimize.minimize(f_scipy, np.zeros([2 * n]), method="L-BFGS-B", jac=True)

        # A gradient-free optimization interface

        f_scipy = tc.interfaces.scipy_optimize_interface(f, shape=[2, n], gradient=False)
        r = optimize.minimize(f_scipy, np.zeros([2 * n]), method="COBYLA")


    :param fun: The quantum function with scalar out that to be optimized
    :type fun: Callable[..., Any]
    :param shape: the shape of parameters that ``fun`` accepts, defaults to None
    :type shape: Optional[Tuple[int, ...]], optional
    :param jit: whether to jit ``fun``, defaults to True
    :type jit: bool, optional
    :param gradient: whether using gradient-based or gradient free scipy optimize interface,
        defaults to True
    :type gradient: bool, optional
    :return: The scipy interface compatible version of ``fun``
    :rtype: Callable[..., Any]
    """
    if gradient:
        vg = backend.value_and_grad(fun, argnums=0)
        if jit:
            vg = backend.jit(vg)

        def scipy_vg(*args: Any, **kws: Any) -> Tuple[Tensor, Tensor]:
            scipy_args = numpy_args_to_backend(args, dtype=dtypestr)
            if shape is not None:
                scipy_args = list(scipy_args)
                scipy_args[0] = backend.reshape(scipy_args[0], shape)
                scipy_args = tuple(scipy_args)
            vs, gs = vg(*scipy_args, **kws)
            scipy_vs = general_args_to_numpy(vs)
            gs = backend.reshape(gs, [-1])
            scipy_gs = general_args_to_numpy(gs)
            scipy_vs = scipy_vs.astype(np.float64)
            scipy_gs = scipy_gs.astype(np.float64)
            return scipy_vs, scipy_gs

        return scipy_vg
    # no gradient
    if jit:
        fun = backend.jit(fun)

    def scipy_v(*args: Any, **kws: Any) -> Tensor:
        scipy_args = numpy_args_to_backend(args, dtype=dtypestr)
        if shape is not None:
            scipy_args = list(scipy_args)
            scipy_args[0] = backend.reshape(scipy_args[0], shape)
            scipy_args = tuple(scipy_args)
        vs = fun(*scipy_args, **kws)
        scipy_vs = general_args_to_numpy(vs)
        scipy_vs = scipy_vs.astype(np.float64)
        return scipy_vs

    return scipy_v


scipy_interface = scipy_optimize_interface
