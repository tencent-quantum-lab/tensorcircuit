"""
Interface wraps quantum function as a scipy function for optimization
"""

from typing import Any, Callable, Tuple, Optional

import numpy as np

from ..cons import backend, dtypestr
from .tensortrans import general_args_to_numpy, numpy_args_to_backend

Tensor = Any


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
