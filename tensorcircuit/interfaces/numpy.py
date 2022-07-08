"""
Interface wraps quantum function as a numpy function
"""

from typing import Any, Callable
from functools import wraps

from ..cons import backend
from .tensortrans import general_args_to_numpy, numpy_args_to_backend

Tensor = Any


def numpy_interface(
    fun: Callable[..., Any],
    jit: bool = True,
) -> Callable[..., Any]:
    """
    Convert ``fun`` on ML backend into a numpy function

    :Example:

    .. code-block:: python

        K = tc.set_backend("tensorflow")

        def f(params, n):
            c = tc.Circuit(n)
            for i in range(n):
                c.rx(i, theta=params[i])
            for i in range(n-1):
                c.cnot(i, i+1)
            r = K.real(c.expectation_ps(z=[n-1]))
            return r

        n = 3
        f_np = tc.interfaces.numpy_interface(f, jit=True)
        f_np(np.ones([n]), n)  # 0.1577285


    :param fun: The quantum function
    :type fun: Callable[..., Any]
    :param jit: whether to jit ``fun``, defaults to True
    :type jit: bool, optional
    :return: The numpy interface compatible version of ``fun``
    :rtype: Callable[..., Any]
    """
    if jit:
        fun = backend.jit(fun)

    @wraps(fun)
    def numpy_fun(*args: Any, **kws: Any) -> Any:
        backend_args = numpy_args_to_backend(args)
        r = fun(*backend_args, **kws)
        np_r = general_args_to_numpy(r)
        return np_r

    return numpy_fun


np_interface = numpy_interface
