"""
Interface wraps quantum function as a tensorflow function
"""

from typing import Any, Callable, Tuple
from functools import wraps

from ..cons import backend
from ..utils import return_partial
from .tensortrans import general_args_to_backend

Tensor = Any


def tf_wrapper(
    fun: Callable[..., Any], enable_dlpack: bool = False
) -> Callable[..., Any]:
    @wraps(fun)
    def fun_tf(*x: Any) -> Any:
        x = general_args_to_backend(x, enable_dlpack=enable_dlpack)
        y = fun(*x)
        y = general_args_to_backend(
            y, target_backend="tensorflow", enable_dlpack=enable_dlpack
        )
        return y

    return fun_tf


def tf_dtype(dtype: str) -> Any:
    import tensorflow as tf

    if isinstance(dtype, str):
        return getattr(tf, dtype)
    return dtype


def tensorflow_interface(
    fun: Callable[..., Any], ydtype: Any, jit: bool = False, enable_dlpack: bool = False
) -> Callable[..., Any]:
    """
    Wrap a quantum function on different ML backend with a tensorflow interface.

    :Example:

    .. code-block:: python

        K = tc.set_backend("jax")


        def f(params):
            c = tc.Circuit(1)
            c.rx(0, theta=params[0])
            c.ry(0, theta=params[1])
            return K.real(c.expectation([tc.gates.z(), [0]]))


        f = tc.interfaces.tf_interface(f, ydtype=tf.float32, jit=True)

        tfb = tc.get_backend("tensorflow")
        grads = tfb.jit(tfb.grad(f))(tc.get_backend("tensorflow").ones([2]))

    :param fun: The quantum function with tensor in and tensor out
    :type fun: Callable[..., Any]
    :param ydtype: output tf dtype or in str
    :type ydtype: Any
    :param jit: whether to jit ``fun``, defaults to False
    :type jit: bool, optional
    :param enable_dlpack: whether transform tensor backend via dlpack, defaults to False
    :type enable_dlpack: bool, optional
    :return: The same quantum function but now with torch tensor in and torch tensor out
        while AD is also supported
    :rtype: Callable[..., Any]
    """
    import tensorflow as tf

    if jit is True:
        fun = backend.jit(fun)

    fun_tf = tf_wrapper(fun)

    ydtype = backend.tree_map(tf_dtype, ydtype)

    @tf.custom_gradient  # type: ignore
    def fun_wrap(*x: Any) -> Any:
        nx = len(x)

        def vjp_fun(*xv: Tensor) -> Tuple[Tensor, Tensor]:
            x = xv[:nx]
            v = xv[nx:]
            if len(x) == 1:
                x = x[0]
            if len(v) == 1:
                v = v[0]
            return backend.vjp(fun, x, v)  # type: ignore

        if jit is True:
            vjp_fun = backend.jit(vjp_fun)

        vjp_fun_tf = return_partial(tf_wrapper(vjp_fun), 1)

        xdtype = backend.tree_map(lambda x: x.dtype, x)
        # (x, )
        if len(xdtype) == 1:
            xdtype = xdtype[0]
        y = tf.py_function(func=fun_tf, inp=x, Tout=ydtype)
        # if len(x) == 1:
        #     x = x[0]

        def grad(*dy: Any, **kws: Any) -> Any:
            # if len(dy) == 1:
            #     dy = dy[0]
            # g = vjp_fun_tf(*(x+dy))
            g = tf.py_function(func=vjp_fun_tf, inp=x + dy, Tout=xdtype)
            # a redundency due to current vjp API

            return g

        return y, grad

    return fun_wrap  # type: ignore


tf_interface = tensorflow_interface

# TODO(@refraction-ray): overhead and efficiency to be benchmarked
