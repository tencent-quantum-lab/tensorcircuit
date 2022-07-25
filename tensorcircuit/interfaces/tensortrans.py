"""
general function for interfaces transformation
"""

from typing import Any, Callable, Union, Sequence
from functools import partial, wraps

from ..cons import backend, dtypestr
from ..gates import Gate
from ..quantum import QuOperator
from ..backends import get_backend  # type: ignore

Tensor = Any
Array = Any

module2backend = {
    "tensorflow": "tensorflow",
    "numpy": "numpy",
    "jaxlib": "jax",
    "torch": "pytorch",
    "jax": "jax",
}


def which_backend(a: Tensor, return_backend: bool = True) -> Any:
    """
    Given a tensor ``a``, return the corresponding backend

    :param a: the tensor
    :type a: Tensor
    :param return_backend: if true, return backend object, if false, return backend str,
        defaults to True
    :type return_backend: bool, optional
    :return: the backend object or backend str
    :rtype: Any
    """
    module = type(a).__module__.split(".")[0]
    bkstr = module2backend[module]
    if not return_backend:
        return bkstr
    return get_backend(bkstr)


def tensor_to_numpy(t: Tensor) -> Array:
    if isinstance(t, int) or isinstance(t, float):
        return t
    return which_backend(t).numpy(t)


def tensor_to_backend_jittable(t: Tensor) -> Tensor:
    if which_backend(t, return_backend=False) == backend.name:
        return t
    if isinstance(t, int) or isinstance(t, float):
        return t
    return backend.convert_to_tensor(which_backend(t).numpy(t))


def numpy_to_tensor(t: Array, backend: Any) -> Tensor:
    if isinstance(t, int) or isinstance(t, float):
        return t
    return backend.convert_to_tensor(t)


def tensor_to_dtype(t: Tensor) -> str:
    return which_backend(t).dtype(t)  # type: ignore


def tensor_to_dlpack(t: Tensor) -> Any:
    return which_backend(t).to_dlpack(t)


def general_args_to_numpy(args: Any) -> Any:
    """
    Given a pytree, get the corresponding numpy array pytree

    :param args: pytree
    :type args: Any
    :return: the same format pytree with all tensor replaced by numpy array
    :rtype: Any
    """
    return backend.tree_map(tensor_to_numpy, args)


def numpy_args_to_backend(
    args: Any, dtype: Any = None, target_backend: Any = None
) -> Any:
    """
    Given a pytree of numpy arrays, get the corresponding tensor pytree

    :param args: pytree of numpy arrays
    :type args: Any
    :param dtype: str of str of the same pytree shape as args, defaults to None
    :type dtype: Any, optional
    :param target_backend: str or backend object, defaults to None,
        indicating the current default backend
    :type target_backend: Any, optional
    :return: the same format pytree with all numpy array replaced by the tensors
        in the target backend
    :rtype: Any
    """
    if target_backend is None:
        target_backend = backend
    elif isinstance(target_backend, str):
        target_backend = get_backend(target_backend)

    if dtype is None:
        return backend.tree_map(partial(numpy_to_tensor, backend=target_backend), args)
    else:
        if isinstance(dtype, str):
            leaves, treedef = backend.tree_flatten(args)
            dtype = [dtype for _ in range(len(leaves))]
            dtype = backend.tree_unflatten(treedef, dtype)
        t = backend.tree_map(partial(numpy_to_tensor, backend=target_backend), args)
        t = backend.tree_map(target_backend.cast, t, dtype)
        return t


def general_args_to_backend(
    args: Any, dtype: Any = None, target_backend: Any = None, enable_dlpack: bool = True
) -> Any:
    if not enable_dlpack:
        args = general_args_to_numpy(args)
        args = numpy_args_to_backend(args, dtype, target_backend)
        return args

    caps = backend.tree_map(tensor_to_dlpack, args)
    if target_backend is None:
        target_backend = backend
    elif isinstance(target_backend, str):
        target_backend = get_backend(target_backend)
    if dtype is None:
        return backend.tree_map(target_backend.from_dlpack, caps)
    if isinstance(dtype, str):
        leaves, treedef = backend.tree_flatten(args)
        dtype = [dtype for _ in range(len(leaves))]
        dtype = backend.tree_unflatten(treedef, dtype)
    t = backend.tree_map(target_backend.from_dlpack, caps)
    t = backend.tree_map(target_backend.cast, t, dtype)
    return t


def gate_to_matrix(t: Gate, is_reshapem: bool = True) -> Tensor:
    if isinstance(t, Gate):
        t = t.tensor
        if is_reshapem:
            t = backend.reshapem(t)
    return t


def qop_to_matrix(t: QuOperator, is_reshapem: bool = True) -> Tensor:
    if isinstance(t, QuOperator):
        if is_reshapem:
            t = t.copy().eval_matrix()
        else:
            t = t.copy().eval()
    return t


def args_to_tensor(
    f: Callable[..., Any],
    argnums: Union[int, Sequence[int]] = 0,
    tensor_as_matrix: bool = False,
    gate_to_tensor: bool = False,
    gate_as_matrix: bool = True,
    qop_to_tensor: bool = False,
    qop_as_matrix: bool = True,
    cast_dtype: bool = True,
) -> Callable[..., Any]:
    """
    Function decorator that automatically convert inputs to tensors on current backend

    :Example:

    .. code-block:: python

        tc.set_backend("jax")

        @partial(
        tc.interfaces.args_to_tensor,
        argnums=[0, 1, 2],
        gate_to_tensor=True,
        qop_to_tensor=True,
        )
        def f(a, b, c, d):
            return a, b, c, d

        f(
        [tc.Gate(np.ones([2, 2])), tc.Gate(np.ones([2, 2, 2, 2]))],
        tc.QuOperator.from_tensor(np.ones([2, 2, 2, 2, 2, 2])),
        np.ones([2, 2, 2, 2]),
        tf.zeros([1, 2]),
        )

        # ([DeviceArray([[1.+0.j, 1.+0.j],
        #        [1.+0.j, 1.+0.j]], dtype=complex64),
        # DeviceArray([[1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
        #             [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
        #             [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
        #             [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j]], dtype=complex64)],
        # DeviceArray([[1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        #             1.+0.j],
        #             [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        #             1.+0.j],
        #             [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        #             1.+0.j],
        #             [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        #             1.+0.j],
        #             [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        #             1.+0.j],
        #             [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        #             1.+0.j],
        #             [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        #             1.+0.j],
        #             [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        #             1.+0.j]], dtype=complex64),
        # DeviceArray([[[[1.+0.j, 1.+0.j],
        #                 [1.+0.j, 1.+0.j]],

        #             [[1.+0.j, 1.+0.j],
        #                 [1.+0.j, 1.+0.j]]],


        #             [[[1.+0.j, 1.+0.j],
        #                 [1.+0.j, 1.+0.j]],

        #             [[1.+0.j, 1.+0.j],
        #                 [1.+0.j, 1.+0.j]]]], dtype=complex64),
        # <tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[0., 0.]], dtype=float32)>)



    :param f: the wrapped function whose arguments in ``argnums``
        position are expected to be tensor format
    :type f: Callable[..., Any]
    :param argnums: position of args under the auto conversion, defaults to 0
    :type argnums: Union[int, Sequence[int]], optional
    :param tensor_as_matrix: try reshape all input tensor as matrix
        with shape rank 2, defaults to False
    :type tensor_as_matrix: bool, optional
    :param gate_to_tensor: convert ``Gate`` to tensor, defaults to False
    :type gate_to_tensor: bool, optional
    :param gate_as_matrix: reshape tensor from ``Gate`` input as matrix, defaults to True
    :type gate_as_matrix: bool, optional
    :param qop_to_tensor: convert ``QuOperator`` to tensor, defaults to False
    :type qop_to_tensor: bool, optional
    :param qop_as_matrix: reshape tensor from ``QuOperator`` input as matrix, defaults to True
    :type qop_as_matrix: bool, optional
    :param cast_dtype: whether cast to backend dtype, defaults to True
    :type cast_dtype: bool, optional
    :return: The wrapped function
    :rtype: Callable[..., Any]
    """
    if isinstance(argnums, int):
        argnumslist = [argnums]
    else:
        argnumslist = argnums  # type: ignore

    @wraps(f)
    def wrapper(*args: Any, **kws: Any) -> Any:
        nargs = []
        for i, arg in enumerate(args):
            if i in argnumslist:
                if gate_to_tensor:
                    arg = backend.tree_map(
                        partial(gate_to_matrix, is_reshapem=gate_as_matrix), arg
                    )
                if qop_to_matrix:
                    arg = backend.tree_map(
                        partial(qop_to_matrix, is_reshapem=qop_as_matrix), arg
                    )
                arg = backend.tree_map(tensor_to_backend_jittable, arg)
                # arg = backend.tree_map(backend.convert_to_tensor, arg)
                if cast_dtype:
                    arg = backend.tree_map(partial(backend.cast, dtype=dtypestr), arg)
                if tensor_as_matrix:
                    arg = backend.tree_map(backend.reshapem, arg)
            nargs.append(arg)
        return f(*nargs, **kws)

    return wrapper
