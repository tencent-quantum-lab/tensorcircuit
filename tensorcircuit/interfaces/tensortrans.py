"""
general function for interfaces transformation
"""

from typing import Any

from ..cons import backend
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
    return which_backend(t).numpy(t)


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
        return backend.tree_map(target_backend.convert_to_tensor, args)
    else:
        if isinstance(dtype, str):
            leaves, treedef = backend.tree_flatten(args)
            dtype = [dtype for _ in range(len(leaves))]
            dtype = backend.tree_unflatten(treedef, dtype)
        t = backend.tree_map(target_backend.convert_to_tensor, args)
        t = backend.tree_map(target_backend.cast, t, dtype)
        return t


def general_args_to_backend(
    args: Any, dtype: Any = None, target_backend: Any = None, enable_dlpack: bool = True
) -> Any:
    if not enable_dlpack:
        # TODO(@refraction-ray): add device shift for numpy mediate transformation
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
