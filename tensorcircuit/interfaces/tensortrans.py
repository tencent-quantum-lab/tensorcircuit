"""
general function for interfaces transformation
"""

from typing import Any
from functools import partial

from ..cons import backend
from ..backends import get_backend  # type: ignore
from ..utils import is_sequence

Tensor = Any
Array = Any

module2backend = {
    "tensorflow": "tensorflow",
    "numpy": "numpy",
    "jaxlib": "jax",
    "torch": "pytorch",
}


def which_backend(a: Tensor, return_backend: bool = True) -> Any:
    module = type(a).__module__.split(".")[0]
    bkstr = module2backend[module]
    if not return_backend:
        return bkstr
    return get_backend(bkstr)


def tensor_to_numpy(t: Tensor) -> Array:
    return which_backend(t).numpy(t)


def general_args_to_numpy(args: Any, same_pytree: bool = True) -> Any:
    res = []
    alone = False
    if not is_sequence(args):
        args = [args]
        alone = True
    for i in args:
        res.append(backend.tree_map(tensor_to_numpy, i))
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
    elif isinstance(target_backend, str):
        target_backend = get_backend(target_backend)

    res = []
    alone = False
    if not is_sequence(args):
        args = [args]
        alone = True
    if not is_sequence(dtype):
        dtype = [dtype for _ in range(len(args))]
    for i, dt in zip(args, dtype):
        if dt is None:
            res.append(backend.tree_map(target_backend.convert_to_tensor, i))
        else:
            t = backend.tree_map(target_backend.convert_to_tensor, i)
            t = backend.tree_map(partial(target_backend.cast, dtype=dt), t)
            # TODO: (@refraction-ray) when dtype is different in the same pytree
            res.append(t)
    if not same_pytree:
        return res  # all list
    if isinstance(args, tuple):
        return tuple(res)
    if isinstance(args, list) and alone is True:
        return res[0]
    return res  # plain list
