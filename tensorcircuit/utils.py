"""
some helper functions
"""

from typing import Any, Callable, Union, Sequence
from functools import wraps


def return_partial(
    f: Callable[..., Any], return_argnums: Union[int, Sequence[int]] = 0
) -> Callable[..., Any]:
    r"""
    Return a callable function for output ith parts of the original output along the first axis.
    Original output supports List and Tensor.

    Example:

    >>> import tensorcircuit as tc
    >>> from tensorcircuit.utils import return_partial
    >>> import numpy as np
    >>>
    >>> testin = np.array([[1,2],[3,4],[5,6],[7,8]])
    >>>
    >>> # Method 1:
    >>> return_partial(lambda x: x, [1, 3])(testin)
    (array([3, 4]), array([7, 8]))
    >>>
    >>> # Method 2: (As a decorator, can only return
    >>> # the first part of the splited result along the first axis)
    >>> @return_partial
    ... def f(inp):
    ...     return inp
    ...
    >>> f(testin)
    array([1, 2])

    :param f: The function to be applied this method
    :type f: Callable[..., Any]
    :param return_partial: The ith parts of original output along the first axis (axis=0 or dim=0)
    :type return_partial: Union[int, Sequence[int]]
    :returns: The modified callable function
    :rtype: Callable[..., Any]
    """
    if isinstance(return_argnums, int):
        return_argnums = [
            return_argnums,
        ]
        one_input = True
    else:
        one_input = False

    @wraps(f)
    def wrapper(*args: Any, **kws: Any) -> Any:
        r = f(*args, **kws)
        nr = [r[ind] for ind in return_argnums]  # type: ignore
        if one_input:
            return nr[0]
        return tuple(nr)

    return wrapper
