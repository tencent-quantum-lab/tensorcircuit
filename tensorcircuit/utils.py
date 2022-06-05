"""
Helper functions
"""

from typing import Any, Callable, Union, Sequence
from functools import wraps
import platform


def return_partial(
    f: Callable[..., Any], return_argnums: Union[int, Sequence[int]] = 0
) -> Callable[..., Any]:
    r"""
    Return a callable function for output ith parts of the original output along the first axis.
    Original output supports List and Tensor.

    :Example:

    >>> from tensorcircuit.utils import return_partial
    >>> testin = np.array([[1,2],[3,4],[5,6],[7,8]])
    >>> # Method 1:
    >>> return_partial(lambda x: x, [1, 3])(testin)
    (array([3, 4]), array([7, 8]))
    >>> # Method 2:
    >>> from functools import partial
    >>> @partial(return_partial, return_argnums=(0,2))
    ... def f(inp):
    ...     return inp
    ...
    >>> f(testin)
    (array([1, 2]), array([5, 6]))

    :param f: The function to be applied this method
    :type f: Callable[..., Any]
    :param return_partial: The ith parts of original output along the first axis (axis=0 or dim=0)
    :type return_partial: Union[int, Sequence[int]]
    :return: The modified callable function
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


def append(f: Callable[..., Any], *op: Callable[..., Any]) -> Any:
    """
    Functional programming paradigm to build function pipeline

    :Example:

    >>> f = tc.utils.append(lambda x: x**2, lambda x: x+1, tc.backend.mean)
    >>> f(tc.backend.ones(2))
    (2+0j)

    :param f: The function which are attached with other functions
    :type f: Callable[..., Any]
    :param op: Function to be attached
    :type op: Callable[..., Any]
    :return: The final results after function pipeline
    :rtype: Any
    """

    @wraps(f)
    def wrapper(*args: Any, **kws: Any) -> Any:
        rs = f(*args, **kws)
        for opi in op:
            rs = opi(rs)
        return rs

    return wrapper


def is_m1mac() -> bool:
    """
    check whether the running platform is MAC with M1 chip

    :return: True for MAC M1 platform
    :rtype: bool
    """
    if platform.processor() != "arm":
        return False
    if not platform.platform().startswith("macOS"):
        return False
    return True
