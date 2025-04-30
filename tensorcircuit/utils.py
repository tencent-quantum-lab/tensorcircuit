"""
Helper functions
"""

from typing import Any, Callable, Union, Sequence, Tuple, Dict
from functools import wraps
import os
import platform
import re
import time


def gpu_memory_share(flag: bool = True) -> None:
    # TODO(@refraction-ray): the default torch behavior should be True
    # preallocate behavior for torch to be investigated
    if flag is True:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    else:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"


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


def is_sequence(x: Any) -> bool:
    if isinstance(x, list) or isinstance(x, tuple):
        return True
    return False


def is_number(x: Any) -> bool:
    if isinstance(x, int) or isinstance(x, float) or isinstance(x, complex):
        return True
    return False


def arg_alias(
    f: Callable[..., Any],
    alias_dict: Dict[str, Union[str, Sequence[str]]],
    fix_doc: bool = True,
) -> Callable[..., Any]:
    """
    function argument alias decorator with new docstring

    :param f: _description_
    :type f: Callable[..., Any]
    :param alias_dict: _description_
    :type alias_dict: Dict[str, Union[str, Sequence[str]]]
    :param fix_doc: whether to add doc for these new alias arguments, defaults True
    :type fix_doc: bool
    :return: the decorated function
    :rtype: Callable[..., Any]
    """

    @wraps(f)
    def wrapper(*args: Any, **kws: Any) -> Any:
        for k, vs in alias_dict.items():
            if isinstance(vs, str):
                vs = []
            for v in vs:
                if v in kws:
                    # in case it is None by design!
                    kws[k] = kws[v]
                    del kws[v]
        return f(*args, **kws)

    if fix_doc:
        doc = wrapper.__doc__
        if doc is not None:
            doc = doc.split("\n")  # type: ignore
            ndoc = []
            skip = False
            for i, line in enumerate(doc):
                if not skip:
                    ndoc.append(line)
                else:
                    if doc[i].strip().startswith(":type"):
                        skip = False

                if line.strip().startswith(":param "):
                    param = re.findall(r":param\s([^\s]+):", line)[0]
                    if param in alias_dict:
                        j = 1
                        while True:
                            ndoc.append(doc[i + j])
                            if doc[i + j].strip().startswith(":type"):
                                break
                            j += 1
                        skip = True
                        for v in alias_dict[param]:
                            ndoc.append(
                                re.sub(
                                    r"(.*:param\s)([^\s]+)(:.*)",
                                    r"\1%s: alias for the argument ``%s``" % (v, param),
                                    line,
                                )
                            )
                            j = 1
                            while True:
                                if doc[i + j].strip().startswith(":type"):
                                    ndoc.append(
                                        re.sub(
                                            r"(.*:type\s)([^\s]+)(:.*)",
                                            r"\1%s\3" % v,
                                            doc[i + j],
                                        )
                                    )
                                    break
                                j += 1
            ndoc = "\n".join(ndoc)  # type: ignore
            wrapper.__doc__ = ndoc  # type: ignore
    return wrapper


def benchmark(
    f: Any, *args: Any, tries: int = 5, verbose: bool = True
) -> Tuple[Any, float, float]:
    """
    benchmark jittable function with staging time and running time

    :param f: _description_
    :type f: Any
    :param tries: _description_, defaults to 5
    :type tries: int, optional
    :param verbose: _description_, defaults to True
    :type verbose: bool, optional
    :return: _description_
    :rtype: Tuple[Any, float, float]
    """
    time0 = time.time()
    r = f(*args)
    time1 = time.time()
    for _ in range(tries):
        r = f(*args)
    time2 = time.time()
    if tries == 0:
        rt = 0
    else:
        rt = (time2 - time1) / tries  # type: ignore
    if verbose:
        print("staging time: ", time1 - time0, "running time: ", rt)
    return r, time1 - time0, rt
