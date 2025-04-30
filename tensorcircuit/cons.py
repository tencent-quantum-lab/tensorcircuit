"""
Constants and setups
"""

# pylint: disable=invalid-name

import logging
import sys
import time
from contextlib import contextmanager
from functools import partial, reduce, wraps
from operator import mul
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import opt_einsum
import tensornetwork as tn
from tensornetwork.backend_contextmanager import get_default_backend

from .backends.numpy_backend import NumpyBackend
from .backends import get_backend
from .simplify import _multi_remove

logger = logging.getLogger(__name__)

package_name = "tensorcircuit"
thismodule = sys.modules[__name__]
dtypestr = "complex64"
rdtypestr = "float32"
npdtype = np.complex64
backend: NumpyBackend = get_backend("numpy")
contractor = tn.contractors.auto
# these above lines are just for mypy, it is not very good at evaluating runtime object


def set_tensornetwork_backend(
    backend: Optional[str] = None, set_global: bool = True
) -> Any:
    r"""To set the runtime backend of tensorcircuit.

    Note: ``tc.set_backend`` and ``tc.cons.set_tensornetwork_backend`` are the same.

    :Example:

    >>> tc.set_backend("numpy")
    numpy_backend
    >>> tc.gates.num_to_tensor(0.1)
    array(0.1+0.j, dtype=complex64)
    >>>
    >>> tc.set_backend("tensorflow")
    tensorflow_backend
    >>> tc.gates.num_to_tensor(0.1)
    <tf.Tensor: shape=(), dtype=complex64, numpy=(0.1+0j)>
    >>>
    >>> tc.set_backend("pytorch")
    pytorch_backend
    >>> tc.gates.num_to_tensor(0.1)
    tensor(0.1000+0.j)
    >>>
    >>> tc.set_backend("jax")
    jax_backend
    >>> tc.gates.num_to_tensor(0.1)
    DeviceArray(0.1+0.j, dtype=complex64)

    :param backend: "numpy", "tensorflow", "jax", "pytorch". defaults to None,
        which gives the same behavior as ``tensornetwork.backend_contextmanager.get_default_backend()``.
    :type backend: Optional[str], optional
    :param set_global: Whether the object should be set as global.
    :type set_global: bool
    :return: The `tc.backend` object that with all registered universal functions.
    :rtype: backend object
    """
    if not backend:
        backend = get_default_backend()
    backend_obj = get_backend(backend)
    if set_global:
        for module in sys.modules:
            if module.startswith(package_name):
                setattr(sys.modules[module], "backend", backend_obj)
        tn.set_default_backend(backend)
    return backend_obj


set_backend = set_tensornetwork_backend

set_tensornetwork_backend()


def set_function_backend(backend: Optional[str] = None) -> Callable[..., Any]:
    """
    Function decorator to set function-level runtime backend

    :param backend: "numpy", "tensorflow", "jax", "pytorch", defaults to None
    :type backend: Optional[str], optional
    :return: Decorated function
    :rtype: Callable[..., Any]
    """

    def wrapper(f: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(f)
        def newf(*args: Any, **kws: Any) -> Any:
            old_backend = getattr(thismodule, "backend").name
            set_backend(backend)
            r = f(*args, **kws)
            set_backend(old_backend)
            return r

        return newf

    return wrapper


@contextmanager
def runtime_backend(backend: Optional[str] = None) -> Iterator[Any]:
    """
    Context manager to set with-level runtime backend

    :param backend: "numpy", "tensorflow", "jax", "pytorch", defaults to None
    :type backend: Optional[str], optional
    :yield: the backend object
    :rtype: Iterator[Any]
    """
    old_backend = getattr(thismodule, "backend").name
    K = set_backend(backend)
    yield K
    set_backend(old_backend)


def set_dtype(dtype: Optional[str] = None, set_global: bool = True) -> Tuple[str, str]:
    """
    Set the global runtime numerical dtype of tensors.

    :param dtype: "complex64"/"float32" or "complex128"/"float64",
        defaults to None, which is equivalent to "complex64".
    :type dtype: Optional[str], optional
    :return: complex dtype str and the corresponding real dtype str
    :rtype: Tuple[str, str]
    """
    if not dtype:
        dtype = "complex64"

    if dtype == "complex64":
        rdtype = "float32"
    elif dtype == "complex128":
        rdtype = "float64"
    elif dtype == "float32":
        dtype = "complex64"
        rdtype = "float32"
    elif dtype == "float64":
        dtype = "complex128"
        rdtype = "float64"
    else:
        raise ValueError(f"Unsupported data type: {dtype}")

    try:
        from jax import config
    except ImportError:
        config = None  # type: ignore

    if config is not None:
        if dtype == "complex128":
            config.update("jax_enable_x64", True)
        elif dtype == "complex64":
            config.update("jax_enable_x64", False)

    if set_global:
        npdtype = getattr(np, dtype)
        for module in sys.modules:
            if module.startswith(package_name):
                setattr(sys.modules[module], "dtypestr", dtype)
                setattr(sys.modules[module], "rdtypestr", rdtype)
                setattr(sys.modules[module], "npdtype", npdtype)

        from .gates import meta_gate

        meta_gate()
    return dtype, rdtype


get_dtype = partial(set_dtype, set_global=False)

set_dtype()


def set_function_dtype(dtype: Optional[str] = None) -> Callable[..., Any]:
    """
    Function decorator to set function-level numerical dtype

    :param dtype: "complex64" or "complex128", defaults to None
    :type dtype: Optional[str], optional
    :return: The decorated function
    :rtype: Callable[..., Any]
    """

    def wrapper(f: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(f)
        def newf(*args: Any, **kws: Any) -> Any:
            old_dtype = getattr(thismodule, "dtypestr")
            set_dtype(dtype)
            r = f(*args, **kws)
            set_dtype(old_dtype)
            return r

        return newf

    return wrapper


@contextmanager
def runtime_dtype(dtype: Optional[str] = None) -> Iterator[Tuple[str, str]]:
    """
    Context manager to set with-level runtime dtype

    :param dtype: "complex64" or "complex128", defaults to None ("complex64")
    :type dtype: Optional[str], optional
    :yield: complex dtype str and real dtype str
    :rtype: Iterator[Tuple[str, str]]
    """
    old_dtype = getattr(thismodule, "dtypestr")
    dtuple = set_dtype(dtype)
    yield dtuple
    set_dtype(old_dtype)


# here below comes other contractors (just works,
# but correctness has not been extensively tested for some of them)


def _sizen(node: tn.Node, is_log: bool = False) -> int:
    s = reduce(mul, node.tensor.shape + (1,))
    if is_log:
        return int(np.log2(s))
    return s  # type: ignore


def _merge_single_gates(
    nodes: List[Any], total_size: Optional[int] = None
) -> Tuple[List[Any], int]:
    # TODO(@refraction-ray): investigate whether too much copy here so that staging is slow for large circuit
    nodes = list(nodes)
    if total_size is None:
        total_size = sum([_sizen(t) for t in nodes])
    queue = [n for n in nodes if len(n.tensor.shape) <= 2]
    while queue:
        n0 = queue[0]
        try:
            n0[0]
        except IndexError:
            queue = _multi_remove(queue, [0])
            continue
        if n0[0].is_dangling():
            try:
                e0 = n0[1]
                if e0.is_dangling():
                    queue = _multi_remove(queue, [0])
                    continue
            except IndexError:
                queue = _multi_remove(queue, [0])
                continue
        else:
            e0 = n0[0]
        njs = [i for i, n in enumerate(nodes) if id(n) in [id(e0.node1), id(e0.node2)]]
        qjs = [i for i, n in enumerate(queue) if id(n) in [id(e0.node1), id(e0.node2)]]
        new_node = tn.contract(e0)
        total_size += _sizen(new_node)

        logger.debug(
            _sizen(new_node, is_log=True),
        )
        queue = _multi_remove(queue, qjs)
        if len(njs) > 1:
            nodes[njs[1]] = new_node
            nodes = _multi_remove(nodes, [njs[0]])
        else:  # trace edge?
            nodes[njs[0]] = new_node
        if len(new_node.tensor.shape) <= 2:
            # queue.append(new_node)
            queue.insert(0, new_node)
    return nodes, total_size


def experimental_contractor(
    nodes: List[Any],
    output_edge_order: Optional[List[Any]] = None,
    ignore_edge_order: bool = False,
    local_steps: int = 2,
) -> Any:
    total_size = sum([_sizen(t) for t in nodes])
    nodes = list(nodes)
    # merge single qubit gate
    if len(nodes) > 5:
        nodes, total_size = _merge_single_gates(nodes, total_size)

    # further dq fusion
    if len(nodes) > 15:
        for r in range(local_steps):
            if len(nodes) < 10:
                break
            i = 0
            while len(nodes) > i + 1:
                new_node = tn.contract_between(
                    nodes[i], nodes[i + 1], allow_outer_product=True
                )
                total_size += _sizen(new_node)

                logger.debug(
                    r,
                    _sizen(new_node, is_log=True),
                )
                nodes[i] = new_node
                nodes = _multi_remove(nodes, [i + 1])
                i += 1

    logger.info("length of remaining nodes after dq fusion: %s" % len(nodes))
    nodes = list(reversed(nodes))

    while len(nodes) > 1:
        new_node = tn.contract_between(nodes[-1], nodes[-2], allow_outer_product=True)
        nodes = _multi_remove(nodes, [len(nodes) - 2, len(nodes) - 1])
        nodes.append(new_node)
        logger.debug(_sizen(new_node, is_log=True))
        total_size += _sizen(new_node)

    logger.info("----- WRITE: %s --------\n" % np.log2(total_size))

    # if the final node has more than one edge,
    # output_edge_order has to be specified
    final_node = nodes[0]
    if output_edge_order is not None:
        final_node.reorder_edges(output_edge_order)
    return final_node


def plain_contractor(
    nodes: List[Any],
    output_edge_order: Optional[List[Any]] = None,
    ignore_edge_order: bool = False,
) -> Any:
    """
    The naive state-vector simulator contraction path.

    :param nodes: The list of ``tn.Node``.
    :type nodes: List[Any]
    :param output_edge_order: The list of dangling node edges, defaults to be None.
    :type output_edge_order: Optional[List[Any]], optional
    :return: The ``tn.Node`` after contraction
    :rtype: tn.Node
    """
    total_size = sum([_sizen(t) for t in nodes])
    # nodes = list(reversed(list(nodes)))
    nodes = list(nodes)
    nodes = list(reversed(nodes))

    width = 0

    while len(nodes) > 1:
        new_node = tn.contract_between(nodes[-1], nodes[-2], allow_outer_product=True)
        nodes = _multi_remove(nodes, [len(nodes) - 2, len(nodes) - 1])
        nodes.append(new_node)
        im_size = _sizen(new_node, is_log=True)
        logger.debug(im_size)
        width = max(width, im_size)
        total_size += _sizen(new_node)
    logger.info("----- SIZE: %s --------\n" % width)
    logger.info("----- WRITE: %s --------\n" % np.log2(total_size))

    final_node = nodes[0]
    if output_edge_order is not None:
        final_node.reorder_edges(output_edge_order)
    return final_node


# TODO(@refraction-ray): try to make plain contractor follow the pattern of custom?

# TODO(@refraction-ray): consistent logger system for different contractors.


def nodes_to_adj(ns: List[Any]) -> Any:
    ind = {id(n): i for i, n in enumerate(ns)}
    adj = np.zeros([len(ns), len(ns)])
    for node in ns:
        for e in node:
            if not e.is_dangling():
                if id(e.node1) == id(node):
                    onode = e.node2
                else:
                    onode = e.node1
                adj[ind[id(node)], ind[id(onode)]] += np.log10(e.dimension)
    return adj


try:
    _ps = tn.contractors.custom_path_solvers.pathsolvers
    has_ps = True
except AttributeError:
    has_ps = False


def d2s(n: int, dl: List[Any]) -> List[Any]:
    # dynamic to static list
    nums = [i for i in range(n)]
    i = n
    sl = []
    for a, b in dl:
        sl.append([nums[a], nums[b]])
        nums = _multi_remove(nums, [a, b])
        nums.insert(a, i)
        i += 1
    return sl


# seems worse than plain contraction in most cases
def tn_greedy_contractor(
    nodes: List[Any],
    output_edge_order: Optional[List[Any]] = None,
    ignore_edge_order: bool = False,
    max_branch: int = 1,
) -> Any:
    nodes = list(nodes)
    adj = nodes_to_adj(nodes)
    path = _ps.full_solve_complete(adj, max_branch=max_branch)[0]
    dl = []
    for i in range(path.shape[1]):
        a, b = path[:, i]
        dl.append([a, b])
    sl = d2s(len(nodes), dl)
    for a, b in sl:
        new_node = tn.contract_between(nodes[a], nodes[b], allow_outer_product=True)
        nodes.append(new_node)
        # new_node = tn.contract_between(nodes[a], nodes[b], allow_outer_product=True)
        # nodes = _multi_remove(nodes, [a, b])
        # nodes.insert(a, new_node)
    final_node = nodes[-1]
    if output_edge_order is not None:
        final_node.reorder_edges(output_edge_order)
    return final_node


# base = tn.contractors.opt_einsum_paths.path_contractors.base
# utils = tn.contractors.opt_einsum_paths.utils


def _get_path(
    nodes: List[tn.Node], algorithm: Any
) -> Tuple[List[Tuple[int, int]], List[tn.Node]]:
    nodes = list(nodes)
    input_sets = [set([id(e) for e in node.edges]) for node in nodes]
    output_set = set([id(e) for e in tn.get_subgraph_dangling(nodes)])
    size_dict = {id(edge): edge.dimension for edge in tn.get_all_edges(nodes)}

    return algorithm(input_sets, output_set, size_dict), nodes


def _get_path_cache_friendly(
    nodes: List[tn.Node], algorithm: Any
) -> Tuple[List[Tuple[int, int]], List[tn.Node]]:
    nodes = list(nodes)
    mapping_dict = {}
    i = 0
    for n in nodes:
        for e in n:
            if id(e) not in mapping_dict:
                mapping_dict[id(e)] = i
                i += 1
    # TODO(@refraction-ray): may be not that cache friendly, since the edge id correspondence is not that fixed?
    input_sets = [set([mapping_dict[id(e)] for e in node.edges]) for node in nodes]
    placeholder = [[1e20 for _ in range(100)]]
    order = np.argsort(np.array(list(map(sorted, input_sets)) + placeholder, dtype=object))[:-1]  # type: ignore
    nodes_new = [nodes[i] for i in order]
    if isinstance(algorithm, list):
        return algorithm, nodes_new

    input_sets = [set([mapping_dict[id(e)] for e in node.edges]) for node in nodes_new]
    output_set = set([mapping_dict[id(e)] for e in tn.get_subgraph_dangling(nodes_new)])
    size_dict = {
        mapping_dict[id(edge)]: edge.dimension for edge in tn.get_all_edges(nodes_new)
    }
    logger.debug("input_sets: %s" % input_sets)
    logger.debug("output_set: %s" % output_set)
    logger.debug("size_dict: %s" % size_dict)
    logger.debug("path finder algorithm: %s" % algorithm)
    return algorithm(input_sets, output_set, size_dict), nodes_new
    # directly get input_sets, output_set and size_dict by using identity function as algorithm


# some contractor setup usages
"""
import cotengra as ctg
import opt_einsum as oem

sys.setrecursionlimit(10000) # for successfullt ctg parallel

opt = ctg.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    parallel=True,
    minimize="write",
    max_time=30,
    max_repeats=4096,
    progbar=True,
)
tc.set_contractor("custom", optimizer=opt, preprocessing=True)
tc.set_contractor("custom_stateful", optimizer=oem.RandomGreedy, max_time=60, max_repeats=128, minimize="size")
tc.set_contractor("plain-experimental", local_steps=3)

# hyper efficient contractor: though long computation time required, suitable for extra large circuit simulation
opt = ctg.ReusableHyperOptimizer(
    minimize='combo',
    max_repeats=1024,
    max_time='equil:128',
    optlib='nevergrad',
    progbar=True,
)

def opt_reconf(inputs, output, size, **kws):
    tree = opt.search(inputs, output, size)
    tree_r = tree.subtree_reconfigure_forest(progbar=True, num_trees=10, num_restarts=20, subtree_weight_what=("size", ))
    return tree_r.get_path()

tc.set_contractor("custom", optimizer=opt_reconf)
"""


def _base(
    nodes: List[tn.Node],
    algorithm: Any,
    output_edge_order: Optional[Sequence[tn.Edge]] = None,
    ignore_edge_order: bool = False,
    total_size: Optional[int] = None,
    debug_level: int = 0,
) -> tn.Node:
    """
    The base method for all `opt_einsum` contractors.

    :param nodes: A collection of connected nodes.
    :type nodes: List[tn.Node]
    :pram algorithm: `opt_einsum` contraction method to use.
    :type algorithm: Any
    :param output_edge_order: An optional list of edges. Edges of the
        final node in `nodes_set` are reordered into `output_edge_order`;
        if final node has more than one edge, `output_edge_order` must be provided.
    :type output_edge_order: Optional[Sequence[tn.Edge]], optional
    :param ignore_edge_order: An option to ignore the output edge order.
    :type ignore_edge_order: bool
    :param total_size: The total size of the tensor network.
    :type total_size: Optional[int], optional
    :raises ValueError:"The final node after contraction has more than
        one remaining edge. In this case `output_edge_order` has to be provided," or
        "Output edges are not equal to the remaining non-contracted edges of the final node."
    :return: The final node after full contraction.
    :rtype: tn.Node
    """
    # rewrite tensornetwork default to add logging infras
    nodes_set = set(nodes)
    edges = tn.get_all_edges(nodes_set)
    # output edge order has to be determinded before any contraction
    # (edges are refreshed after contractions)

    if not ignore_edge_order:
        if output_edge_order is None:
            output_edge_order = list(tn.get_subgraph_dangling(nodes))
            if len(output_edge_order) > 1:
                raise ValueError(
                    "The final node after contraction has more than "
                    "one remaining edge. In this case `output_edge_order` "
                    "has to be provided."
                )

        if set(output_edge_order) != tn.get_subgraph_dangling(nodes):
            raise ValueError(
                "output edges are not equal to the remaining "
                "non-contracted edges of the final node."
            )

    for edge in edges:
        if not edge.is_disabled:  # if its disabled we already contracted it
            if edge.is_trace():
                idx = [i for i, n in enumerate(nodes) if id(n) == id(edge.node1)]
                nodes = _multi_remove(nodes, idx)
                nodes.append(tn.contract_parallel(edge))
                # nodes_set.remove(edge.node1)
                # nodes_set.add(tn.contract_parallel(edge))

    if len(nodes) == 1:
        # There's nothing to contract.
        if ignore_edge_order:
            return list(nodes)[0]
        return list(nodes)[0].reorder_edges(output_edge_order)

    # nodes = list(nodes_set)

    # Then apply `opt_einsum`'s algorithm
    # if isinstance(algorithm, list):
    #     path = algorithm
    # else:
    path, nodes = _get_path_cache_friendly(nodes, algorithm)
    if debug_level == 2:  # do nothing
        if output_edge_order:
            shape = [e.dimension for e in output_edge_order]
        else:
            shape = []
        return tn.Node(backend.zeros(shape))
    logger.info("the contraction path is given as %s" % str(path))
    if total_size is None:
        total_size = sum([_sizen(t) for t in nodes])
    for ab in path:
        if len(ab) < 2:
            logger.warning("single element tuple in contraction path!")
            continue
        a, b = ab

        if debug_level == 1:
            from .simplify import pseudo_contract_between

            new_node = pseudo_contract_between(nodes[a], nodes[b])
        else:
            new_node = tn.contract_between(nodes[a], nodes[b], allow_outer_product=True)
        nodes.append(new_node)
        # nodes[a] = backend.zeros([1])
        # nodes[b] = backend.zeros([1])
        nodes = _multi_remove(nodes, [a, b])

        logger.debug(_sizen(new_node, is_log=True))
        total_size += _sizen(new_node)
    logger.info("----- WRITE: %s --------\n" % np.log2(total_size))

    # if the final node has more than one edge,
    # output_edge_order has to be specified
    final_node = nodes[0]  # nodes were connected, we checked this
    if not ignore_edge_order:
        final_node.reorder_edges(output_edge_order)
    return final_node


def custom(
    nodes: List[Any],
    optimizer: Any,
    memory_limit: Optional[int] = None,
    output_edge_order: Optional[List[Any]] = None,
    ignore_edge_order: bool = False,
    **kws: Any,
) -> Any:
    if len(nodes) < 5:
        alg = opt_einsum.paths.optimal
        # not good at minimize WRITE actually...
        return _base(nodes, alg, output_edge_order, ignore_edge_order)

    total_size = None
    if kws.get("preprocessing", None):
        # nodes = _full_light_cone_cancel(nodes)
        nodes, total_size = _merge_single_gates(nodes)
    if not isinstance(optimizer, list):
        alg = partial(optimizer, memory_limit=memory_limit)
    else:
        alg = optimizer
    debug_level = kws.get("debug_level", 0)
    return _base(
        nodes,
        alg,
        output_edge_order,
        ignore_edge_order,
        total_size,
        debug_level=debug_level,
    )


def custom_stateful(
    nodes: List[Any],
    optimizer: Any,
    memory_limit: Optional[int] = None,
    opt_conf: Optional[Dict[str, Any]] = None,
    output_edge_order: Optional[List[Any]] = None,
    ignore_edge_order: bool = False,
    **kws: Any,
) -> Any:
    if len(nodes) < 5:
        alg = opt_einsum.paths.optimal
        # dynamic_programming has a potential bug for outer product
        # not good at minimize WRITE actually...
        return _base(nodes, alg, output_edge_order, ignore_edge_order)

    total_size = None
    if kws.get("preprocessing", None):
        nodes, total_size = _merge_single_gates(nodes)
    if opt_conf is None:
        opt_conf = {}
    opt = optimizer(**opt_conf)  # reinitiate the optimizer each time
    if kws.get("contraction_info", None):
        opt = contraction_info_decorator(opt)
    alg = partial(opt, memory_limit=memory_limit)
    debug_level = kws.get("debug_level", 0)

    return _base(
        nodes,
        alg,
        output_edge_order,
        ignore_edge_order,
        total_size,
        debug_level=debug_level,
    )


# only work for custom
def contraction_info_decorator(algorithm: Callable[..., Any]) -> Callable[..., Any]:
    from cotengra import ContractionTree

    def new_algorithm(
        input_sets: Dict[Any, int],
        output_set: Dict[Any, int],
        size_dict: Dict[Any, int],
        **kws: Any,
    ) -> Any:
        t0 = time.time()
        path = algorithm(input_sets, output_set, size_dict, **kws)
        path_finding_time = time.time() - t0
        tree = ContractionTree.from_path(input_sets, output_set, size_dict, path=path)
        print("------ contraction cost summary ------")
        print(
            "log10[FLOPs]: ",
            "%.3f" % np.log10(float(tree.total_flops())),
            " log2[SIZE]: ",
            "%.0f" % tree.contraction_width(),
            " log2[WRITE]: ",
            "%.3f" % np.log2(float(tree.total_write())),
            " PathFindingTime: ",
            "%.3f" % path_finding_time,
        )
        return path

    return new_algorithm


def set_contractor(
    method: Optional[str] = None,
    optimizer: Optional[Any] = None,
    memory_limit: Optional[int] = None,
    opt_conf: Optional[Dict[str, Any]] = None,
    set_global: bool = True,
    contraction_info: bool = False,
    debug_level: int = 0,
    **kws: Any,
) -> Callable[..., Any]:
    """
    To set runtime contractor of the tensornetwork for a better contraction path.
    For more information on the usage of contractor, please refer to independent tutorial.

    :param method: "auto", "greedy", "branch", "plain", "tng", "custom", "custom_stateful". defaults to None ("auto")
    :type method: Optional[str], optional
    :param optimizer: Valid for "custom" or "custom_stateful" as method, defaults to None
    :type optimizer: Optional[Any], optional
    :param memory_limit: It is not very useful, as ``memory_limit`` leads to ``branch`` contraction
        instead of ``greedy`` which is rather slow, defaults to None
    :type memory_limit: Optional[int], optional
    :raises Exception: Tensornetwork version is too low to support some of the contractors.
    :raises ValueError: Unknown method options.
    :return: The new tensornetwork with its contractor set.
    :rtype: tn.Node
    """
    if not method:
        method = "greedy"
        # auto for small size fallbacks to dp, which has bug for now
        # see: https://github.com/dgasmith/opt_einsum/issues/172
    if method.startswith("cotengra"):
        # cotengra shortcut
        import cotengra

        if method == "cotengra":
            method = "custom"
            optimizer = cotengra.ReusableHyperOptimizer(
                methods=["greedy", "kahypar"],
                parallel=True,
                minimize="combo",
                max_time=30,
                max_repeats=64,
                progbar=True,
            )
        else:  # "cotengra-30-64"
            _, mt, mr = method.split("-")
            method = "custom"
            optimizer = cotengra.ReusableHyperOptimizer(
                methods=["greedy", "kahypar"],
                parallel=True,
                minimize="combo",
                max_time=int(mt),
                max_repeats=int(mr),
                progbar=True,
            )

    if method == "plain":
        cf = plain_contractor
    elif method == "plain-experimental":
        cf = partial(experimental_contractor, local_steps=kws.get("local_steps", 2))
    elif method == "tng":  # don't use, deprecated, no guarantee
        if has_ps:
            cf = tn_greedy_contractor
        else:
            raise Exception(
                "current version of tensornetwork doesn't support tng contraction"
            )
    elif method == "custom_stateful":
        cf = custom_stateful  # type: ignore
        cf = partial(
            cf,
            optimizer=optimizer,
            opt_conf=opt_conf,
            contraction_info=contraction_info,
            debug_level=debug_level,
            **kws,
        )

    else:
        # cf = getattr(tn.contractors, method, None)
        # if not cf:
        # raise ValueError("Unknown contractor type: %s" % method)

        if method != "custom":
            optimizer = getattr(opt_einsum.paths, method)
        if contraction_info is True:
            optimizer = contraction_info_decorator(optimizer)  # type: ignore
        cf = partial(
            custom,
            optimizer=optimizer,
            memory_limit=memory_limit,
            debug_level=debug_level,
            **kws,
        )
    if set_global:
        for module in sys.modules:
            if module.startswith(package_name):
                setattr(sys.modules[module], "contractor", cf)
    return cf


set_contractor("greedy", preprocessing=True)

get_contractor = partial(set_contractor, set_global=False)


def set_function_contractor(*confargs: Any, **confkws: Any) -> Callable[..., Any]:
    """
    Function decorate to change function-level contractor

    :return: _description_
    :rtype: Callable[..., Any]
    """

    def wrapper(f: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(f)
        def newf(*args: Any, **kws: Any) -> Any:
            old_contractor = getattr(thismodule, "contractor")
            set_contractor(*confargs, **confkws)
            r = f(*args, **kws)
            for module in sys.modules:
                if module.startswith(package_name):
                    setattr(sys.modules[module], "contractor", old_contractor)
            return r

        return newf

    return wrapper


@contextmanager
def runtime_contractor(*confargs: Any, **confkws: Any) -> Iterator[Any]:
    """
    Context manager to change with-levek contractor

    :yield: _description_
    :rtype: Iterator[Any]
    """
    old_contractor = getattr(thismodule, "contractor")
    nc = set_contractor(*confargs, **confkws)
    yield nc
    for module in sys.modules:
        if module.startswith(package_name):
            setattr(sys.modules[module], "contractor", old_contractor)


def split_rules(
    max_singular_values: Optional[int] = None,
    max_truncation_err: Optional[float] = None,
    relative: bool = False,
) -> Any:
    """
    Obtain the direcionary of truncation rules

    :param max_singular_values: The maximum number of singular values to keep.
    :type max_singular_values: int, optional
    :param max_truncation_err: The maximum allowed truncation error.
    :type max_truncation_err: float, optional
    :param relative: Multiply `max_truncation_err` with the largest singular value.
    :type relative: bool, optional
    """
    rules: Any = {}
    if max_singular_values is not None:
        rules["max_singular_values"] = max_singular_values
    if max_truncation_err is not None:
        rules["max_truncattion_err"] = max_truncation_err
    if relative is not None:
        rules["relative"] = relative
    return rules
