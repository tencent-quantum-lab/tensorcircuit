"""
some constants and setups
"""
import logging
import sys
from functools import partial, reduce
from operator import mul
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import opt_einsum
import tensornetwork as tn
from tensornetwork.backend_contextmanager import get_default_backend

from .backends import get_backend

logger = logging.getLogger(__name__)


modules = [
    "tensorcircuit",
    "tensorcircuit.cons",
    "tensorcircuit.gates",
    "tensorcircuit.circuit",
    "tensorcircuit.backends",
    "tensorcircuit.densitymatrix",
    "tensorcircuit.densitymatrix2",
    "tensorcircuit.channels",
    "tensorcircuit.keras",
    "tensorcircuit.quantum",
    "tensorcircuit.simplify",
]

dtypestr = "complex64"
npdtype = np.complex64
backend = get_backend("numpy")
contractor = tn.contractors.auto
# these above lines are just for mypy, it is not very good at evaluating runtime object


def set_tensornetwork_backend(backend: Optional[str] = None) -> None:
    """
    set the runtime backend of tensorcircuit

    :param backend: "numpy", "tensorflow", "jax", "pytorch". defaults to None,
        which gives the same behavior as ``tensornetwork.backend_contextmanager.get_default_backend()``
    :type backend: Optional[str], optional
    """
    if not backend:
        backend = get_default_backend()
    backend_obj = get_backend(backend)
    for module in modules:
        if module in sys.modules:
            setattr(sys.modules[module], "backend", backend_obj)
    tn.set_default_backend(backend)


set_backend = set_tensornetwork_backend


set_tensornetwork_backend()


def set_dtype(dtype: Optional[str] = None) -> None:
    """
    set the runtime numerical dtype of tensors

    :param dtype: "complex64" or "complex128", defaults to None, which is equivalent to "complex64"
    :type dtype: Optional[str], optional
    """
    if not dtype:
        dtype = "complex64"
    if dtype == "complex64":
        rdtype = "float32"
    else:
        rdtype = "float64"
    if backend.name == "jax" and dtype == "complex128":
        from jax.config import config  # type: ignore

        config.update("jax_enable_x64", True)

    npdtype = getattr(np, dtype)
    for module in modules:
        if module in sys.modules:
            setattr(sys.modules[module], "dtypestr", dtype)
            setattr(sys.modules[module], "rdtypestr", rdtype)
            setattr(sys.modules[module], "npdtype", npdtype)
    from .gates import meta_gate

    meta_gate()


set_dtype()

## here below comes other contractors (just works,
## but correctness has not been extensively tested for some of them)


def _multi_remove(elems: List[Any], indices: List[int]) -> List[Any]:
    """Remove multiple indicies in a list at once."""
    return [i for j, i in enumerate(elems) if j not in indices]


def _sizen(node: tn.Node, is_log: bool = False) -> int:
    s = reduce(mul, node.tensor.shape + (1,))
    if is_log:
        return int(np.log2(s))
    return s  # type: ignore


def _merge_single_gates(
    nodes: List[Any], total_size: Optional[int] = None
) -> Tuple[List[Any], int]:
    if total_size is None:
        total_size = sum([_sizen(t) for t in nodes])
    queue = [n for n in nodes if len(n.tensor.shape) <= 2]
    while queue:
        n0 = queue[0]
        if n0[0].is_dangling():
            e0 = n0[1]
        else:
            e0 = n0[0]
        njs = [i for i, n in enumerate(nodes) if id(n) in [id(e0.node1), id(e0.node2)]]
        qjs = [i for i, n in enumerate(queue) if id(n) in [id(e0.node1), id(e0.node2)]]

        new_node = tn.contract(e0)
        total_size += _sizen(new_node)  # type: ignore

        logger.debug(
            _sizen(new_node, is_log=True),
        )
        queue = _multi_remove(queue, qjs)
        nodes[njs[1]] = new_node
        nodes = _multi_remove(nodes, [njs[0]])
        if len(new_node.tensor.shape) <= 2:
            # queue.append(new_node)
            queue.insert(0, new_node)
    return nodes, total_size  # type: ignore


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
    naive statevector simulator contraction path

    :param nodes: list of ``tn.Node``
    :type nodes: List[Any]
    :param output_edge_order: list of dangling node edges, defaults to None
    :type output_edge_order: Optional[List[Any]], optional
    :return: ``tn.Node`` after contraction
    :rtype: tn.Node
    """
    total_size = sum([_sizen(t) for t in nodes])
    # nodes = list(reversed(list(nodes)))
    nodes = list(nodes)
    nodes = list(reversed(nodes))

    while len(nodes) > 1:
        new_node = tn.contract_between(nodes[-1], nodes[-2], allow_outer_product=True)
        nodes = _multi_remove(nodes, [len(nodes) - 2, len(nodes) - 1])
        nodes.append(new_node)
        logger.info(_sizen(new_node, is_log=True))
        total_size += _sizen(new_node)
    logger.info("----- WRITE: %s --------\n" % np.log2(total_size))

    final_node = nodes[0]
    if output_edge_order is not None:
        final_node.reorder_edges(output_edge_order)
    return final_node


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

    return algorithm(input_sets, output_set, size_dict), nodes  # type: ignore


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
    input_sets = [set([mapping_dict[id(e)] for e in node.edges]) for node in nodes]
    order = np.argsort(list(map(sorted, input_sets)) + [[1e10]])[:-1]  # type: ignore
    # TODO(@refraction-ray): more stable and unwarning arg sorting here
    nodes_new = [nodes[i] for i in order]
    input_sets = [set([mapping_dict[id(e)] for e in node.edges]) for node in nodes_new]
    output_set = set([mapping_dict[id(e)] for e in tn.get_subgraph_dangling(nodes_new)])
    size_dict = {
        mapping_dict[id(edge)]: edge.dimension for edge in tn.get_all_edges(nodes_new)
    }
    logger.debug("input_sets: %s" % input_sets)
    logger.debug("output_set: %s" % output_set)
    logger.debug("path finder algorithm: %s" % algorithm)
    return algorithm(input_sets, output_set, size_dict), nodes_new  # type: ignore


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
"""


def _base(
    nodes: List[tn.Node],
    algorithm: Any,
    output_edge_order: Optional[Sequence[tn.Edge]] = None,
    ignore_edge_order: bool = False,
    total_size: Optional[int] = None,
) -> tn.Node:
    """Base method for all `opt_einsum` contractors.

    Args:
      nodes: A collection of connected nodes.
      algorithm: `opt_einsum` contraction method to use.
      output_edge_order: An optional list of edges. Edges of the
        final node in `nodes_set`
        are reordered into `output_edge_order`;
        if final node has more than one edge,
        `output_edge_order` must be provided.
      ignore_edge_order: An option to ignore the output edge
        order.

    Returns:
      Final node after full contraction.
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
    if isinstance(algorithm, list):
        path = algorithm
    else:
        path, nodes = _get_path_cache_friendly(nodes, algorithm)
    if total_size is None:
        total_size = sum([_sizen(t) for t in nodes])
    for ab in path:
        if len(ab) < 2:
            logger.warning("single element tuple in contraction path!")
            continue
        a, b = ab
        new_node = tn.contract_between(nodes[a], nodes[b], allow_outer_product=True)
        nodes.append(new_node)
        # nodes[a] = backend.zeros([1])
        # nodes[b] = backend.zeros([1])
        nodes = _multi_remove(nodes, [a, b])

        logger.debug(_sizen(new_node, is_log=True))
        total_size += _sizen(new_node)  # type: ignore
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
    **kws: Any
) -> Any:
    if len(nodes) < 5:
        alg = opt_einsum.paths.optimal
        # not good at minimize WRITE actually...
        return _base(nodes, alg, output_edge_order, ignore_edge_order)

    total_size = None
    if kws.get("preprocessing", None):
        nodes, total_size = _merge_single_gates(nodes)
    alg = partial(optimizer, memory_limit=memory_limit)
    return _base(nodes, alg, output_edge_order, ignore_edge_order, total_size)


def custom_stateful(
    nodes: List[Any],
    optimizer: Any,
    memory_limit: Optional[int] = None,
    opt_conf: Optional[Dict[str, Any]] = None,
    output_edge_order: Optional[List[Any]] = None,
    ignore_edge_order: bool = False,
    **kws: Any
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
    alg = partial(opt, memory_limit=memory_limit)
    return _base(nodes, alg, output_edge_order, ignore_edge_order, total_size)


def set_contractor(
    method: Optional[str] = None,
    optimizer: Optional[Any] = None,
    memory_limit: Optional[int] = None,
    opt_conf: Optional[Dict[str, Any]] = None,
    set_global: bool = True,
    **kws: Any
) -> Callable[..., Any]:
    """
    set runtime contractor of the tensornetwork for a better contraction path

    :param method: "auto", "greedy", "branch", "plain", "tng", "custom", "custom_stateful". defaults to None ("auto")
    :type method: Optional[str], optional
    :param optimizer: valid for "custom" or "custom_stateful" as method, defaults to None
    :type optimizer: Optional[Any], optional
    :param memory_limit: not very useful, as ``memory_limit`` leads to ``branch`` contraction
        instead of ``greedy`` which is rather slow, defaults to None
    :type memory_limit: Optional[int], optional
    :raises Exception: tensornetwork version is too low to support some of the contractors
    :raises ValueError: unknown method options
    """
    if not method:
        method = "greedy"
        # auto for small size fallbacks to dp, which has bug for now
        # see: https://github.com/dgasmith/opt_einsum/issues/172
    if method == "plain":
        cf = plain_contractor
    elif method == "plain-experimental":
        cf = partial(experimental_contractor, local_steps=kws.get("local_steps", 2))
    elif method == "tng":
        if has_ps:
            cf = tn_greedy_contractor
        else:
            raise Exception(
                "current version of tensornetwork doesn't support tng contraction"
            )
    elif method == "custom_stateful":
        cf = custom_stateful  # type: ignore
        cf = partial(cf, optimizer=optimizer, opt_conf=opt_conf, **kws)

    else:
        # cf = getattr(tn.contractors, method, None)
        # if not cf:
        # raise ValueError("Unknown contractor type: %s" % method)
        if method == "custom":
            cf = partial(custom, optimizer=optimizer, **kws)
        else:
            cf = partial(custom, optimizer=getattr(opt_einsum.paths, method))
        cf = partial(cf, memory_limit=memory_limit)
    if set_global:
        for module in modules:
            if module in sys.modules:
                setattr(sys.modules[module], "contractor", cf)
    return cf


set_contractor()

get_contractor = partial(set_contractor, set_global=False)

# TODO(@refraction-ray): contractor at Circuit and instruction level setup
