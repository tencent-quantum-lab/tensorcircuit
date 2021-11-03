"""
some constants and setups
"""

import sys
from functools import partial
from typing import Optional, Any, List

import numpy as np
import tensornetwork as tn
from tensornetwork.backend_contextmanager import get_default_backend

from .backends import get_backend


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

    npdtype = getattr(np, dtype)
    for module in modules:
        if module in sys.modules:
            setattr(sys.modules[module], "dtypestr", dtype)
            setattr(sys.modules[module], "rdtypestr", rdtype)
            setattr(sys.modules[module], "npdtype", npdtype)
    from .gates import meta_gate

    meta_gate()


set_dtype()

## here below comes other contractors (just works, but correctness has not been tested)


def _multi_remove(elems: List[Any], indices: List[int]) -> List[Any]:
    """Remove multiple indicies in a list at once."""
    return [i for j, i in enumerate(elems) if j not in indices]


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
    nodes = list(reversed(list(nodes)))
    while len(nodes) > 1:
        new_node = tn.contract_between(nodes[-1], nodes[-2], allow_outer_product=True)
        nodes = _multi_remove(nodes, [len(nodes) - 2, len(nodes) - 1])
        nodes.append(new_node)
    # if the final node has more than one edge,
    # output_edge_order has to be specified
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


base = tn.contractors.opt_einsum_paths.path_contractors.base

# designed for cotengra.HyperOptimizer
# usage ``opt = ctg.HyperOptimizer``
# ``tc.set_contractor("custom_stateful", optimizer=opt, parallel=False)``


def custom_stateful(
    nodes: List[Any],
    optimizer: Any,
    output_edge_order: Optional[List[Any]] = None,
    ignore_edge_order: bool = False,
    **kws: Any
) -> Any:
    opt = optimizer(**kws)
    alg = opt
    # alg = partial(tn.contractors.custom, optimizer)
    return base(nodes, alg, output_edge_order, ignore_edge_order)


def set_contractor(
    method: Optional[str] = None,
    optimizer: Optional[Any] = None,
    memory_limit: Optional[int] = None,
    **kws: Any
) -> None:
    """
    set runtim contractor of the tensornetwork for a better contraction path

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
    elif method == "tng":
        if has_ps:
            cf = tn_greedy_contractor
        else:
            raise Exception(
                "current version of tensornetwork doesn't support tng contraction"
            )
    elif method == "custom_stateful":
        cf = custom_stateful  # type: ignore
        cf = partial(cf, optimizer=optimizer, **kws)
    else:
        cf = getattr(tn.contractors, method, None)
        if not cf:
            raise ValueError("Unknown contractor type: %s" % method)
        if method == "custom":
            cf = partial(cf, optimizer=optimizer)
        cf = partial(cf, memory_limit=memory_limit)
    for module in modules:
        if module in sys.modules:
            setattr(sys.modules[module], "contractor", cf)


set_contractor()
