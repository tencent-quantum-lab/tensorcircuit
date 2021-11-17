"""
tensornetwork simplification
"""
# part of the implementations and ideas are inspired from
# https://github.com/jcmgray/quimb/blob/a2968050eba5a8a04ced4bdaa5e43c4fb89edc33/quimb/tensor/tensor_core.py#L7309-L8293
# (Apache 2.0)
# We here more focus on tensornetwork derived from circuit simulation
# and consider less on general tensornetwork topology.
# Note we have no direct hyperedge support in tensornetwork package

from typing import Tuple, List, Any

import numpy as np
import tensornetwork as tn

from .cons import _multi_remove


def infer_new_size(a: tn.Node, b: tn.Node, include_old: bool = True) -> Any:
    shared_edges = tn.get_shared_edges(a, b)
    a_dim = np.prod([e.dimension for e in a])
    b_dim = np.prod([e.dimension for e in b])
    new_dim = np.prod([e.dimension for e in a if e not in shared_edges]) * np.prod(
        [e.dimension for e in b if e not in shared_edges]
    )
    if include_old is True:
        return new_dim, a_dim, b_dim
    return new_dim


def infer_new_shape(a: tn.Node, b: tn.Node, include_old: bool = True) -> Any:
    # return: Union[Tuple[int, ...], Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]
    shared_edges = tn.get_shared_edges(a, b)
    a_shape = tuple(sorted([e.dimension for e in a]))
    b_shape = tuple(sorted([e.dimension for e in b]))
    new_shape = tuple(
        sorted(
            ([e.dimension for e in a if e not in shared_edges])
            + ([e.dimension for e in b if e not in shared_edges])
        )
    )
    if include_old is True:
        return new_shape, a_shape, b_shape
    return new_shape


def _rank_simplify(nodes: List[Any]) -> Tuple[List[Any], bool]:
    # if total_size is None:
    # total_size = sum([_sizen(t) for t in nodes])
    is_changed = False
    l = len(nodes)
    for i in range(l):
        if i < len(nodes):
            n = nodes[i]
            for e in n:
                if not e.is_dangling():
                    nd, ad, bd = infer_new_shape(e.node1, e.node2)
                    if nd <= ad or nd <= bd:
                        njs = [
                            i
                            for i, n in enumerate(nodes)
                            if id(n) in [id(e.node1), id(e.node2)]
                        ]
                        new_node = tn.contract_between(e.node1, e.node2)
                        # contract(e) is not enough for multi edges between two tensors
                        nodes[njs[0]] = new_node
                        nodes = _multi_remove(nodes, [njs[1]])
                        is_changed = True
                        break  # switch to the next node
        else:
            break
    return nodes, is_changed


def _full_rank_simplify(nodes: List[Any]) -> List[Any]:
    nodes, is_changed = _rank_simplify(nodes)
    while is_changed:
        nodes, is_changed = _rank_simplify(nodes)
    return nodes
