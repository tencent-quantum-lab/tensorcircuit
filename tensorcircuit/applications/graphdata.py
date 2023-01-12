"""
Modules for graph instance data and more
"""

import itertools
from functools import partial
from typing import Any, Dict, Iterator, Sequence, Tuple, Optional

import networkx as nx
import numpy as np

Graph = Any

graph_instances = {
    "16A": {
        3: {15: {"weight": 1.0}, 11: {"weight": 1.0}, 0: {"weight": 1.0}},
        15: {3: {"weight": 1.0}, 1: {"weight": 1.0}, 0: {"weight": 1.0}},
        0: {14: {"weight": 1.0}, 3: {"weight": 1.0}, 15: {"weight": 1.0}},
        14: {0: {"weight": 1.0}, 4: {"weight": 1.0}, 5: {"weight": 1.0}},
        2: {8: {"weight": 1.0}, 6: {"weight": 1.0}, 7: {"weight": 1.0}},
        8: {2: {"weight": 1.0}, 6: {"weight": 1.0}, 7: {"weight": 1.0}},
        11: {3: {"weight": 1.0}, 9: {"weight": 1.0}, 5: {"weight": 1.0}},
        5: {13: {"weight": 1.0}, 11: {"weight": 1.0}, 14: {"weight": 1.0}},
        13: {5: {"weight": 1.0}, 10: {"weight": 1.0}, 9: {"weight": 1.0}},
        1: {15: {"weight": 1.0}, 10: {"weight": 1.0}, 4: {"weight": 1.0}},
        4: {12: {"weight": 1.0}, 14: {"weight": 1.0}, 1: {"weight": 1.0}},
        12: {4: {"weight": 1.0}, 10: {"weight": 1.0}, 9: {"weight": 1.0}},
        6: {7: {"weight": 1.0}, 2: {"weight": 1.0}, 8: {"weight": 1.0}},
        7: {6: {"weight": 1.0}, 2: {"weight": 1.0}, 8: {"weight": 1.0}},
        10: {12: {"weight": 1.0}, 1: {"weight": 1.0}, 13: {"weight": 1.0}},
        9: {11: {"weight": 1.0}, 13: {"weight": 1.0}, 12: {"weight": 1.0}},
    },
    "10A": {
        1: {2: {"weight": 1.0}, 6: {"weight": 1.0}, 7: {"weight": 1.0}},
        2: {1: {"weight": 1.0}, 6: {"weight": 1.0}, 0: {"weight": 1.0}},
        5: {9: {"weight": 1.0}, 4: {"weight": 1.0}, 0: {"weight": 1.0}},
        9: {5: {"weight": 1.0}, 4: {"weight": 1.0}, 3: {"weight": 1.0}},
        6: {2: {"weight": 1.0}, 8: {"weight": 1.0}, 1: {"weight": 1.0}},
        4: {9: {"weight": 1.0}, 5: {"weight": 1.0}, 0: {"weight": 1.0}},
        8: {6: {"weight": 1.0}, 3: {"weight": 1.0}, 7: {"weight": 1.0}},
        3: {8: {"weight": 1.0}, 9: {"weight": 1.0}, 7: {"weight": 1.0}},
        0: {4: {"weight": 1.0}, 5: {"weight": 1.0}, 2: {"weight": 1.0}},
        7: {1: {"weight": 1.0}, 3: {"weight": 1.0}, 8: {"weight": 1.0}},
    },
    "8A": {
        0: {
            1: {"weight": 1.0},
            4: {"weight": 1.0},
            5: {"weight": 1.0},
            7: {"weight": 1.0},
            2: {"weight": 1.0},
        },
        1: {0: {"weight": 1.0}, 2: {"weight": 1.0}, 5: {"weight": 1.0}},
        2: {
            1: {"weight": 1.0},
            3: {"weight": 1.0},
            6: {"weight": 1.0},
            0: {"weight": 1.0},
        },
        3: {2: {"weight": 1.0}, 7: {"weight": 1.0}},
        4: {0: {"weight": 1.0}, 5: {"weight": 1.0}},
        5: {
            1: {"weight": 1.0},
            4: {"weight": 1.0},
            6: {"weight": 1.0},
            0: {"weight": 1.0},
        },
        6: {2: {"weight": 1.0}, 5: {"weight": 1.0}, 7: {"weight": 1.0}},
        7: {3: {"weight": 1.0}, 6: {"weight": 1.0}, 0: {"weight": 1.0}},
    },
    "6A": {
        0: {4: {"weight": 1.0}, 5: {"weight": 1.0}},
        1: {4: {"weight": 1.0}, 5: {"weight": 1.0}},
        2: {5: {"weight": 1.0}},
        3: {4: {"weight": 1.0}},
        4: {
            0: {"weight": 1.0},
            1: {"weight": 1.0},
            3: {"weight": 1.0},
            5: {"weight": 1.0},
        },
        5: {
            0: {"weight": 1.0},
            1: {"weight": 1.0},
            2: {"weight": 1.0},
            4: {"weight": 1.0},
        },
    },
    "8B": {
        0: {1: {"weight": 1.0}, 7: {"weight": 1.0}, 3: {"weight": 1.0}},
        1: {0: {"weight": 1.0}, 2: {"weight": 1.0}, 3: {"weight": 1.0}},
        2: {1: {"weight": 1.0}, 3: {"weight": 1.0}, 5: {"weight": 1.0}},
        4: {7: {"weight": 1.0}, 6: {"weight": 1.0}, 5: {"weight": 1.0}},
        7: {4: {"weight": 1.0}, 6: {"weight": 1.0}, 0: {"weight": 1.0}},
        3: {1: {"weight": 1.0}, 2: {"weight": 1.0}, 0: {"weight": 1.0}},
        6: {7: {"weight": 1.0}, 4: {"weight": 1.0}, 5: {"weight": 1.0}},
        5: {6: {"weight": 1.0}, 4: {"weight": 1.0}, 2: {"weight": 1.0}},
    },
    "8BW": {
        0: {
            1: {"weight": -0.08015762930148482},
            7: {"weight": 0.5760374347070247},
            3: {"weight": 2.130555257631913},
        },
        1: {
            0: {"weight": -0.08015762930148482},
            2: {"weight": -1.3460396434739554},
            3: {"weight": 0.44190622039475597},
        },
        2: {
            1: {"weight": -1.3460396434739554},
            3: {"weight": -0.9906762424422002},
            5: {"weight": 0.00023144222616794873},
        },
        4: {
            7: {"weight": -0.08800323266198477},
            6: {"weight": -0.4554192225796251},
            5: {"weight": -0.8386283543725825},
        },
        7: {
            0: {"weight": 0.5760374347070247},
            4: {"weight": -0.08800323266198477},
            6: {"weight": 1.7363683549872604},
        },
        3: {
            0: {"weight": 2.130555257631913},
            1: {"weight": 0.44190622039475597},
            2: {"weight": -0.9906762424422002},
        },
        6: {
            4: {"weight": -0.4554192225796251},
            7: {"weight": 1.7363683549872604},
            5: {"weight": -1.232667082395665},
        },
        5: {
            2: {"weight": 0.00023144222616794873},
            4: {"weight": -0.8386283543725825},
            6: {"weight": -1.232667082395665},
        },
    },
    "8BP": {
        0: {
            1: {"weight": 1.194287393754959},
            7: {"weight": 1.0245561910654257},
            3: {"weight": 0.7068030998131771},
        },
        1: {
            0: {"weight": 1.194287393754959},
            2: {"weight": 0.6799813983516487},
            3: {"weight": 1.0241436821477117},
        },
        2: {
            1: {"weight": 0.6799813983516487},
            3: {"weight": 0.8195552276675298},
            5: {"weight": 1.1544896529892292},
        },
        4: {
            7: {"weight": 1.35187907862147},
            6: {"weight": 1.0893647845617211},
            5: {"weight": 0.8339167938538325},
        },
        7: {
            0: {"weight": 1.0245561910654257},
            4: {"weight": 1.35187907862147},
            6: {"weight": 1.0711550783285257},
        },
        3: {
            0: {"weight": 0.7068030998131771},
            1: {"weight": 1.0241436821477117},
            2: {"weight": 0.8195552276675298},
        },
        6: {
            4: {"weight": 1.0893647845617211},
            7: {"weight": 1.0711550783285257},
            5: {"weight": 1.3692516705972504},
        },
        5: {
            2: {"weight": 1.1544896529892292},
            4: {"weight": 0.8339167938538325},
            6: {"weight": 1.3692516705972504},
        },
    },
    "8C": {
        0: {
            1: {"weight": 1.0},
            3: {"weight": 1.0},
            4: {"weight": 1.0},
            6: {"weight": 1.0},
            7: {"weight": 1.0},
        },
        1: {
            0: {"weight": 1.0},
            2: {"weight": 1.0},
            3: {"weight": 1.0},
            4: {"weight": 1.0},
            5: {"weight": 1.0},
        },
        2: {
            1: {"weight": 1.0},
            3: {"weight": 1.0},
            4: {"weight": 1.0},
            5: {"weight": 1.0},
            6: {"weight": 1.0},
        },
        3: {
            0: {"weight": 1.0},
            1: {"weight": 1.0},
            2: {"weight": 1.0},
            7: {"weight": 1.0},
        },
        4: {
            0: {"weight": 1.0},
            1: {"weight": 1.0},
            2: {"weight": 1.0},
            5: {"weight": 1.0},
            6: {"weight": 1.0},
        },
        5: {
            1: {"weight": 1.0},
            2: {"weight": 1.0},
            4: {"weight": 1.0},
            7: {"weight": 1.0},
        },
        6: {0: {"weight": 1.0}, 2: {"weight": 1.0}, 4: {"weight": 1.0}},
        7: {0: {"weight": 1.0}, 3: {"weight": 1.0}, 5: {"weight": 1.0}},
    },
    "3C": {
        0: {1: {"weight": 1.0}, 2: {"weight": 1.0}},
        1: {0: {"weight": 1.0}, 2: {"weight": 2.0}},
        2: {0: {"weight": 1.0}, 1: {"weight": 2.0}},
    },
}


def dict2graph(d: Dict[Any, Any]) -> Graph:
    """
    ```python
    d = nx.to_dict_of_dicts(g)
    ```

    :param d:
    :return:
    """
    g = nx.to_networkx_graph(d)
    for e in g.edges:
        if not g[e[0]][e[1]].get("weight"):
            g[e[0]][e[1]]["weight"] = 1.0
    return g


def get_graph(c: str) -> Graph:
    graph_recipe = graph_instances.get(c, "3C")
    return dict2graph(graph_recipe)  # type: ignore


def _generate_random_graph(x: int, p: float = 0.3, weights: bool = False) -> Graph:
    g = nx.erdos_renyi_graph(x, p=p)
    for e in g.edges:
        g[e[0]][e[1]]["weight"] = np.random.uniform() if weights else 1.0
    return g


def all_nodes_covered(g: Graph) -> bool:
    for _, adj in g.adj.items():
        if len(set(adj)) == 0:
            return False
    return True


def erdos_graph_generator(*args, **kwargs) -> Iterator[Graph]:  # type: ignore
    while True:
        g = _generate_random_graph(*args, **kwargs)
        while len(g.edges) < 3:  # ensure there are at least some edges
            # not there is no need to call all_nodes_covered(g), since measurements qubits
            # coming with the edges, so separate disconnected nodes doesn't PQC making
            # namely either ops on all qubits, or ops on edge qubits
            # but when NNN is included as shown below, nnlayers are not necessarily share
            # the full overlap with measurement qubits which may lead to failuer of PQC building
            g = _generate_random_graph(*args, **kwargs)
        yield g


def regular_graph_generator(d: int, n: int, weights: bool = False) -> Iterator[Graph]:
    while True:
        g = nx.random_regular_graph(d, n)
        # no need to check edge number since it is fixed
        for e in g.edges:
            g[e[0]][e[1]]["weight"] = np.random.uniform() if weights else 1.0
        yield g


def _maxcut(g: Graph, values: Sequence[int]) -> float:
    r"""
    To get the max-cut for the graph g by given values $$\pm 1$$ on each vertex as a list.

    :param g: The graph
    :type g: Graph
    :param values: The list of values to be cut on each vertex
    :type values: Sequence[int]
    :return: The resulted graph
    :rtype: g
    """
    cost = 0
    for e in g.edges:
        cost += g[e[0]][e[1]].get("weight", 1.0) / 2 * (1 - values[e[0]] * values[e[1]])
    return cost


def maxcut_solution_bruteforce(g: Graph) -> Tuple[float, Sequence[int]]:
    l = len(g.nodes)
    result = _maxcut(g, [1 for _ in range(l)])
    result_value = [1 for _ in range(l)]
    for v in itertools.product(*[[1, -1] for _ in range(l)]):
        nr = _maxcut(g, v)
        if result < nr:
            result = nr
            result_value = v  # type: ignore
    return result, result_value


def ensemble_maxcut_solution(g: Graph, samples: int = 100) -> Tuple[float, float]:
    r = []
    for _ in range(samples):
        r.append(maxcut_solution_bruteforce(g.send(None))[0])
    return np.mean(r), np.std(r) / np.sqrt(len(r))  # type: ignore


def reduce_edges(g: Graph, m: int = 1) -> Sequence[Graph]:
    """

    :param g:
    :param m:
    :return: all graphs with m edge out from g
    """
    n = len(g.nodes)
    e = len(g.edges)
    el = list(g.edges)
    glist = []
    for missing_edges in itertools.product(*[[i for i in range(e)] for _ in range(m)]):
        missing_set = set(list(missing_edges))
        if len(missing_set) == m:  # no replication
            ng = nx.Graph()
            for i in range(n):
                ng.add_node(i)
            for i, edge in enumerate(el):
                if i not in missing_set:
                    ng.add_edge(*edge, weight=g[edge[0]][edge[1]]["weight"])
            ng.__repr__ = str(missing_set)
            glist.append(ng)

    return glist


def reduced_ansatz(g: Graph, ratio: Optional[int] = None) -> Graph:
    """
    Generate a reduced graph with given ratio of edges compared to the original graph g.

    :param g: The base graph
    :type g: Graph
    :param ratio: number of edges kept, default half of the edges
    :return: The resulted reduced graph
    :rtype: Graph
    """
    nn = len(g.nodes)
    ne = len(g.edges)
    if ratio is None:
        ratio = int(ne / 2)
    edges = np.array(g.edges)[np.random.choice(len(g.edges), size=ratio, replace=False)]
    ng = nx.Graph()
    for i in range(nn):
        ng.add_node(i)
    for j, k in edges:
        ng.add_edge(j, k, weight=g[j][k].get("weight", 1))
    return ng


def split_ansatz(g: Graph, split: int = 2) -> Sequence[Graph]:
    """
    Split the graph in exactly ``split`` piece evenly.

    :param g: The mother graph
    :type g: Graph
    :param split: The number of the graph we want to divide into, defaults to 2
    :type split: int, optional
    :return: List of graph instance of size ``split``
    :rtype: Sequence[Graph]
    """
    edges = np.array(g.edges)
    ne = len(edges)
    np.random.shuffle(edges)
    gs = [nx.Graph() for _ in range(split)]
    for i in range(split):
        for j, k in edges[int(i * ne / split) : int((i + 1) * ne / split)]:
            gs[i].add_edge(j, k, weight=g[j][k].get("weight", 1))
    return gs


def graph1D(n: int, pbc: bool = True) -> Graph:
    """
    1D PBC chain with n sites.

    :param n: The number of nodes
    :type n: int
    :return: The resulted graph g
    :rtype: Graph
    """

    g = nx.Graph()
    for i in range(n):
        g.add_node(i)
    for i in range(n - 1):
        g.add_edge(i, i + 1, weight=1.0)
    if pbc is True:
        g.add_edge(n - 1, 0, weight=1.0)
    return g


def even1D(n: int, s: int = 0) -> Graph:
    g = nx.Graph()
    for i in range(n):
        g.add_node(i)
    for i in range(s, n, 2):
        g.add_edge(i, (i + 1) % n, weight=1.0)
    return g


odd1D = partial(even1D, s=1)


def Grid2D(m: int, n: int, pbc: bool = True) -> Graph:
    def one2two(i: int) -> Tuple[int, int]:
        x = i // n
        y = i % n
        return x, y

    def two2one(x: int, y: int) -> int:
        return x * n + y

    g = nx.Graph()
    for i in range(m * n):
        g.add_node(i)
    for i in range(m * n):
        x, y = one2two(i)
        if pbc is False and x - 1 < 0:
            pass
        else:
            g.add_edge(i, two2one((x - 1) % m, y), weight=1)
        if pbc is False and y - 1 < 0:
            pass
        else:
            g.add_edge(i, two2one(x, (y - 1) % n), weight=1)
    return g


def Triangle2D(m: int, n: int) -> Graph:
    def one2two(i: int) -> Tuple[int, int]:
        y = i // m
        x = i % m
        return x, y

    def two2one(x: int, y: int) -> int:
        return x + y * m

    g = nx.Graph()
    for i in range(m * n):
        g.add_node(i)
    for i in range(m * n):
        x, y = one2two(i)
        g.add_edge(i, two2one((x + 1) % m, y), weight=1)
        g.add_edge(i, two2one(x, (y + 1) % n), weight=1)
        g.add_edge(i, two2one((x + 1) % m, (y - 1) % n), weight=1)
    return g


def dress_graph_with_cirq_qubit(g: Graph) -> Graph:
    # deprecated cirq related utilities
    import cirq

    for i in range(len(g.nodes)):
        g.nodes[i]["qubit"] = cirq.GridQubit(i, 0)
    return g
