"""
modules for graph instance data and more
"""

import itertools
import networkx as nx
import numpy as np
from typing import Any, Dict, Iterator, Sequence, Tuple

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
    for i, adj in g.adj.items():
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
    """
    cut by given values $$\pm 1$$ on each vertex as a list

    :param g:
    :param values:
    :return:
    """
    cost = 0
    for e in g.edges:
        cost += g[e[0]][e[1]]["weight"] / 2 * (1 - values[e[0]] * values[e[1]])
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
    return np.mean(r), np.std(r) / np.sqrt(len(r))
