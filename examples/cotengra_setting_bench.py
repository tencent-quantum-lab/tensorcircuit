"""
Optimization for performance comparison with different cotengra settings.
"""

import itertools
import sys
import warnings

import cotengra as ctg
import networkx as nx
import numpy as np

sys.path.insert(0, "../")
import tensorcircuit as tc

try:
    import kahypar
except ImportError:
    print("kahypar not installed, please install it to run this script.")
    exit()


# suppress the warning from cotengra
warnings.filterwarnings(
    "ignore",
    message="The inputs or output of this tree are not ordered."
    "Costs will be accurate but actually contracting requires "
    "ordered indices corresponding to array axes.",
)

K = tc.set_backend("jax")


def generate_circuit(param, g, n, nlayers):
    # construct the circuit ansatz
    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)
    for j in range(nlayers):
        c = tc.templates.blocks.QAOA_block(c, g, param[j, 0], param[j, 1])
    return c


def trigger_cotengra_optimization(n, nlayers, graph):

    # define the loss function
    def loss_f(params, n, nlayers):

        c = generate_circuit(params, graph, n, nlayers)

        loss = c.expectation_ps(z=[0, 1, 2], reuse=False)

        return K.real(loss)

    params = K.implicit_randn(shape=[nlayers, 2])

    # run only once to trigger the compilation
    K.jit(
        loss_f,
        static_argnums=(1, 2),
    )(params, n, nlayers)


# define the benchmark parameters
n = 12
nlayers = 12

# define the cotengra optimizer parameters
graph_args = {
    "1D lattice": nx.convert_node_labels_to_integers(
        nx.grid_graph((n, 1))
    ),  # 1D lattice
    "2D lattice": nx.convert_node_labels_to_integers(
        nx.grid_graph((n // 5, n // (n // 5)))
    ),  # 2D lattice
    "all-to-all connected": nx.convert_node_labels_to_integers(
        nx.complete_graph(n)
    ),  # all-to-all connected
}

methods_args = [  # https://cotengra.readthedocs.io/en/latest/advanced.html#drivers
    "greedy",
    "kahypar",
    # "labels",
    # "spinglass",  # requires igraph
    # "labelprop",  # requires igraph
    # "betweenness",  # requires igraph
    # "walktrap",  # requires igraph
    # "quickbb",  # requires https://github.com/dechterlab/quickbb
    # "flowcutter",  # requires https://github.com/kit-algo/flow-cutter-pace17
]

optlib_args = [  # https://cotengra.readthedocs.io/en/latest/advanced.html#optimization-library
    "optuna",  # pip install optuna
    # "random",  # default when no library is installed
    # "baytune",  # pip install baytune
    # "nevergrad",  # pip install nevergrad
    # "chocolate",  # pip install git+https://github.com/AIworx-Labs/chocolate@master
    # "skopt",  # pip install scikit-optimize
]

post_processing_args = [  # https://cotengra.readthedocs.io/en/latest/advanced.html#slicing-and-subtree-reconfiguration
    (None, None),
    # ("slicing_opts", {"target_size": 2**28}),
    # ("slicing_reconf_opts", {"target_size": 2**28}),
    ("reconf_opts", {}),
    ("simulated_annealing_opts", {}),
]

minimize_args = [  # https://cotengra.readthedocs.io/en/main/advanced.html#objective
    # "flops",  # minimize the total number of scalar operations
    # "size",  # minimize the size of the largest intermediate tensor
    # "write",  # minimize the sum of sizes of all intermediate tensors
    "combo",  # minimize the sum of FLOPS + α * WRITE where α is 64
]


def get_optimizer(method, optlib, post_processing, minimize):
    if post_processing[0] is None:
        return ctg.HyperOptimizer(
            methods=method,
            optlib=optlib,
            minimize=minimize,
            parallel=True,
            max_time=60,
            max_repeats=128,
            progbar=True,
        )
    else:
        return ctg.HyperOptimizer(
            methods=method,
            optlib=optlib,
            minimize=minimize,
            parallel=True,
            max_time=60,
            max_repeats=128,
            progbar=True,
            **{post_processing[0]: post_processing[1]},
        )


if __name__ == "__main__":
    for graph, method, optlib, post_processing, minimize in itertools.product(
        graph_args.keys(),
        methods_args,
        optlib_args,
        post_processing_args,
        minimize_args,
    ):
        print(
            f"graph: {graph}, method: {method}, optlib: {optlib}, "
            f"post_processing: {post_processing}, minimize: {minimize}"
        )
        tc.set_contractor(
            "custom",
            optimizer=get_optimizer(method, optlib, post_processing, minimize),
            contraction_info=True,
            preprocessing=True,
            debug_level=2,  # no computation
        )
        trigger_cotengra_optimization(n, nlayers, graph_args[graph])
        print("-------------------------")
