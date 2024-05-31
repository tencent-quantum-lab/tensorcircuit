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


def trigger_cotengra_optimization(n, nlayers, d):
    g = nx.random_regular_graph(d, n)

    # define the loss function
    def loss_f(params, n, nlayers):

        c = generate_circuit(params, g, n, nlayers)

        # calculate the loss function, max cut
        loss = 0.0
        for e in g.edges:
            loss += c.expectation_ps(z=[e[0], e[1]])

        return K.real(loss)

    params = K.implicit_randn(shape=[nlayers, 2])

    # run only once to trigger the compilation
    K.jit(
        K.value_and_grad(loss_f, argnums=0),
        static_argnums=(1, 2),
    )(params, n, nlayers)


# define the cotengra optimizer parameters
methods_args = [  # https://cotengra.readthedocs.io/en/latest/advanced.html#drivers
    "greedy",
    "kahypar",
    "labels",
    # "spinglass",  # requires igraph
    # "labelprop",  # requires igraph
    # "betweenness",  # requires igraph
    # "walktrap",  # requires igraph
    # "quickbb",  # requires https://github.com/dechterlab/quickbb
    # "flowcutter",  # requires https://github.com/kit-algo/flow-cutter-pace17
]

optlib_args = [  # https://cotengra.readthedocs.io/en/latest/advanced.html#optimization-library
    "optuna",  # pip install optuna
    "random",
    # "baytune",  # pip install baytune
    # "nevergrad",  # pip install nevergrad
    # "chocolate",  # pip install git+https://github.com/AIworx-Labs/chocolate@master
    # "skopt",  # pip install scikit-optimize
]

post_processing_args = [  # https://cotengra.readthedocs.io/en/latest/advanced.html#slicing-and-subtree-reconfiguration
    (None, None),
    ("slicing_opts", {"target_size": 2**28}),
    ("slicing_reconf_opts", {"target_size": 2**28}),
    ("reconf_opts", {}),
    ("simulated_annealing_opts", {}),
]


def get_optimizer(method, optlib, post_processing):
    if post_processing[0] is None:
        return ctg.HyperOptimizer(
            methods=method,
            optlib=optlib,
            minimize="flops",
            parallel=True,
            max_time=30,
            max_repeats=30,
            progbar=True,
        )
    else:
        return ctg.HyperOptimizer(
            methods=method,
            optlib=optlib,
            minimize="flops",
            parallel=True,
            max_time=30,
            max_repeats=30,
            progbar=True,
            **{post_processing[0]: post_processing[1]},
        )


if __name__ == "__main__":
    # define the parameters
    n = 20
    nlayers = 15
    d = 3

    for method, optlib, post_processing in itertools.product(
        methods_args, optlib_args, post_processing_args
    ):
        print(f"method: {method}, optlib: {optlib}, post_processing: {post_processing}")
        tc.set_contractor(
            "custom",
            optimizer=get_optimizer(method, optlib, post_processing),
            contraction_info=True,
            preprocessing=True,
            debug_level=2,  # no computation
        )
        trigger_cotengra_optimization(n, nlayers, d)
        print("-------------------------")
