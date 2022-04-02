"""
A plain QAOA optimization example with given graphs using networkx.
"""
import sys

sys.path.insert(0, "../")
import networkx as nx
import tensorflow as tf
import tensorcircuit as tc

K = tc.set_backend("tensorflow")

## 1. define the graph


def dict2graph(d):
    g = nx.to_networkx_graph(d)
    for e in g.edges:
        if not g[e[0]][e[1]].get("weight"):
            g[e[0]][e[1]]["weight"] = 1.0
    return g


# a graph instance

example_graph_dict = {
    0: {1: {"weight": 1.0}, 7: {"weight": 1.0}, 3: {"weight": 1.0}},
    1: {0: {"weight": 1.0}, 2: {"weight": 1.0}, 3: {"weight": 1.0}},
    2: {1: {"weight": 1.0}, 3: {"weight": 1.0}, 5: {"weight": 1.0}},
    4: {7: {"weight": 1.0}, 6: {"weight": 1.0}, 5: {"weight": 1.0}},
    7: {4: {"weight": 1.0}, 6: {"weight": 1.0}, 0: {"weight": 1.0}},
    3: {1: {"weight": 1.0}, 2: {"weight": 1.0}, 0: {"weight": 1.0}},
    6: {7: {"weight": 1.0}, 4: {"weight": 1.0}, 5: {"weight": 1.0}},
    5: {6: {"weight": 1.0}, 4: {"weight": 1.0}, 2: {"weight": 1.0}},
}

example_graph = dict2graph(example_graph_dict)

# 2. define the quantum ansatz

nlayers = 3


def QAOAansatz(gamma, beta, g=example_graph):
    n = len(g.nodes)
    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)
    for j in range(nlayers):
        for e in g.edges:
            c.exp1(
                e[0],
                e[1],
                unitary=tc.gates._zz_matrix,
                theta=g[e[0]][e[1]].get("weight", 1.0) * gamma[j],
            )
        for i in range(n):
            c.rx(i, theta=beta[j])

    # calculate the loss function, max cut
    loss = 0.0
    for e in g.edges:
        loss += c.expectation([tc.gates.z(), [e[0]]], [tc.gates.z(), [e[1]]])

    return loss


# 3. get compiled function for QAOA ansatz and its gradient

QAOA_vg = K.jit(K.value_and_grad(QAOAansatz, argnums=(0, 1)), static_argnums=2)


# 4. optimization loop

beta = tf.Variable(tf.random.normal(shape=[nlayers], stddev=0.1))
gamma = tf.Variable(tf.random.normal(shape=[nlayers], stddev=0.1))
opt = K.optimizer(tf.keras.optimizers.Adam(1e-2))

for i in range(100):
    loss, grads = QAOA_vg(gamma, beta, example_graph)
    print(K.numpy(loss))
    gamma, beta = opt.update(grads, [gamma, beta])  # gradient descent
