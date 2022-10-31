"""
QAOA with finite measurement shot noise
"""
from functools import partial
import numpy as np
from scipy import optimize
import networkx as nx
import optax
import cotengra as ctg
import tensorcircuit as tc
from tensorcircuit import experimental as E
from tensorcircuit.applications.graphdata import maxcut_solution_bruteforce

K = tc.set_backend("jax")
# note this script only supports jax backend

opt_ctg = ctg.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    parallel="ray",
    minimize="combo",
    max_time=10,
    max_repeats=128,
    progbar=True,
)

tc.set_contractor("custom", optimizer=opt_ctg, preprocessing=True)


def get_graph(n, d, weights=None):
    g = nx.random_regular_graph(d, n)
    if weights is not None:
        i = 0
        for e in g.edges:
            g[e[0]][e[1]]["weight"] = weights[i]
            i += 1
    return g


def get_exact_maxcut_loss(g):
    cut, _ = maxcut_solution_bruteforce(g)
    totalw = 0
    for e in g.edges:
        totalw += g[e[0]][e[1]].get("weight", 1)
    loss = totalw - 2 * cut
    return loss


def get_pauli_string(g):
    n = len(g.nodes)
    pss = []
    ws = []
    for e in g.edges:
        l = [0 for _ in range(n)]
        l[e[0]] = 3
        l[e[1]] = 3
        pss.append(l)
        ws.append(g[e[0]][e[1]].get("weight", 1))
    return pss, ws


def generate_circuit(param, g, n, nlayers):
    # construct the circuit ansatz
    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)
    for j in range(nlayers):
        c = tc.templates.blocks.QAOA_block(c, g, param[j, 0], param[j, 1])
    return c


def ps2xyz(psi):
    # ps2xyz([1, 2, 2, 0]) = {"x": [0], "y": [1, 2], "z": []}
    xyz = {"x": [], "y": [], "z": []}
    for i, j in enumerate(psi):
        if j == 1:
            xyz["x"].append(i)
        if j == 2:
            xyz["y"].append(i)
        if j == 3:
            xyz["z"].append(i)
    return xyz


rkey = K.get_random_state(42)


def main_benchmark_suite(n, nlayers, d=3, init=None):
    g = get_graph(n, d, weights=np.random.uniform(size=[int(d * n / 2)]))
    loss_exact = get_exact_maxcut_loss(g)
    print("exact minimal loss by max cut bruteforce: ", loss_exact)
    pss, ws = get_pauli_string(g)
    if init is None:
        init = np.random.normal(scale=0.1, size=[nlayers, 2])

    @partial(K.jit, static_argnums=(2))
    def exp_val(param, key, shots=10000):
        # expectation with shot noise
        # ps, w: H = \sum_i w_i ps_i
        # describing the system Hamiltonian as a weighted sum of Pauli string
        c = generate_circuit(param, g, n, nlayers)
        loss = 0
        s = c.state()
        mc = tc.quantum.measurement_counts(
            s,
            counts=shots,
            format="sample_bin",
            random_generator=key,
            jittable=True,
            is_prob=False,
        )
        for psi, wi in zip(pss, ws):
            xyz = ps2xyz(psi)
            loss += wi * tc.quantum.correlation_from_samples(xyz["z"], mc, c._nqubits)
        return K.real(loss)

    @K.jit
    def exp_val_analytical(param):
        c = generate_circuit(param, g, n, nlayers)
        loss = 0
        for psi, wi in zip(pss, ws):
            xyz = ps2xyz(psi)
            loss += wi * c.expectation_ps(**xyz)
        return K.real(loss)

    # for i in range(3):
    # print(exp_val(init, K.get_random_state(i)))
    # print(exp_val_analytical(init))

    # 0. Exact result double check

    hm = tc.quantum.PauliStringSum2COO(pss, ws, numpy=True)
    hm = K.to_dense(hm)
    e, _ = np.linalg.eigh(hm)
    print("exact minimal loss via eigenstate: ", e[0])

    # 1.1 QAOA with numerically exact expectation: gradient free

    print("QAOA without shot noise")

    exp_val_analytical_sp = tc.interfaces.scipy_interface(
        exp_val_analytical, shape=[nlayers, 2], gradient=False
    )

    r = optimize.minimize(
        exp_val_analytical_sp,
        init,
        method="Nelder-Mead",
        options={"maxiter": 5000},
    )
    print(r)
    print("double check the value?: ", exp_val_analytical_sp(r["x"]))
    # cobyla seems to have issue to given consistent x and cobyla

    # 1.2 QAOA with numerically exact expectation: gradient based

    exponential_decay_scheduler = optax.exponential_decay(
        init_value=1e-2, transition_steps=500, decay_rate=0.9
    )
    opt = K.optimizer(optax.adam(exponential_decay_scheduler))
    param = init  # zeros stall the gradient
    param = tc.array_to_tensor(init, dtype=tc.rdtypestr)
    exp_val_grad_analytical = K.jit(K.value_and_grad(exp_val_analytical))
    for i in range(1000):
        e, gs = exp_val_grad_analytical(param)
        param = opt.update(gs, param)
        if i % 100 == 99:
            print(e)
    print("QAOA energy after gradient descent:", e)

    # 2.1 QAOA with finite shot noise: gradient free

    print("QAOA with shot noise")

    def exp_val_wrapper(param):
        global rkey
        rkey, skey = K.random_split(rkey)
        # maintain stateless randomness in scipy optimize interface
        return exp_val(param, skey)

    exp_val_sp = tc.interfaces.scipy_interface(
        exp_val_wrapper, shape=[nlayers, 2], gradient=False
    )

    r = optimize.minimize(
        exp_val_sp,
        init,
        method="Nelder-Mead",
        options={"maxiter": 5000},
    )
    print(r)

    # the real energy position after optimization

    print("converged as: ", exp_val_analytical_sp(r["x"]))

    # 2.2 QAOA with finite shot noise: gradient based

    exponential_decay_scheduler = optax.exponential_decay(
        init_value=1e-2, transition_steps=500, decay_rate=0.9
    )
    opt = K.optimizer(optax.adam(exponential_decay_scheduler))
    param = tc.array_to_tensor(init, dtype=tc.rdtypestr)
    exp_grad = E.parameter_shift_grad_v2(
        exp_val, argnums=0, random_argnums=1, shifts=(0.001, 0.002)
    )
    # parameter shift doesn't directly apply in QAOA case
    rkey = K.get_random_state(42)

    for i in range(1000):
        rkey, skey = K.random_split(rkey)
        gs = exp_grad(param, skey)
        param = opt.update(gs, param)
        if i % 100 == 99:
            rkey, skey = K.random_split(rkey)
            print(exp_val(param, skey))

    # the real energy position after optimization

    print("converged as:", exp_val_analytical(param))


if __name__ == "__main__":
    main_benchmark_suite(8, 4)
