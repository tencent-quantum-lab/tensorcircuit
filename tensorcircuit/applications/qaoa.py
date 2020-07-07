"""
modules for QAOA application and its variants
layers and graphdata module are utilized
"""

import sys
from functools import lru_cache
import numpy as np
import tensorflow as tf
from typing import List, Sequence, Any, Tuple, Callable, Iterator
import networkx as nx

from ..gates import array_to_tensor, num_to_tensor
from .. import (
    cons,
)  # don't directly import backend, as it is supposed to change at runtime
from ..circuit import Circuit
from .layers import *

Array = Any  # np.array

thismodule = sys.modules[__name__]

lbd = 2.0


@lru_cache()
def energy(i: int, n: int, g: Graph) -> float:
    """
    maxcut energy for n qubit wavefunction i-th basis

    :param i: ranged from 0 to 2**n-1
    :param n: number of qubits
    :param g: nx.Graph
    :return:
    """
    basis = bin(i)[2:].zfill(n)
    r = 0
    for e in g.edges:
        r += g[e[0]][e[1]].get("weight", 1.0) * int(basis[e[0]] != basis[e[1]])
    return r


def ave_func(
    state: Array, g: Graph, f: Callable[[float], float] = lambda s: s
) -> Tensor:
    """

    :param state: 1D array for full wavefunction, the basis is in lexcical order
    :param g: nx.Graph
    :param f: transformation function before averaged
    :return:
    """
    n = int(np.log2(len(state)))
    r = [f(energy(i, n, g)) for i in range(len(state))]
    r = array_to_tensor(np.array(r))
    return cons.backend.real(
        cons.backend.tensordot(
            r,
            cons.backend.cast(cons.backend.abs(state) ** 2, dtype="complex64"),
            [0, 0],
        )
    )


def exp_forward(
    theta: Tensor,
    preset: Sequence[int],
    g: Graph,
    lbd: float = 1.0,
    overlap_threhold: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    n = len(g.nodes)
    ci = Circuit(n)
    cset = get_choice()
    for i, j in enumerate(preset):
        cset[j](ci, theta[i], g)

    state = ci.wavefunction()[0]
    loss = cons.backend.log(ave_func(state, g, f=lambda s: cons.backend.exp(-lbd * s)))  # type: ignore
    loss2 = ave_func(state, g)  # energy
    proba = cons.backend.abs(state) ** 2
    probasum = num_to_tensor(0.0, dtype="float32")
    if overlap_threhold > 0:  # only compute overlap when this argument is positive
        for i in range(2 ** n):
            if energy(i, n, g) >= overlap_threhold:
                probasum += proba[i]
    return loss, -loss2, probasum


def evaluate_vag(
    params: Array,
    preset: Sequence[int],
    g: Graph,
    lbd: float = 0.0,
    overlap_threhold: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    value and gradient, currently on tensorflow backend is supported
    jax and numpy seems to be slow in circuit simulation anyhow.

    :param params:
    :param preset:
    :param g:
    :param lbd: if lbd=0, take energy as objective
    :param overlap_threhold: if as default 0, overlap will not compute in the process
    :return:
    """
    params = array_to_tensor(params)  # complexify params
    with tf.GradientTape() as t:
        t.watch(params)
        expe, ene, probasum = exp_forward(
            params, preset, g, lbd=lbd, overlap_threhold=overlap_threhold
        )
    if lbd == 0:
        gr = t.gradient(ene, params)
    else:
        gr = t.gradient(expe, params)
    #     return forward(beta)
    return expe, ene, cons.backend.real(gr), probasum


def qaoa_train(
    preset: Sequence[int],
    g: Union[Graph, Iterator[Graph]],
    *,
    epochs: int = 100,
    batch: int = 1,
    initial_param: Optional[Array] = None,
    opt: Any = None,
    lbd: float = 0.0,
    overlap_threhold: float = 0.0,
    verbose: bool = True
) -> Tuple[Array, Sequence[float], Sequence[float], Sequence[float]]:
    if initial_param is None:
        initial_param = np.random.normal(loc=0.3, scale=0.05, size=[len(preset)])
    theta = tf.Variable(initial_value=initial_param, dtype=tf.float32)
    # this initialization is near optimal as this parallels QA protocol
    if opt is None:
        opt = tf.keras.optimizers.Adam(learning_rate=0.02)
    gibbs_history = []
    mean_history = []
    overlap_history = []
    if isinstance(g, nx.Graph):

        def one_generator():
            while True:
                yield g

        gi = one_generator()
    else:
        gi = g
    for i in range(epochs):
        means = []
        grs = []
        overlaps = []
        gibbs = []
        for _ in range(batch):
            gdata = gi.send(None)
            value, ene, gra, probasum = evaluate_vag(
                theta.numpy(),
                preset,
                gdata,
                lbd=lbd,
                overlap_threhold=overlap_threhold,
            )

            gibbs.append(value.numpy())
            means.append(ene.numpy())
            overlaps.append(probasum.numpy())
            grs.append(gra.numpy())
        overlap_history.append(np.mean(overlaps))
        gibbs_history.append(np.mean(gibbs))
        mean_history.append(np.mean(means))
        gr = tf.reduce_mean(tf.constant(grs), axis=0)
        opt.apply_gradients([(gr, theta)])
        if verbose:
            print("epoch:", i)
            print("Gibbs objective:", overlap_history[-1])
            print("mean energy:", mean_history[-1])
            print("overlap:", overlap_history[-1])
            print("trainable weights:", theta.numpy())
    return theta.numpy(), mean_history, gibbs_history, overlap_history
