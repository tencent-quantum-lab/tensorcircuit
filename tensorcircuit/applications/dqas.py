"""
modules for DQAS, including kernels on various applications
"""

import sys
import inspect
from functools import lru_cache, partial
from multiprocessing import Pool, get_context
import functools
import operator
import numpy as np
import scipy as sp
import sympy
import tensorflow as tf
from typing import (
    List,
    Sequence,
    Any,
    Tuple,
    Callable,
    Iterator,
    Optional,
    Union,
    Iterable,
    Dict,
)
import networkx as nx
import cirq
import tensorflow_quantum as tfq
from cirq.contrib.svg import SVGCircuit  # type: ignore

from ..gates import array_to_tensor, num_to_tensor
from .. import cons

# don't directly import backend, as it is supposed to change at runtime
from ..circuit import Circuit
from .layers import *

Array = Any  # np.array
Opt = Any  # tf.keras.optimizer
Model = Any  # tf.keras.Model

thismodule = sys.modules[__name__]


_op_pool: Sequence[Any] = []


def set_op_pool(l: Sequence[Any]) -> None:
    # sometimes, to make parallel mode work, one should set_op_pool in global level of the script
    global _op_pool
    _op_pool = l


def get_op_pool() -> Sequence[Any]:
    global _op_pool
    return _op_pool


## GHZ circuit application


def GHZ_vag(
    gdata: Any, nnp: Tensor, preset: Sequence[int], verbose: bool = False, n: int = 3
) -> Tuple[Tensor, Tensor]:
    # gdata = None
    reference_state = np.zeros([2 ** n])
    #     W states benchmarks
    #     for i in range(n):
    #         reference_state[2**(i)] = 1/np.sqrt(n)
    reference_state[0] = 1 / np.sqrt(2)
    reference_state[-1] = 1 / np.sqrt(2)
    reference_state = tf.constant(reference_state, dtype=tf.complex64)
    nnp = nnp.numpy()  # real
    pnnp = [nnp[i, j] for i, j in enumerate(preset)]
    pnnp = array_to_tensor(np.array(pnnp))  # complex
    circuit = Circuit(n)
    cset = get_op_pool()

    with tf.GradientTape() as t:
        t.watch(pnnp)
        for i, j in enumerate(preset):
            gate = cset[j]
            if gate[0].startswith("r"):
                getattr(circuit, gate[0])(gate[1], theta=pnnp[i])
            elif len(gate[0]) == 1:
                getattr(circuit, gate[0])(gate[1])
            elif gate[0] == "CNOT":
                circuit.CNOT(gate[1], gate[2])  # type: ignore
        s = circuit.wavefunction()
        s = tf.reshape(s, [2 ** n,])
        loss = tf.math.reduce_sum(
            tf.math.abs(s - reference_state)
        )  # better for overlap objective to optimize
    #   loss = (tf.math.abs(tf.tensordot(s, reference_state, [0,0]))-1.)**2
    if verbose:
        print(s.numpy())
    gr = t.gradient(loss, pnnp)
    if gr is None:
        # if all gates in preset are not trainable, then gr returns None instead of 0s
        gr = tf.zeros_like(pnnp)
    gr = cons.backend.real(gr)
    gr = tf.where(tf.math.is_nan(gr), 0.0, gr)
    gmatrix = np.zeros_like(nnp)
    for i, j in enumerate(preset):
        gmatrix[i, j] = gr[i]
    gmatrix = tf.constant(gmatrix)
    return loss, gmatrix


def double_qubits_initial() -> Iterator[Sequence[Any]]:
    while True:
        yield [
            cirq.Circuit(
                [cirq.rx(0.0)(cirq.GridQubit(0, 0)), cirq.rx(0.0)(cirq.GridQubit(1, 0))]
            ),  # 00 +xx +zz
            cirq.Circuit(
                [cirq.X(cirq.GridQubit(1, 0)), cirq.rx(0.0)(cirq.GridQubit(0, 0))]
            ),  # 01 +xx -zz
            cirq.Circuit(
                [cirq.X(cirq.GridQubit(0, 0)), cirq.rx(0.0)(cirq.GridQubit(1, 0))]
            ),  # 10 -xx +zz
            cirq.Circuit(
                [cirq.X(cirq.GridQubit(0, 0)), cirq.X(cirq.GridQubit(1, 0))]
            ),  # 11 -xx -zz
        ]


def GHZ_vag_tfq(
    gdata: Any,
    nnp: Tensor,
    preset: Sequence[int],
    verbose: bool = False,
    index: Tuple[int, int, int, int, int, int, int, int] = (1, 1, 1, 0, 0, 1, 0, 0),
) -> Tuple[Tensor, Tensor]:
    # gdata = quantum_circuit

    circuit = cirq.Circuit()
    cset = get_op_pool()
    for i, j in enumerate(preset):
        circuit.append(cset[j])
    input_circuits = [c + circuit for c in gdata]
    measurements = [
        cirq.Z(cirq.GridQubit(0, 0)) * cirq.Z(cirq.GridQubit(1, 0)),
        cirq.X(cirq.GridQubit(0, 0)) * cirq.X(cirq.GridQubit(1, 0)),
    ]
    expl = tfq.layers.Expectation()
    res = expl(input_circuits, operators=measurements)
    if verbose:
        print(res.numpy())
    loss = (
        (-1.0) ** index[0] * res[0, 0]
        + (-1.0) ** index[1] * res[0, 1]
        + (-1.0) ** index[2] * res[1, 0]
        + (-1.0) ** index[3] * res[1, 1]
        + (-1.0) ** index[4] * res[2, 0]
        + (-1.0) ** index[5] * res[2, 1]
        + (-1.0) ** index[6] * res[3, 0]
        + (-1.0) ** index[7] * res[3, 1]
    )
    #     loss = -tf.reduce_sum(tf.abs(res)) # for more general case
    return loss, tf.zeros_like(nnp)


## QFT QEM application


def q(i: int) -> cirq.LineQubit:
    """
    short cut for ``cirq.LineQubit(i)``

    :param i:
    :return:
    """
    return cirq.LineQubit(i)


@lru_cache()
def qft_circuit(n: int) -> cirq.Circuit:

    circuit = cirq.Circuit()
    for i in reversed(range(n)):
        circuit.append(cirq.H(q(i)))
        for d, j in enumerate(reversed(range(i))):
            circuit.append(
                cirq.ControlledGate(cirq.Z ** (1 / 2 ** (d + 1)))(q(j), q(i))
            )

    return circuit


def gapfilling(circuit: cirq.Circuit, placeholder: Sequence[Any]) -> cirq.Circuit:
    """
    Fill single qubit gates according to placeholder on circuit

    :param circuit:
    :param placeholder:
    :return:
    """
    n_circuit = cirq.Circuit()
    all_qubits = sorted(circuit.all_qubits())
    i = 0
    for m in circuit.moments:
        n_circuit.append(m)
        occupied_qubits = set()
        for g in m:
            for q in g.qubits:
                occupied_qubits.add(q)
        for q in all_qubits:
            if q not in occupied_qubits:
                if placeholder[i] != cirq.I:
                    n_circuit.append(placeholder[i](q))
                i += 1
    return n_circuit


def noisyfy(
    circuit: cirq.Circuit,
    error_model: str = "bit_flip",
    p_idle: float = 0.2,
    p_sep: float = 0.02,
) -> cirq.Circuit:
    noise_circuit = cirq.Circuit()
    error = getattr(cirq, error_model)
    all_qubits = circuit.all_qubits()
    for m in circuit.moments:
        noise_circuit.append(m)
        occupied_qubits = set()
        for g in m:
            for q in g.qubits:
                occupied_qubits.add(q)
        for q in all_qubits:
            if q not in occupied_qubits:
                noise_circuit.append(error(p_idle)(q))
        noise_circuit.append(error(p_sep).on_each(*all_qubits))
    return noise_circuit


def unitary_design_block(circuit: cirq.Circuit, n: int) -> cirq.Circuit:
    """
    random Haar measure approximation

    :param circuit: cirq.Circuit, empty circuit
    :param n: # of qubit
    :return:
    """
    for i in range(n):
        theta = np.random.choice([0, 2 / 3, 4 / 3])
        circuit.append(cirq.ZPowGate(exponent=theta)(q(i)))
    for i in range(n - 1):
        for j in range(i + 1, n):
            if np.random.choice([0, 1]) < 0.5:
                circuit.append(cirq.CZ(q(i), q(j)))
    for i in range(n):
        circuit.append(cirq.H(q(i)))
    return circuit


def unitary_design(n: int, l: int = 3) -> cirq.Circuit:
    """
    generate random wavefunction from approximately Haar measure,
    reference:  https://doi.org/10.1063/1.4983266

    :param n: number of qubits
    :param l: repetition of the blocks
    :return:
    """
    circuit = cirq.Circuit()
    for i in range(n):
        circuit.append(cirq.H(q(i)))  # the first block only final H layer has effect
    for _ in range(l):
        unitary_design_block(circuit, n)
    return circuit


def qft_qem_vag(
    gdata: Any,
    nnp: Tensor,
    preset: Sequence[int],
    n: int = 3,
    p_idle: float = 0.2,
    p_sep: float = 0.02,
) -> Tuple[Tensor, Tensor]:
    # gdata = None
    if gdata is None:
        prepend = unitary_design(n)
    else:
        prepend = gdata
    s = cirq.DensityMatrixSimulator()
    cset = get_op_pool()
    placeholder = [cset[j] for j in preset]
    qftc = qft_circuit(n)

    ideal = prepend + qftc
    pdm = s.simulate(ideal).final_density_matrix
    ncq = gapfilling(qftc, placeholder)
    ncq = noisyfy(ncq, p_idle=p_idle, p_sep=p_sep)
    ncq = prepend + ncq
    ndm = s.simulate(ncq).final_density_matrix
    loss = -cirq.fidelity(pdm, ndm)
    return tf.constant(loss, dtype=tf.float32), tf.zeros_like(nnp)


## QAOA application


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
    state: Array,
    g: Graph,
    *fs: Union[
        Tuple[Callable[[float], float], Callable[[Tensor], Tensor]],
        Tuple[
            Callable[[float], float],
            Callable[[Tensor], Tensor],
            Callable[[Tensor, Tensor], Tensor],
        ],
    ],
) -> Sequence[Tensor]:
    """

    :param state: 1D array for full wavefunction, the basis is in lexcical order
    :param g: nx.Graph
    :param fs: transformation functions before averaged
    :return:
    """
    n = int(np.log2(len(state)))
    ebasis = [energy(i, n, g) for i in range(len(state))]
    result = []
    for ftuple in fs:
        if len(ftuple) == 2:
            f, f2 = ftuple  # type: ignore
        else:
            f, f2, f3 = ftuple  # type: ignore
        r = [f(e) for e in ebasis]
        if len(ftuple) == 3:
            r = f3(r, cons.backend.abs(state) ** 2)
        r = array_to_tensor(np.array(r))

        result.append(
            f2(
                cons.backend.real(
                    cons.backend.tensordot(
                        r,
                        cons.backend.cast(
                            cons.backend.abs(state) ** 2, dtype="complex64"
                        ),
                        [0, 0],
                    )
                )
            )
        )
    return result


def exp_forward(
    theta: Tensor,
    preset: Sequence[int],
    g: Graph,
    *fs: Tuple[Callable[[float], float], Callable[[Tensor], Tensor]],
) -> Sequence[Tensor]:
    n = len(g.nodes)
    ci = Circuit(n)
    cset = get_op_pool()
    for i, j in enumerate(preset):
        if callable(cset[j]):
            cset[j](ci, theta[i], g)
        else:
            layer, graph = cset[j]
            layer(ci, theta[i], graph)

    state = ci.wavefunction()[0]
    losses = ave_func(state, g, *fs)
    return losses


# def graph_forward(
#     theta: Tensor,
#     preset: Sequence[int],
#     g: Graph,
#     *fs: Tuple[Callable[[float], float], Callable[[Tensor], Tensor]],
# ) -> Sequence[Tensor]:
#     n = len(g.nodes)
#     ci = Circuit(n)
#     cset = get_op_pool()  # [(Hlayer, nx.Graph), ...]
#
#     for i, j in enumerate(preset):
#         layer, graph = cset[j]
#         layer(ci, theta[i], graph)
#
#     state = ci.wavefunction()[0]
#     losses = ave_func(state, g, *fs)  # objective graph
#     return losses


def _identity(s: Any) -> Any:
    return s


def _neg(s: Any) -> Any:
    return -s


def _exp_fun(s: Any, lbd: float = 1.0) -> Tensor:
    return cons.backend.exp(-lbd * s)


def _overlap_fun(s: Any, overlap_threhold: float = 0.0) -> Tensor:
    if s >= overlap_threhold > 0:
        return 1.0
    return 0.0


def cvar(r: List[float], p: Tensor, percent: float = 0.2) -> Sequence[float]:
    """
    as f3

    :param r:
    :param p:
    :param percent:
    :return:
    """

    rs = sorted(
        [(i, j) for i, j in enumerate(r)], key=lambda s: -s[1]
    )  # larger to smaller
    sump = 0.0
    count = 0
    while sump < percent:
        if sump + p[rs[count][0]] > percent:
            r[rs[count][0]] = (percent - sump) / (p[rs[count][0]]) * r[rs[count][0]]
            count += 1
            break
        else:
            sump += p[rs[count][0]]
            count += 1

    for i in range(count, len(r)):
        r[rs[i][0]] = 0
    r = [k / percent for k in r]
    return r


def qaoa_vag(
    gdata: Graph,
    nnp: Tensor,
    preset: Sequence[int],
    f: Optional[Tuple[Callable[[float], float], Callable[[Tensor], Tensor]]] = None,
    forward_func: Optional[
        Callable[
            [
                Tensor,
                Sequence[int],
                Graph,
                Tuple[Callable[[float], float], Callable[[Tensor], Tensor]],
            ],
            Tuple[Tensor, Tensor],
        ]
    ] = None,
) -> Tuple[Tensor, Tensor]:
    if forward_func is None:
        forward_func = exp_forward  # type: ignore
    if f is None:
        f = (_identity, _neg)
    nnp = nnp.numpy()  # real
    pnnp = [nnp[i, j] for i, j in enumerate(preset)]
    pnnp = array_to_tensor(np.array(pnnp))  # complex
    with tf.GradientTape() as t:
        t.watch(pnnp)
        loss = forward_func(pnnp, preset, gdata, f)  # type: ignore
    gr = t.gradient(loss, pnnp)
    if gr is None:
        # if all gates in preset are not trainable, then gr returns None instead of 0s
        gr = tf.zeros_like(pnnp)
    gr = cons.backend.real(gr)
    gr = tf.where(tf.math.is_nan(gr), 0.0, gr)
    gmatrix = np.zeros_like(nnp)
    for i, j in enumerate(preset):
        gmatrix[i, j] = gr[i]
    gmatrix = tf.constant(gmatrix)
    return loss[0], gmatrix


# qaoa_graph_vag = partial(qaoa_vag, forward_func=graph_forward)


def qaoa_block_vag(
    gdata: Graph,
    nnp: Tensor,
    preset: Sequence[int],
    f: Tuple[Callable[[float], float], Callable[[Tensor], Tensor]],
) -> Tuple[Tensor, Tensor]:
    """
    QAOA block encoding kernel, support 2 params in one op

    :param gdata:
    :param nnp:
    :param preset:
    :param f:
    :return:
    """
    # for universality, nnp always occupy two rows for each block
    # for a more general approach of multi params in one op, see ``quantum_mp_qaoa_vag``

    nnp = nnp.numpy()  # real
    pnnp = []
    ops = get_op_pool()
    for i, j in enumerate(preset):
        # print(ops[j].__repr__)
        if ops[j].__repr__.endswith("_block"):
            pnnp.append([nnp[2 * i, j], nnp[2 * i + 1, j]])
        else:
            pnnp.append([nnp[2 * i, j]])
    # pnnp = array_to_tensor(np.array(pnnp))  # complex
    pnnp = tf.ragged.constant(pnnp, dtype=getattr(tf, cons.dtypestr))
    with tf.GradientTape() as t:
        t.watch(pnnp.values)  # type: ignore
        loss = exp_forward(pnnp, preset, gdata, f)
    gr = t.gradient(loss, pnnp.values)  # type:ignore
    if gr is None:
        # if all gates in preset are not trainable, then gr returns None instead of 0s
        gr = tf.zeros_like(pnnp)
    else:
        gr = cons.backend.real(gr)
        gr = tf.where(tf.math.is_nan(gr), 0.0, gr)
        # if all gates in preset are not trainable, then gr returns None instead of 0s
        gr = pnnp.with_values(gr)  # type:ignore

    gr = cons.backend.real(gr)

    gmatrix = np.zeros_like(nnp)
    for j in range(gr.shape[0]):
        if gr[j].shape[0] == 2:
            gmatrix[2 * j, preset[j]] = gr[j][0]
            gmatrix[2 * j + 1, preset[j]] = gr[j][1]
        else:  # 1
            gmatrix[2 * j, preset[j]] = gr[j][0]

    gmatrix = tf.constant(gmatrix)
    return loss[0], gmatrix


def evaluate_vag(
    params: Array,
    preset: Sequence[int],
    g: Graph,
    lbd: float = 0.0,
    overlap_threhold: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    value and gradient, currently only tensorflow backend is supported
    jax and numpy seems to be slow in circuit simulation anyhow.

    :param params:
    :param preset:
    :param g:
    :param lbd: if lbd=0, take energy as objective
    :param overlap_threhold: if as default 0, overlap will not compute in the process
    :return:
    """
    params = array_to_tensor(params)  # complexify params
    _exp_fun_partial = partial(_exp_fun, lbd=lbd)
    _overlap_fun_partial = partial(_overlap_fun, overlap_threhold=overlap_threhold)
    with tf.GradientTape() as t:
        t.watch(params)
        expe, ene, probasum = exp_forward(
            params,
            preset,
            g,
            (_exp_fun_partial, cons.backend.log),  # gibbs objective
            (_identity, _neg),  # energy
            (_overlap_fun_partial, _identity),  # probability
        )

    if lbd == 0:
        gr = t.gradient(ene, params)
    else:
        gr = t.gradient(expe, params)
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
    verbose: bool = True,
) -> Tuple[Array, Sequence[float], Sequence[float], Sequence[float]]:
    """
    training QAOA with only optimizing circuit parameters, can be well replaced with more general function `DQAS_search`

    :param preset:
    :param g:
    :param epochs:
    :param batch:
    :param initial_param:
    :param opt:
    :param lbd:
    :param overlap_threhold:
    :param verbose:
    :return:
    """
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

        def one_generator() -> Iterator[Graph]:
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
            gdata = gi.send(None)  # type: ignore
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
            print("Gibbs objective:", gibbs_history[-1])
            print("mean energy:", mean_history[-1])
            print("overlap:", overlap_history[-1])
            print("trainable weights:", theta.numpy())
    return theta.numpy(), mean_history, gibbs_history, overlap_history


## functions for quantum Hamiltonian QAOA with tensorflow quantum backend


v = sympy.symbols("v_{0:64}")
vv = sympy.symbols(["v_" + str(i) + "_0:32" for i in range(32)])
# symbol pool


@lru_cache()
def tfim_measurements(
    g: Graph, hzz: float = 1, hx: float = 0, hz: float = 0, one: bool = True
) -> Any:
    """
    Hamiltonian for tfim on lattice defined by graph g

    :param g:
    :param hzz:
    :param hx:
    :param hz:
    :param one:
    :return: cirq.PauliSum as operators for tfq expectation layer
    """
    # one=True and False seems give no speed difference
    measurements = []
    qubits = generate_qubits(g)
    for e in g.edges:
        measurements.append(
            hzz * g[e[0]][e[1]]["weight"] * cirq.Z(qubits[e[0]]) * cirq.Z(qubits[e[1]])
        )

    if hx != 0:
        for i in range(len(g.nodes)):
            measurements.append(hx * cirq.X(qubits[i]))
    if hz != 0:
        for i in range(len(g.nodes)):
            measurements.append(hz * cirq.Z(qubits[i]))
    if one:
        measurements = functools.reduce(operator.add, measurements)
    return measurements


@lru_cache()
def heisenberg_measurements(
    g: Graph, hxx: float = 1.0, hyy: float = 1.0, hzz: float = 1.0, one: bool = True
) -> Any:
    """
    Hamiltonian measurements for Heisenberg model on graph lattice g

    :param g:
    :param hxx:
    :param hyy:
    :param hzz:
    :param one:
    :return:
    """
    measurements = []
    qubits = generate_qubits(g)
    for e in g.edges:
        measurements.append(
            hzz * g[e[0]][e[1]]["weight"] * cirq.Z(qubits[e[0]]) * cirq.Z(qubits[e[1]])
        )
        measurements.append(
            hxx * g[e[0]][e[1]]["weight"] * cirq.X(qubits[e[0]]) * cirq.X(qubits[e[1]])
        )
        measurements.append(
            hyy * g[e[0]][e[1]]["weight"] * cirq.Y(qubits[e[0]]) * cirq.Y(qubits[e[1]])
        )
    if one:
        measurements = functools.reduce(operator.add, measurements)
    return measurements


def quantum_qaoa_vag(
    gdata: Graph,
    nnp: Tensor,
    preset: Sequence[int],
    measurements_func: Optional[Callable[..., Any]] = None,
    **kws: Any,
) -> Tuple[Tensor, Tensor]:
    """
    tensorflow quantum backend compare to qaoa_vag which is tensorcircuit backend

    :param gdata:
    :param nnp:
    :param preset:
    :param measurements_func:
    :param kws: kw arguments for measurements_func
    :return:
    """
    if measurements_func is None:
        measurements_func = tfim_measurements
    ep = tfq.layers.Expectation()
    nnp = nnp.numpy()  # real
    pnnp = [nnp[i, j] for i, j in enumerate(preset)]
    pnnp = array_to_tensor(np.array(pnnp))  # complex
    ci = cirq.Circuit()
    cset = get_op_pool()
    for i, j in enumerate(preset):
        if callable(cset[j]):
            cset[j](ci, gdata, v[i])
        else:  # op is a tuple with graph info as (op, g)
            cset[j][0](ci, cset[j][1], v[i])

    with tf.GradientTape() as t:
        t.watch(pnnp)
        loss = ep(
            inputs=[ci],
            symbol_names=v[: len(preset)],
            symbol_values=[pnnp],
            operators=measurements_func(gdata, **kws),
        )[0]
        gr = t.gradient(loss, pnnp)
    if gr is None:
        # if all gates in preset are not trainable, then gr returns None instead of 0s
        gr = tf.zeros_like(pnnp)
    gr = cons.backend.real(gr)
    gr = tf.where(tf.math.is_nan(gr), 0.0, gr)
    gmatrix = np.zeros_like(nnp)
    for i, j in enumerate(preset):
        gmatrix[i, j] = gr[i]
    gmatrix = tf.constant(gmatrix)
    return loss[0], gmatrix


def quantum_mp_qaoa_vag(
    gdata: Graph,
    nnp: Tensor,
    preset: Sequence[int],
    measurements_func: Optional[Callable[..., Any]] = None,
    **kws: Any,
) -> Tuple[Tensor, Tensor]:
    """
    multi parameter for one layer

    :param gdata:
    :param nnp:
    :param preset:
    :param measurements_func:
    :param kws: kw arguments for measurements_func
    :return: loss function, gradient of nnp
    """
    if measurements_func is None:
        measurements_func = tfim_measurements
    assert len(nnp.shape) == 3  # p * c * l
    nnp = nnp.numpy()  # real
    p, c, l = nnp.shape[0], nnp.shape[1], nnp.shape[2]
    pnnp = np.empty(dtype=np.float32, shape=[p, l])
    for i, j in enumerate(preset):
        pnnp[i, :] = nnp[i, j, :]
    pnnp = array_to_tensor(np.array(pnnp))  # complex
    ci = cirq.Circuit()
    cset = get_op_pool()
    for i, j in enumerate(preset):
        if callable(cset[j]):
            cset[j](ci, gdata, vv[i][:l])
        else:  # op is a tuple with graph info as (op, g)
            cset[j][0](ci, cset[j][1], vv[i][:l])
    ep = tfq.layers.Expectation()
    symbol_names = []
    for i in range(len(preset)):
        symbol_names.extend(vv[i][:l])
    with tf.GradientTape() as t:
        t.watch(pnnp)
        loss = ep(
            inputs=[ci],
            symbol_names=symbol_names,
            symbol_values=[tf.reshape(pnnp, [-1])],
            operators=measurements_func(gdata, **kws),
        )[0]
    gr = t.gradient(loss, pnnp)
    if gr is None:
        # if all gates in preset are not trainable, then gr returns None instead of 0s
        gr = tf.zeros_like(pnnp)
    gr = cons.backend.real(gr)
    gr = tf.where(tf.math.is_nan(gr), 0.0, gr)
    gmatrix = np.zeros_like(nnp)
    for i, j in enumerate(preset):
        gmatrix[i, j, :] = gr[i, :]
    gmatrix = tf.constant(gmatrix)
    return loss[0], gmatrix


## infrastrcture for DQAS search


def get_var(name: str) -> Any:
    """
    call in customized functions and grab variable within DQAF framework function by var name str

    :param name:
    :return:
    """
    return inspect.stack()[2][0].f_locals[name]


def verbose_output(max_prob: bool = True, weight: bool = True) -> None:
    """
    doesn't support prob model DQAS search

    :param max_prob:
    :param weight:
    :return:
    """
    if max_prob:
        prob = get_var("prob")
        print("max probability for each layer:")
        print(np.max(prob.numpy(), axis=1))
    if weight:
        nnp = get_var("nnp")
        stp = get_var("stp")
        cand_weight = get_weights(nnp, stp).numpy()
        print(
            "associating weights:", cand_weight,
        )


def preset_byprob(prob: Tensor) -> Sequence[int]:
    preset = []
    p = prob.shape[0]
    c = prob.shape[1]
    for i in range(p):
        j = np.random.choice(np.arange(c), p=np.array(prob[i]))
        preset.append(j)
    return preset


def get_preset(stp: Tensor) -> Tensor:
    return tf.argmax(stp, axis=1)


def get_weights(
    nnp: Tensor, stp: Tensor = None, preset: Optional[Sequence[int]] = None
) -> Tensor:
    """
    works only when nnp has the same shape as stp, i.e. one parameter for each op

    :param nnp:
    :param stp:
    :param preset:
    :return:
    """
    if preset is None:
        preset = get_preset(stp)
    p = nnp.shape[0]
    ind_ = tf.stack([tf.cast(tf.range(p), tf.int32), tf.cast(preset, tf.int32)])
    return tf.gather_nd(nnp, tf.transpose(ind_))


def get_weights_v2(nnp: Tensor, preset: Sequence[int]) -> Tensor:
    if len(nnp.shape) == 3:
        l = nnp.shape[-1]
    else:
        l = 1
        nnp = nnp[..., tf.newaxis]
    p, c = nnp.shape[0], nnp.shape[1]
    weights = np.empty(dtype=np.float32, shape=[p, l])
    for i, j in enumerate(preset):
        weights[i, :] = nnp[i, j, :]
    if l == 1:
        weights = weights.reshape([p])
    return tf.constant(weights)


def parallel_kernel(
    prob: Tensor,
    gdata: Any,
    nnp: Tensor,
    kernel_func: Callable[[Any, Tensor, Sequence[int]], Tuple[Tensor, Tensor]],
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    kernel for multiprocess to run parallel in DQAS function

    :param prob:
    :param gdata:
    :param nnp:
    :param kernel_func:
    :return:
    """
    sp.random.seed()  # make each subprocess run with different random state
    # see https://stackoverflow.com/a/6914470/9062180
    # it is still not the best way to corporate numpy random and multiprocessing
    # see more in https://github.com/numpy/numpy/issues/9650
    dtype = tf.float32
    p = prob.shape[0]
    preset = preset_byprob(prob)
    loss, gnnp = kernel_func(gdata, nnp, preset)

    gs = tf.tensor_scatter_nd_add(
        tf.cast(-prob, dtype=dtype),
        tf.constant(list(zip(range(p), preset))),
        tf.ones([p], dtype=dtype),
    )  # \nabla lnp
    return loss, gnnp, gs


def void_generator() -> Iterator[Any]:
    while True:
        yield None


def single_generator(g: Any) -> Iterator[Any]:
    while True:
        yield g


def history_loss() -> Array:
    return get_var("avcost1").numpy()


def repr_op(element: Any) -> str:
    if isinstance(element, str):
        return element
    if isinstance(element, list) or isinstance(element, tuple):
        return str(tuple([repr_op(e) for e in element]))
    if callable(element.__repr__):
        return element.__repr__()  # type: ignore
    else:
        return element.__repr__  # type: ignore


def DQAS_search(
    kernel_func: Callable[[Any, Tensor, Sequence[int]], Tuple[Tensor, Tensor]],
    *,
    g: Optional[Iterator[Any]] = None,
    op_pool: Optional[Sequence[Any]] = None,
    p: Optional[int] = None,
    p_nnp: Optional[int] = None,
    p_stp: Optional[int] = None,
    batch: int = 300,
    prethermal: int = 0,
    epochs: int = 100,
    parallel_num: int = 0,
    verbose: bool = False,
    verbose_func: Optional[Callable[[], None]] = None,
    history_func: Optional[Callable[[], Any]] = None,
    prob_clip: Optional[float] = None,
    baseline_func: Optional[Callable[[Sequence[float]], float]] = None,
    pertubation_func: Optional[Callable[[], Tensor]] = None,
    nnp_initial_value: Optional[Array] = None,
    stp_initial_value: Optional[Array] = None,
    network_opt: Optional[Opt] = None,
    structure_opt: Optional[Opt] = None,
    prethermal_opt: Optional[Opt] = None,
    prethermal_preset: Optional[Sequence[int]] = None,
    stp_regularization: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    nnp_regularization: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
) -> Tuple[Tensor, Tensor, Sequence[Any]]:
    """
    DQAS framework entrypoint

    :param kernel_func: function with input of data instance, circuit parameters theta and structural paramter k,
                    return tuple of objective value and gradient with respect to theta
    :param g: data generator as dataset
    :param op_pool: list of operations as primitive operator pool
    :param p: the default layer number of the circuit ansatz
    :param p_nnp: shape of circuit parameter pool, in general p_stp*l, where l is the max number of circuit parameters for
            op in the operator pool
    :param p_stp: the same as p in the most times
    :param batch: batch size of one epoch
    :param prethermal: prethermal update times
    :param epochs: training epochs
    :param parallel_num: parallel thread number, 0 to disable multiprocessing model by default
    :param verbose: set verbose log to print
    :param vebose_func: function to output verbose information
    :param history_func: function return intermiediate result for final history list
    :param prob_clip: cutoff probability to avoid peak distribution
    :param baseline_func: function accepting list of objective values and return the baseline value used in the next round
    :param pertubation_func: return noise with the same shape as circuit parameter pool
    :param nnp_initial_value: initial values for circuit parameter pool
    :param stp_initial_value: initial values for probabilistic model parameters
    :param network_opt: optimizer for circuit parameters theta
    :param structure_opt: optimizer for model parameters alpha
    :param prethermal_opt: optimizer for circuit parameters in prethermal stage
    :param prethermal_preset: fixed structural parameters for prethermal training
    :param stp_regularization: regularization function for model parameters alpha
    :param nnp_regularization: regularization function for circuit parameters theta
    :return:
    """

    # shape of nnp and stp is not necessarily compatible in complicated settings
    dtype = tf.float32  # caution, simply changing this is not guranteed to work
    if op_pool is None:
        op_pool = get_op_pool()
    c = len(op_pool)
    set_op_pool(op_pool)
    if g is None:
        g = void_generator()
    if parallel_num > 0:
        pool = get_context("spawn").Pool(parallel_num)
        global parallel_kernel
        p_parallel_kernel = partial(parallel_kernel, kernel_func=kernel_func)

    if network_opt is None:
        network_opt = tf.keras.optimizers.Adam(learning_rate=0.1)  # network
    if structure_opt is None:
        structure_opt = tf.keras.optimizers.Adam(
            learning_rate=0.1, beta_1=0.8, beta_2=0.99
        )  # structure
    if prethermal_opt is None:
        prethermal_opt = tf.keras.optimizers.Adam(learning_rate=0.1)  # prethermal

    if nnp_initial_value is None:
        if p_nnp is None:
            if p is not None:
                p_nnp = p
            else:
                raise ValueError("Please give the shape information on nnp")
        nnp_initial_value = np.random.uniform(size=[p_nnp, c])
    if stp_initial_value is None:
        if p_stp is None:
            if p is not None:
                p_stp = p
            else:
                raise ValueError("Please give the shape information on stp")
        stp_initial_value = np.zeros([p_stp, c])
    if p is None:
        p = stp_initial_value.shape[0]
    if baseline_func is None:
        baseline_func = np.mean
    nnp = tf.Variable(initial_value=nnp_initial_value, dtype=dtype)
    stp = tf.Variable(initial_value=stp_initial_value, dtype=dtype)

    history = []

    prob = tf.math.exp(stp) / tf.tile(
        tf.math.reduce_sum(tf.math.exp(stp), axis=1)[:, tf.newaxis], [1, c]
    )  # softmax categorical probability
    avcost1 = 0

    for _, gdata in zip(range(prethermal), g):  # prethermal for nn param
        if prethermal_preset is None:
            preset = preset_byprob(prob)
        else:
            preset = prethermal_preset
        forwardv, gnnp = kernel_func(gdata, nnp, preset)
        prethermal_opt.apply_gradients([(gnnp, nnp)])

    if verbose:
        print("network parameter after prethermalization: \n", nnp.numpy())

    try:
        for epoch in range(epochs):  # iteration to update strcuture param
            # for data spliting case, odd update network, even update structure
            prob = tf.math.exp(stp) / tf.tile(
                tf.math.reduce_sum(tf.math.exp(stp), axis=1)[:, tf.newaxis], [1, c]
            )
            if prob_clip is not None:
                prob = tf.clip_by_value(prob, (1 - prob_clip) / c, prob_clip)
                prob = prob / tf.tile(
                    tf.reshape(tf.reduce_sum(prob, axis=1), [prob.shape[0], 1]),
                    tf.constant([1, prob.shape[1]]),
                )

            if verbose:
                print("probability: \n", prob.numpy())

            print("----------new epoch %s-----------" % epoch)

            deri_stp = []
            deri_nnp = []
            # avcost2 = tf.convert_to_tensor(avcost1 / batch) * baseline_scale
            avcost2 = avcost1
            costl = []
            # nnpg = tf.zeros_like(nnp)
            # collect nn param graident on the matrix with the same form as nnp
            if stp_regularization is not None:
                stp_penalty_gradient = stp_regularization(stp, nnp)
                if verbose:
                    print("stp_penalty_gradient:", stp_penalty_gradient.numpy())
            else:
                stp_penalty_gradient = 0.0
            if nnp_regularization is not None:
                nnp_penalty_gradient = nnp_regularization(stp, nnp)
                if verbose:
                    print("nnpp_penalty_gradient:", nnp_penalty_gradient.numpy())

            else:
                nnp_penalty_gradient = 0.0

            if parallel_num == 0:
                for _, gdata in zip(range(batch), g):
                    preset = preset_byprob(prob)
                    if pertubation_func is not None:
                        loss, gnnp = kernel_func(
                            gdata, nnp + pertubation_func(), preset
                        )
                    else:
                        loss, gnnp = kernel_func(gdata, nnp, preset)

                    gs = tf.tensor_scatter_nd_add(
                        tf.cast(-prob, dtype=dtype),
                        tf.constant(list(zip(range(p), preset))),
                        tf.ones([p], dtype=dtype),
                    )  # \nabla lnp
                    deri_stp.append(
                        (tf.cast(loss, dtype=dtype) - tf.cast(avcost2, dtype=dtype))
                        * tf.cast(gs, dtype=dtype)
                    )
                    deri_nnp.append(gnnp)
                    costl.append(loss.numpy())
            else:  ## parallel mode for batch evaluation
                args_list = []
                for _, gdata in zip(range(batch), g):
                    if pertubation_func is not None:
                        args_list.append((prob, gdata, nnp + pertubation_func()))
                    else:
                        args_list.append((prob, gdata, nnp))
                parallel_result = pool.starmap(p_parallel_kernel, args_list)
                # [(loss, gnnp, gs), ...]
                deri_nnp = []
                deri_stp = []
                costl = []
                for loss, gnnp, gs in parallel_result:
                    costl.append(loss.numpy())
                    deri_nnp.append(gnnp)
                    deri_stp.append(
                        (tf.cast(loss, dtype=dtype) - tf.cast(avcost2, dtype=dtype))
                        * tf.cast(gs, dtype=dtype)
                    )

            avcost1 = tf.convert_to_tensor(baseline_func(costl))

            print(
                "batched average loss: ",
                np.mean(costl),
                " batched loss std: ",
                np.std(costl),
                "\nnew baseline: ",
                avcost1.numpy(),  # type: ignore
            )
            batched_gs = tf.math.reduce_mean(
                tf.convert_to_tensor(deri_stp, dtype=dtype), axis=0
            )
            batched_gnnp = tf.math.reduce_mean(
                tf.convert_to_tensor(deri_nnp, dtype=dtype), axis=0
            )
            if verbose:
                print("batched gradient of stp: \n", batched_gs.numpy())
                print("batched gradient of nnp: \n", batched_gnnp.numpy())

            network_opt.apply_gradients(
                zip([batched_gnnp + nnp_penalty_gradient], [nnp])
            )
            structure_opt.apply_gradients(
                zip([batched_gs + stp_penalty_gradient], [stp])
            )
            if verbose:
                print(
                    "strcuture parameter: \n",
                    stp.numpy(),
                    "\n network parameter: \n",
                    nnp.numpy(),
                )

            if verbose_func is not None:
                verbose_func()

            cand_preset = get_preset(stp).numpy()
            cand_preset_repr = [repr_op(op_pool[f]) for f in cand_preset]
            print("best candidates so far:", cand_preset_repr)
            # TODO, more general repr
            if nnp.shape == stp.shape and verbose:
                cand_weight = get_weights(nnp, stp).numpy()
                print(
                    "And associating weights:", cand_weight,
                )

            if history_func is not None:
                history.append(history_func())
        if parallel_num > 0:
            pool.close()
        return stp, nnp, history  # TODO: history list trackings
    except KeyboardInterrupt:
        if parallel_num > 0:
            pool.close()
        return stp, nnp, history


## training based on DQAS

qaoa_vag_energy = partial(qaoa_vag, f=(_identity, _neg))


def qaoa_simple_train(
    preset: Sequence[int],
    graph: Union[Sequence[Graph], Iterator[Graph]],
    vag_func: Optional[
        Callable[[Any, Tensor, Sequence[int]], Tuple[Tensor, Tensor]]
    ] = None,
    epochs: int = 60,
    batch: int = 1,
    nnp_shape: Optional[Array] = None,
    nnp_initial_value: Optional[Array] = None,
    opt: Optional[Opt] = None,
    search_func: Optional[Callable[..., Any]] = None,
    kws: Optional[Dict[Any, Any]] = None,
) -> Tuple[Array, float]:
    sp.random.seed()
    # TODO: the best practice combine mulprocessing and random generator still needs further investigation
    p = len(preset)
    c = len(get_op_pool())
    stp_train = np.zeros([p, c])
    for i, j in enumerate(preset):
        stp_train[i, j] = 10.0
    if nnp_initial_value is None and nnp_shape is None:
        nnp_initial_value = np.random.normal(loc=0.23, scale=0.8, size=[p, c])
    elif nnp_shape is not None and nnp_initial_value is None:
        nnp_initial_value = np.random.normal(loc=0.23, scale=0.8, size=nnp_shape)
    if vag_func is None:
        vag_func = qaoa_vag_energy
    if kws is None:
        kws = {}
    if "prob_model_func" in kws:
        pmf = kws["prob_model_func"]
        del kws["prob_model_func"]
        kws[
            "prob_model"
        ] = pmf()  # in case keras model cannot pickled for multiprocessing map
    if isinstance(graph, list):

        def graph_generator() -> Iterator[Graph]:
            i = 0
            l = len(graph)  # type: ignore
            while True:
                if i < l:
                    yield graph[i]  # type: ignore
                    i += 1
                else:
                    i = 0
                    yield graph[i]  # type: ignore

        graph_g = graph_generator()
    else:
        graph_g = graph  # type: ignore

    if search_func is None:
        search_func = DQAS_search
        kws.update({"stp_initial_value": stp_train})

    stp, nnp, h = search_func(
        vag_func,
        g=graph_g,
        p=p,
        batch=batch,
        prethermal=0,
        epochs=epochs,
        history_func=history_loss,
        nnp_initial_value=nnp_initial_value,
        network_opt=opt,
        **kws,
    )
    return (get_weights_v2(nnp, preset=preset).numpy(), h[-1])


def parallel_qaoa_train(
    preset: Sequence[int],
    g: Any,
    vag_func: Any = None,
    opt: Opt = None,
    epochs: int = 60,
    tries: int = 16,
    batch: int = 1,
    cores: int = 8,
    loc: float = 0.0,
    scale: float = 1.0,
    nnp_shape: Optional[Sequence[int]] = None,
    search_func: Optional[Callable[..., Any]] = None,
    kws: Optional[Dict[Any, Any]] = None,
) -> Sequence[Any]:
    """
    parallel variational parameter training and search to avoid local minimum
    not limited to qaoa setup as the function name indicates,
    as long as you provided suitable `vag_func`

    :param preset:
    :param g: data input generator for vag_func
    :param vag_func: vag_kernel
    :param opt:
    :param epochs:
    :param tries: number of tries
    :param batch: for optimization problem the input is in general fixed so batch is often 1
    :param cores: number of parallel jobs
    :param loc: mean value of normal distribution for nnp
    :param scale: std deviation of normal distribution for nnp
    :return:
    """

    if not opt:
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    p = len(preset)
    c = len(get_op_pool())
    glist = []
    for _ in range(epochs * batch):
        glist.append(g.send(None))  # pickle doesn't support generators even in dill
    if vag_func is None:
        vag_func = qaoa_vag_energy
    if nnp_shape is None:
        nnp_shape = [p, c]
    pool = Pool(cores)
    args_list = [
        (
            preset,
            glist,
            vag_func,
            epochs,
            batch,
            None,
            np.random.normal(loc=loc, scale=scale, size=nnp_shape),
            opt,
            search_func,
            kws,
        )
        for _ in range(tries)
    ]
    result_list = pool.starmap(qaoa_simple_train, args_list)
    pool.close()
    result_list = sorted(result_list, key=lambda s: s[1])
    print(result_list)
    print("the optimal result is %s" % result_list[0][1])
    return result_list


## probabilisitic model based DQAS


def van_sample(
    prob_model: Model, batch_size: int
) -> Tuple[List[Tensor], List[List[Tensor]]]:
    glnprob_list = []
    with tf.GradientTape(persistent=True) as t:
        sample, xhat = prob_model.sample(batch_size)
        lnprob = prob_model._log_prob(sample, xhat)
        for i in range(batch_size):
            glnprob_list.append(t.gradient(lnprob[i], prob_model.variables))
    sample = tf.argmax(sample, axis=-1)
    sample_list = [sample[i] for i in range(batch_size)]
    del t
    return sample_list, glnprob_list


def van_regularization(
    prob_model: Model, nnp: Tensor = None, lbd_w: float = 0.01, lbd_b: float = 0.01
) -> Tensor:
    return prob_model.regularization(lbd_w=lbd_w, lbd_b=lbd_b)


def micro_sample(
    prob_model: Model, batch_size: int, repetitions: Optional[List[int]] = None,
) -> Tuple[List[Tensor], List[List[Tensor]]]:
    glnprob_list = []
    with tf.GradientTape(persistent=True) as t:
        sample, xhat = prob_model.sample(batch_size)
        lnprob = prob_model._log_prob(sample, xhat)
        for i in range(batch_size):
            glnprob_list.append(t.gradient(lnprob[i], prob_model.variables))
    sample = tf.argmax(sample, axis=-1)
    sample_list = sample.numpy()
    del t

    if not repetitions:
        return tf.constant(sample_list), glnprob_list
    else:
        ns = np.empty(shape=[batch_size, len(repetitions)], dtype=np.int32)
        for i, j in enumerate(repetitions):
            ns[:, i] = sample_list[:, j]
        return tf.constant(ns), glnprob_list


def DQAS_search_pmb(
    kernel_func: Callable[[Any, Tensor, Sequence[int]], Tuple[Tensor, Tensor]],
    prob_model: Model,
    *,
    sample_func: Optional[
        Callable[[Model, int], Tuple[List[Tensor], List[List[Tensor]]]]
    ] = None,
    g: Optional[Iterator[Any]] = None,
    op_pool: Optional[Sequence[Any]] = None,
    p: Optional[int] = None,
    batch: int = 300,
    prethermal: int = 0,
    epochs: int = 100,
    parallel_num: int = 0,
    verbose: bool = False,
    verbose_func: Optional[Callable[[], None]] = None,
    history_func: Optional[Callable[[], Any]] = None,
    baseline_func: Optional[Callable[[Sequence[float]], float]] = None,
    pertubation_func: Optional[Callable[[], Tensor]] = None,
    nnp_initial_value: Optional[Array] = None,
    stp_regularization: Optional[Callable[[Model, Tensor], Tensor]] = None,
    network_opt: Optional[Opt] = None,
    structure_opt: Optional[Opt] = None,
    prethermal_opt: Optional[Opt] = None,
) -> Tuple[Tensor, Tensor, Sequence[Any]]:
    """
    probabilistic model based DQAS, can use extensively for DQAS case for ``NMF`` probabilistic model

    :param kernel_func: vag func, return loss and nabla lnp
    :param prob_model: keras model
    :param sample_func: sample func of logic with keras model input
    :param g: input data pipeline generator
    :param op_pool: operation pool
    :param p: depth for DQAS
    :param batch:
    :param prethermal:
    :param epochs:
    :param parallel_num: parallel kernels
    :param verbose:
    :param verbose_func:
    :param history_func:
    :param baseline_func:
    :param pertubation_func:
    :param nnp_initial_value:
    :param stp_regularization:
    :param network_opt:
    :param structure_opt:
    :param prethermal_opt:
    :return:
    """

    # shape of nnp and stp is not necessarily compatible in complicated settings
    dtype = tf.float32  # caution, simply changing this is not guranteed to work
    if op_pool is None:
        op_pool = get_op_pool()
    c = len(op_pool)
    set_op_pool(op_pool)
    if sample_func is None:
        sample_func = van_sample
    if g is None:
        g = void_generator()
    if parallel_num > 0:
        pool = get_context("spawn").Pool(parallel_num)
        # use spawn model instead of default fork which has threading lock issues

    if network_opt is None:
        network_opt = tf.keras.optimizers.Adam(learning_rate=0.1)  # network
    if structure_opt is None:
        structure_opt = tf.keras.optimizers.Adam(
            learning_rate=0.1, beta_1=0.8, beta_2=0.99
        )  # structure
    if prethermal_opt is None:
        prethermal_opt = tf.keras.optimizers.Adam(learning_rate=0.1)  # prethermal
    if p is None:
        p = nnp_initial_value.shape[0]  # type: ignore
    if nnp_initial_value is None:
        nnp_initial_value = np.random.normal(loc=0, scale=0.3, size=[p, c])

    if baseline_func is None:
        baseline_func = np.mean
    nnp = tf.Variable(initial_value=nnp_initial_value, dtype=dtype)

    history = []

    avcost1 = 0

    if prethermal > 0:
        presets, glnprobs = sample_func(prob_model, prethermal)
    for i, gdata in zip(range(prethermal), g):  # prethermal for nn param
        forwardv, gnnp = kernel_func(gdata, nnp, presets[i])
        prethermal_opt.apply_gradients([(gnnp, nnp)])

    if verbose:
        print("network parameter after prethermalization: \n", nnp.numpy())

    try:
        for epoch in range(epochs):  # iteration to update strcuture param

            print("----------new epoch %s-----------" % epoch)

            deri_stp = []
            deri_nnp = []
            avcost2 = avcost1
            costl = []

            presets, glnprobs = sample_func(prob_model, batch)

            if stp_regularization is not None:
                with tf.GradientTape() as t:
                    stp_penalty = stp_regularization(prob_model, nnp)
                gr = t.gradient(stp_penalty, prob_model.variables)
                g_stp_penalty = []
                for v, gi in zip(prob_model.variables, gr):
                    if gi is not None:
                        g_stp_penalty.append(gi)
                    else:
                        g_stp_penalty.append(tf.zeros_like(v))

                if verbose:
                    print(
                        "typical scale of gradient from stp variable regularization:",
                        [tf.reduce_mean(tf.math.abs(w)).numpy() for w in g_stp_penalty],
                    )

            else:
                g_stp_penalty = []
                for v in prob_model.variables:
                    g_stp_penalty.append(tf.zeros_like(v))

            if parallel_num == 0:

                for i, gdata in zip(range(batch), g):
                    if pertubation_func is not None:
                        loss, gnnp = kernel_func(
                            gdata, nnp + pertubation_func(), presets[i]
                        )
                    else:
                        loss, gnnp = kernel_func(gdata, nnp, presets[i])

                    deri_stp.append(
                        [
                            (tf.cast(loss, dtype=dtype) - tf.cast(avcost2, dtype=dtype))
                            * w
                            for w in glnprobs[i]
                        ]
                    )
                    deri_nnp.append(gnnp)
                    costl.append(loss.numpy())
            else:  ## parallel mode for batch evaluation
                args_list = []
                for i, gdata in zip(range(batch), g):
                    if pertubation_func is not None:
                        args_list.append(
                            (gdata, nnp + pertubation_func(), presets[i].numpy())
                        )
                    else:
                        args_list.append((gdata, nnp, presets[i].numpy()))
                # print(args_list)
                parallel_result = pool.starmap(kernel_func, args_list)
                # [(loss, gnnp), ...]
                deri_nnp = []
                deri_stp = []
                costl = []
                for i, r in enumerate(parallel_result):
                    loss, gnnp = r
                    costl.append(loss.numpy())
                    deri_nnp.append(gnnp)
                    deri_stp.append(
                        [
                            (tf.cast(loss, dtype=dtype) - tf.cast(avcost2, dtype=dtype))
                            * w
                            for w in glnprobs[i]
                        ]
                    )

            avcost1 = tf.convert_to_tensor(baseline_func(costl))

            print(
                "batched average loss: ",
                np.mean(costl),
                " batched loss std: ",
                np.std(costl),
                "\nnew baseline: ",
                avcost1.numpy(),  # type: ignore
            )
            batched_gs = []
            batched_gs_std = []
            for i in range(len(glnprobs[0])):
                batched_gs.append(
                    tf.math.reduce_mean(
                        tf.convert_to_tensor([w[i] for w in deri_stp], dtype=dtype),
                        axis=0,
                    )
                    + g_stp_penalty[i]
                )
                if verbose:  # check on baseline fluctuation reduction effect
                    batched_gs_std.append(
                        tf.math.reduce_std(
                            tf.convert_to_tensor([w[i] for w in deri_stp], dtype=dtype),
                            axis=0,
                        )
                    )

            batched_gnnp = tf.math.reduce_mean(
                tf.convert_to_tensor(deri_nnp, dtype=dtype), axis=0
            )
            if verbose:
                print("batched gradient of nnp: \n", batched_gnnp.numpy())
                print(
                    "typical scale of batched graident of stp: \n",
                    [tf.reduce_mean(tf.math.abs(w)).numpy() for w in batched_gs],
                )

            network_opt.apply_gradients(zip([batched_gnnp], [nnp]))
            structure_opt.apply_gradients(zip(batched_gs, prob_model.variables))
            if verbose:
                print(
                    "\n network parameter: \n", nnp.numpy(),
                )
                print(
                    "typical scale of stp parameter: \n",
                    [
                        tf.reduce_mean(tf.math.abs(w)).numpy()
                        for w in prob_model.variables
                    ],
                )
                print(
                    "typical scale standard deviation of batched gradient (ratio to mean): \n",
                    [
                        tf.reduce_mean(tf.math.abs(w1)).numpy()
                        / tf.reduce_mean(tf.math.abs(w2) + 1.0e-20).numpy()
                        for w1, w2 in zip(batched_gs_std, prob_model.variables)
                    ],
                )

            if verbose_func is not None:
                verbose_func()

            if history_func is not None:
                history.append(history_func())
        if parallel_num > 0:
            pool.close()
        return prob_model, nnp, history
    except KeyboardInterrupt:
        if parallel_num > 0:
            pool.close()
        return prob_model, nnp, history


## some utils


def color_svg(circuit: cirq.Circuit, *coords: Tuple[int, int]) -> Any:
    """
    color cirq circuit SVG for given gates

    :param circuit:
    :param coords: integer coordinate which gate is colored
    :return:
    """
    import xml

    svg_str = SVGCircuit(circuit)._repr_svg_()
    DOMTree = xml.dom.minidom.parseString(svg_str)  # type: ignore
    xpos = []
    ypos = []
    for r in DOMTree.getElementsByTagName("rect"):  # [0].setAttribute("fill", "gray")

        xpos.append(int(float(r.getAttribute("x"))))
        ypos.append(int(float(r.getAttribute("y"))))
    xpos = sorted(list(set(xpos)))
    ypos = sorted(list(set(ypos)))
    # xpos_dict = dict(zip(range(len(xpos)), xpos))
    # ypos_dict = dict(zip(range(len(ypos)), ypos))
    i_xpos_dict = dict(zip(xpos, range(len(xpos))))
    i_ypos_dict = dict(zip(ypos, range(len(ypos))))
    for r in DOMTree.getElementsByTagName("rect"):
        x, y = int(float(r.getAttribute("x"))), int(float(r.getAttribute("y")))
        if (i_xpos_dict[x], i_ypos_dict[y]) in coords:
            r.setAttribute("fill", "gray")
    return DOMTree.toxml()


def repr2array(inputs: str) -> Array:
    """
    transform repr form of an array to real numpy array

    :param inputs:
    :return:
    """
    inputs = inputs.split("]")  # type: ignore
    inputs = [l.strip().strip("[") for l in inputs if l.strip()]  # type: ignore
    outputs = []
    for l in inputs:
        o = [float(c.strip()) for c in l.split(" ") if c.strip()]
        outputs.append(o)
    return np.array(outputs)
