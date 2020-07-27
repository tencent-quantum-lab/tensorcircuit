"""
modules for DQAS, including kernels on various applications
"""

import sys
import inspect
from functools import lru_cache, partial
from multiprocessing import Pool
import numpy as np
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

thismodule = sys.modules[__name__]


_op_pool: Sequence[Any] = []


def set_op_pool(l: Sequence[Any]) -> None:
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
            f, f2 = ftuple # type: ignore
        else:
            f, f2, f3 = ftuple # type: ignore
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
        cset[j](ci, theta[i], g)

    state = ci.wavefunction()[0]
    losses = ave_func(state, g, *fs)
    return losses


def graph_forward(
    theta: Tensor,
    preset: Sequence[int],
    g: Graph,
    *fs: Tuple[Callable[[float], float], Callable[[Tensor], Tensor]],
) -> Sequence[Tensor]:
    n = len(g.nodes)
    ci = Circuit(n)
    cset = get_op_pool()  # [(Hlayer, nx.Graph), ...]

    for i, j in enumerate(preset):
        layer, graph = cset[j]
        layer(ci, theta[i], graph)

    state = ci.wavefunction()[0]
    losses = ave_func(state, g, *fs)  # objective graph
    return losses


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
        count += 1
        if sump + p[rs[count][0]] > percent:
            r[rs[count][0]] = (percent - sump) / (p[rs[count][0]]) * r[rs[count][0]]
            break
        else:
            sump += p[rs[count][0]]
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


qaoa_graph_vag = partial(qaoa_vag, forward_func=graph_forward)


def qaoa_block_vag(
    gdata: Graph,
    nnp: Tensor,
    preset: Sequence[int],
    f: Tuple[Callable[[float], float], Callable[[Tensor], Tensor]],
) -> Tuple[Tensor, Tensor]:
    # for universality, nnp always occupy two rows for each block

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


## infrastrcture for DQAS search


def get_var(name: str) -> Any:
    return inspect.stack()[2][0].f_locals[name]


def verbose_output(max_prob: bool = True) -> None:
    if max_prob:
        prob = get_var("prob")
        print("max probability for each layer:")
        print(np.max(prob.numpy(), axis=1))


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


def parallel_kernel(
    prob: Tensor,
    gdata: Any,
    nnp: Tensor,
    kernel_func: Callable[[Any, Tensor, Sequence[int]], Tuple[Tensor, Tensor]],
) -> Tuple[Tensor, Tensor, Tensor]:
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
        pool = Pool(parallel_num)
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
                raise ValueError("Please give the shape information on nnp")
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
                parallel_kernel = getattr(thismodule, "parallel_kernel")
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
                "\n new baseline: ",
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
    nnp_initial_value: Optional[Array] = None,
    opt: Optional[Opt] = None,
) -> Tuple[Array, float]:
    p = len(preset)
    c = len(get_op_pool())
    stp_train = np.zeros([p, c])
    for i, j in enumerate(preset):
        stp_train[i, j] = 10.0
    if nnp_initial_value is not None:
        nnp_initial_value = np.random.normal(loc=0.23, scale=0.8, size=[p, c])
    if vag_func is None:
        vag_func = qaoa_vag_energy
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

    stp, nnp, h = DQAS_search(
        vag_func,
        g=graph_g,
        p=p,
        batch=batch,
        prethermal=0,
        epochs=epochs,
        history_func=history_loss,
        nnp_initial_value=nnp_initial_value,
        stp_initial_value=stp_train,
        network_opt=opt,
    )
    return (get_weights(nnp, preset=preset).numpy(), h[-1])


def parallel_qaoa_train(
    preset: Sequence[int],
    g: Any,
    vag_func: Any = None,
    opt: Opt = None,
    epochs: int = 60,
    tries: int = 16,
    batch: int = 1,
    cores: int = 8,
    scale: float = 1.0,
) -> Sequence[Any]:

    if not opt:
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    p = len(preset)
    c = len(get_op_pool())
    glist = []
    for _ in range(epochs * batch):
        glist.append(g.send(None))  # pickle doesn't support generators even in dill
    if vag_func is None:
        vag_func = qaoa_vag_energy
    pool = Pool(cores)
    args_list = [
        (
            preset,
            glist,
            vag_func,
            epochs,
            batch,
            np.random.normal(loc=0.23, scale=scale, size=[p, c]),
            opt,
        )
        for _ in range(tries)
    ]
    result_list = pool.starmap(qaoa_simple_train, args_list)
    pool.close()
    result_list = sorted(result_list, key=lambda s: s[1])
    print(result_list)
    print("the optimal result is %s" % result_list[0][1])
    return result_list


## some utils


def color_svg(circuit: cirq.Circuit, *coords: Tuple[int, int]) -> Any:
    """
    color cirq circuit given gates

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
