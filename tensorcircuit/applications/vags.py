"""
DQAS application kernels as vag functions
"""
from functools import lru_cache, partial, reduce
import logging
import operator
from typing import (
    List,
    Sequence,
    Any,
    Tuple,
    Callable,
    Iterator,
    Optional,
    Union,
)

import networkx as nx
import numpy as np
import tensorflow as tf

from ..gates import array_to_tensor
from .. import cons
from .. import gates as G
from ..circuit import Circuit
from ..densitymatrix import DMCircuit

from .layers import generate_qubits
from .dqas import get_op_pool

logger = logging.getLogger(__name__)

try:
    import cirq
    import sympy
    import tensorflow_quantum as tfq

except ImportError as e:
    logger.warning(e)
    logger.warning("Therefore some functionality in %s may not work" % __name__)


Array = Any  # np.array
Opt = Any  # tf.keras.optimizer
Model = Any  # tf.keras.Model
Tensor = Any
Graph = Any


## GHZ circuit application


def GHZ_vag(
    gdata: Any, nnp: Tensor, preset: Sequence[int], verbose: bool = False, n: int = 3
) -> Tuple[Tensor, Tensor]:
    # gdata = None
    reference_state = np.zeros([2**n])
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
        s = tf.reshape(
            s,
            [2**n],
        )
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

    state = ci.wavefunction()
    losses = ave_func(state, g, *fs)
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
    verbose_fs: Optional[
        Sequence[Tuple[Callable[[float], float], Callable[[Tensor], Tensor]]]
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
    if verbose_fs:
        for vf in verbose_fs:
            print(forward_func(pnnp, preset, gdata, vf))  # type: ignore
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


qaoa_vag_energy = partial(qaoa_vag, f=(_identity, _neg))
qaoa_block_vag_energy = partial(qaoa_block_vag, f=(_identity, _neg))


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
    *deprecated*

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


## noise qaoa tensorcircuits setup


def noise_forward(
    theta: Tensor,
    preset: Sequence[int],
    g: Graph,
    measure_func: Callable[[DMCircuit, Graph], Tensor],
    is_mc: bool = False,
) -> Tensor:
    n = len(g.nodes)
    if is_mc is True:
        ci = Circuit(n)
    else:
        ci = DMCircuit(n)  # type: ignore
    cset = get_op_pool()
    for i, j in enumerate(preset):
        if len(cset[j]) == 3:  # (noiselayer, graph, [0.2])
            layer, graph, params = cset[j]
            layer(ci, theta[i], graph, *params)
        elif len(cset[j]) == 4:  # (rxlayer, graph, noiselayer, [0.1])
            layer, graph, noisemodel, params = cset[j]
            layer(ci, theta[i], graph)
            noisemodel(ci, g, *params)
        elif len(cset[j]) == 2:  # (noiselayer, [0.2]) no noise
            layer, params = cset[j]
            layer(ci, theta[i], g, *params)
        else:  # len == 1
            cset[j][0](ci, theta[i], g)

    loss = measure_func(ci, g)  # type: ignore
    return loss


def maxcut_measurements_tc(c: Union[Circuit, DMCircuit], g: Graph) -> Tensor:
    loss = 0.0
    for e in g.edges:
        loss += (
            g[e[0]][e[1]]["weight"]
            * 0.5
            * (c.expectation((G.z(), [e[0]]), (G.z(), [e[1]])) - 1)  # type: ignore
        )
    return loss


def tfim_measurements_tc(
    c: Union[Circuit, DMCircuit],
    g: Graph,
    hzz: float = 1.0,
    hx: float = 0.0,
    hz: float = 0.0,
) -> Tensor:
    loss = 0.0
    for e in g.edges:
        loss += (
            g[e[0]][e[1]]["weight"]
            * hzz
            * c.expectation((G.z(), [e[0]]), (G.z(), [e[1]]))  # type: ignore
        )
    if hx != 0.0:
        for i in range(len(g.nodes)):
            loss += hx * c.expectation((G.x(), [i]))  # type: ignore
    if hz != 0.0:
        for i in range(len(g.nodes)):
            loss += hz * c.expectation((G.z(), [i]))  # type: ignore
    return loss


def heisenberg_measurements_tc(
    c: Union[Circuit, DMCircuit],
    g: Graph,
    hzz: float = 1.0,
    hxx: float = 1.0,
    hyy: float = 1.0,
    hz: float = 0.0,
    hx: float = 0.0,
    hy: float = 0.0,
    reuse: bool = True,
) -> Tensor:
    loss = 0.0
    for e in g.edges:
        loss += (
            g[e[0]][e[1]]["weight"]
            * hzz
            * c.expectation((G.z(), [e[0]]), (G.z(), [e[1]]), reuse=reuse)  # type: ignore
        )
        loss += (
            g[e[0]][e[1]]["weight"]
            * hyy
            * c.expectation((G.y(), [e[0]]), (G.y(), [e[1]]), reuse=reuse)  # type: ignore
        )
        loss += (
            g[e[0]][e[1]]["weight"]
            * hxx
            * c.expectation((G.x(), [e[0]]), (G.x(), [e[1]]), reuse=reuse)  # type: ignore
        )
    if hx != 0:
        for i in range(len(g.nodes)):
            loss += hx * c.expectation((G.x(), [i]), reuse=reuse)  # type: ignore
    if hy != 0:
        for i in range(len(g.nodes)):
            loss += hy * c.expectation((G.y(), [i]), reuse=reuse)  # type: ignore
    if hz != 0:
        for i in range(len(g.nodes)):
            loss += hz * c.expectation((G.z(), [i]), reuse=reuse)  # type: ignore
    return loss


def qaoa_noise_vag(
    gdata: Graph,
    nnp: Tensor,
    preset: Sequence[int],
    measure_func: Optional[Callable[[DMCircuit, Graph], Tensor]] = None,
    forward_func: Optional[
        Callable[
            [Tensor, Sequence[int], Graph, Callable[[DMCircuit, Graph], Tensor]],
            Tensor,
        ]
    ] = None,
    **kws: Any,
) -> Tuple[Tensor, Tensor]:
    if measure_func is None:
        measure_func = maxcut_measurements_tc
    if forward_func is None:
        forward_func = noise_forward
    nnp = nnp.numpy()  # real
    pnnp = [nnp[i, j] for i, j in enumerate(preset)]
    pnnp = array_to_tensor(np.array(pnnp))  # complex
    with tf.GradientTape() as t:
        t.watch(pnnp)
        loss = forward_func(pnnp, preset, gdata, measure_func, **kws)
        loss = cons.backend.real(loss)
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
    # print("loss", loss, "g\n", gmatrix)
    return loss, gmatrix


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
    ## depracated qaoa train

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


def compose_tc_circuit_with_multiple_pools(
    theta: Tensor,
    preset: Sequence[int],
    g: Graph,
    pool_choice: Sequence[int],
    cset: Optional[Sequence[Any]] = None,
    measure_func: Optional[Callable[[DMCircuit, Graph], Tensor]] = None,
) -> Circuit:
    n = len(g.nodes)
    ci = Circuit(n)
    if cset is None:
        cset = get_op_pool()
    mem = 0
    for i, j in enumerate(preset):
        ele = cset[pool_choice[i]][j]
        if isinstance(ele, tuple) or isinstance(ele, list):
            gate = ele[0]
            index = ele[1]
        else:
            index = [mem % n]
            gate = ele
            mem += 1
        if gate.lower() in ["cnot"]:
            getattr(ci, gate)(*index)
        else:
            getattr(ci, gate)(*index, theta=theta[i])
    return ci


def gatewise_vqe_vag(
    gdata: Graph,
    nnp: Tensor,
    preset: Sequence[int],
    pool_choice: Sequence[int],
    measure_func: Optional[Callable[[Union[Circuit, DMCircuit], Graph], Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    nnp = nnp.numpy()  # real
    pnnp = []
    cset = get_op_pool()
    if measure_func is None:
        measure_func = maxcut_measurements_tc
    for i, j in enumerate(preset):
        k = pool_choice[i]
        if j >= len(cset[k]):
            j = len(cset[k]) - 1
            preset[i] = j  # type: ignore
        pnnp.append(nnp[i, j])
    pnnp = array_to_tensor(np.array(pnnp))  # complex
    with tf.GradientTape() as tape:
        tape.watch(pnnp)
        ci = compose_tc_circuit_with_multiple_pools(pnnp, preset, gdata, pool_choice)
        loss = measure_func(ci, gdata)
        loss = cons.backend.real(loss)
    gr = tape.gradient(loss, pnnp)
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


## functions for quantum Hamiltonian QAOA with tensorflow quantum backend
## deprecated tfq related vags

try:
    v = sympy.symbols("v_{0:64}")
    vv = sympy.symbols(["v_" + str(i) + "_0:32" for i in range(32)])

    # symbol pool
    def double_qubits_initial() -> Iterator[Sequence[Any]]:
        while True:
            yield [
                cirq.Circuit(
                    [
                        cirq.rx(0.0)(cirq.GridQubit(0, 0)),
                        cirq.rx(0.0)(cirq.GridQubit(1, 0)),
                    ]
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
        for j in preset:
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
            circuit.append(
                cirq.H(q(i))
            )  # the first block only final H layer has effect
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
                hzz
                * g[e[0]][e[1]]["weight"]
                * cirq.Z(qubits[e[0]])
                * cirq.Z(qubits[e[1]])
            )

        if hx != 0:
            for i in range(len(g.nodes)):
                measurements.append(hx * cirq.X(qubits[i]))
        if hz != 0:
            for i in range(len(g.nodes)):
                measurements.append(hz * cirq.Z(qubits[i]))
        if one:
            measurements = reduce(operator.add, measurements)
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
                hzz
                * g[e[0]][e[1]]["weight"]
                * cirq.Z(qubits[e[0]])
                * cirq.Z(qubits[e[1]])
            )
            measurements.append(
                hxx
                * g[e[0]][e[1]]["weight"]
                * cirq.X(qubits[e[0]])
                * cirq.X(qubits[e[1]])
            )
            measurements.append(
                hyy
                * g[e[0]][e[1]]["weight"]
                * cirq.Y(qubits[e[0]])
                * cirq.Y(qubits[e[1]])
            )
        if one:
            measurements = reduce(operator.add, measurements)
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
        # p, c, l = nnp.shape[0], nnp.shape[1], nnp.shape[2]
        p, l = nnp.shape[0], nnp.shape[2]
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

except NameError as e:
    logger.warning(e)
    logger.warning("tfq related vags disabled due to missing packages")

## some quantum quantities using tf ops
## deprecated, see new functions implemented in backend agnostic way
## in ``tc.quantum`` module


def entropy(rho: Tensor, eps: float = 1e-12) -> Tensor:
    """
    deprecated, current version in tc.quantum
    """
    lbd = tf.math.real(tf.linalg.eigvals(rho))
    entropy = -tf.math.reduce_sum(lbd * tf.math.log(lbd + eps))
    return tf.math.real(entropy)
    # e = np.linalg.eigvals(rho)
    # eps = 1e-12
    # return -np.real(np.dot(e, np.log(e + eps)))
    # return np.trace(rho@LA.logm(rho))


def renyi_entropy(rho: Tensor, k: int = 2, eps: float = 1e-12) -> Tensor:
    # no matrix power in tf?
    rhok = rho
    for _ in range(k - 1):
        rhok = rhok @ rho
    return 1 / (1 - k) * tf.math.real(tf.linalg.trace(rhok))


def reduced_density_matrix(
    state: Tensor, freedom: int, cut: Union[int, List[int]], p: Optional[Tensor] = None
) -> Tensor:
    """
    deprecated, current version in tc.quantum
    """
    if isinstance(cut, list) or isinstance(cut, tuple):
        traceout = cut
    else:
        traceout = [i for i in range(cut)]
    w = state / tf.linalg.norm(state)
    perm = [i for i in range(freedom) if i not in traceout]
    perm = perm + traceout
    w = tf.reshape(w, [2 for _ in range(freedom)])
    w = tf.transpose(w, perm=perm)
    w = tf.reshape(w, [-1, 2 ** len(traceout)])
    if p is None:
        rho = w @ tf.transpose(w, conjugate=True)
    else:
        rho = w @ tf.linalg.diag(p) @ tf.transpose(w, conjugate=True)
        rho /= tf.linalg.trace(rho)
    return rho


def entanglement_entropy(state: Tensor) -> Tensor:
    """
    deprecated as non tf and non flexible, use the combination of ``reduced_density_matrix`` and ``entropy`` instead.
    """
    state = state.reshape([-1])
    state = state / np.linalg.norm(state)
    t = state.shape[0]
    ht = int(np.sqrt(t))
    square = state.reshape([ht, ht])
    rho = square @ np.conj(np.transpose(square))
    return entropy(rho)


def free_energy(rho: Tensor, h: Tensor, beta: float = 1, eps: float = 1e-12) -> Tensor:
    energy = tf.math.real(tf.linalg.trace(rho @ h))
    s = entropy(rho, eps)
    return tf.math.real(energy - s / beta)


def renyi_free_energy(rho: Tensor, h: Tensor, beta: float = 1) -> Tensor:
    energy = tf.math.real(tf.linalg.trace(rho @ h))
    s = -tf.math.real(tf.math.log(tf.linalg.trace(rho @ rho)))
    return tf.math.real(energy - s / beta)


def taylorlnm(x: Tensor, k: int) -> Tensor:
    dtype = x.dtype
    s = x.shape[-1]
    y = 1 / k * (-1) ** (k + 1) * tf.eye(s, dtype=dtype)
    for i in reversed(range(k)):
        y = y @ x
        if i > 0:
            y += 1 / (i) * (-1) ** (i + 1) * tf.eye(s, dtype=dtype)
    return y


def truncated_free_energy(
    rho: Tensor, h: Tensor, beta: float = 1, k: int = 2, eps: float = 1e-12
) -> Tensor:
    dtype = rho.dtype
    s = rho.shape[-1]
    tyexpand = rho @ taylorlnm(rho - tf.eye(s, dtype=dtype), k - 1)
    renyi = -tf.math.real(tf.linalg.trace(tyexpand))
    energy = tf.math.real(tf.linalg.trace(rho @ h))
    print(energy, renyi, renyi / beta)
    return tf.math.real(energy - renyi / beta)


def trace_distance(rho: Tensor, rho0: Tensor, eps: float = 1e-12) -> Tensor:
    d2 = rho - rho0
    d2 = tf.transpose(d2, conjugate=True) @ d2
    lbds = tf.math.real(tf.linalg.eigvals(d2))
    return 0.5 * tf.reduce_sum(tf.sqrt(lbds + eps))


def fidelity(rho: Tensor, rho0: Tensor) -> Tensor:
    rhosqrt = tf.linalg.sqrtm(rho)
    return tf.math.real(tf.linalg.trace(tf.linalg.sqrtm(rhosqrt @ rho0 @ rhosqrt)) ** 2)


def gibbs_state(h: Tensor, beta: float = 1) -> Tensor:
    rho = tf.linalg.expm(-beta * h)
    rho /= tf.linalg.trace(rho)
    return rho


def double_state(h: Tensor, beta: float = 1) -> Tensor:
    rho = tf.linalg.expm(-beta / 2 * h)
    state = tf.reshape(rho, [-1])
    norm = tf.linalg.norm(state)
    return state / norm


def correlation(m: Tensor, rho: Tensor) -> Tensor:
    return tf.math.real(tf.linalg.trace(rho @ m))
