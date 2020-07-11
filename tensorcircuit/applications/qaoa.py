"""
modules for QAOA application and its variants
layers and graphdata module are utilized
"""

import sys
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

from ..gates import array_to_tensor, num_to_tensor
from .. import cons

# don't directly import backend, as it is supposed to change at runtime
from ..circuit import Circuit
from .layers import *

Array = Any  # np.array
Opt = Any  # tf.keras.optimizer

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
    state: Array,
    g: Graph,
    *fs: Tuple[Callable[[float], float], Callable[[Tensor], Tensor]],
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
    for f, g in fs:
        r = [f(e) for e in ebasis]
        r = array_to_tensor(np.array(r))
        result.append(
            g(
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


def qas_vag_factory(
    f: Tuple[Callable[[float], float], Callable[[Tensor], Tensor]]
) -> Callable[[Graph, Tensor, Sequence[int]], Tuple[Tensor, Tensor]]:
    def qas_vag(
        gdata: Graph, nnp: Tensor, preset: Sequence[int]
    ) -> Tuple[Tensor, Tensor]:
        nnp = nnp.numpy()  # real
        pnnp = [nnp[i, j] for i, j in enumerate(preset)]
        pnnp = array_to_tensor(np.array(pnnp))  # complex
        with tf.GradientTape() as t:
            t.watch(pnnp)
            loss = exp_forward(pnnp, preset, gdata, f)
        gr = t.gradient(loss, pnnp)
        if gr is None:
            # if all gates in preset are not trainable, then gr returns None instead of 0s
            gr = tf.zeros_like(pnnp)
        gr = cons.backend.real(gr)

        gmatrix = np.zeros_like(nnp)
        for i, j in enumerate(preset):
            gmatrix[i, j] = gr[i]
        gmatrix = tf.constant(gmatrix)
        return loss[0], gmatrix

    return qas_vag


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
            print("Gibbs objective:", overlap_history[-1])
            print("mean energy:", mean_history[-1])
            print("overlap:", overlap_history[-1])
            print("trainable weights:", theta.numpy())
    return theta.numpy(), mean_history, gibbs_history, overlap_history


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
    if preset is None:
        preset = get_preset(stp)
    p = nnp.shape[0]
    ind_ = tf.stack([tf.cast(tf.range(p), tf.int32), tf.cast(preset, tf.int32)])
    return tf.gather_nd(nnp, tf.transpose(ind_))


_op_pool: Sequence[Any] = []


def set_op_pool(l: Sequence[Any]) -> None:
    global _op_pool
    _op_pool = l


def get_op_pool() -> Sequence[Any]:
    global _op_pool
    return _op_pool


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


def DQAS_search(
    g: Iterator[Any],
    p: int,
    op_pool: Sequence[Any],
    kernel_func: Callable[
        [Iterator[Any], Tensor, Sequence[int]], Tuple[Tensor, Tensor]
    ],
    *,
    batch: int = 300,
    prethermal: int = 100,
    epochs: int = 100,
    parallel_num: int = 0,
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
) -> Tuple[Tensor, Tensor]:
    dtype = tf.float32  # caution, simply changing this is not guranteed to work
    c = len(op_pool)
    set_op_pool(op_pool)
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
        nnp_initial_value = np.random.uniform(size=[p, c])
    if stp_initial_value is None:
        stp_initial_value = np.zeros([p, c])
    if baseline_func is None:
        baseline_func = np.mean
    nnp = tf.Variable(initial_value=nnp_initial_value, shape=[p, c], dtype=dtype)
    stp = tf.Variable(initial_value=stp_initial_value, shape=[p, c], dtype=dtype)

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

    print("network parameter after prethermalization: \n", nnp.numpy())

    try:
        for epoch in range(epochs):  # iteration to update strcuture param
            # for data spliting case, odd update network, even update structure
            prob = tf.math.exp(stp) / tf.tile(
                tf.math.reduce_sum(tf.math.exp(stp), axis=1)[:, tf.newaxis], [1, c]
            )
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
                print("stp_penalty_gradient:", stp_penalty_gradient.numpy())
            else:
                stp_penalty_gradient = 0.0
            if nnp_regularization is not None:
                nnp_penalty_gradient = nnp_regularization(stp, nnp)
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
            print("batched gradient of stp: \n", batched_gs.numpy())
            print("batched gradient of nnp: \n", batched_gnnp.numpy())

            network_opt.apply_gradients(
                zip([batched_gnnp + nnp_penalty_gradient], [nnp])
            )
            structure_opt.apply_gradients(
                zip([batched_gs + stp_penalty_gradient], [stp])
            )
            print(
                "strcuture parameter: \n",
                stp.numpy(),
                "\n network parameter: \n",
                nnp.numpy(),
            )
            cand_preset = get_preset(stp).numpy()
            cand_preset_repr = [op_pool[f].__repr__ for f in cand_preset]
            # TODO, more general repr
            cand_weight = get_weights(nnp, stp).numpy()
            print(
                "best candidates so far:",
                cand_preset_repr,
                "\n and associating weights:",
                cand_weight,
            )
        if parallel_num > 0:
            pool.close()
        return stp, nnp  # TODO: history list trackings
    except KeyboardInterrupt:
        if parallel_num > 0:
            pool.close()
        return stp, nnp
