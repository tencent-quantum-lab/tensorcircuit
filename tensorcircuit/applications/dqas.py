"""
Modules for DQAS framework
"""
# possibly deprecated, multiprocessing is not the recommended way to do DQAS task now, using vmap!

import sys
import inspect
from functools import partial
from multiprocessing import Pool, get_context
from typing import (
    List,
    Sequence,
    Any,
    Tuple,
    Callable,
    Iterator,
    Optional,
    Union,
    Dict,
)

import numpy as np
import scipy as sp
import tensorflow as tf


Array = Any  # np.array
Opt = Any  # tf.keras.optimizer
Model = Any  # tf.keras.Model
Tensor = Any
Graph = Any
thismodule = sys.modules[__name__]


_op_pool: Sequence[Any] = []


def set_op_pool(l: Sequence[Any]) -> None:
    # sometimes, to make parallel mode work, one should set_op_pool in global level of the script
    global _op_pool
    _op_pool = l


def get_op_pool() -> Sequence[Any]:
    global _op_pool
    return _op_pool


## infrastrcture for DQAS search


def get_var(name: str) -> Any:
    """
    Call in customized functions and grab variables within DQAS framework function by var name str.

    :param name: The DQAS framework function
    :type name: str
    :return: Variables within the DQAS framework
    :rtype: Any
    """
    return inspect.stack()[2][0].f_locals[name]


def verbose_output(max_prob: bool = True, weight: bool = True) -> None:
    """
    Doesn't support prob model DQAS search.

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
            "associating weights:",
            cand_weight,
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
    This function works only when nnp has the same shape as stp, i.e. one parameter for each op.

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
    p, _ = nnp.shape[0], nnp.shape[1]
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
    The kernel for multiprocess to run parallel in DQAS function/

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
    :param p_nnp: shape of circuit parameter pool, in general p_stp*l,
        where l is the max number of circuit parameters for op in the operator pool
    :param p_stp: the same as p in the most times
    :param batch: batch size of one epoch
    :param prethermal: prethermal update times
    :param epochs: training epochs
    :param parallel_num: parallel thread number, 0 to disable multiprocessing model by default
    :param verbose: set verbose log to print
    :param vebose_func: function to output verbose information
    :param history_func: function return intermiediate result for final history list
    :param prob_clip: cutoff probability to avoid peak distribution
    :param baseline_func: function accepting list of objective values and
        return the baseline value used in the next round
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
        _, gnnp = kernel_func(gdata, nnp, preset)
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
            # TODO(@referction-ray): more general repr
            if nnp.shape == stp.shape and verbose:
                cand_weight = get_weights(nnp, stp).numpy()
                print(
                    "And associating weights:",
                    cand_weight,
                )

            if history_func is not None:
                history.append(history_func())
        if parallel_num > 0:
            pool.close()
        return stp, nnp, history
        # TODO(@refraction-ray): history list trackings
    except KeyboardInterrupt:
        if parallel_num > 0:
            pool.close()
        return stp, nnp, history


## training based on DQAS


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
    # TODO(@refraction-ray): the best practice combine multiprocessing and random generator still
    # needs further investigation
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
        from .vags import qaoa_vag_energy

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

    _, nnp, h = search_func(
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
    return (get_weights_v2(nnp, preset=preset).numpy(), np.mean(h[-10:]))


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
        from .vags import qaoa_vag_energy

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


def evaluate_everyone(
    vag_func: Any,
    gdata: Iterator[Any],
    nnp: Tensor,
    presets: Sequence[Sequence[List[int]]],
    batch: int = 1,
) -> Sequence[Tuple[Tensor, Tensor]]:
    losses = []
    if not isinstance(nnp, tf.Tensor):
        nnp = tf.Variable(initial_value=nnp)

    for preset in presets:
        loss = 0
        for _, g in zip(range(batch), gdata):
            loss += vag_func(g, nnp, preset)[0]
        loss /= batch  # type: ignore
        losses.append((preset, loss.numpy()))  # type: ignore
    return losses


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
    prob_model: Model,
    batch_size: int,
    repetitions: Optional[List[int]] = None,
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
    loss_func: Optional[Callable[[Tensor], Tensor]] = None,
    loss_derivative_func: Optional[Callable[[Tensor], Tensor]] = None,
    validate_period: int = 0,
    validate_batch: int = 1,
    validate_func: Optional[
        Callable[[Any, Tensor, Sequence[int]], Tuple[Tensor, Tensor]]
    ] = None,
    vg: Optional[Iterator[Any]] = None,
) -> Tuple[Tensor, Tensor, Sequence[Any]]:
    """
    The probabilistic model based DQAS, can use extensively for DQAS case for ``NMF`` probabilistic model.

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
    :param loss_func: final loss function in terms of average of sub loss for each circuit
    :param loss_derivative_func: derivative function for ``loss_func``
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
    if vg is None:
        vg = void_generator()
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

    if loss_func is None:
        loss_func = lambda s: s

    if loss_derivative_func is None:
        loss_derivative_func = lambda s: tf.constant(1.0)

    history = []

    avcost1 = 0

    if prethermal > 0:
        presets, glnprobs = sample_func(prob_model, prethermal)
    for i, gdata in zip(range(prethermal), g):  # prethermal for nn param
        _, gnnp = kernel_func(gdata, nnp, presets[i])
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
                    # gnnp \equiv \partial L_i/\partial \theta
                    # batched_gnnp = sum_{i\in batch} \partial \mathcal{L}/\partial L_i \partial L_i/\partial \theta
                    # batched_gstp = \partial \mathcal{L}/\partial \bar{L} (\sum_i (L-\bar{L})\nabla \ln p)
                    # \partial \mathcal{L}/\partial L_i = \partial \mathcal{L}/\partial \bar{L} 1/n
                    deri_stp.append(
                        [
                            (tf.cast(loss, dtype=dtype) - tf.cast(avcost2, dtype=dtype))
                            * w
                            for w in glnprobs[i]
                        ]
                    )
                    deri_nnp.append(gnnp)
                    costl.append(loss.numpy())
                if validate_period != 0 and epoch % validate_period == 0:
                    accuracy = []
                    validate_presets, _ = sample_func(prob_model, validate_batch)
                    for i, gdata in zip(range(validate_batch), vg):
                        accuracy.append(validate_func(gdata, nnp, validate_presets[i]))  # type: ignore
                    print("accuracy on validation set:", np.mean(accuracy))

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
            loss_bar = tf.reduce_mean(costl)
            loss_bar_d = loss_derivative_func(
                loss_bar
            )  # \partial \mathcal{L} /\partial \bar{L}
            for i in range(len(glnprobs[0])):
                batched_gs.append(
                    loss_bar_d
                    * tf.math.reduce_mean(
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

            batched_gnnp = loss_bar_d * tf.math.reduce_mean(
                tf.convert_to_tensor(deri_nnp, dtype=dtype), axis=0
            )
            if verbose:
                print(
                    "final loss:",
                    loss_func(loss_bar),
                    " final loss derivative multiplier:",
                    loss_bar_d,
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
                    "\n network parameter: \n",
                    nnp.numpy(),
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

            if validate_period != 0 and (epoch + 1) % validate_period == 0:
                args_list = []
                validate_presets, _ = sample_func(prob_model, validate_batch)

                for i, gdata in zip(range(validate_batch), vg):
                    args_list.append((gdata, nnp, validate_presets[i].numpy()))
                # print(args_list)
                parallel_validation_result = pool.starmap(validate_func, args_list)  # type: ignore
                print("--------")
                if isinstance(parallel_validation_result[0], dict):
                    for kk in parallel_validation_result[0]:
                        print(
                            "%s on validation set:" % kk,
                            np.mean([p[kk] for p in parallel_validation_result]),
                        )
                else:
                    print(
                        "accuracy on validation set:",
                        np.mean(parallel_validation_result),
                    )

        if parallel_num > 0:
            pool.close()
        return prob_model, nnp, history
    except KeyboardInterrupt:
        if parallel_num > 0:
            pool.close()
        return prob_model, nnp, history
