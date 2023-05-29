"""
higher level wrapper shortcut for submit_task
"""
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import logging
import time

import numpy as np

from ..circuit import Circuit
from ..results import counts
from ..results.readout_mitigation import ReadoutMit
from ..utils import is_sequence
from ..cons import backend
from ..quantum import ps2xyz
from ..compiler import DefaultCompiler
from ..compiler.simple_compiler import simple_compile
from .apis import submit_task, get_device
from .abstraction import Device


logger = logging.getLogger(__name__)
Tensor = Any


def batch_submit_template(
    device: str, batch_limit: int = 64, **kws: Any
) -> Callable[..., List[counts.ct]]:
    def run(
        cs: Union[Circuit, Sequence[Circuit]], shots: int = 8192, **nkws: Any
    ) -> List[counts.ct]:
        """
        batch circuit running alternative
        """
        single = False
        if not is_sequence(cs):
            cs = [cs]  # type: ignore
            single = True
        # for c in cs:  # type: ignore
        #     ts.append(submit_task(circuit=c, shots=shots, device=device))
        #     time.sleep(0.3)
        kws.update(nkws)
        if len(cs) <= batch_limit:  # type: ignore
            css = [cs]
        else:
            ntimes = len(cs) // batch_limit  # type: ignore
            if ntimes * batch_limit == len(cs):  # type: ignore
                css = [
                    cs[i * batch_limit : (i + 1) * batch_limit] for i in range(ntimes)  # type: ignore
                ]
            else:
                css = [
                    cs[i * batch_limit : (i + 1) * batch_limit] for i in range(ntimes)  # type: ignore
                ] + [
                    cs[ntimes * batch_limit :]  # type: ignore
                ]
        tss = []
        logger.info(f"submit task on {device} for {len(cs)} circuits")  # type: ignore
        time0 = time.time()
        for i, cssi in enumerate(css):
            tss += submit_task(circuit=cssi, shots=shots, device=device, **kws)
            if i < len(css) - 1:
                time.sleep(1.5)
            # TODO(@refraction-ray) whether the sleep time is enough for tquk?
            # incase duplicae request error protection
        l = [t.results() for t in tss]
        time1 = time.time()
        logger.info(
            f"finished collecting count results of {len(cs)} tasks in {round(time1-time0, 4)} seconds"  # type: ignore
        )
        if single is False:
            return l
        return l[0]  # type: ignore

    return run


def sample_expectation_ps(
    c: Circuit,
    x: Optional[Sequence[int]] = None,
    y: Optional[Sequence[int]] = None,
    z: Optional[Sequence[int]] = None,
    shots: int = 1024,
    device: Optional[Device] = None,
    **kws: Any,
) -> float:
    """
    Deprecated, please use :py:meth:`tensorcircuit.cloud.wrapper.batch_expectation_ps`.
    """
    logger.warning(
        "This method is deprecated and not maintained, \
        please use `tensorcircuit.cloud.wrapper.batch_expectation_ps` instead"
    )
    c1 = Circuit.from_qir(c.to_qir())
    if x is None:
        x = []
    if y is None:
        y = []
    if z is None:
        z = []
    for i in x:
        c1.H(i)  # type: ignore
    for i in y:
        c1.rx(i, theta=np.pi / 2)  # type: ignore
    if device is None:
        device = get_device()
    t = submit_task(circuit=c1, device=device, shots=shots)
    raw_counts = t.results(blocked=True)  # type: ignore
    x, y, z = list(x), list(y), list(z)
    return counts.expectation(raw_counts, x + y + z)


def reduce_and_evaluate(
    cs: List[Circuit], shots: int, run: Callable[..., Any]
) -> List[counts.ct]:
    reduced_cs = []
    reduced_dict = {}
    recover_dict = {}
    # merge the same circuit
    for j, c in enumerate(cs):
        key = hash(c.to_openqasm())
        if key not in reduced_dict:
            reduced_dict[key] = [j]
            reduced_cs.append(c)
            recover_dict[key] = len(reduced_cs) - 1
        else:
            reduced_dict[key].append(j)

    # for j, ps in enumerate(pss):
    #     ps = [i if i in [1, 2] else 0 for i in ps]
    #     if tuple(ps) not in reduced_dict:
    #         reduced_dict[tuple(ps)] = [j]
    #         reduced_cs.append(cs[j])
    #         recover_dict[tuple(ps)] = len(reduced_cs) - 1
    #     else:
    #         reduced_dict[tuple(ps)].append(j)

    reduced_raw_counts = run(reduced_cs, shots)
    raw_counts: List[Dict[str, int]] = [None] * len(cs)  # type: ignore
    for i, c in enumerate(cs):
        key = hash(c.to_openqasm())
        raw_counts[i] = reduced_raw_counts[recover_dict[key]]
    return raw_counts


def batch_expectation_ps(
    c: Circuit,
    pss: List[List[int]],
    device: Any = None,
    ws: Optional[List[float]] = None,
    shots: int = 8192,
    with_rem: bool = True,
    compile_func: Optional[Callable[[Circuit], Tuple[Circuit, Dict[str, Any]]]] = None,
    batch_limit: int = 64,
    batch_submit_func: Optional[Callable[..., List[counts.ct]]] = None,
) -> Union[Any, List[Any]]:
    """
    Unified interface to compute the Pauli string expectation lists or sums via simulation or on real qpu.
    Error mitigation, circuit compilation and Pauli string grouping are all built-in.

    One line access to unlock the whole power or real quantum hardware on quantum cloud.

    :Example:

    .. code-block:: python

        c = tc.Circuit(2)
        c.h(0)
        c.x(1)
        tc.cloud.wrapper.batch_expectation_ps(c, [[1, 0], [0, 3]], device=None)
        # array([ 0.99999994, -0.99999994], dtype=float32)
        tc.cloud.wrapper.batch_expectation_ps(c, [[1, 0], [0, 3]], device="tencent::9gmon")
        # array([ 1.03093477, -1.11715944])

    :param c: The target circuit to compute expectation
    :type c: Circuit
    :param pss: List of Pauli string list, eg. [[0, 1, 0], [2, 3, 3]] represents [X1, Y0Z1Z2].
    :type pss: List[List[int]]
    :param device: The device str or object for quantum cloud module,
        defaults to None, None is for analytical exact simulation
    :type device: Any, optional
    :param ws: List of float to indicate the final return is the weighted sum of Pauli string expectations,
        e.g. [2., -0.3] represents the final results is 2* ``pss`` [0]-0.3* ``pss`` [1]
        defaults to None, None indicate the list of expectations for ``pss`` are all returned
    :type ws: Optional[List[float]], optional
    :param shots: measurement shots for each expectation estimation, defaults to 8192
    :type shots: int, optional
    :param with_rem: whether enable readout error mitigation for the result, defaults to True
    :type with_rem: bool, optional
    :return: List of Pauli string expectation or a weighted sum float for Pauli strings, depending on ``ws``
    :rtype: Union[Any, List[Any]]
    """
    if device is None:
        results = []
        for ps in pss:
            results.append(c.expectation_ps(**ps2xyz(ps)))  # type: ignore
        if ws is None:
            return backend.real(backend.stack(results))
        else:
            sumr = sum([w * r for w, r in zip(ws, results)])
            return backend.convert_to_tensor(sumr)
    cs = []
    infos = []
    exps = []
    if isinstance(device, str):
        device = get_device(device)

    if compile_func is None:
        try:
            coupling_map = device.topology()
            compile_func = DefaultCompiler(
                {
                    "coupling_map": coupling_map,
                }
            )
        except (AttributeError, ValueError):
            compile_func = DefaultCompiler()
    c1, info = compile_func(c)  # type: ignore
    if not info.get("logical_physical_mapping", None):
        info["logical_physical_mapping"] = {i: i for i in range(c._nqubits)}
    for ps in pss:
        # TODO(@refraction-ray): Pauli string grouping
        # https://docs.pennylane.ai/en/stable/_modules/pennylane/pauli/grouping/group_observables.html
        c2 = Circuit.from_qir(c1.to_qir())
        exp = []
        for j, i in enumerate(ps):
            if i == 1:
                c2.H(info["logical_physical_mapping"][j])  # type: ignore
                c2, _ = simple_compile(c2)
                exp.append(j)

            elif i == 2:
                c2.rx(info["logical_physical_mapping"][j], theta=np.pi / 2)  # type: ignore
                c2, _ = simple_compile(c2)
                exp.append(j)
            elif i == 3:
                exp.append(j)
        for i in range(c._nqubits):
            c2.measure_instruction(info["logical_physical_mapping"][i])
        # c1, info = compile_func(c1)  # type: ignore
        # TODO(@refraction-ray): two steps compiling with pre compilation:
        # basically done, require some fine tuning for performance
        cs.append(c2)
        infos.append(info)
        exps.append(exp)

    # reduced_cs = []
    # reduced_dict = {}
    # recover_dict = {}
    # # merge the same circuit
    # for j, ps in enumerate(pss):
    #     ps = [i if i in [1, 2] else 0 for i in ps]
    #     if tuple(ps) not in reduced_dict:
    #         reduced_dict[tuple(ps)] = [j]
    #         reduced_cs.append(cs[j])
    #         recover_dict[tuple(ps)] = len(reduced_cs) - 1
    #     else:
    #         reduced_dict[tuple(ps)].append(j)

    if batch_submit_func is None:
        run = batch_submit_template(
            device,
            batch_limit,
            enable_qos_qubit_mapping=False,
            enable_qos_gate_decomposition=False,
        )
    else:
        run = batch_submit_func

    raw_counts = reduce_and_evaluate(cs, shots, run)  # type: ignore

    # def run(cs: List[Any], shots: int) -> List[Dict[str, int]]:
    #     logger.info(f"submit task on {device.name} for {len(cs)} circuits")
    #     time0 = time.time()
    #     ts = submit_task(
    #         circuit=cs,
    #         device=device,
    #         shots=shots,
    #         enable_qos_qubit_mapping=False,
    #         enable_qos_gate_decomposition=False,
    #     )
    #     if not is_sequence(ts):
    #         ts = [ts]  # type: ignore
    #     raw_counts = [t.results(blocked=True) for t in ts]
    #     time1 = time.time()
    #     logger.info(
    #         f"finished collecting count results of {len(cs)} tasks in {round(time1-time0, 4)} seconds"
    #     )
    #     return raw_counts

    # reduced_raw_counts = run(reduced_cs, shots)
    # raw_counts: List[Dict[str, int]] = [None] * len(cs)  # type: ignore
    # for i in range(len(cs)):
    #     ps = [i if i in [1, 2] else 0 for i in pss[i]]
    #     raw_counts[i] = reduced_raw_counts[recover_dict[tuple(ps)]]

    if with_rem:
        if getattr(device, "readout_mit", None) is None:
            mit = ReadoutMit(run)
            # TODO(@refraction-ray) only work for tencent provider
            nq = device.list_properties().get("qubits", None)
            if nq is None:
                nq = c._nqubits
            mit.cals_from_system(nq, shots=shots)
            device.readout_mit = mit
        else:
            mit = device.readout_mit

        results = [
            mit.expectation(raw_counts[i], exps[i], **infos[i])
            for i in range(len(raw_counts))
        ]
    else:
        results = [
            counts.expectation(raw_counts[i], exps[i]) for i in range(len(raw_counts))
        ]
    if ws is not None:
        sumr = sum([w * r for w, r in zip(ws, results)])
        return backend.convert_to_tensor(sumr)
    return backend.stack(results)
