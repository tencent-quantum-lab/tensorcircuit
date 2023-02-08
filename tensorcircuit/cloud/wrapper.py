"""
higher level wrapper shortcut for submit_task
"""
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from ..circuit import Circuit
from ..results import counts
from ..results.readout_mitigation import ReadoutMit
from ..utils import is_sequence
from ..compiler.qiskit_compiler import qiskit_compile
from .apis import submit_task, get_device
from .abstraction import Device

Tensor = Any


def batch_submit_template(device: str) -> Callable[..., List[counts.ct]]:
    # TODO(@refraction-ray): fixed when batch submission really works
    def run(cs: Union[Circuit, Sequence[Circuit]], shots: int) -> List[counts.ct]:
        """
        batch circuit running alternative
        """
        single = False
        if not is_sequence(cs):
            cs = [cs]  # type: ignore
            single = True
        ts = []
        # for c in cs:  # type: ignore
        #     ts.append(submit_task(circuit=c, shots=shots, device=device))
        #     time.sleep(0.3)
        ts = submit_task(circuit=cs, shots=shots, device=device)
        l = [t.results(blocked=True) for t in ts]  # type: ignore
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
    # deprecated
    # TODO(@refraction-ray): integrated error mitigation
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


def batch_sample_expectation_ps(
    c: Circuit,
    device: Any,
    pss: List[List[int]],
    ws: Optional[List[float]] = None,
    shots: int = 8192,
    with_rem: bool = True,
) -> float:
    cs = []
    infos = []
    exps = []
    if isinstance(device, str):
        device = get_device(device)
    for ps in pss:
        # TODO(@refraction-ray): Pauli string grouping
        c1 = Circuit.from_qir(c.to_qir())
        exp = []
        for j, i in enumerate(ps):
            if i == 1:
                c1.H(i)  # type: ignore
                exp.append(j)
            elif i == 2:
                c1.rx(i, theta=np.pi / 2)  # type: ignore
                exp.append(j)
            elif i == 3:
                exp.append(j)
        c1, info = qiskit_compile(
            c1,
            compiled_options={
                "basis_gates": device.native_gates(),
                "optimization_level": 3,
                "coupling_map": device.topology(),
            },
        )
        cs.append(c1)
        infos.append(info)
        exps.append(exp)
    if ws is None:
        ws = [1.0 for _ in range(len(pss))]

    def run(cs: List[Any], shots: int) -> List[Dict[str, int]]:
        ts = submit_task(
            circuit=cs,
            device=device,
            shots=shots,
            enable_qos_qubit_mapping=False,
            enable_qos_gate_decomposition=False,
        )
        if not is_sequence(ts):
            ts = [ts]
        raw_counts = [t.results(blocked=True) for t in ts]
        return raw_counts

    raw_counts = run(cs, shots)

    if with_rem:
        if getattr(device, "readout_mit", None) is None:
            mit = ReadoutMit(run)
            # TODO(@refraction-ray) only work for tencent provider
            mit.cals_from_system(device.list_properties()["qubits"], shots=shots)
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
    sumr = sum([w * r for w, r in zip(ws, results)])
    return sumr
