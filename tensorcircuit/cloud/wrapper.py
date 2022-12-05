"""
higher level wrapper shortcut for submit_task
"""
from typing import Any, Optional, Sequence

import numpy as np

from ..circuit import Circuit
from ..results import counts
from .apis import submit_task, get_device
from .abstraction import Device

Tensor = Any


def sample_expectation_ps(
    c: Circuit,
    x: Optional[Sequence[int]] = None,
    y: Optional[Sequence[int]] = None,
    z: Optional[Sequence[int]] = None,
    shots: int = 1024,
    device: Optional[Device] = None,
    **kws: Any,
) -> float:
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
    return counts.correlation(raw_counts, x + y + z)


# TODO(@refraction-ray): batch support
# def batch_sample_expectation_ps(c, pss, ws, device, shots)
