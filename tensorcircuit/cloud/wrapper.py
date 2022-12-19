"""
higher level wrapper shortcut for submit_task
"""
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np

from ..circuit import Circuit
from ..results import counts
from ..utils import is_sequence
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


# TODO(@refraction-ray): batch support
# def batch_sample_expectation_ps(c, pss, ws, device, shots)
