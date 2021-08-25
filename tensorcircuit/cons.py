"""
some cons and sets.
"""

from typing import Optional, Any
import sys
from functools import partial

import numpy as np
import tensorflow as tf
import tensornetwork as tn
from tensornetwork.backend_contextmanager import get_default_backend

from .backends import get_backend


modules = [
    "tensorcircuit",
    "tensorcircuit.cons",
    "tensorcircuit.gates",
    "tensorcircuit.circuit",
    "tensorcircuit.backends",
    "tensorcircuit.densitymatrix",
    "tensorcircuit.channels",
]

dtypestr = "complex64"
npdtype = np.complex64
backend = get_backend("numpy")
contractor = tn.contractors.auto
# these above lines are just for mypy, it is not very good at evaluating runtime object


def set_tensornetwork_backend(backend: Optional[str] = None) -> None:
    """
    set the runtime backend of tensornetwork

    :param backend: numpy, tensorflow, jax, pytorch
    :return:
    """
    if not backend:
        backend = get_default_backend()
    backend_obj = get_backend(backend)
    for module in modules:
        if module in sys.modules:
            setattr(sys.modules[module], "backend", backend_obj)
    tn.set_default_backend(backend)


set_backend = set_tensornetwork_backend


set_tensornetwork_backend()


def set_dtype(dtype: Optional[str] = None) -> None:
    if not dtype:
        dtype = "complex64"
    npdtype = getattr(np, dtype)
    for module in modules:
        if module in sys.modules:
            setattr(sys.modules[module], "dtypestr", dtype)
            setattr(sys.modules[module], "npdtype", npdtype)
    from .gates import meta_gate

    meta_gate()


set_dtype()


def set_contractor(
    method: Optional[str] = None, optimizer: Optional[Any] = None
) -> None:
    if not method:
        method = "auto"
    cf = getattr(tn.contractors, method, None)
    if not cf:
        raise ValueError("Unknown contractor type: %s" % method)
    if method == "custom":
        cf = partial(cf, optimizer=optimizer)
    for module in modules:
        if module in sys.modules:
            setattr(sys.modules[module], "contractor", cf)


set_contractor()


def set_seed() -> None:
    try:
        global_r = tf.random.Generator.from_non_deterministic_state()
    except AttributeError:
        global_r = (
            tf.random.experimental.Generator.from_non_deterministic_state()
        )  # early version of tf2
    for module in modules:
        if module in sys.modules:
            setattr(sys.modules[module], "global_r", global_r)


set_seed()
