"""
some cons and sets.
"""

from typing import Optional
import sys

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
]


dtypestr = "complex64"
npdtype = np.complex64
tfdtype = tf.complex64
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
    tfdtype = getattr(tf, dtype)
    for module in modules:
        if module in sys.modules:
            setattr(sys.modules[module], "dtypestr", dtype)
            setattr(sys.modules[module], "npdtype", npdtype)
            setattr(sys.modules[module], "tfdtype", tfdtype)


set_dtype()


def set_contractor(method: Optional[str] = None) -> None:
    if not method:
        method = "auto"
    cf = getattr(tn.contractors, method, None)
    if not cf:
        raise ValueError("Unknown contractor type: %s" % method)
    for module in modules:
        if module in sys.modules:
            setattr(sys.modules[module], "contractor", cf)


set_contractor()
