from typing import Optional
import sys

import numpy as np
from tensornetwork.backend_contextmanager import get_default_backend
from tensornetwork.backends import backend_factory


modules = [
    "tensorcircuit",
    "tensorcircuit.cons",
    "tensorcircuit.gates",
    "tensorcircuit.circuit",
]


dtype = "complex64"
npdtype = np.complex64
backend = backend_factory.get_backend("numpy")
# these above lines are just for mypy, it is not very good at evaluating runtime object


def set_tensornetwork_backend(backend: Optional[str] = None) -> None:
    if not backend:
        backend = get_default_backend()
    backend_obj = backend_factory.get_backend(backend)
    for module in modules:
        if module in sys.modules:
            setattr(sys.modules[module], "backend", backend_obj)


set_backend = set_tensornetwork_backend


set_tensornetwork_backend()


def set_dtype(dtype: Optional[str] = None) -> None:
    if not dtype:
        dtype = "complex64"
    npdtype = getattr(np, dtype)
    for module in modules:
        if module in sys.modules:
            setattr(sys.modules[module], "npdtype", npdtype)


set_dtype()
