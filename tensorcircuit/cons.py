import numpy as np
from tensornetwork.backend_contextmanager import get_default_backend
from tensornetwork.backends import backend_factory

dtype = "complex64"
npdtype = getattr(np, dtype)

b = get_default_backend()
backend = backend_factory.get_backend(b)
