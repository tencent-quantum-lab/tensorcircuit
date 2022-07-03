"""
Interfaces bridging different backends
"""

from . import tensortrans
from .tensortrans import numpy_args_to_backend, general_args_to_numpy
from .scipy import scipy_interface, scipy_optimize_interface
from .torch import torch_interface, pytorch_interface
from .tensorflow import tensorflow_interface, tf_interface
