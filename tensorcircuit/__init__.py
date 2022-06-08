__version__ = "0.1.2"
__author__ = "TensorCircuit Authors"
__creator__ = "refraction-ray"

from .cons import (
    set_backend,
    set_dtype,
    set_contractor,
    set_function_backend,
    set_function_dtype,
    set_function_contractor,
    runtime_backend,
    runtime_dtype,
    runtime_contractor,
)  # prerun of set hooks
from . import gates
from .circuit import Circuit, expectation
from .mpscircuit import MPSCircuit
from .densitymatrix import DMCircuit as DMCircuit_reference
from .densitymatrix2 import DMCircuit2

DMCircuit = DMCircuit2  # compatibility issue to still expose DMCircuit2
from .gates import num_to_tensor, array_to_tensor
from .vis import qir2tex, render_pdf
from . import interfaces
from . import templates
from . import quantum

try:
    from . import keras
except ModuleNotFoundError:
    pass  # in case tf is not installed

# just for fun
from .asciiart import set_ascii
