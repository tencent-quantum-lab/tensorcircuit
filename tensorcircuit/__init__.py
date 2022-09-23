__version__ = "0.4.1"
__author__ = "TensorCircuit Authors"
__creator__ = "refraction-ray"

from .cons import (
    set_backend,
    set_dtype,
    set_contractor,
    get_backend,
    get_dtype,
    get_contractor,
    set_function_backend,
    set_function_dtype,
    set_function_contractor,
    runtime_backend,
    runtime_dtype,
    runtime_contractor,
)  # prerun of set hooks
from . import gates
from . import basecircuit
from .gates import Gate
from .circuit import Circuit, expectation
from .mpscircuit import MPSCircuit
from .densitymatrix import DMCircuit as DMCircuit_reference
from .densitymatrix import DMCircuit2

DMCircuit = DMCircuit2  # compatibility issue to still expose DMCircuit2
from .gates import num_to_tensor, array_to_tensor
from .vis import qir2tex, render_pdf
from . import interfaces
from . import templates
from . import quantum
from .quantum import QuOperator, QuVector, QuAdjointVector, QuScalar

try:
    from . import keras
    from .keras import QuantumLayer as KerasLayer
except ModuleNotFoundError:
    pass  # in case tf is not installed

try:
    from . import torchnn
    from .torchnn import QuantumNet as TorchLayer
except ModuleNotFoundError:
    pass  # in case torch is not installed

# just for fun
from .asciiart import set_ascii
