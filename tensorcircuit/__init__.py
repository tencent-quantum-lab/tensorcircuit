__version__ = "0.12.0"
__author__ = "TensorCircuit Authors"
__creator__ = "refraction-ray"

from .utils import gpu_memory_share

gpu_memory_share()

from .about import about
from .cons import (
    backend,
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
from . import waveforms
from .gates import Gate
from .circuit import Circuit, expectation, Param
from .mpscircuit import MPSCircuit
from .densitymatrix import DMCircuit as DMCircuit_reference
from .densitymatrix import DMCircuit2

DMCircuit = DMCircuit2  # compatibility issue to still expose DMCircuit2
from .gates import num_to_tensor, array_to_tensor
from .vis import qir2tex, render_pdf
from . import interfaces
from . import templates
from . import results
from . import quantum
from .quantum import QuOperator, QuVector, QuAdjointVector, QuScalar
from . import compiler
from . import cloud
from . import fgs
from .fgs import FGSSimulator

try:
    from . import keras
    from .keras import KerasLayer, KerasHardwareLayer
except ModuleNotFoundError:
    pass  # in case tf is not installed

try:
    from . import torchnn
    from .torchnn import TorchLayer, TorchHardwareLayer
except ModuleNotFoundError:
    pass  # in case torch is not installed

try:
    import qiskit

    qiskit.QuantumCircuit.cnot = qiskit.QuantumCircuit.cx
    qiskit.QuantumCircuit.toffoli = qiskit.QuantumCircuit.ccx
    qiskit.QuantumCircuit.fredkin = qiskit.QuantumCircuit.cswap

    # amazing qiskit 1.0 nonsense...
except ModuleNotFoundError:
    pass

# just for fun
from .asciiart import set_ascii
