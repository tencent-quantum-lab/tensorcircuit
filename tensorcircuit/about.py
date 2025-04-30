"""
Prints the information for tensorcircuit installation and environment.
"""

import platform
import sys
import numpy


def about() -> None:
    """
    Prints the information for tensorcircuit installation and environment.
    """
    print(f"OS info: {platform.platform(aliased=True)}")
    print(
        f"Python version: {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"
    )
    print(f"Numpy version: {numpy.__version__}")

    try:
        import scipy

        print(f"Scipy version: {scipy.__version__}")
    except ModuleNotFoundError:
        print(f"Scipy is not installed")

    try:
        import pandas

        print(f"Pandas version: {pandas.__version__}")
    except ModuleNotFoundError:
        print(f"Pandas is not installed")

    try:
        import tensornetwork as tn

        print(f"TensorNetwork version: {tn.__version__}")
    except ModuleNotFoundError:
        print(f"TensorNetwork is not installed")

    try:
        import cotengra

        try:
            print(f"Cotengra version: {cotengra.__version__}")
        except AttributeError:
            print(f"Cotengra: installed")
    except ModuleNotFoundError:
        print(f"Cotengra is not installed")

    try:
        import tensorflow as tf

        print(f"TensorFlow version: {tf.__version__}")
        print(f"TensorFlow GPU: {tf.config.list_physical_devices('GPU')}")
        print(f"TensorFlow CUDA infos: {dict(tf.sysconfig.get_build_info())}")
    except ModuleNotFoundError:
        print(f"TensorFlow is not installed")

    try:
        import jax

        print(f"Jax version: {jax.__version__}")
        try:
            device = jax.devices("gpu")
            print(f"Jax GPU: {device}")
        except RuntimeError:
            print(f"Jax installation doesn't support GPU")
    except ModuleNotFoundError:
        print(f"Jax is not installed")

    try:
        import jaxlib

        print(f"JaxLib version: {jaxlib.__version__}")
    except ModuleNotFoundError:
        print(f"JaxLib is not installed")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch GPU support: {torch.cuda.is_available()}")
        print(
            f"PyTorch GPUs: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}"
        )
        if torch.version.cuda is not None:
            print(f"Pytorch cuda version: {torch.version.cuda}")
    except ModuleNotFoundError:
        print(f"PyTorch is not installed")

    try:
        import cupy

        print(f"Cupy version: {cupy.__version__}")
    except ModuleNotFoundError:
        print(f"Cupy is not installed")

    try:
        import qiskit

        print(f"Qiskit version: {qiskit.__version__}")
    except ModuleNotFoundError:
        print(f"Qiskit is not installed")

    try:
        import cirq

        print(f"Cirq version: {cirq.__version__}")
    except ModuleNotFoundError:
        print(f"Cirq is not installed")

    from tensorcircuit import __version__

    print(f"TensorCircuit version {__version__}")


if __name__ == "__main__":
    about()
