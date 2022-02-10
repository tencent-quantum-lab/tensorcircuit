=================================
TensorCircuit: What is inside?
=================================

This part of documentation is mainly for advanced users and developers who want to learn more about what happened behind the scene and delve into the codebase.


Overview of modules
-----------------------

**Core modules:**

- :py:mod:`tensorcircuit.circuit`: The core object :py:obj:`tensorcircuit.circuit.Circuit`. It supports circuit construction, simulation, representation and visualization without noise or with noise using Monte Carlo trajectory approach.

- :py:mod:`tensorcircuit.cons`: Runtime ML backend, dtype and contractor setups. We provide three sets of set methods for global setup, function level setup using function decorators and context setup using ``with`` context managers. We also include customized contractor infrastructures in this module.

- :py:mod:`tensorcircuit.gates`: Definition of quantum gates, either fixed ones or parameterized ones, as well as :py:obj:`tensorcircuit.gates.GateF` class for gates.

**Backend agnostic abstraction:**

- :py:mod:`tensorcircuit.backends` provides a set of backend API and the coreesponding implementation on Numpy, Jax, TensorFlow and PyTorch backends. These backends are inherited fron TensorNetwork package and highly customized.

**Noisy simulation related modules:**

- :py:mod:`tensorcircuit.channels`: Definition of quantum noise channels.

- :py:mod:`tensorcircuit.densitymatrix`: Referenced implementation of ``tc.DMCircuit`` class, with similar set API of ``tc.Circuit`` while simulating the noise in the full form of density matrix.

- :py:mod:`tensorcircuit.densitymatrix2`: Highly efficient implementation of :py:obj:`tensorcircuit.densitymatrix2.DMCircuit2` class, always preferred than the referenced implementation.

**ML interfaces related modules:**

- :py:mod:`tensorcircuit.interfaces`: Provide interfaces when quantum simulation backend is different from neural libraries. Currently include PyTorch and scipy optimizer interfaces.

- :py:mod:`tensorcircuit.keras`: Provide TensorFlow Keras layers as well as wrappers of jitted function save/load from tf side.

**MPS and MPO utilities modules:**

- :py:mod:`tensorcircuit.quantum`: Provide definition and classes for Matrix Product States as well as Matrix Product Operators, we also include various quantum physics and quantum information quantities in this module.

**MPS based simulator modules:**

- :py:mod:`tensorcircuit.mps_base`: Customized and jit/AD compatible MPS class from TensorNetwork package.

- :py:mod:`tensorcircuit.mpscircuit`: :py:obj:`tensorcircuit.mpscircuit.MPSCircuit` class with similar (but subtley different) APIs as ``tc.Circuit``, where the simulation engine is based on MPS TEBD.

**Supplemental modules:**

- :py:mod:`tensorcircuit.simplify`: Provide tools and utility functions to simplify the tensornetworks before the real contractions.

- :py:mod:`tensorcircuit.experimental`: Experimental functions, long and stable support is not guaranteed.

- :py:mod:`tensorcircuit.utils`: Some and general function tools that is not quantum at all.

- :py:mod:`tensorcircuit.vis`: Visualization code for circuit drawing.

**Shortcuts and templates for circuit manipulation:**

- :py:mod:`tensorcircuit.templates`: provide handy shortcuts functions for expectation or circuit building patterns.

**Applications:**

- :py:mod:`tensorcircuit.applications`: most code here are not maintained and deprecated, use on your own risk.

.. note::

    Recommend reading order -- only read the part of code you care about on your purpose. 
    If you want to get and overview of the codebase, please read ``tc.circuit`` following by ``tc.cons`` and ``tc.gates``.