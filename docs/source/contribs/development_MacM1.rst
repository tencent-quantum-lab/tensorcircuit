Run TensorCircuit on TensorlowBackend with Apple M1 Pro
========================================================
Contributed by (Yuqin Chen)

Why We Can't Run TensorCircuit on TensorlowBackend with Apple M1 Pro
---------------------------------------------------------------
TensorCircuit requires Tensorflow to support TensorflowBackend. However for Apple M1 Pro, Tensorflow package cannot be properly installed by a usual method like "pip install tensorflow". 
All we need is to correctly install tensorflow on Apple M1 Pro.

Install tensorflow on Apple M1 Pro
---------------------------------------------------------------
According to the instructions below or the installation manual on Apple's official website `tensorflow-metal PluggableDevice <https://developer.apple.com/metal/tensorflow-plugin/>`_, you can install tensorflow step by step.

**Step1: Environment setup**

x86 : AMD
Create virtual environment (recommended):
.. code-block:: bash

    python3 -m venv ~/tensorflow-metal
    source ~/tensorflow-metal/bin/activate
    python -m pip install -U pip

NOTE: python version 3.8 required

arm64 : Apple Silicon
Download and install Conda env:
.. code-block:: bash

    chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
    sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
    source ~/miniforge3/bin/activate
    
Install the TensorFlow dependencies:
.. code-block:: bash
    conda install -c apple tensorflow-deps

- When upgrading to new base TensorFlow version, recommend:
 .. code-block:: bash
    # uninstall existing tensorflow-macos and tensorflow-metal
    python -m pip uninstall tensorflow-macos
    python -m pip uninstall tensorflow-metal
    # Upgrade tensorflow-deps
    conda install -c apple tensorflow-deps --force-reinstall
    # or point to specific conda environment
    conda install -c apple tensorflow-deps --force-reinstall -n my_env

- tensorflow-deps versions are following base TensorFlow versions so:
for v2.5
.. code-block:: bash
    conda install -c apple tensorflow-deps==2.5.0

for v2.6
.. code-block:: bash
    conda install -c apple tensorflow-deps==2.6.0


**Step2: Install base TensorFlow**

.. code-block:: bash
    python -m pip install tensorflow-macos

**Step3: Install tensorflow-metal plugin**

.. code-block:: bash
    python -m pip install tensorflow-metal


Test your installation
---------------------------------------------------------------
After properly install tensorflow, you can test it using:
.. code-block:: bash
    import tensorflow

And then you can install and run TensorCircuit on TensorlowBackend with Apple M1 Pro.