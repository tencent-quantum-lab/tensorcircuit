Run TensorCircuit on TensorlowBackend with Apple M1
========================================================
Contributed by (Yuqin Chen)


.. warning::
    This page is deprecated. Please visit `the update tutorial <development_MacARM.html>`_ for latest information.


Why We Can't Run TensorCircuit on TensorlowBackend with Apple M1
-----------------------------------------------------------------------
TensorCircuit requires Tensorflow to support TensorflowBackend. However for Apple M1, Tensorflow package cannot be properly installed by a usual method like "pip install tensorflow". As well, the TensorCircuit package cannot be properly installed by a usual method "pip install tensorcircuit"
All we need is to properly install tensorflow on Apple M1 Pro and then download the TensorCircuit package to the local and install it. 

Install tensorflow on Apple M1
------------------------------------
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


Install TensorCircuit on Apple M1
-----------------------------------
After properly install tensorflow, you can continue install TensorCircuit. 
Up to now, for Apple M1, the Tensorcircuit package can not be installed by simply
conducting "pip install tensorcircuit", which will lead to improper way for Tensorflow installation.
One need to download the installation package to the local, only in this way the installation proceess can recognize the Apple M1 environment. 

One should download the TensorCircuit package to local at first. 

.. code-block:: bash

    git clone https://github.com/tencent-quantum-lab/tensorcircuit.git


Then unpackage it, and cd into the folder with "setup.py". Conducting

.. code-block:: bash

    python setup.py build

    python setup.py install



