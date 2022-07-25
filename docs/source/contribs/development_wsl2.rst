Run TensorCirit on Windows with WSL2 (Windows Subsystem for Linux 2)
===========================================================================

Contributed by `YHPeter <https://github.com/YHPeter>`_ (Peter Yu)

Reminder, if you are not supposed to use JAX, you can still use Numpy/Tensorflow/Pytorch backend to run demonstrations.

Step 1.
Install WSL2, follow the official installation instruction: https://docs.microsoft.com/en-us/windows/wsl/install

Step 2.
Install CUDA for GPU support, if you want to used GPU accelerator.
The official CUDA installation for WSL2: https://docs.nvidia.com/cuda/wsl-user-guide/index.html#ch02-getting-started

Step 3.
Follow the Linux Installation Instructions to finish installing.

.. list-table:: **System Support Summary**
   :header-rows: 1

   * - Backend
     - Numpy
     - TensorFlow
     - JAX
     - Pytorch
   * - Suggested Package Version
     - >= 1.20.0
     - >= 2.7.0
     - >= 0.3.0
     - >= 1.12
   * - OS Support without GPU Accelerator
     - Windows/MacOS/Linux
     - Windows/MacOS/Linux
     - Windows/MacOS/Linux
     - Windows/MacOS/Linux
   * - OS Support with GPU Accelerator
     - No Support for GPU
     - Windows(WSL2, docker)/`MacOS <https://developer.apple.com/metal/tensorflow-plugin>`_/Linux
     - Windows(WSL2, docker)/MacOS/Linux
     - Windows(WSL2, docker)/MacOS(torch>=1.12)/Linux
   * - Platform with TPU Accelerator
     - No Support for TPU
     - `GCP - Tensorflow with TPU <https://cloud.google.com/tpu/docs/run-calculation-tensorflow>`_
     - `GCP - JAX with TPU <https://cloud.google.com/tpu/docs/run-calculation-jax>`_
     - `GCP - Pytorch with TPU <https://cloud.google.com/tpu/docs/run-calculation-pytorch>`_

Tips: Currently, we don't suggest you to use TPU accelerator.