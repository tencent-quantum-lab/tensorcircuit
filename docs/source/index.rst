TensorCircuit Documentation
===========================================================

.. image:: https://github.com/tencent-quantum-lab/tensorcircuit/blob/master/docs/source/statics/logov2.jpg?raw=true
    :target: https://github.com/tencent-quantum-lab/tensorcircuit


**Welcome and congratulations! You have found TensorCircuit.** 👏 

Introduction
---------------

TensorCircuit is an open-source high-performance quantum computing software framework in Python.

* It is built for humans. 👽

* It is designed for speed, flexibility and elegance. 🚀

* It is empowered by advanced tensor network simulator engine. 🔋

* It is ready for quantum hardware access with CPU/GPU/QPU (local/cloud) hybrid solutions. 🖥

* It is implemented with industry-standard machine learning frameworks: TensorFlow, JAX, and PyTorch. 🤖

* It is compatible with machine learning engineering paradigms: automatic differentiation, just-in-time compilation, vectorized parallelism and GPU acceleration. 🛠

With the help of TensorCircuit, now get ready to efficiently and elegantly solve interesting and challenging quantum computing problems: from academic research prototype to industry application deployment.




Relevant Links
--------------------

TensorCircuit is created and maintained by `Shi-Xin Zhang <https://github.com/refraction-ray>`_ and this version is released by `Tencent Quantum Lab <https://quantum.tencent.com/>`_.

The current core authors of TensorCircuit are `Shi-Xin Zhang <https://github.com/refraction-ray>`_ and `Yu-Qin Chen <https://github.com/yutuer21>`_.
We also thank `contributions <https://github.com/tencent-quantum-lab/tensorcircuit/graphs/contributors>`_ from the open source community.

If you have any further questions or collaboration ideas, please use the issue tracker or forum below, or send email to shixinzhang#tencent.com.


.. card-carousel:: 2

   .. card:: Source code
      :link: https://github.com/tencent-quantum-lab/tensorcircuit
      :shadow: md

      GitHub


   .. card:: Documentation
      :link: https://tensorcircuit.readthedocs.io
      :shadow: md

      Readthedocs


   .. card:: Whitepaper
      :link: https://quantum-journal.org/papers/q-2023-02-02-912/
      :shadow: md

      *Quantum* journal


   .. card:: Issue Tracker
      :link: https://github.com/tencent-quantum-lab/tensorcircuit/issues
      :shadow: md

      GitHub Issues


   .. card:: Forum
      :link: https://github.com/tencent-quantum-lab/tensorcircuit/discussions
      :shadow: md

      GitHub Discussions


   .. card:: PyPI
      :link:  https://pypi.org/project/tensorcircuit
      :shadow: md

      ``pip install``


   .. card:: DockerHub
      :link: https://hub.docker.com/repository/docker/tensorcircuit/tensorcircuit
      :shadow: md

      ``docker pull``
      

   .. card:: Application
      :link: https://github.com/tencent-quantum-lab/tensorcircuit#research-and-applications
      :shadow: md

      Research using TC


   .. card:: Cloud
      :link: https://quantum.tencent.com/cloud

      Tencent Quantum Cloud




..
   * Source code: https://github.com/tencent-quantum-lab/tensorcircuit

   * Documentation: https://tensorcircuit.readthedocs.io

   * Software Whitepaper (published in Quantum): https://quantum-journal.org/papers/q-2023-02-02-912/

   * Issue Tracker: https://github.com/tencent-quantum-lab/tensorcircuit/issues

   * Forum: https://github.com/tencent-quantum-lab/tensorcircuit/discussions

   * PyPI page: https://pypi.org/project/tensorcircuit

   * DockerHub page: https://hub.docker.com/repository/docker/tensorcircuit/tensorcircuit

   * Research and projects based on TensorCircuit: https://github.com/tencent-quantum-lab/tensorcircuit#research-and-applications

   * Tencent Quantum Cloud Service: https://quantum.tencent.com/cloud/



Unified Quantum Programming
------------------------------

TensorCircuit is unifying infrastructures and interfaces for quantum computing.

.. grid:: 1 2 4 4
   :margin: 0
   :padding: 0
   :gutter: 2

   .. grid-item-card:: Unified Backends
      :columns: 12 6 3 3
      :shadow: md

      Jax/TensorFlow/PyTorch/Numpy/Cupy

   .. grid-item-card:: Unified Devices
      :columns: 12 6 3 3
      :shadow: md

      CPU/GPU/TPU

   .. grid-item-card:: Unified Providers
      :columns: 12 6 3 3
      :shadow: md

      QPUs from different vendors

   .. grid-item-card:: Unified Resources
      :columns: 12 6 3 3
      :shadow: md

      local/cloud/HPC


.. grid:: 1 2 4 4
   :margin: 0
   :padding: 0
   :gutter: 2

   .. grid-item-card:: Unified Interfaces
      :columns: 12 6 3 3
      :shadow: md

      numerical sim/hardware exp

   .. grid-item-card:: Unified Engines
      :columns: 12 6 3 3
      :shadow: md

      ideal/noisy/approximate simulation

   .. grid-item-card:: Unified Representations
      :columns: 12 6 3 3
      :shadow: md

      from/to_IR/qiskit/openqasm/json

   .. grid-item-card:: Unified Pipelines
      :columns: 12 6 3 3
      :shadow: md

      stateless functional programming/stateful ML models




Reference Documentation
----------------------------

The following documentation sections briefly introduce TensorCircuit to the users and developpers.

.. toctree::
   :maxdepth: 2

   quickstart.rst
   advance.rst
   faq.rst
   sharpbits.rst
   infras.rst
   contribution.rst

Tutorials
---------------------

The following documentation sections include integrated examples in the form of Jupyter Notebook.

.. toctree-filt::
   :maxdepth: 2

   :zh:tutorial.rst
   :zh:whitepapertoc.rst
   :en:tutorial_cn.rst
   :en:whitepapertoc_cn.rst
   :en:textbooktoc.rst



API References
=======================

.. toctree::
   :maxdepth: 2
    
   modules.rst
    

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
