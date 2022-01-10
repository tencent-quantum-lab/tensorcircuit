================
Quick Start
================

Install from GitHub
--------------------------

For beta version usage, one needs to install tensorcircuit package from GitHub. For development and PR workflow, please refer to `contribution <contribution.html>`__ instead.

For private tensorcircuit-dev repo, one needs to firstly configure the SSH key on GitHub and locally, please refer to `GitHub doc <https://docs.github.com/en/authentication/connecting-to-github-with-ssh>`__

Then try ``pip3 install --force-reinstall git+ssh://git@github.com/quclub/tensorcircuit-dev.git`` in shell.

Depending on one's need, one may further pip install tensorflow (for tensorflow backend) or jax and jaxlib (for jax backend) or `cotengra <https://github.com/jcmgray/cotengra>`__ (for more advanced tensornetwork contraction path solver).

If one needs circuit visualization on Jupyter lab, python package `wand <https://docs.wand-py.org/en/0.6.7/>`__ and its binary bindings as well as LaTeX installation is required.


Circuit Object
------------------

The basic object for TensorCircuit is ``tc.Circuit``. 

Initialize the circuit with the number of qubits ``c=tc.Circuit(n)``.

**Input states:**

The default input function for the circuit is :math:`\vert 0^n \rangle`. One can change this to other wavefunctions by directly feed the inputs state vectors w: ``c=tc.Circuit(n, inputs=w)``.

One can also feed matrix product state as input states for the circuit, but we leave MPS/MPO usage for future sections.

**Quantum gates:**

We can apply gates on the circuit object as: ``c.H(1)`` or ``c.rx(2, theta=0.2)`` which are for apply Hadamard gate on qubit 1 (0-based) or apply Rx gate on qubit 2 as :math:`e^{-i\theta/2 X}`.

The same rules apply to multi-qubit gates, such as ``c.cnot(0, 1)``.

There are also highly customizable gates, two representatives are:

- ``c.exp1(0, 1, unitary=m, theta=0.2)`` which is for the exponential gate :math:`e^{i\theta m}` of any matrix m as long as :math:`m^2=1`.

- ``c.any(0, 1, unitary=m)`` which is for applying the unitary gate m on the circuit.

These two examples are flexible and support gate on any number of qubits.

**Measurements and expectations:**

The most directly way to get the output from the circuit object is just getting the output wavefunction in vector form as ``c.state()``.

For bitstring sampling, we have ``c.perfect_sampling()`` which returns the bitstring and the corresponding probability amplitude.

To measure part of the qubits, we can use ``c.measure(0, 1)``, if we want to know the corresponding probability of the measurement output, try ``c.measure(0, 1, with_prob=True)``. The measure API is by default non-jittable, but we also have a jittable version as ``c.measure_jit(0, 1)``.

To compute expectation values for local observables, we have ``c.expectation([tc.gates.z(), [0]], [tc.gates.z(), [1]])`` for :math:`\langle Z_0Z_1 \rangle` or ``c.expectation([tc.gates.x(), [0]])`` for :math:`\langle X_0 \rangle`.

This expectation API is rather flexible, as one can measure any matrix m on several qubits as ``c.expectation([m, [0, 1, 2]])``.

**Circuit visualization:**


Programming Paradigm
-------------------------

The most common use case and the most typical programming paradigm for TensorCircuit is to evaluate the circuit output and the corresponding quantum gradients, which is common in variational quantum algorithms.

.. code-block:: python

    import tensorcircuit as tc

    K = tc.set_backend("tensorflow")

    n = 1


    def loss(params, n):
        c = tc.Circuit(n)
        for i in range(n):
            c.rx(i, theta=params[0, i])
        for i in range(n):
            c.rz(i, theta=params[1, i])
        loss = 0.0
        for i in range(n):
            loss += c.expectation([tc.gates.z(), [i]])
        return K.real(loss)


    vagf = K.jit(K.value_and_grad(loss), static_argnums=1)
    params = K.implicit_randn([2, n])
    print(vagf(params, n))  # get the quantum loss and the gradient

If the users have no intension to maintain the application code in a backend agnostic fashion, the API for ML frameworks can be more freely used and interleaved with TensorCircuit API.

.. code-block:: python

    import tensorcircuit as tc
    import tensorflow as tf

    K = tc.set_backend("tensorflow")

    n = 1


    def loss(params, n):
        c = tc.Circuit(n)
        for i in range(n):
            c.rx(i, theta=params[0, i])
        for i in range(n):
            c.rz(i, theta=params[1, i])
        loss = 0.0
        for i in range(n):
            loss += c.expectation([tc.gates.z(), [i]])
        return tf.math.real(loss)

    def vagf(params, n):
        with tf.GradientTape() as tape:
            tape.watch(params)
            l = loss(params, n)
        return l, tape.gradient(l, params)

    vagf = tf.function(vagf)
    params = tf.random.normal([2, n])
    print(vagf(params, n))  # get the quantum loss and the gradient


Automatic differentiation, JIT and vectorized parallelism
-------------------------------------------------------------

For the concepts of AD, JIT and VMAP, please refer to `Jax documentation <https://jax.readthedocs.io/en/latest/jax-101/index.html>`__ . 

The related API design in TensorCircuit closely follows the design pattern in Jax with some small differences.

- AD support: gradients, vjps, jvps, natural gradients, Jacobians and Hessians

- JIT support: parameterized quantum circuit can run in a blink

- VMAP support: inputs, parameters, measurements, circuit structures, noise can all be parallelly evaluate


Backend Agnosticism
-------------------------

TensorCircuit support TensorFlow, Jax and PyTorch backends. We recommend to use TensorFlow or Jax backend, since PyTorch lacks advanced jit and vmap features.

The backend can be set as ``K=tc.set_backend("jax")`` and ``K`` is the backend with a full set of APIs as a conventional ML framework, which can also be accessed by ``tc.backend``.

.. code-block:: python

    >>> import tensorcircuit as tc
    >>> K = tc.set_backend("tensorflow")
    >>> K.ones([2,2])
    <tf.Tensor: shape=(2, 2), dtype=complex64, numpy=
    array([[1.+0.j, 1.+0.j],
        [1.+0.j, 1.+0.j]], dtype=complex64)>
    >>> tc.backend.eye(3)
    <tf.Tensor: shape=(3, 3), dtype=complex64, numpy=
    array([[1.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 1.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 1.+0.j]], dtype=complex64)>
    >>> tc.set_backend("jax")
    <tensorcircuit.backends.jax_backend.JaxBackend object at 0x7fb00e0fd6d0>
    >>> tc.backend.name
    'jax'
    >>> tc.backend.implicit_randu()
    WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
    DeviceArray([0.7400521], dtype=float32)

The supported APIs in backend come from two sources, one part is implemented in `TensorNetwork package <https://github.com/google/TensorNetwork/blob/master/tensornetwork/backends/abstract_backend.py>`__
and the other part is implemented in `TensorCircuit package <modules.html#module-tensorcircuit.backends>`__.

Switch the dtype
--------------------

TensorCircuit supports simulation using 32/64 bit percesion. The default dtype is 32-bit as "complex64".
Change this by `tc.set_dtype("complex128")`.

`tc.dtypestr` always return the current dtype str: either "complex64" or "complex128".


Setup the contractor
------------------------

TensorCircuit is a tensornetwork contraction based quantum circuit simulator. A contractor is for searching the optimal contraction path of the circuit tensornetwork.

There are various advanced contractor provided by the third-party packages, such as `opt-einsum <https://github.com/dgasmith/opt_einsum>`__ and `cotengra <https://github.com/jcmgray/cotengra>`__.


Noisy Circuit simulation
----------------------------

**Monte Carlo State Simulator:**

**Density Matrix Simulator:**


MPS and MPO
----------------


Interfaces
-------------


Templates as Shortcuts
------------------------