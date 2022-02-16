================
Quick Start
================

Install from GitHub
--------------------------

For beta version usage, one needs to install tensorcircuit package from GitHub. For development and PR workflow, please refer to `contribution <contribution.html>`__ instead.

For private tensorcircuit-dev repo, one needs to firstly configure the SSH key on GitHub and locally, please refer to `GitHub doc <https://docs.github.com/en/authentication/connecting-to-github-with-ssh>`__

Then try ``pip3 install --force-reinstall git+ssh://git@github.com/quclub/tensorcircuit-dev.git`` in shell.

Depending on one's need, one may further pip install tensorflow (for tensorflow backend) or jax and jaxlib (for jax backend) or `cotengra <https://github.com/jcmgray/cotengra>`__ (for more advanced tensornetwork contraction path solver).

If one needs circuit visualization on JupyterLab, python package `wand <https://docs.wand-py.org/en/0.6.7/>`__ and its binary bindings as well as LaTeX installation is required.


Circuit Object
------------------

The basic object for TensorCircuit is ``tc.Circuit``. 

Initialize the circuit with the number of qubits ``c=tc.Circuit(n)``.

**Input states:**

The default input function for the circuit is :math:`\vert 0^n \rangle`. One can change this to other wavefunctions by directly feeding the inputs state vectors w: ``c=tc.Circuit(n, inputs=w)``.

One can also feed matrix product states as input states for the circuit, but we leave MPS/MPO usage for future sections.

**Quantum gates:**

We can apply gates on circuit objects. For example, using ``c.H(1)`` or ``c.rx(2, theta=0.2)``, we can apply Hadamard gate on qubit 1 (0-based) or apply Rx gate on qubit 2 as :math:`e^{-i\theta/2 X}`.

The same rule also apply to multi-qubit gates, such as ``c.cnot(0, 1)``.

There are also highly customizable gates, two instances are:

- ``c.exp1(0, 1, unitary=m, theta=0.2)`` which is for the exponential gate :math:`e^{i\theta m}` of any matrix m as long as :math:`m^2=1`.

- ``c.any(0, 1, unitary=m)`` which is for applying the unitary gate m on the circuit.

These two examples are flexible and support gates on any number of qubits.

**Measurements and Expectations:**

The most straightforward way to get the output from the circuit object is by getting the output wavefunction in vector form as ``c.state()``.

For bitstring sampling, we have ``c.perfect_sampling()`` which returns the bitstring and the corresponding probability amplitude.

To measure part of the qubits, we can use ``c.measure(0, 1)``, if we want to know the corresponding probability of the measurement output, try ``c.measure(0, 1, with_prob=True)``. The measure API is by default non-jittable, but we also have a jittable version as ``c.measure_jit(0, 1)``.

The measurement and sampling utilize advanced algorithms based on tensornetwork and thus requires no knowledge or space for the full wavefunction.

See example below:

.. code-block:: python

    K = tc.set_backend("jax")
    @K.jit
    def sam(key):
        K.set_random_state(key)
        n = 50
        c = tc.Circuit(n)
        for i in range(n):
            c.H(i)
        return c.perfect_sampling()

    sam(jax.random.PRNGKey(42))
    sam(jax.random.PRNGKey(43))


To compute expectation values for local observables, we have ``c.expectation([tc.gates.z(), [0]], [tc.gates.z(), [1]])`` for :math:`\langle Z_0Z_1 \rangle` or ``c.expectation([tc.gates.x(), [0]])`` for :math:`\langle X_0 \rangle`.

This expectation API is rather flexible, as one can measure an m on several qubits as ``c.expectation([m, [0, 1, 2]])``.

**Circuit visualization:** 

``c.vis_tex()`` can generate tex code for circuit visualization based on LaTeX `quantikz <https://arxiv.org/abs/1809.03842>`__ package.

There are also some automatic pipeline helper functions to directly generate figures from tex code, but they require extra installations in the environment.

``render_pdf(tex)`` function requires full installation of LaTeX locally. And in Jupyter environment, we may prefer ``render_pdf(tex, notebook=True)`` to return jpg figures, which further require wand magicwand library installed, see `here <https://docs.wand-py.org/en/latest/>`__.

**Circuit Intermediate Representation:**

TensorCircuit provides its own circuit IR as a python list of dicts. This IR can be further utilized to run compiling, generate serialization qasm or render circuit figures.

The IR is given as a list, each element is a dict containing information on one gate that applied on the circuit. Note gate attr in the dict is actually a python function that returns the gate's node.

.. code-block:: python

    >>> c = tc.Circuit(2)
    >>> c.cnot(0, 1)
    >>> c.crx(1, 0, theta=0.2)
    >>> c.to_qir()
    [{'gate': cnot, 'index': (0, 1), 'name': 'cnot', 'split': None}, {'gate': crx, 'index': (1, 0), 'name': 'crx', 'split': None, 'parameters': {'theta': 0.2}}]


Programming Paradigm
-------------------------

The most common case and the most typical programming paradigm for TensorCircuit is to evaluate the circuit output and the corresponding quantum gradients, which is common in variational quantum algorithms.

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

Also for a non-quantum example (linear regression) demonstrating the backend agnostic feature, variables with pytree support, AD/jit/vmap usage, and variational optimization loops. Please refer to the example script: `linear regression example <https://github.com/quclub/tensorcircuit-dev/blob/master/examples/universal_lr.py>`_.
This example might be more friendly to machine learning community since it is purely classical while also showcasing the main features and paradigms of tensorcircuit.

If the user has no intention to maintain the application code in a backend agnostic fashion, the API for ML frameworks can be more handily used and interleaved with the TensorCircuit API.

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


Automatic differentiation, JIT and Vectorized Parallelism
-------------------------------------------------------------

For concepts of AD, JIT and VMAP, please refer to `Jax documentation <https://jax.readthedocs.io/en/latest/jax-101/index.html>`__ .

The related API design in TensorCircuit closely follows the functional programming design pattern in Jax with some slight differences. So we strongly recommend users learning some basics about Jax no matter which ML backend they intend to use.

**AD support:**

Gradients, vjps, jvps, natural gradients, Jacobians and Hessians.
AD is the base for all modern machine learning libraries.


**JIT support:**

Parameterized quantum circuits can run in a blink. Always use jit if the circuit will get evaluations multiple times, it can greatly boost the simulation with two or three order time reduction. But also be cautious, users need to be familiar with jit, otherwise, the jitted function may return unexpected results or recompile on every hit (wasting lots of time).
To learn more about the jit mechanism, one can refer to documentation or blogs on ``tf.function`` or ``jax.jit``, though these two still have subtle differences.


**VMAP support:**

Inputs, parameters, measurements, circuit structures, Monte Carlo noise can all be evaluated in parallel.
To learn more about vmap mechanism, one can refer to documentation or blogs on ``tf.vectorized_map`` or ``jax.vmap``.


Backend Agnosticism
-------------------------

TensorCircuit supports TensorFlow, Jax, and PyTorch backends. We recommend using TensorFlow or Jax backend since PyTorch lacks advanced jit and vmap features.

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


Switch the Dtype
--------------------

TensorCircuit supports simulation using 32/64 bit precession. The default dtype is 32-bit as "complex64".
Change this by ``tc.set_dtype("complex128")``.

``tc.dtypestr`` always returns the current dtype string: either "complex64" or "complex128".


Setup the Contractor
------------------------

TensorCircuit is a tensornetwork contraction based quantum circuit simulator. A contractor is for searching the optimal contraction path of the circuit tensornetwork.

There are various advanced contractors provided by third-party packages, such as `opt-einsum <https://github.com/dgasmith/opt_einsum>`__ and `cotengra <https://github.com/jcmgray/cotengra>`__.

`opt-einsum` is shipped with TensorNetwork package. To use cotengra, one need to pip install it; kahypar is also recommended to install with cotengra.

Some setup cases:

.. code-block:: python

    import tensorcircuit as tc
    
    # 1. cotengra contractors, has better and consistent performance for large circuit simulation
    import cotengra as ctg

    optr = ctg.ReusableHyperOptimizer(
        methods=["greedy", "kahypar"],
        parallel=True,
        minimize="flops",
        max_time=120,
        max_repeats=4096,
        progbar=True,
    )
    tc.set_contractor("custom", optimizer=optr, preprocessing=True)
    # by preprocessing set as True, tensorcircuit will automatically merge all single-qubit gates into entangling gates

    # 2.  RandomGreedy contractor
    tc.set_contractor("custom_stateful", optimizer=oem.RandomGreedy, max_time=60, max_repeats=128, minimize="size")

    # 3. state simulator like contractor provided by tensorcircuit, maybe better when there is ring topology for two-qubit gate layout
    tc.set_contractor("plain-experimental")

For advanced configurations on cotengra contractors, please refer to cotengra `doc <https://cotengra.readthedocs.io/en/latest/advanced.html>`__ and more fancy examples can be found at `contractor tutorial <https://github.com/quclub/tensorcircuit-tutorials/blob/master/tutorials/contractors.ipynb>`__.

**Setup in function or context level**

Beside global level setup, we can also setup the backend, the dtype, and the contractor in function level or context manager level:

.. code-block:: python

    with tc.runtime_backend("tensorflow"):
        with tc.runtime_dtype("complex128"):
            m = tc.backend.eye(2)
    n = tc.backend.eye(2)
    print(m, n) # m is tf tensor while n is numpy array

    @tc.set_function_backend("tensorflow")
    @tc.set_function_dtype("complex128")
    def f():
        return tc.backend.eye(2)
    print(f()) # complex128 tf tensor


Noisy Circuit Simulation
----------------------------

**Monte Carlo State Simulator:**

For the Monte Carlo trajectory noise simulator, the unitary Kraus channel can be handled easily. TensorCircuit also supports fully jittable and differentiable general Kraus channel Monte Carlo simulation, though.

.. code-block:: python

    >>> c = tc.Circuit(2)
    >>> c.unitary_kraus(tc.channels.depolarizingchannel(0.2, 0.2, 0.2), 0)
    0.0
    >>> c.general_kraus(tc.channels.resetchannel(), 1)
    0.0
    >>> c.state()
    array([0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j], dtype=complex64)

**Density Matrix Simulator:**

Density matrix simulator ``tc.DMCircuit`` simulates the noise in a full form, but takes twice qubits to do noiseless simulation. The API is basically the same as ``tc.Circuit``.


MPS and MPO
----------------

TensorCircuit has its own class for MPS and MPO originally defined in TensorNetwork as ``tc.QuVector``, ``tc.QuOperator``.

``tc.QuVector`` can be extracted from ``tc.Circuit`` as the tensor network form for the output state (uncontracted) by ``c.quvector()``.

The QuVector forms a wavefunction w, which can also be fed into Circuit as the inputs state as ``c=tc.Circuit(n, mps_inputs=w)``.

For example, a quick way to calculate the wavefunction overlap without explicitly computing the state amplitude is given as below:

.. code-block:: python

    >>> c = tc.Circuit(3)
    >>> [c.H(i) for i in range(3)]
    [None, None, None]
    >>> c.cnot(0, 1)
    >>> c2 = tc.Circuit(3)
    >>> [c2.H(i) for i in range(3)]
    [None, None, None]
    >>> c2.cnot(1, 0)
    >>> q = c.quvector()
    >>> q2 = c2.quvector().adjoint()
    >>> (q2@q).eval_matrix()
    array([[0.9999998+0.j]], dtype=complex64)


Interfaces
-------------

**PyTorch interface to hybrid with PyTorch modules:**

As we have mentioned in the backend section, the PyTorch backend may lack the advanced features. This doesn't mean we cannot hybrid advanced circuit module with PyTorch neural module. We can run the quantum function on TensorFlow or Jax backend, while wrapping it with a Torch interface.

.. code-block:: python

    import tensorcircuit as tc
    from tensorcircuit.interfaces import torch_interface
    import torch

    tc.set_backend("tensorflow")


    def f(params):
        c = tc.Circuit(1)
        c.rx(0, theta=params[0])
        c.ry(0, theta=params[1])
        return c.expectation([tc.gates.z(), [0]])


    f_torch = torch_interface(f, jit=True)

    a = torch.ones([2], requires_grad=True)
    b = f_torch(a)
    c = b ** 2
    c.backward()

    print(a.grad)


**Scipy interface to utilize scipy optimizers:**

Automatically transform quantum functions as scipy-compatible values and grad functions as provided for scipy interface with ``jac=True``.

.. code-block:: python

    n = 3

    def f(param):
        c = tc.Circuit(n)
        for i in range(n):
            c.rx(i, theta=param[0, i])
            c.rz(i, theta=param[1, i])
        loss = c.expectation(
            [
                tc.gates.y(),
                [
                    0,
                ],
            ]
        )
        return tc.backend.real(loss)

    f_scipy = tc.interfaces.scipy_optimize_interface(f, shape=[2, n])
    r = optimize.minimize(f_scipy, np.zeros([2 * n]), method="L-BFGS-B", jac=True)


Templates as Shortcuts
------------------------

**Measurements:**

**Circuit blocks:**