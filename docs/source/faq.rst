Frequently Asked Questions
============================

How can I run TensorCircuit on GPU?
-----------------------------------------

This is done directly through the ML backend. GPU support is totally determined by whether ML libraries are can run on GPU, we don't handle this within tensorcircuit.
It is the users' responsibility to configure an GPU compatible environment for these ML packages. Please refer to the installation documentation for these ML packages and directly use official dockerfiles provided by TensorCircuit.
With GPU compatible enviroment, we can switch the use of GPU or CPU by a backend agnostic environment variable ``CUDA_VISIBLE_DEVICES``.

Which ML framework backend should I use?
--------------------------------------------

Since Numpy backend has no support for AD, if you want to evaluate the circuit gradient, you must set the backend as one of the ML framework beyond Numpy.

Since PyTorch has very limited support for vectorization and jit while our package strongly depends on these features, it is not recommend to use. Though one can always wrap a quantum function on other backend using a PyTorch interface, say :py:meth:`tensorcircuit.interfaces.torch_interface`.

In terms of the choice between TensorFlow and Jax backend, the better one may depend on the use cases and one may want to benchmark both to pick the better one. There is no one-for-all recommendation and this is why we maintain the backend agnostic form of our software.

Some general rule of thumb:

* On both CPU and GPU, the running time of jitted function is faster for jax backend.

* But on GPU, jit staging time is usually much longer for jax backend.

* For hybrid machine learning task, TensorFlow has a better ML ecosystem and reusable classical ML models.

* Jax has some built-in advanced features that is lack in TensorFlow, such as checkpoint in AD and pmap for distributed computing.

* Jax is much insensitive to dtype where type promotion is handled automatically which means easier debugging.

* TensorFlow can cached the jitted function on the disk via SavedModel, which further amortize the staging time.


What is the counterpart of ``QuantumLayer`` for PyTorch and Jax backend?
----------------------------------------------------------------------------

Since PyTorch doesn't have mature vmap and jit support and Jax doesn't have native classical ML layers, we highly recommend TensorFlow as the backend for quantum-classical hybrid machine learning tasks, where ``QuantumLayer`` plays an important role.
For PyTorch, we can in pricinple wrap the corresponding quantum function into a PyTorch module, but we currently has no built-in support for this wrapper.
In terms of Jax backend, we highly suggested to keep the functional programming paradigm for such machine learning task.
Besides, it is worthing noting that, jit and vmap is automatically taken care of in ``QuantumLayer``.


Is there some API less cumbersome than ``expectation`` for Pauli string?
----------------------------------------------------------------------------

Say we want to measure something like :math:`\langle X_0Z_1Y_2Z_4 \rangle` for a six-qubit system, the general ``expectation`` API may seems to be cumbersome.
So one can try one of the following options:

* ``c.expectation_ps(x=[0], y=[2], z=[1, 4])`` 

* ``tc.templates.measurements.parameterized_measurements(c, np.array([1, 3, 2, 0, 3, 0]), onehot=True)``

Can I apply quantum operation based on previous classical measurement result in TensorCircuit?
----------------------------------------------------------------------------------------------------

Try the following: (the pipeline is even fully jittable!)

.. code-block:: python

    c = tc.Circuit(2)
    c.H(0)
    r = c.cond_measurement(0)
    c.conditional_gate(r, [tc.gates.i(), tc.gates.x()], 1)

``cond_measurement`` will return 0 or 1 based on the measurement result on z-basis, and ``conditional_gate`` applies gate_list[r] on the circuit.