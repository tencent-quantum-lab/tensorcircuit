=================================
TensorCircuit: The Sharp Bits 🔪
=================================

Be fast is never for free, though much cheaper in TensorCircuit, but you have to be cautious especially in terms of AD, JIT compatibility.
We will go through the main sharp edges 🔪 in this note.

Jit Compatibility
---------------------

Non tensor input or varying shape tensor input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The input must be in tensor form and the input tensor shape must be fixed otherwise the recompilation is incurred which is time-consuming.
Therefore, if there are input args that are non-tensor or varying shape tensors and frequently change, jit is not recommend.

.. code-block:: python

    K = tc.set_backend("tensorflow")

    @K.jit
    def f(a):
        print("compiling")
        return 2*a

    f(K.ones([2]))
    # compiling
    # <tf.Tensor: shape=(2,), dtype=complex64, numpy=array([2.+0.j, 2.+0.j], dtype=complex64)>

    f(K.zeros([2]))
    # <tf.Tensor: shape=(2,), dtype=complex64, numpy=array([0.+0.j, 0.+0.j], dtype=complex64)>

    f(K.ones([3]))
    # compiling
    # <tf.Tensor: shape=(3,), dtype=complex64, numpy=array([2.+0.j, 2.+0.j, 2.+0.j], dtype=complex64)>

Mix use of numpy and ML backend APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To make the function jittable and ad-aware, every ops in the function should be called via ML backend (``tc.backend`` API or direct API for the chosen backend ``tf`` or ``jax``).
This is because the ML backend has to create the computational graph to carry out AD and JIT transformation. For numpy ops, they will be only called in jit staging time (the first run).

.. code-block:: python

    K = tc.set_backend("tensorflow")

    @K.jit
    def f(a):
        return np.dot(a, a)

    f(K.ones([2]))
    # NotImplementedError: Cannot convert a symbolic Tensor (a:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported

Numpy call inside jitted function can be helpful if you are sure of the behavior is what you expect.

.. code-block:: python

    K = tc.set_backend("tensorflow")

    @K.jit
    def f(a):
        print("compiling")
        n = a.shape[0]
        m = int(np.log(n)/np.log(2))
        return K.reshape(a, [2 for _ in range(m)])

    f(K.ones([4]))
    # compiling
    # <tf.Tensor: shape=(2, 2), dtype=complex64, numpy=
    # array([[1.+0.j, 1.+0.j],
    #        [1.+0.j, 1.+0.j]], dtype=complex64)>

    f(K.zeros([4]))
    # <tf.Tensor: shape=(2, 2), dtype=complex64, numpy=
    # array([[0.+0.j, 0.+0.j],
    #        [0.+0.j, 0.+0.j]], dtype=complex64)>

    f(K.zeros([2]))
    # compiling
    # <tf.Tensor: shape=(2,), dtype=complex64, numpy=array([0.+0.j, 0.+0.j], dtype=complex64)>

list append under if
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Append something to a Python list within if whose condition is based on tensor values will lead to wrong results.
Actually values of both branch will be attached to the list. See example below.

.. code-block:: python

    K = tc.set_backend("tensorflow")

    @K.jit
    def f(a):
        l = []
        one = K.ones([])
        zero = K.zeros([])
        if a > 0:
            l.append(one)
        else:
            l.append(zero)
        return l

    f(-K.ones([], dtype="float32"))

    # [<tf.Tensor: shape=(), dtype=complex64, numpy=(1+0j)>,
    # <tf.Tensor: shape=(), dtype=complex64, numpy=0j>]

The above code raise ``ConcretizationTypeError`` exception directly for Jax backend since Jax jit doesn't support tensor value if condition.

Similarly, conditional gate application must be takend carefully.

.. code-block:: python

    K = tc.set_backend("tensorflow")

    @K.jit
    def f():
        c = tc.Circuit(1)
        c.h(0)
        a = c.cond_measure(0)
        if a > 0.5:
            c.x(0)
        else:
            c.z(0)
        return c.state()

    f()
    # InaccessibleTensorError: tf.Graph captured an external symbolic tensor.

    # The correct implementation is

    @K.jit
    def f():
        c = tc.Circuit(1)
        c.h(0)
        a = c.cond_measure(0)
        c.conditional_gate(a, [tc.gates.z(), tc.gates.x()], 0)
        return c.state()

    f()
    # <tf.Tensor: shape=(2,), dtype=complex64, numpy=array([0.99999994+0.j, 0.        +0.j], dtype=complex64)>


Tensor variables consistency
-------------------------------------------------------


All tensor variables' backend (tf vs jax vs ..), dtype (float vs complex), shape and device (cpu vs gpu) must be compatible/consistent.

Inspect the backend, dtype, shape and device using the following codes.

.. code-block:: python

    for backend in ["numpy", "tensorflow", "jax", "pytorch"]:
        with tc.runtime_backend(backend):
            a = tc.backend.ones([2, 3])
            print("tensor backend:", tc.interfaces.which_backend(a))
            print("tensor dtype:", tc.backend.dtype(a))
            print("tensor shape:", tc.backend.shape_tuple(a))
            print("tensor device:", tc.backend.device(a))

If the backend is inconsistent, one can convert the tensor backend via :py:meth:`tensorcircuit.interfaces.tensortrans.general_args_to_backend`.

.. code-block:: python

    for backend in ["numpy", "tensorflow", "jax", "pytorch"]:
        with tc.runtime_backend(backend):
            a = tc.backend.ones([2, 3])
            print("tensor backend:", tc.interfaces.which_backend(a))
            b = tc.interfaces.general_args_to_backend(a, target_backend="jax", enable_dlpack=False)
            print("tensor backend:", tc.interfaces.which_backend(b))

If the dtype is inconsistent, one can convert the tensor dtype using ``tc.backend.cast``.

.. code-block:: python

    for backend in ["numpy", "tensorflow", "jax", "pytorch"]:
        with tc.runtime_backend(backend):
            a = tc.backend.ones([2, 3])
            print("tensor dtype:", tc.backend.dtype(a))
            b = tc.backend.cast(a, dtype="float64")
            print("tensor dtype:", tc.backend.dtype(b))

Also note the jax issue on float64/complex128, see `jax gotcha <https://github.com/google/jax#current-gotchas>`_.

If the shape is not consistent, one can convert the shape by ``tc.backend.reshape``.

If the device is not consistent, one can move the tensor between devices by ``tc.backend.device_move``.


AD Consistency
---------------------

TF and JAX backend manage the differentiation rules differently for complex-valued function (actually up to a complex conjuagte). See issue discussion `tensorflow issue <https://github.com/tensorflow/tensorflow/issues/3348>`_.

In TensorCircuit, currently we make the difference in AD transparent, namely, when switching the backend, the AD behavior and result for complex valued function can be different and determined by the nature behavior of the corresponding backend framework.
All AD relevant ops such as ``grad`` or ``jacrev`` may be affected. Therefore, the user must be careful when dealing with AD on complex valued function in a backend agnostic way in TensorCircuit.

See example script on computing Jacobian with different modes on different backends: `jacobian_cal.py <https://github.com/tencent-quantum-lab/tensorcircuit/blob/master/examples/jacobian_cal.py>`_.
Also see the code below for a reference:

.. code-block:: python

    bks = ["tensorflow", "jax"]
    n = 2
    for bk in bks:
        print(bk, "backend")
        with tc.runtime_backend(bk) as K:
            def wfn(params):
                c = tc.Circuit(n)
                for i in range(n):
                    c.H(i)
                for i in range(n):
                    c.rz(i, theta=params[i])
                    c.rx(i, theta=params[i])
                return K.real(c.expectation_ps(z=[0])+c.expectation_ps(z=[1]))
            print(K.grad(wfn)(K.ones([n], dtype="complex64"))) # default
            print(K.grad(wfn)(K.ones([n], dtype="float32")))

    # tensorflow backend
    # tf.Tensor([0.90929717+0.9228758j 0.90929717+0.9228758j], shape=(2,), dtype=complex64)
    # tf.Tensor([0.90929717 0.90929717], shape=(2,), dtype=float32)
    # jax backend
    # [0.90929747-0.9228759j 0.90929747-0.9228759j]
    # [0.90929747 0.90929747]