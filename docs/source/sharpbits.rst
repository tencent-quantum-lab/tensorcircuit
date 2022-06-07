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