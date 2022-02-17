================
Advanced Usage
================

MPS Simulator
----------------

(Still experimental support)

Split Two-qubit Gates
-------------------------

The two-qubit gates applied on the circuit can be decomposed via SVD, which may further improve the optimality the contraction path finding.

`split` configuration can be set in circuit level or gate level.

.. code-block:: python

    split_conf = {
        "max_singular_values": 2,  # how many singular values are kept
        "fixed_choice": 1, # 1 for normal one, 2 for swapped one
    }

    c = tc.Circuit(nwires, split=split_conf)

    # or

    c.exp1(
            i,
            (i + 1) % nwires,
            theta=paramc[2 * j, i],
            unitary=tc.gates._zz_matrix,
            split=split_conf
        )

Note ``max_singular_values`` must be specified to make the whole procedure static and thus jittable.


Jitted Function Save/Load
-----------------------------

To reuse the jitted function, we can save it on the disk via support from TensorFlow `SavedModel <https://www.tensorflow.org/guide/saved_model>`_. That is to say, only jitted quantum function on TensorFlow backend can be saved on the disk. 

For jax-backend quantum function, one can first transform them into tf-backend function via jax experimental support: `jax2tf <https://github.com/google/jax/tree/main/jax/experimental/jax2tf>`_.

We wrap the tf-backend `SavedModel` as very easy-to-use function :py:meth:`tensorcircuit.keras.save_func` and :py:meth:`tensorcircuit.keras.load_func`.

Parameterized Measurements
-----------------------------

For plain measurements API on a ``tc.Circuit``, eg. `c = tc.Circuit(n=3)`, if we want to evaluate the expectation :math:`<Z_1Z_2>`, we need to call the API as ``c.expectation((tc.gates.z(), [1]), (tc.gates.z(), [2]))``. 

In some cases, we may want to tell the software what to measure but in a tensor fashion. For example, if we want to get the above expectation, we can use the following API: :py:meth:`tensorcircuit.templates.measurements.parameterized_measurements`.

.. code-block:: python

    c = tc.Circuit(3)
    z1z2 = tc.templates.measurements.parameterized_measurements(c, tc.array_to_tensor([0, 3, 3, 0]), onehot=True) # 1

This API corresponds to measure :math:`I_0Z_1Z_2I_3` where 0, 1, 2, 3 are for local I, X, Y, Z operators respectively.

Sparse Matrix
----------------


Randoms, Jit, Backend Agnostic and Their Interplay
--------------------------------------------------------

.. code-block:: python

    import tensorcircuit as tc
    K = tc.set_backend("tensorflow")
    K.set_random_state(42)

    @K.jit
    def r():
        return K.implicit_randn()

    print(r(), r()) # different, correct

.. code-block:: python

    import tensorcircuit as tc
    K = tc.set_backend("jax")
    K.set_random_state(42)

    @K.jit
    def r():
        return K.implicit_randn()

    print(r(), r()) # the same, wrong


.. code-block:: python

    import tensorcircuit as tc
    import jax
    K = tc.set_backend("jax")
    key = K.set_random_state(42)

    @K.jit
    def r(key):
        K.set_random_state(key)
        return K.implicit_randn()

    key1, key2 = K.random_split(key)

    print(r(key1), r(key2)) # different, correct

Therefore, a unified jittable random infrastructure with backend agnostic can be formulatted as 

.. code-block:: python

    import tensorcircuit as tc
    import jax
    K = tc.set_backend("tensorflow")

    def ba_key(key):
        if tc.backend.name == "tensorflow":
            return None
        if tc.backend.name == "jax":
            return jax.random.PRNGKey(key)
        raise ValueError("unsupported backend %s"%tc.backend.name)

        
    @K.jit
    def r(key=None):
        if key is not None:
            K.set_random_state(key)
        return K.implicit_randn()

    key = ba_key(42)

    key1, key2 = K.random_split(key)

    print(r(key1), r(key2))