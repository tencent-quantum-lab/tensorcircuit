================
Advanced Usage
================

MPS simulator
----------------

Split two-qubit gates
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


SavedModel
-----------------

Parameterized measurements
-----------------------------

Randoms, jit, backend agnostic and their interplay
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