================
Advanced Usage
================

MPS Simulator
----------------

(Still experimental support)

Split Two-qubit Gates
-------------------------

The two-qubit gates applied on the circuit can be decomposed via SVD, which may further improve the optimality of the contraction pathfinding.

`split` configuration can be set at circuit-level or gate-level.

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

To reuse the jitted function, we can save it on the disk via support from the TensorFlow `SavedModel <https://www.tensorflow.org/guide/saved_model>`_. That is to say, only jitted quantum function on the TensorFlow backend can be saved on the disk. 

For the JAX-backend quantum function, one can first transform them into the tf-backend function via JAX experimental support: `jax2tf <https://github.com/google/jax/tree/main/jax/experimental/jax2tf>`_.

We wrap the tf-backend `SavedModel` as very easy-to-use function :py:meth:`tensorcircuit.keras.save_func` and :py:meth:`tensorcircuit.keras.load_func`.

Parameterized Measurements
-----------------------------

For plain measurements API on a ``tc.Circuit``, eg. `c = tc.Circuit(n=3)`, if we want to evaluate the expectation :math:`<Z_1Z_2>`, we need to call the API as ``c.expectation((tc.gates.z(), [1]), (tc.gates.z(), [2]))``. 

In some cases, we may want to tell the software what to measure but in a tensor fashion. For example, if we want to get the above expectation, we can use the following API: :py:meth:`tensorcircuit.templates.measurements.parameterized_measurements`.

.. code-block:: python

    c = tc.Circuit(3)
    z1z2 = tc.templates.measurements.parameterized_measurements(c, tc.array_to_tensor([0, 3, 3, 0]), onehot=True) # 1

This API corresponds to measure :math:`I_0Z_1Z_2I_3` where 0, 1, 2, 3 are for local I, X, Y, and Z operators respectively.

Sparse Matrix
----------------

We support COO format sparse matrix as most backends only support this format, and some common backend methods for sparse matrices are listed below:

.. code-block:: python

    def sparse_test():
        m = tc.backend.coo_sparse_matrix(indices=np.array([[0, 1],[1, 0]]), values=np.array([1.0, 1.0]), shape=[2, 2])
        n = tc.backend.convert_to_tensor(np.array([[1.0], [0.0]]))
        print("is sparse: ", tc.backend.is_sparse(m), tc.backend.is_sparse(n))
        print("sparse matmul: ", tc.backend.sparse_dense_matmul(m, n))

    for K in ["tensorflow", "jax", "numpy"]:
        with tc.runtime_backend(K):
            print("using backend: ", K)
            sparse_test()

The sparse matrix is specifically useful to evaluate Hamiltonian expectation on the circuit, where sparse matrix representation has a good tradeoff between space and time.
Please refer to :py:meth:`tensorcircuit.templates.measurements.sparse_expectation` for more detail.

For different representations to evaluate Hamiltonian expectation in tensorcircuit, please refer to :doc:`tutorials/tfim_vqe_diffreph`.

Randoms, Jit, Backend Agnostic, and Their Interplay
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

Therefore, a unified jittable random infrastructure with backend agnostic can be formulated as 

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

And a more neat approach to achieve this is as follows:

.. code-block:: python

    key = K.get_random_state(42)

    @K.jit
    def r(key):
        K.set_random_state(key)
        return K.implicit_randn()

    key1, key2 = K.random_split(key)

    print(r(key1), r(key2))

It is worth noting that since ``Circuit.unitary_kraus`` and ``Circuit.general_kraus`` call ``implicit_rand*`` API, the correct usage of these APIs is the same as above.

One may wonder why random numbers are dealt in such a complicated way, please refer to the `Jax design note <https://github.com/google/jax/blob/main/docs/design_notes/prng.md>`_ for some hints.

If vmap is also involved apart from jit, I currently find no way to maintain the backend agnosticity as TensorFlow seems to have no support of vmap over random keys (ping me on GitHub if you think you have a way to do this). I strongly recommend the users using Jax backend in the vmap+random setup.