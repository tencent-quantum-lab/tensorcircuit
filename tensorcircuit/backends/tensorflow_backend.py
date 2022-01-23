"""
backend magic inherited from tensornetwork: tensorflow backend
"""
# pylint: disable=invalid-name

from functools import reduce, partial
from operator import mul
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import tensornetwork
from tensornetwork.backends.tensorflow import tensorflow_backend

try:  # old version tn compatiblity
    from tensornetwork.backends import base_backend

    tnbackend = base_backend.BaseBackend

except ImportError:
    from tensornetwork.backends import abstract_backend

    tnbackend = abstract_backend.AbstractBackend

dtypestr: str
Tensor = Any
RGenerator = Any  # tf.random.Generator
pytree = Any

tf: Any


class keras_optimizer:
    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer
        self.is_variable = True

    def _c2v(self, v: Tensor) -> Tensor:
        if not isinstance(v, tf.Variable):
            v = tf.Variable(v)
            self.is_variable = False
        return v

    def _apply_gradients(self, grads: Tensor, params: Tensor) -> None:
        self.optimizer.apply_gradients([(grads, params)])

    def update(self, grads: pytree, params: pytree) -> pytree:
        params = TensorFlowBackend.tree_map(None, self._c2v, params)
        # don't do the () initialization since cache is in upper level of backend_factory
        TensorFlowBackend.tree_map(None, self._apply_gradients, grads, params)
        if not self.is_variable:
            return TensorFlowBackend.tree_map(None, tf.convert_to_tensor, params)
        return params

    # TODO(@refraction-ray): complex compatible for opt interface
    # (better not use complex variables though)


def _tensordot_tf(
    self: Any, a: Tensor, b: Tensor, axes: Union[int, Sequence[Sequence[int]]]
) -> Tensor:
    return tf.tensordot(a, b, axes)


def _outer_product_tf(self: Any, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tf.tensordot(tensor1, tensor2, 0)


def _matmul_tf(self: Any, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    if (len(tensor1.shape) <= 1) or (len(tensor2.shape) <= 1):
        raise ValueError("inputs to `matmul` have to be a tensors of order > 1,")

    return tf.matmul(tensor1, tensor2)


def _random_choice_tf(
    g: RGenerator,
    a: Union[int, Sequence[int], Tensor],
    shape: Union[int, Sequence[int]],
    p: Optional[Union[Sequence[float], Tensor]] = None,
) -> Tensor:
    # only replace=True support, replace=False is not implemented
    # for stateless random module, tf has corresponding categorical function similar to choice
    # however, such utility is not implemented with ``tf.random.Generator``
    # part of the code below is inspired by corresponding implementation in jax (Apache 2.0)
    if isinstance(a, int):
        assert a > 0
        a = tf.range(a)
    if not (isinstance(a, tf.Tensor) or isinstance(a, tf.Variable)):
        a = tf.constant(a)
    assert len(a.shape) == 1
    if isinstance(shape, int):
        shape = (shape,)
    if p is None:
        dtype = tf.float32
        p = tf.ones_like(a)
        p = tf.cast(p, dtype=dtype)
        p /= tf.reduce_sum(p)
    else:
        if not (isinstance(p, tf.Tensor) or isinstance(p, tf.Variable)):
            p = tf.constant(p)
        dtype = p.dtype
    shape1 = reduce(mul, shape)
    p_cuml = tf.cumsum(p)
    r = p_cuml[-1] * (1 - g.uniform([shape1], dtype=dtype))
    ind = tf.searchsorted(p_cuml, r)
    res = tf.gather(a, ind)
    return tf.reshape(res, shape)


# temporary hot replace until new version of tensorflow is released,
# see issue: https://github.com/google/TensorNetwork/issues/940
# avoid buggy tensordot2 in tensornetwork

tensornetwork.backends.tensorflow.tensorflow_backend.TensorFlowBackend.tensordot = (
    _tensordot_tf
)
tensornetwork.backends.tensorflow.tensorflow_backend.TensorFlowBackend.outer_product = (
    _outer_product_tf
)
tensornetwork.backends.tensorflow.tensorflow_backend.TensorFlowBackend.matmul = (
    _matmul_tf
)


class TensorFlowBackend(tensorflow_backend.TensorFlowBackend):  # type: ignore
    """
    see the original backend API at `tensorflow backend
    <https://github.com/google/TensorNetwork/blob/master/tensornetwork/backends/tensorflow/tensorflow_backend.py>`_
    """

    def __init__(self) -> None:
        global tf
        super(TensorFlowBackend, self).__init__()
        try:
            import tensorflow
        except ImportError:
            raise ImportError(
                "Tensorflow not installed, please switch to a "
                "different backend or install Tensorflow."
            )
        tf = tensorflow
        self.minor = int(tf.__version__.split(".")[1])
        self.name = "tensorflow"

    def eye(
        self, N: int, dtype: Optional[str] = None, M: Optional[int] = None
    ) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        r = tf.eye(num_rows=N, num_columns=M)
        return self.cast(r, dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        r = tf.ones(shape=shape)
        return self.cast(r, dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        r = tf.zeros(shape=shape)
        return self.cast(r, dtype)

    def copy(self, a: Tensor) -> Tensor:
        return tf.identity(a)

    def expm(self, a: Tensor) -> Tensor:
        return tf.linalg.expm(a)

    def sin(self, a: Tensor) -> Tensor:
        return tf.math.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return tf.math.cos(a)

    def size(self, a: Tensor) -> Tensor:
        return tf.size(a)

    def kron(self, a: Tensor, b: Tensor) -> Tensor:
        # array more than 2d consistency is not guranteed for different backends
        return tf.reshape(
            tf.reshape(a, [a.shape[0], 1, a.shape[1], 1])
            * tf.reshape(b, [1, b.shape[0], 1, b.shape[1]]),
            [a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]],
        )

    def numpy(self, a: Tensor) -> Tensor:
        return a.numpy()  # only valid in eager mode

    def i(self, dtype: Any = None) -> Tensor:
        if not dtype:
            dtype = getattr(tf, dtypestr)  # type: ignore
        if isinstance(dtype, str):
            dtype = getattr(tf, dtype)
        return tf.constant(1j, dtype=dtype)

    def min(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return tf.reduce_min(a, axis=axis)

    def max(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return tf.reduce_max(a, axis=axis)

    def argmax(self, a: Tensor, axis: int = 0) -> Tensor:
        return tf.math.argmax(a, axis=axis)

    def argmin(self, a: Tensor, axis: int = 0) -> Tensor:
        return tf.math.argmin(a, axis=axis)

    def unique_with_counts(self, a: Tensor) -> Tuple[Tensor, Tensor]:
        r = tf.unique_with_counts(a)
        order = tf.argsort(r.y)
        return tf.gather(r.y, order), tf.gather(r.count, order)

    def stack(self, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return tf.stack(a, axis=axis)

    def concat(self, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return tf.concat(a, axis=axis)

    def tile(self, a: Tensor, rep: Tensor) -> Tensor:
        return tf.tile(a, rep)

    def relu(self, a: Tensor) -> Tensor:
        return tf.nn.relu(a)

    def onehot(self, a: Tensor, num: int) -> Tensor:
        return tf.one_hot(a, num)

    def softmax(self, a: Sequence[Tensor], axis: Optional[int] = None) -> Tensor:
        if axis is None:  # make the default behavior consistent
            r = tf.keras.activations.softmax(tf.reshape(a, [1, -1]), axis=axis)
            return tf.reshape(r, a.shape)  # type: ignore
        return tf.keras.activations.softmax(a, axis=axis)

    def cumsum(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        if axis is None:
            a = tf.reshape(a, [-1])
            return tf.cumsum(a)
        else:
            return tf.cumsum(a, axis)

    def is_tensor(self, a: Any) -> bool:
        if isinstance(a, tf.Tensor) or isinstance(a, tf.Variable):
            return True
        return False

    def abs(self, a: Tensor) -> Tensor:
        return tf.math.abs(a)

    def real(self, a: Tensor) -> Tensor:
        return tf.math.real(a)

    def imag(self, a: Tensor) -> Tensor:
        return tf.math.imag(a)

    def cast(self, a: Tensor, dtype: str) -> Tensor:
        if isinstance(dtype, str):
            return tf.cast(a, dtype=getattr(tf, dtype))
        return tf.cast(a, dtype=dtype)

    def solve(self, A: Tensor, b: Tensor, **kws: Any) -> Tensor:
        if b.shape[-1] == A.shape[-1]:
            b = b[..., tf.newaxis]
            vector = True
        else:
            vector = False
        x = tf.linalg.solve(A, b)
        if vector:
            return self.reshape(x, x.shape[:-1])
        return x

    def set_random_state(
        self, seed: Optional[Union[int, RGenerator]] = None, get_only: bool = False
    ) -> Any:
        if seed is None:
            g = tf.random.Generator.from_non_deterministic_state()
        elif isinstance(seed, int):
            g = tf.random.Generator.from_seed(seed)
        else:
            g = seed
        if get_only is False:
            self.g = g
        return g

    def stateful_randn(
        self,
        g: RGenerator,
        shape: Union[int, Sequence[int]] = 1,
        mean: float = 0,
        stddev: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        if isinstance(dtype, str):
            dtype = dtype[-2:]
        if isinstance(shape, int):
            shape = (shape,)
        if dtype == "32":
            dtyper = tf.float32
        elif dtype == "64":
            dtyper = tf.float64
        elif not isinstance(dtype, str):
            dtyper = dtype
        return g.normal(shape, mean, stddev, dtype=dtyper)

    def stateful_randu(
        self,
        g: RGenerator,
        shape: Union[int, Sequence[int]] = 1,
        low: float = 0,
        high: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        if isinstance(dtype, str):
            dtype = dtype[-2:]
        if isinstance(shape, int):
            shape = (shape,)
        if dtype == "32":
            dtyper = tf.float32
        elif dtype == "64":
            dtyper = tf.float64
        elif not isinstance(dtype, str):
            dtyper = dtype
        return g.uniform(shape, minval=low, maxval=high, dtype=dtyper)

    def stateful_randc(
        self,
        g: RGenerator,
        a: Union[int, Sequence[int], Tensor],
        shape: Union[int, Sequence[int]],
        p: Optional[Union[Sequence[float], Tensor]] = None,
    ) -> Tensor:
        return _random_choice_tf(g, a, shape, p)

    def scatter(self, operand: Tensor, indices: Tensor, updates: Tensor) -> Tensor:
        return tf.tensor_scatter_nd_update(operand, indices, updates)

    def coo_sparse_matrix(
        self, indices: Tensor, values: Tensor, shape: Tensor
    ) -> Tensor:
        return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)

    def sparse_dense_matmul(
        self,
        sp_a: Tensor,
        b: Tensor,
    ) -> Tensor:
        return tf.sparse.sparse_dense_matmul(sp_a, b)

    def _densify(self) -> Tensor:
        @partial(self.jit, jit_compile=True)
        def densify(sp_a: Tensor) -> Tensor:
            return tf.sparse.to_dense(sp_a)

        return densify

    def to_dense(self, sp_a: Tensor) -> Tensor:
        return self._densify()(sp_a)

    # very weirdly, at least for cpu, tf.sparse.to_dense only works within tf.function(jit_compile=True)
    # and will fail with tf.function or bare function
    # on the contrary, tf.sparse.sparse_dense_matmul only fails within tf.function(jit_compile=True)
    # and works well with tf.function or bare function...

    def is_sparse(self, a: Tensor) -> bool:
        return isinstance(a, tf.SparseTensor)

    def cond(
        self,
        pred: bool,
        true_fun: Callable[[], Tensor],
        false_fun: Callable[[], Tensor],
    ) -> Tensor:
        return tf.cond(pred, true_fun, false_fun)

    def switch(
        self: Any, index: Tensor, branches: Sequence[Callable[[], Tensor]]
    ) -> Tensor:
        return tf.switch_case(index, branches)

    def stop_gradient(self, a: Tensor) -> Tensor:
        return tf.stop_gradient(a)

    def grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Any]:
        # experimental attempt
        # Note: tensorflow grad is gradient while jax grad is derivative, they are different with a conjugate!
        # And we DONT make them consitent by mannually set conjugate of the returns.
        def wrapper(*args: Any, **kws: Any) -> Any:
            with tf.GradientTape() as t:
                if isinstance(argnums, int):
                    x = args[argnums]
                else:
                    x = [args[i] for i in argnums]
                t.watch(x)
                y = f(*args, **kws)
                if has_aux:
                    g = t.gradient(y[0], x)
                else:
                    g = t.gradient(y, x)
            if has_aux:
                return (g, y[1:])
            return g

        return wrapper

    def value_and_grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Tuple[Any, Any]]:
        def wrapper(*args: Any, **kws: Any) -> Any:
            with tf.GradientTape() as t:
                if isinstance(argnums, int):
                    x = args[argnums]
                else:
                    x = [args[i] for i in argnums]
                t.watch(x)
                y = f(*args, **kws)
                if has_aux:
                    g = t.gradient(y[0], x)
                else:
                    g = t.gradient(y, x)
            return y, g

        return wrapper

    def jvp(
        self,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        if not (isinstance(inputs, list) or isinstance(inputs, tuple)):
            # one input tensor
            inputs = [inputs]
        if not (isinstance(v, list) or isinstance(v, tuple)):
            v = [v]
        with tf.autodiff.ForwardAccumulator(inputs, v) as t:
            y = f(*inputs)
        g = t.jvp(y)
        return y, g

    def vjp(
        self,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        if not (isinstance(inputs, list) or isinstance(inputs, tuple)):
            # one input tensor
            one_input = True
            inputs = [inputs]
        else:
            one_input = False
        with tf.GradientTape() as t:
            t.watch(inputs)
            y = f(*inputs)
        g = t.gradient(y, inputs, v)
        if one_input:
            g = g[0]
        return y, g

    def jit(
        self,
        f: Callable[..., Any],
        static_argnums: Optional[Union[int, Sequence[int]]] = None,
        jit_compile: Optional[bool] = None,
    ) -> Any:
        # static_argnums not supported in tf case, this is only for a consistent interface
        # for more on static_argnums in tf.function, see issue: https://github.com/tensorflow/tensorflow/issues/52193
        # tf.function works with dict pytree but fails at list pytree, hmm...
        # no full jittable pytree support in tf
        # another difference from jax.jit
        if self.minor < 5:
            return tf.function(f, experimental_compile=jit_compile)
        else:
            return tf.function(f, jit_compile=jit_compile)

    # old vmap impl before I know pytrees support in tf
    # def vmap(
    #     self, f: Callable[..., Any], vectorized_argnums: Union[int, Sequence[int]] = 0
    # ) -> Any:
    #     if isinstance(vectorized_argnums, int):
    #         vectorized_argnums = (vectorized_argnums,)
    #     if vectorized_argnums == (0,):  # fast shortcut

    #         def wrapper(*args: Any, **kws: Any) -> Tensor:
    #             def pf(x: Tensor) -> Tensor:
    #                 return f(x, *args[1:], **kws)

    #             return tf.vectorized_map(pf, args[0])

    #     else:

    #         @self.jit
    #         def wrapper(*args: Any, **kws: Any) -> Tensor:
    #             shapes = []
    #             l = len(args)
    #             seps = [0]
    #             batch = args[vectorized_argnums[0]].shape[0]  # type: ignore
    #             nargs = []
    #             for i, arg in enumerate(args):
    #                 if i in vectorized_argnums:  # type: ignore
    #                     shapes.append(arg.shape[1:])
    #                     assert (
    #                         arg.shape[0] == batch
    #                     ), "different tensors has different batch dimensions!"
    #                     arg = tf.reshape(arg, [batch, -1])
    #                     nargs.append(arg)
    #                     seps.append(seps[-1] + arg.shape[-1])
    #             sargs = tf.concat(nargs, 1)

    #             def sf(sarg: Any) -> Any:
    #                 vargs = []
    #                 for i in range(len(shapes)):
    #                     arg = sarg[seps[i] : seps[i + 1]]
    #                     arg = tf.reshape(arg, shapes[i])
    #                     vargs.append(arg)
    #                 vvargs = []
    #                 j = 0
    #                 for i in range(l):
    #                     if i in vectorized_argnums:  # type: ignore
    #                         vvargs.append(vargs[j])
    #                         j += 1
    #                     else:
    #                         vvargs.append(args[i])
    #                 return f(*vvargs, **kws)

    #             return tf.vectorized_map(sf, sargs)

    #     return wrapper

    # def wrapper(f: Callable[..., Any], args: Sequence[Any]) -> Any:
    #     return f(*args)

    # wrapper = partial(wrapper, f)

    # def own_vectorized_map(f: Callable[..., Any], *args: Any) -> Any:
    #     return tf.vectorized_map(f, args)

    # return partial(own_vectorized_map, wrapper)

    def vmap(
        self, f: Callable[..., Any], vectorized_argnums: Union[int, Sequence[int]] = 0
    ) -> Any:
        if isinstance(vectorized_argnums, int):
            vectorized_argnums = (vectorized_argnums,)
        if vectorized_argnums == (0,):  # fast shortcut

            def wrapper(*args: Any, **kws: Any) -> Tensor:
                def pf(x: Tensor) -> Tensor:
                    return f(x, *args[1:], **kws)

                return tf.vectorized_map(pf, args[0])

        else:

            @self.jit  # otherwise, vectorized_map claim on retracing
            def wrapper(*args: Any, **kws: Any) -> Tensor:
                def sf(sarg: Any) -> Any:
                    vvargs = []
                    j = 0
                    for i in range(len(args)):
                        if i in vectorized_argnums:  # type: ignore
                            vvargs.append(sarg[j])
                            j += 1
                        else:
                            vvargs.append(args[i])
                    return f(*vvargs, **kws)

                sarg = []
                for i in vectorized_argnums:  # type: ignore
                    sarg.append(args[i])
                return tf.vectorized_map(sf, sarg)

        return wrapper

    def vectorized_value_and_grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        vectorized_argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Tuple[Any, Any]]:
        # note how tf only works in this order, due to the bug reported as:
        # https://github.com/google/TensorNetwork/issues/940
        vf = self.vmap(f, vectorized_argnums=vectorized_argnums)

        def wrapper(
            *args: Any, **kws: Any
        ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
            with tf.GradientTape() as tape:
                if isinstance(argnums, int):
                    x = args[argnums]
                else:
                    x = [args[i] for i in argnums]
                tape.watch(x)
                vs = vf(*args, **kws)
            if has_aux:
                grad = tape.gradient(vs[0], x)
            else:
                grad = tape.gradient(vs, x)
            return vs, grad

        return wrapper

        # f = self.vmap(f)
        # f = self.value_and_grad(f, argnums=argnums)
        # f = self.jit(f)
        # return f

    vvag = vectorized_value_and_grad

    optimizer = keras_optimizer
