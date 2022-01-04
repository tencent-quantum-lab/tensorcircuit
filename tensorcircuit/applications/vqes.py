"""
relevant classes for VQNHE
"""
from functools import lru_cache
from itertools import product
from typing import (
    List,
    Any,
    Tuple,
    Callable,
    Optional,
    Dict,
)

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..circuit import Circuit
from ..quantum import generate_local_hamiltonian
from .. import gates as G

# from .graphdata import *
# from .vags import *
# TODO(@refraction-ray): ensure the remove of these two wildcard import is safe

Tensor = Any
Array = Any
Model = Any
dtype = np.complex128
# only guarantee compatible with float64 mode
# only support tf backend

x = np.array([[0, 1.0], [1.0, 0]], dtype=dtype)
y = np.array([[0, -1j], [1j, 0]], dtype=dtype)
z = np.array([[1, 0], [0, -1]], dtype=dtype)
xx = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=dtype)
yy = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=dtype)
zz = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=dtype)
swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=dtype)

pauli = [np.eye(2), x, y, z]


@lru_cache()
def paulistring(term: Tuple[int, ...]) -> Array:
    n = len(term)
    if n == 1:
        return pauli[term[0]]
    ps = np.kron(pauli[term[0]], paulistring(tuple(list(term)[1:])))
    return ps


def construct_matrix(ham: List[List[float]]) -> Array:
    # only suitable for matrix of small Hilbert dimensions, say <14 qubits
    h = 0.0j
    for term in ham:
        term = list(term)
        for i, t in enumerate(term):
            if i > 0:
                term[i] = int(t)
        h += term[0] * paulistring(tuple(term[1:]))
    return h


# replace with QuOperator tensor_product approach for Hamiltonian construction
# i.e. using ``construct_matrix_v2``


def construct_matrix_tf(ham: List[List[float]], dtype: Any = tf.complex128) -> Tensor:
    # deprecated
    h = 0.0j
    for term in ham:
        term = list(term)
        for i, t in enumerate(term):
            if i > 0:
                term[i] = int(t)
        h += tf.cast(term[0], dtype=dtype) * tf.constant(
            paulistring(tuple(term[1:])), dtype=dtype
        )
    return h


_generate_local_hamiltonian = tf.function(generate_local_hamiltonian)


def construct_matrix_v2(ham: List[List[float]], dtype: Any = tf.complex128) -> Tensor:
    s = len(ham[0]) - 1
    h = tf.zeros([2 ** s, 2 ** s], dtype=dtype)
    for term in tqdm(ham, desc="Hamiltonian building"):
        term = list(term)
        for i, t in enumerate(term):
            if i > 0:
                term[i] = int(t)
        local_matrix_list = [tf.constant(pauli[t], dtype=dtype) for t in term[1:]]
        h += tf.cast(term[0], dtype=dtype) * tf.cast(
            _generate_local_hamiltonian(*local_matrix_list), dtype=dtype
        )
    return h


def construct_matrix_v3(ham: List[List[float]], dtype: Any = tf.complex128) -> Tensor:
    from ..quantum import PauliStringSum2COO
    from ..cons import backend

    sparsem = PauliStringSum2COO([h[1:] for h in ham], [h[0] for h in ham])  # type: ignore
    densem = backend.to_dense(sparsem)
    return tf.cast(densem, dtype)


def vqe_energy(c: Circuit, h: List[List[float]], reuse: bool = True) -> Tensor:
    loss = 0.0
    for term in h:
        ep = []
        for i, t in enumerate(term[1:]):
            if t == 3:
                ep.append((G.z(), [i]))  # type: ignore
            elif t == 1:
                ep.append((G.x(), [i]))  # type: ignore
            elif t == 2:
                ep.append((G.y(), [i]))  # type: ignore
        if ep:
            loss += term[0] * c.expectation(*ep, reuse=reuse)
        else:
            loss += term[0]

    return loss


def vqe_energy_shortcut(c: Circuit, h: Tensor) -> Tensor:
    w = c.wavefunction()
    e = (tf.math.conj(tf.reshape(w, [1, -1])) @ h @ tf.reshape(w, [-1, 1]))[0, 0]
    return e


class Linear(tf.keras.layers.Layer):  # type: ignore
    """
    Dense layer but with complex weights, used for building complex RBM
    """

    def __init__(self, units: int, input_dim: int, stddev: float = 0.1) -> None:
        super().__init__()
        self.wr = tf.Variable(
            initial_value=tf.random.normal(
                shape=[input_dim, units], stddev=stddev, dtype=tf.float64
            ),
            trainable=True,
        )
        self.wi = tf.Variable(
            initial_value=tf.random.normal(
                shape=[input_dim, units], stddev=stddev, dtype=tf.float64
            ),
            trainable=True,
        )
        self.br = tf.Variable(
            initial_value=tf.random.normal(
                shape=[units], stddev=stddev, dtype=tf.float64
            ),
            trainable=True,
        )
        self.bi = tf.Variable(
            initial_value=tf.random.normal(
                shape=[units], stddev=stddev, dtype=tf.float64
            ),
            trainable=True,
        )

    def call(self, inputs: Tensor) -> Tensor:
        inputs = tf.cast(inputs, dtype=dtype)
        return (
            tf.matmul(
                inputs,
                tf.cast(self.wr, dtype=dtype) + 1.0j * tf.cast(self.wi, dtype=dtype),
            )
            + tf.cast(self.br, dtype=dtype)
            + 1.0j * tf.cast(self.bi, dtype=dtype)
        )


class JointSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):  # type: ignore
    def __init__(
        self,
        steps: int = 300,
        pre_rate: float = 0.1,
        pre_decay: int = 400,
        post_rate: float = 0.001,
        post_decay: int = 4000,
        dtype: Any = tf.float64,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.pre_rate = pre_rate
        self.pre_decay = pre_decay
        self.post_rate = post_rate
        self.post_decay = post_decay
        self.dtype = dtype

    def __call__(self, step: Tensor) -> Tensor:
        if step < self.steps:
            return tf.constant(self.pre_rate, dtype=self.dtype) * (1 / 2.0) ** (
                (tf.cast(step, dtype=self.dtype)) / self.pre_decay
            )
        else:
            return tf.constant(self.post_rate, dtype=self.dtype) * (1 / 2.0) ** (
                (tf.cast(step, dtype=self.dtype) - self.steps) / self.post_decay
            )


class VQNHE:
    def __init__(
        self,
        n: int,
        hamiltonian: List[List[float]],
        model_params: Optional[Dict[str, Any]] = None,
        circuit_params: Optional[Dict[str, Any]] = None,
        shortcut: bool = False,
    ) -> None:
        self.n = n
        if not model_params:
            model_params = {}
        self.model = self.create_model(**model_params)
        self.model_params = model_params
        if not circuit_params:
            circuit_params = {"epochs": 2, "stddev": 0.1}
        self.epochs = circuit_params.get("epochs", 2)
        self.circuit_stddev = circuit_params.get("stddev", 0.1)

        self.circuit_variable = tf.Variable(
            tf.random.normal(
                mean=0.0,
                stddev=self.circuit_stddev,
                shape=[2 * self.epochs, self.n],
                dtype=tf.float64,
            )
        )
        self.circuit = self.create_circuit(**circuit_params)
        self.base = tf.constant(list(product(*[[0, 1] for _ in range(n)])))
        self.hamiltonian = hamiltonian
        self.shortcut = shortcut
        if shortcut:
            hmatrix = construct_matrix_v3(self.hamiltonian)
            self.hop = hmatrix
            # self.hop = tf.constant(hmatrix, dtype=tf.complex128)
            # self.hop = G.any(hmatrix.copy().reshape([2 for _ in range(2 * n)]))

    def assign(
        self, c: Optional[List[Tensor]] = None, q: Optional[Tensor] = None
    ) -> None:
        if c:
            self.model.set_weights(c)
        if q:
            self.circuit_variable.assign(q)

    def create_model(self, choose: str = "real", **kws: Any) -> Model:
        if choose == "real":
            return self.create_real_model(**kws)
        if choose == "complex":
            return self.create_complex_model(**kws)
        if choose == "real-rbm":
            return self.create_real_rbm_model(**kws)
        if choose == "complex-rbm":
            return self.create_complex_rbm_model(**kws)

    def create_real_model(
        self,
        init_value: float = 0.0,
        max_value: float = 1.0,
        min_value: float = 0,
        depth: int = 2,
        width: int = 2,
        **kws: Any,
    ) -> Model:

        model = tf.keras.Sequential()
        for _ in range(depth):
            model.add(tf.keras.layers.Dense(width * self.n, activation="relu"))

        model.add(tf.keras.layers.Dense(1, activation="tanh"))
        model.add(
            tf.keras.layers.Dense(
                1,
                activation=None,
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(value=init_value),
                kernel_constraint=tf.keras.constraints.MinMaxNorm(
                    min_value=min_value,
                    max_value=max_value,
                ),
            )
        )
        model.build([None, self.n])
        return model

    def create_complex_model(
        self,
        init_value: float = 0.0,
        max_value: float = 1.0,
        min_value: float = 0,
        **kws: Any,
    ) -> Model:
        inputs = tf.keras.layers.Input(shape=[self.n])
        x = tf.keras.layers.Dense(2 * self.n, activation="relu")(inputs)
        x = tf.keras.layers.Dense(4 * self.n, activation="relu")(x)

        lnr = tf.keras.layers.Dense(2 * self.n, activation="relu")(x)
        lnr = tf.keras.layers.Dense(self.n, activation=None)(lnr)
        lnr = tf.math.log(tf.math.cosh(lnr))
        # lnr = lnr + inputs

        phi = tf.keras.layers.Dense(2 * self.n, activation="relu")(x)
        phi = tf.keras.layers.Dense(self.n, activation="relu")(phi)
        # phi = phi + inputs

        lnr = tf.keras.layers.Dense(1, activation=None)(lnr)
        # lnr = tf.keras.layers.Dense(
        #     1,
        #     activation=None,
        #     use_bias=False,
        #     kernel_initializer=tf.keras.initializers.Constant(value=init_value),
        #     kernel_constraint=tf.keras.constraints.MinMaxNorm(
        #         min_value=min_value,
        #         max_value=max_value,
        #     ),
        # )(lnr)
        phi = tf.keras.layers.Dense(1, activation="tanh")(phi)
        lnr = tf.cast(lnr, dtype=dtype)
        phi = tf.cast(phi, dtype=dtype)
        phi = 3.14159j * phi
        outputs = lnr + phi
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def create_real_rbm_model(
        self, stddev: float = 0.1, width: int = 2, **kws: Any
    ) -> Model:
        inputs = tf.keras.layers.Input(shape=[self.n])
        x = tf.keras.layers.Dense(width * self.n, activation=None)(inputs)
        x = tf.math.log(2 * tf.math.cosh(x))
        x = tf.reduce_sum(x, axis=-1)
        x = tf.reshape(x, [-1, 1])
        y = tf.keras.layers.Dense(1, activation=None)(inputs)
        outputs = y + x
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def create_complex_rbm_model(
        self, stddev: float = 0.1, width: int = 2, **kws: Any
    ) -> Model:
        inputs = tf.keras.layers.Input(shape=[self.n])
        x = Linear(width * self.n, self.n, stddev=stddev)(inputs)
        x = tf.math.log(2 * tf.math.cosh(x))
        x = tf.reduce_sum(x, axis=-1)
        x = tf.reshape(x, [-1, 1])
        y = Linear(1, self.n, stddev=stddev)(inputs)
        outputs = y + x
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def create_circuit(
        self, choose: str = "hea", **kws: Any
    ) -> Callable[[Tensor], Tensor]:
        if choose == "hea":
            return self.create_hea_circuit(**kws)
        if choose == "hn":
            return self.create_hn_circuit(**kws)
        raise ValueError("no such choose option: %s" % choose)

    def create_hn_circuit(self, **kws: Any) -> Callable[[Tensor], Tensor]:
        def circuit(a: Tensor) -> Tensor:
            c = Circuit(self.n)
            for i in range(self.n):
                c.H(i)  # type: ignore
            return c

        return circuit

    def create_hea_circuit(
        self, epochs: int = 2, filled_qubit: Optional[List[int]] = None, **kws: Any
    ) -> Callable[[Tensor], Tensor]:
        if filled_qubit is None:
            filled_qubit = [0]

        def circuit(a: Tensor) -> Tensor:
            c = Circuit(self.n)
            if filled_qubit:
                for i in filled_qubit:
                    c.X(i)  # type: ignore
            for epoch in range(epochs):
                for i in range(self.n):
                    c.rx(i, theta=tf.cast(a[2 * epoch, i], dtype=dtype))  # type: ignore
                for i in range(self.n):
                    c.rz(i, theta=tf.cast(a[2 * epoch + 1, i], dtype=dtype))  # type: ignore
                for i in range(self.n - 1):
                    c.cnot(i, (i + 1))  # type: ignore
            return c

        return circuit

    @tf.function  # type: ignore
    def evaluation(self, cv: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        VQNHE

        :param cv: [description]
        :type cv: Tensor
        :return: [description]
        :rtype: Tuple[Tensor, Tensor, Tensor]
        """
        with tf.GradientTape() as tape:
            tape.watch(cv)
            w = self.circuit(cv).wavefunction()
            f2 = tf.reshape(self.model(self.base), [-1])
            f2 = tf.exp(f2)
            w1 = tf.multiply(tf.cast(f2, dtype=dtype), w)
            nm = tf.linalg.norm(w1)
            w1 = w1 / nm
            c2 = Circuit(self.n, inputs=w1)
            if not self.shortcut:
                loss = tf.math.real(vqe_energy(c2, self.hamiltonian))
            else:
                loss = tf.math.real(vqe_energy_shortcut(c2, self.hop))
        grad = tape.gradient(loss, [cv, self.model.variables])
        for i, gr in enumerate(grad):
            if gr is None:
                if i == 0:
                    grad[i] = tf.zeros_like(cv)
                else:
                    grad[i] = tf.zeros_like(self.model.variables[i - 1])
        return loss, grad, nm

    @tf.function  # type: ignore
    def plain_evaluation(self, cv: Tensor) -> Tensor:
        """
        VQE

        :param cv: [description]
        :type cv: Tensor
        :return: [description]
        :rtype: Tensor
        """
        with tf.GradientTape() as tape:

            c = self.circuit(cv)
            if not self.shortcut:
                loss = tf.math.real(vqe_energy(c, self.hamiltonian))
            else:
                loss = tf.math.real(vqe_energy_shortcut(c, self.hop))
        grad = tape.gradient(loss, cv)
        return loss, grad

    def training(
        self,
        maxiter: int = 5000,
        learn_q: Optional[Callable[[], Any]] = None,
        learn_c: Optional[Callable[[], Any]] = None,
        threshold: float = 1e-8,
        debug: int = 150,
        onlyq: int = 0,
        checkpoints: Optional[List[Tuple[int, float]]] = None,
    ) -> Tuple[Array, Array, Array, int]:
        loss_prev = 0
        ccount = 0

        if not learn_q:
            learnq = 0.1
        else:
            learnq = learn_q()
        if not learn_c:
            learnc = 0.0
        else:
            learnc = learn_c()

        optc = tf.keras.optimizers.Adam(learning_rate=learnc)
        optq = tf.keras.optimizers.Adam(learning_rate=learnq)
        nm = tf.constant(1.0)
        for j in range(maxiter):
            if j < onlyq:
                loss, grad = self.plain_evaluation(self.circuit_variable)
            else:
                loss, grad, nm = self.evaluation(self.circuit_variable)
            if not tf.math.is_finite(loss):
                print("failed optimization since nan in %s tries" % j)
                # in case numerical instability
                break
            if np.abs(loss - loss_prev) < threshold:  # 0.3e-7
                ccount += 1
                if ccount >= 100:
                    print("finished in %s round" % j)
                    break
            if debug > 0:
                if j % debug == 0:
                    print(loss.numpy())
                    if checkpoints is not None:
                        bk = False
                        for rd, tv in checkpoints:
                            if j > rd and loss.numpy() > tv:
                                print("failed optimization, as checkpoint is not met")
                                bk = True
                                break
                        if bk:
                            break
                    quantume, _ = self.plain_evaluation(self.circuit_variable)
                    print("quantum part energy: ", quantume.numpy())
            loss_prev = loss
            if j < onlyq:
                gradofc = [grad]
            else:
                gradofc = grad[0:1]
            optq.apply_gradients(zip(gradofc, [self.circuit_variable]))
            if j >= onlyq:
                optc.apply_gradients(zip(grad[1], self.model.variables))
        quantume, _ = self.plain_evaluation(self.circuit_variable)
        print(
            "vqnhe prediction: ",
            loss.numpy(),
            "quantum part energy: ",
            quantume.numpy(),
            "classical model scale: ",
            self.model.variables[-1].numpy(),
        )
        print("----------END TRAINING------------")
        return loss.numpy(), np.real(nm.numpy()), quantume.numpy(), j

    def multi_training(
        self,
        maxiter: int = 5000,
        learn_q: Optional[Callable[[], Any]] = None,
        learn_c: Optional[Callable[[], Any]] = None,
        threshold: float = 1e-8,
        debug: int = 150,
        onlyq: int = 0,
        checkpoints: Optional[List[Tuple[int, float]]] = None,
        tries: int = 10,
        initialization_func: Optional[Callable[[int], Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        # TODO(@refraction-ray): old multiple training implementation, we can use vmap infra for batched training
        rs = []
        for t in tqdm(range(tries), desc="multiple round of training"):
            if initialization_func is not None:
                weights = initialization_func(t)
            else:
                weights = {}
            qweights = weights.get("q", None)
            cweights = weights.get("c", None)
            if cweights is None:
                cweights = self.create_model(**self.model_params).get_weights()
            self.model.set_weights(cweights)
            if qweights is None:
                self.circuit_variable = tf.Variable(
                    tf.random.normal(
                        mean=0.0,
                        stddev=self.circuit_stddev,
                        shape=[2 * self.epochs, self.n],
                        dtype=tf.float64,
                    )
                )
            else:
                self.circuit_variable = qweights
            r = self.training(
                maxiter, learn_q, learn_c, threshold, debug, onlyq, checkpoints
            )
            rs.append(
                {
                    "energy": r[0],
                    "norm": r[1],
                    "quantum_energy": r[2],
                    "iterations": r[-1],
                    "model_weights": self.model.get_weights(),
                    "circuit_weights": self.circuit_variable.numpy(),
                }
            )
        return rs
