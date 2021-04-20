import cirq
import numpy as np
import scipy
import networkx as nx
import tensorflow as tf
from functools import partial, lru_cache
from itertools import product
from typing import (
    List,
    Sequence,
    Any,
    Tuple,
    Callable,
    Iterator,
    Optional,
    Union,
    Iterable,
    Dict,
)

from ..circuit import Circuit
from .. import gates as G
from .layers import *
from .graphdata import *
from .vags import *


dtype = np.complex128  # only guarantee compatible with float64
x = np.array([[0, 1.0], [1.0, 0]], dtype=dtype)
y = np.array([[0, -1j], [1j, 0]], dtype=dtype)
z = np.array([[1, 0], [0, -1]], dtype=dtype)
xx = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=dtype)
yy = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=dtype)
zz = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=dtype)
swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=dtype)

pauli = [
    np.eye(2),
    np.array([[0, 1.0], [1.0, 0]]),
    np.array([[0, -1j], [1j, 0]]),
    np.array([[1.0, 0], [0, -1.0]]),
]


@lru_cache()
def paulistring(term: Tuple[int, ...]) -> Array:
    n = len(term)
    if n == 1:
        return pauli[term[0]]
    ps = np.kron(pauli[term[0]], paulistring(tuple(list(term)[1:])))
    return ps


def contruct_matrix(ham: List[List[float]]) -> Array:
    h = 0.0j
    for j, term in enumerate(ham):
        term = list(term)
        for i, t in enumerate(term):
            if i > 0:
                term[i] = int(t)
        h += term[0] * paulistring(tuple(term[1:]))
    return h


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


class JointSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):  # type: ignore
    def __init__(
        self,
        steps: int = 300,
        pre_rate: float = 0.1,
        pre_decay: int = 400,
        post_rate: float = 0.001,
        post_decay: int = 4000,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.pre_rate = pre_rate
        self.pre_decay = pre_decay
        self.post_rate = post_rate
        self.post_decay = post_decay

    def __call__(self, step: Tensor) -> Tensor:
        if step < self.steps:
            return tf.constant(self.pre_rate, dtype=tf.float64) * (1 / 2.0) ** (
                (tf.cast(step, dtype=tf.float64)) / self.pre_decay
            )
        else:
            return tf.constant(self.post_rate, dtype=tf.float64) * (1 / 2.0) ** (
                (tf.cast(step, dtype=tf.float64) - self.steps) / self.post_decay
            )


class VQNHE:
    def __init__(
        self,
        n: int,
        hamiltonian: List[List[float]],
        model_params: Optional[Dict[str, Any]] = None,
        circuit_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n = n
        if not model_params:
            model_params = {}
        self.model = self.create_model(**model_params)
        self.model_params = model_params
        if not circuit_params:
            circuit_params = {"epochs": 2}
        self.epochs = circuit_params.get("epochs", 2)

        self.circuit_variable = tf.Variable(
            tf.random.normal(
                mean=0.0, stddev=0.2, shape=[2 * self.epochs, self.n], dtype=tf.float64
            )
        )
        self.circuit = self.create_circuit(**circuit_params)
        self.base = tf.constant(list(product(*[[0, 1] for _ in range(n)])))
        self.hamiltonian = hamiltonian

    def assign(
        self, c: Optional[List[Tensor]] = None, q: Optional[Tensor] = None
    ) -> None:
        if c:
            self.model.set_weights(c)
        if q:
            self.circuit_variable = q

    def create_model(
        self,
        init_value: float = 0.0,
        max_value: float = 1.0,
        min_value: float = 0,
        depth: int = 2,
    ) -> Model:

        model = tf.keras.Sequential()
        for _ in range(depth):
            model.add(tf.keras.layers.Dense(2 * self.n, activation="relu"))

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

    def create_circuit(
        self, epochs: int = 2, filled_qubit: List[int] = [0]
    ) -> Callable[[Tensor], Tensor]:
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
        with tf.GradientTape() as tape:
            tape.watch(cv)
            w = self.circuit(cv).wavefunction()
            f2 = tf.reshape(self.model(self.base), [-1])
            f2 = tf.exp(f2)
            w1 = tf.multiply(tf.cast(f2, dtype=dtype), w)
            nm = tf.linalg.norm(w1)
            w1 = w1 / nm
            c2 = Circuit(self.n, inputs=w1)
            loss = tf.math.real(vqe_energy(c2, self.hamiltonian))
        grad = tape.gradient(loss, [cv, self.model.variables])
        return loss, grad, nm

    @tf.function  # type: ignore
    def plain_evaluation(self, cv: Tensor) -> Tensor:
        c = self.circuit(cv)
        loss = tf.math.real(vqe_energy(c, self.hamiltonian))
        return loss

    def training(
        self,
        maxiter: int = 5000,
        learn_q: Optional[Callable[[], Any]] = None,
        learn_c: Optional[Callable[[], Any]] = None,
        threshold: float = 1e-8,
        debug: int = 150,
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

        for j in range(maxiter):
            loss, grad, nm = self.evaluation(self.circuit_variable)

            if np.abs(loss - loss_prev) < threshold:  # 0.3e-7
                ccount += 1
                if ccount >= 100:
                    print("finished in %s round" % j)
                    break
            if debug > 0:
                if j % debug == 0:
                    print(loss.numpy())
                    quantume = self.plain_evaluation(self.circuit_variable)
                    print("quantum part: ", quantume.numpy())
            loss_prev = loss
            optq.apply_gradients(zip(grad[0:1], [self.circuit_variable]))
            optc.apply_gradients(zip(grad[1], self.model.variables))
        quantume = self.plain_evaluation(self.circuit_variable)
        print(
            loss.numpy(), nm.numpy(), quantume.numpy(), self.model.variables[-1].numpy()
        )
        print("----------END TRAINING------------")
        return loss.numpy(), nm.numpy(), quantume.numpy(), j

    def multi_training(
        self,
        maxiter: int = 5000,
        learn_q: Optional[Callable[[], Any]] = None,
        learn_c: Optional[Callable[[], Any]] = None,
        threshold: float = 1e-8,
        debug: int = 150,
        tries: int = 10,
        initialization_func: Optional[Callable[[int], Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        rs = []
        for t in range(tries):
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
                        stddev=0.2,
                        shape=[2 * self.epochs, self.n],
                        dtype=tf.float64,
                    )
                )
            else:
                self.circuit_variable = qweights
            r = self.training(maxiter, learn_q, learn_c, threshold, debug)
            rs.append(
                {
                    "energy": r[0],
                    "norm": r[1],
                    "quantum_energy": r[2],
                    "iterations": r[-1],
                    "model_weights": self.model.get_weights(),
                    "circuit_weights": self.circuit_variable,
                }
            )
        return rs
