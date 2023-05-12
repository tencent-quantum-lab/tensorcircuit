"""
VQNHE application
"""
from functools import lru_cache
from itertools import product
import time
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


Tensor = Any
Array = Any
Model = Any
dtype = np.complex128
# only guarantee compatible with float64 mode
# only support tf backend
# i.e. use the following setup in the code
# tc.set_backend("tensorflow")
# tc.set_dtype("complex128")

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
    # deprecated
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
    # deprecated
    s = len(ham[0]) - 1
    h = tf.zeros([2**s, 2**s], dtype=dtype)
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

    sparsem = PauliStringSum2COO([h[1:] for h in ham], [h[0] for h in ham])  # type: ignore
    return sparsem
    # densem = backend.to_dense(sparsem)
    # return tf.cast(densem, dtype)


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
    from ..templates.measurements import operator_expectation

    return operator_expectation(c, h)


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
        self.cut = model_params.get("cut", 20)
        if not circuit_params:
            circuit_params = {"epochs": 2, "stddev": 0.1}
        self.epochs = circuit_params.get("epochs", 2)
        self.circuit_stddev = circuit_params.get("stddev", 0.1)
        self.channel = circuit_params.get("channel", 2)

        self.circuit_variable = tf.Variable(
            tf.random.normal(
                mean=0.0,
                stddev=self.circuit_stddev,
                shape=[self.epochs, self.n, self.channel],
                dtype=tf.float64,
            )
        )
        self.circuit = self.create_circuit(**circuit_params)
        self.base = tf.constant(list(product(*[[0, 1] for _ in range(n)])))
        self.hamiltonian = hamiltonian
        self.shortcut = shortcut
        self.history: List[float] = []

    def assign(
        self, c: Optional[List[Tensor]] = None, q: Optional[Tensor] = None
    ) -> None:
        if c is not None:
            self.model.set_weights(c)
        if q is not None:
            self.circuit_variable.assign(q)

    def recover(self) -> None:
        try:
            self.assign(self.ccache, self.qcache)
        except AttributeError:
            pass

    def load(self, path: str) -> None:
        w = np.load(path, allow_pickle=True)
        self.assign(w["c"], w["q"])
        self.history = list(w["h"])

    def save(self, path: str) -> None:
        np.savez(path, c=self.ccache, q=self.qcache, h=np.array(self.history))

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
        if choose == "hea2":
            return self.create_hea2_circuit(**kws)
        if choose == "hn":
            return self.create_hn_circuit(**kws)
        return self.create_functional_circuit(**kws)  # type: ignore

    def create_functional_circuit(self, **kws: Any) -> Callable[[Tensor], Tensor]:
        func = kws.get("func")
        return func  # type: ignore

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
                    c.rx(i, theta=a[epoch, i, 0])  # type: ignore
                for i in range(self.n):
                    c.rz(i, theta=a[epoch, i, 1])  # type: ignore
                for i in range(self.n - 1):
                    c.cnot(i, (i + 1))  # type: ignore
            return c

        return circuit

    def create_hea2_circuit(
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
                    c.rx(i, theta=a[epoch, i, 0])  # type: ignore
                for i in range(self.n):
                    c.rz(i, theta=a[epoch, i, 1])  # type: ignore
                for i in range(self.n):
                    c.rx(i, theta=a[epoch, i, 2])  # type: ignore
                for i in range(self.n - 1):
                    c.exp1(i, (i + 1), theta=a[epoch, i, 3], unitary=G._zz_matrix)  # type: ignore
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
            f2 -= tf.math.reduce_mean(f2)
            f2 = (
                tf.cast(
                    tf.clip_by_value(tf.math.real(f2), -self.cut, self.cut),
                    tf.complex128,
                )
                + tf.cast(tf.math.imag(f2), tf.complex128) * 1.0j
            )
            f2 = tf.exp(f2)
            w1 = tf.multiply(tf.cast(f2, dtype=dtype), w)
            nm = tf.linalg.norm(w1)
            w1 = w1 / nm
            c2 = Circuit(self.n, inputs=w1)
            if not self.shortcut:
                loss = tf.math.real(vqe_energy(c2, self.hamiltonian))
            else:
                loss = tf.math.real(vqe_energy_shortcut(c2, self.hamiltonian))
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
                loss = tf.math.real(vqe_energy_shortcut(c, self.hamiltonian))
        grad = tape.gradient(loss, cv)
        return loss, grad

    def training(
        self,
        maxiter: int = 5000,
        optq: Optional[Callable[[int], Any]] = None,
        optc: Optional[Callable[[int], Any]] = None,
        threshold: float = 1e-8,
        debug: int = 100,
        onlyq: int = 0,
        checkpoints: Optional[List[Tuple[int, float]]] = None,
    ) -> Tuple[Array, Array, Array, int, List[float]]:
        loss_prev = 0
        loss_opt = 1e8
        j_opt = 0
        nm_opt = 1
        ccount = 0
        self.qcache = self.circuit_variable.numpy()
        self.ccache = self.model.get_weights()

        if not optc:
            optc = 0.001  # type: ignore
        if isinstance(optc, float) or isinstance(
            optc, tf.keras.optimizers.schedules.LearningRateSchedule
        ):
            optc = tf.keras.optimizers.Adam(optc)
        if isinstance(optc, tf.keras.optimizers.Optimizer):
            optcf = lambda _: optc
        else:
            optcf = optc  # type: ignore
        if not optq:
            optq = 0.01  # type: ignore
        if isinstance(optq, float) or isinstance(
            optq, tf.keras.optimizers.schedules.LearningRateSchedule
        ):
            optq = tf.keras.optimizers.Adam(optq)
        if isinstance(optq, tf.keras.optimizers.Optimizer):
            optqf = lambda _: optq
        else:
            optqf = optq  # type: ignore

        nm = tf.constant(1.0)
        times = []
        history = []

        try:
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
                history.append(loss.numpy())
                if checkpoints is not None:
                    bk = False
                    for rd, tv in checkpoints:
                        if j > rd and loss.numpy() > tv:
                            print("failed optimization, as checkpoint is not met")
                            bk = True
                            break
                    if bk:
                        break
                if debug > 0:
                    if j % debug == 0:
                        times.append(time.time())
                        print("energy estimation:", loss.numpy())

                        quantume, _ = self.plain_evaluation(self.circuit_variable)
                        print("quantum part energy: ", quantume.numpy())
                        if len(times) > 1:
                            print("running time: ", (times[-1] - times[-2]) / debug)
                loss_prev = loss
                if j < onlyq:
                    gradofc = [grad]
                else:
                    gradofc = grad[0:1]
                if loss < loss_opt:
                    loss_opt = loss
                    nm_opt = nm
                    j_opt = j
                    self.qcache = self.circuit_variable.numpy()
                    self.ccache = self.model.get_weights()
                optqf(j).apply_gradients(zip(gradofc, [self.circuit_variable]))
                if j >= onlyq:
                    optcf(j).apply_gradients(zip(grad[1], self.model.variables))

        except KeyboardInterrupt:
            pass

        quantume, _ = self.plain_evaluation(self.circuit_variable)
        print(
            "vqnhe prediction: ",
            loss_prev.numpy(),  # type: ignore
            "quantum part energy: ",
            quantume.numpy(),
            "classical model scale: ",
            self.model.variables[-1].numpy(),
        )
        print("----------END TRAINING------------")
        self.history += history[:j_opt]
        return (
            loss_opt.numpy(),  # type: ignore
            np.real(nm_opt.numpy()),  # type: ignore
            quantume.numpy(),
            j_opt,
            history[:j_opt],
        )

    def multi_training(
        self,
        maxiter: int = 5000,
        optq: Optional[Callable[[int], Any]] = None,
        optc: Optional[Callable[[int], Any]] = None,
        threshold: float = 1e-8,
        debug: int = 150,
        onlyq: int = 0,
        checkpoints: Optional[List[Tuple[int, float]]] = None,
        tries: int = 10,
        initialization_func: Optional[Callable[[int], Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        # TODO(@refraction-ray): old multiple training implementation, we can use vmap infra for batched training
        rs = []
        try:
            for t in range(tries):
                print("---------- %s training loop ----------" % t)
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
                            shape=[self.epochs, self.n, self.channel],
                            dtype=tf.float64,
                        )
                    )
                else:
                    self.circuit_variable = qweights
                self.history = []  # refresh the history
                r = self.training(
                    maxiter, optq, optc, threshold, debug, onlyq, checkpoints
                )
                rs.append(
                    {
                        "energy": r[0],
                        "norm": r[1],
                        "quantum_energy": r[2],
                        "iterations": r[-2],
                        "history": r[-1],
                        "model_weights": self.ccache,
                        "circuit_weights": self.qcache,
                    }
                )
        except KeyboardInterrupt:
            pass

        if rs:
            es = [r["energy"] for r in rs]
            ind = np.argmin(es)
            self.assign(rs[ind]["model_weights"], rs[ind]["circuit_weights"])
            self.history = rs[ind]["history"]
        return rs
