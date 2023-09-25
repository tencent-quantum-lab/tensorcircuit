"""
modules for QUBO problems in QAOA
"""

from typing import List, Callable, Any, Optional, Tuple
from functools import partial

import tensorflow as tf
import scipy.optimize as optimize

from ..cons import backend
from ..quantum import measurement_results
from ..interfaces import scipy_interface
from ..templates.ansatz import QAOA_ansatz_for_Ising
from ..templates.conversions import QUBO_to_Ising

Circuit = Any
Tensor = Any
Array = Any


def Ising_loss(c: Circuit, pauli_terms: Tensor, weights: List[float]) -> Any:
    """
    computes the loss function for the Ising model based on a given quantum circuit,
    a list of Pauli terms, and corresponding weights.
    The offset is ignored.

    :param c: A quantum circuit object generating the state.
    :param pauli_terms: A list of Pauli terms, where each term is represented as a list of 0/1 series.
    :param weights: A list of weights corresponding to each Pauli term.
    :return loss: A real number representing the computed loss value.
    """
    loss = 0.0
    for k in range(len(pauli_terms)):
        term = pauli_terms[k]
        index_of_ones = []

        for l in range(len(term)):
            if term[l] == 1:
                index_of_ones.append(l)

        # Compute expectation value based on the number of qubits involved in the Pauli term
        if len(index_of_ones) == 1:
            delta_loss = weights[k] * c.expectation_ps(z=[index_of_ones[0]])
            # Compute expectation value for a single-qubit Pauli term
        else:
            delta_loss = weights[k] * c.expectation_ps(
                z=[index_of_ones[0], index_of_ones[1]]
            )
            # Compute expectation value for a two-qubit Pauli term

        loss += delta_loss

    return backend.real(loss)


def QAOA_loss(
    nlayers: int,
    pauli_terms: Tensor,
    weights: List[float],
    params: List[float],
    full_coupling: bool = False,
    mixer: str = "X",
) -> Any:
    """
    computes the loss function for the Quantum Approximate Optimization Algorithm (QAOA) applied to the Ising model.

    :param nlayers: The number of layers in the QAOA ansatz.
    :param pauli_terms: A list of Pauli terms, where each term is represented as a list of 0/1 series.
    :param weights: A list of weights corresponding to each Pauli term.
    :param params: A list of parameter values used in the QAOA ansatz.
    :param full_coupling (optional): A flag indicating whether to use all-to-all coupling in mixers. Default is False.
    :paran mixer (optional): The mixer operator to use. Default is "X". The other options are "XY" and "ZZ".
    :return: The computed loss value.
    """
    c = QAOA_ansatz_for_Ising(
        params, nlayers, pauli_terms, weights, mixer=mixer, full_coupling=full_coupling
    )
    # Obtain the quantum circuit using QAOA_from_Ising function

    return Ising_loss(c, pauli_terms, weights)
    # Compute the Ising loss using Ising_loss function on the obtained circuit


def QUBO_QAOA(
    Q: Tensor,
    nlayers: int,
    iterations: int,
    vvag: bool = False,
    ncircuits: int = 10,
    init_params: Optional[List[float]] = None,
    mixer: str = "X",
    learning_rate: float = 1e-2,
    callback: Optional[Optional[Callable[[List[float], float], None]]] = None,
    full_coupling: bool = False,
) -> Array:
    """
    Performs the QAOA on a given QUBO problem.
    Adam optimizer from TensorFlow is used.

    :param Q: The n-by-n square and symmetric Q-matrix representing the QUBO problem.
    :param nlayers: The number of layers (depth) in the QAOA ansatz.
    :param iterations: The number of iterations to run the optimization.
    :param vvag (optional): A flag indicating whether to use vectorized variational adjoint gradient. Default is False.
    :param ncircuits (optional): The number of circuits when using vectorized variational adjoint gradient.
        Default is 10.
    :param init_params (optional): The initial parameters for the ansatz circuit.
        Default is None, which initializes the parameters randomly.
    :paran mixer (optional): The mixer operator to use. Default is "X". The other options are "XY" and "ZZ".
    :param learning_rate (optional): The learning rate for the Adam optimizer. Default is 1e-2.
    :param callback (optional): A callback function that is executed during each iteration. Default is None.
    :param full_coupling (optional): A flag indicating whether to use all-to-all coupling in mixers. Default is False.
    :return params: The optimized parameters for the ansatz circuit.
    """

    pauli_terms, weights, _ = QUBO_to_Ising(Q)

    loss_val_grad = backend.value_and_grad(
        partial(
            QAOA_loss,
            nlayers,
            pauli_terms,
            weights,
            mixer=mixer,
            full_coupling=full_coupling,
        )
    )
    loss_val_grad = backend.jit(loss_val_grad, static_argnums=(1, 2))
    # Define the loss and gradients function using value_and_grad, which calculates both the loss value and gradients.

    if init_params is None:
        params = backend.implicit_randn(shape=[2 * nlayers], stddev=0.5)
        if vvag is True:
            loss_val_grad = backend.vvag(loss_val_grad, argnums=0, vectorized_argnums=0)
            params = backend.implicit_randn(shape=[ncircuits, 2 * nlayers], stddev=0.1)
            # If init_params is not provided, initialize the parameters randomly.
            # If vvag flag is set to True, use vectorized variational adjoint gradient (vvag) with multiple circuits.
    else:
        params = init_params
        # If init_params is provided, use the provided parameters.
    # Initialize the parameters for the ansatz circuit.

    # This can improve the performance by pre-compiling the loss and gradients function.

    opt = backend.optimizer(tf.keras.optimizers.Adam(learning_rate))
    # Define the optimizer (Adam optimizer) with the specified learning rate.

    for _ in range(iterations):
        loss, grads = loss_val_grad(params)
        # Calculate the loss and gradients using the loss_val_grad_jit function.

        params = opt.update(grads, params)
        # Update the parameters using the optimizer and gradients.

        if callback is not None:
            callback(loss, params)
        # Execute the callback function with the current loss and parameters.

    return params
    # Return the optimized parameters for the ansatz circuit.


def cvar_value(r: List[float], p: List[float], percent: float) -> Any:
    """
    Compute the Conditional Value at Risk (CVaR) based on the measurement results.

    :param r: The observed outcomes after measurements.
    :param p: Probabilities associated with each observed outcome.
    :param percent: The cut-off percentage for CVaR computation.
    :return: The calculated CVaR value.
    """
    sorted_indices = tf.argsort(r)
    p_sorted = tf.cast(tf.gather(p, sorted_indices), dtype=tf.float32)
    r_sorted = tf.cast(tf.gather(r, sorted_indices), dtype=tf.float32)

    # Calculate the cumulative sum of sorted probabilities.
    cumsum_p = tf.math.cumsum(p_sorted)

    # Create a tensor that evaluates to 1 if the condition is met, otherwise 0.
    mask = tf.cast(tf.math.less(cumsum_p, percent), dtype=tf.float32)

    # Use mask to filter and sum the required elements for CVaR.
    cvar_numerator = tf.reduce_sum(mask * p_sorted * r_sorted)

    # Compute the last remaining portion that exceeds the cut-off percentage.
    last_portion_index = tf.math.argmax(tf.math.greater_equal(cumsum_p, percent))
    last_portion = (percent - cumsum_p[last_portion_index - 1]) * r_sorted[
        last_portion_index
    ]

    # Calculate the final CVaR.
    cvar_result = (cvar_numerator + last_portion) / percent

    return cvar_result


def cvar_from_circuit(circuit: Circuit, nsamples: int, Q: Tensor, alpha: float) -> Any:
    """
    Directly calculate the Conditional Value at Risk (CVaR) from a circuit.
    The CVaR depends on a bunch of measurements.

    :param circuit: The quantum circuit used to prepare the state.
    :param nsamples: The number of samples to take for measurements.
    :param Q: The Q-matrix representing the Quadratic Unconstrained Binary Optimization (QUBO) problem.
    :param alpha: The cut-off percentage for CVaR.
    :return: The calculated CVaR value.
    """
    # Obtain and normalize measurement results
    measurement_data = circuit.state()
    results = measurement_results(
        measurement_data, counts=nsamples, format="sample_int", jittable=True
    )
    n_counts = tf.shape(results)[0]

    # Determine the number of qubits in the circuit and generate all possible states
    n_qubits = len(Q)
    all_states = tf.constant([format(i, f"0{n_qubits}b") for i in range(2**n_qubits)])
    all_binary = tf.reshape(
        tf.strings.to_number(tf.strings.bytes_split(all_states), tf.float32),
        (2**n_qubits, n_qubits),
    )
    all_decimal = tf.range(2**n_qubits, dtype=tf.int32)

    # Convert the Q matrix to a TensorFlow tensor
    Q_tensor = tf.convert_to_tensor(Q, dtype=tf.float32)

    # calculate cost values
    values = tf.reduce_sum(all_binary * tf.matmul(all_binary, Q_tensor), axis=1)

    # Count the occurrences of each state and calculate probabilities
    state_counts = tf.reduce_sum(
        tf.cast(tf.equal(tf.reshape(results, [-1, 1]), all_decimal), tf.int32), axis=0
    )
    probabilities = tf.cast(state_counts, dtype=tf.float32) / tf.cast(
        n_counts, dtype=tf.float32
    )

    # Calculate CVaR
    cvar_result = cvar_value(values, probabilities, alpha)

    return cvar_result


def cvar_from_expectation(circuit: Circuit, Q: Tensor, alpha: float) -> Any:
    """
    Calculate the Conditional Value at Risk (CVaR) from the expectation values of a quantum circuit.

    :param circuit: The quantum circuit.
    :param Q: The Q-matrix representing the Quadratic Unconstrained Binary Optimization (QUBO) problem.
    :param alpha: The cut-off percentage for CVaR.
    :return: The calculated CVaR value.
    """

    # Calculate the probability amplitudes for quantum circuit outcomes.
    prob = tf.convert_to_tensor(circuit.probability(), dtype=tf.float32)

    # Generate all possible binary states for the given Q-matrix.
    n_qubits = len(Q)
    all_states = tf.constant(
        [format(i, "0" + str(n_qubits) + "b") for i in range(2 ** len(Q))]
    )
    all_binary = tf.reshape(
        tf.strings.to_number(tf.strings.bytes_split(all_states), tf.float32),
        (2**n_qubits, n_qubits),
    )

    # Convert the Q-matrix to a TensorFlow tensor.
    Q_tensor = tf.convert_to_tensor(Q, dtype=tf.float32)

    # calculate cost values
    elementwise_product = tf.multiply(all_binary, tf.matmul(all_binary, Q_tensor))
    values = tf.reduce_sum(elementwise_product, axis=1)

    # Calculate the CVaR value using the computed values and the probability distribution.
    cvar_result = cvar_value(values, prob, alpha)

    return cvar_result


def cvar_loss(
    nlayers: int,
    Q: Tensor,
    nsamples: int,
    alpha: float,
    expectation_based: bool,
    params: List[float],
) -> Any:
    """
    Calculate the CVaR loss for a given QUBO problem using the QAOA ansatz.

    :param nlayers: The number of layers (depth) in the QAOA ansatz.
    :param Q: The Q-matrix representing the Quadratic Unconstrained Binary Optimization (QUBO) problem.
    :param nsamples: The number of samples to take for measurements in the CVaR calculation.
    :param alpha: The cut-off percentage for CVaR.
    :param expectation_based: A flag indicating the type of CVaR ansatz (measurement-based or expectation-based).
    :param params: The parameters for the QAOA ansatz circuit.
    :return: The calculated CVaR loss.
    """

    pauli_terms, weights, _ = QUBO_to_Ising(Q)

    c = QAOA_ansatz_for_Ising(params, nlayers, pauli_terms, weights)
    # Generate the QAOA ansatz circuit for the given parameters.

    if expectation_based is False:
        return cvar_from_circuit(c, nsamples, Q, alpha)
        # Calculate CVaR using circuit-based measurement results.
    elif expectation_based is True:
        return cvar_from_expectation(c, Q, alpha)
        # Calculate CVaR using expectation values of the circuit.
    else:
        raise ValueError("Invalid CVaR ansatz type.")
        # Raise an error if an invalid CVaR ansatz type is provided.


def QUBO_QAOA_cvar(
    Q: Tensor,
    nlayers: int,
    alpha: float,
    nsamples: int = 1000,
    callback: Optional[Callable[[List[float], float], None]] = None,
    expectation_based: bool = False,
    maxiter: int = 1000,
    init_params: Optional[Tuple[float,]] = None,
) -> Array:
    """
    Perform the QUBO QAOA optimization with CVaR as the loss function.

    :param Q: The n-by-n square and symmetric Q-matrix representing the QUBO problem.
    :param ansatz: The ansatz function to be used for QAOA.
    :param nlayers: The number of layers (depth) in the QAOA ansatz.
    :param alpha: The cut-off percentage for CVaR.
    :param nsamples: The number of samples for measurements in the CVaR calculation. Default is 1000.
    :param callback: A callback function to be called after each iteration. Default is None.
    :param expectation_based: A flag indicating the type of CVaR ansatz (measurement-based or expectation-based).
        Default is False.
    :param maxiter: The maximum number of iterations for the optimization. Default is 1000.
    :return: The optimized parameters for the ansatz circuit.
    """
    loss = partial(cvar_loss, nlayers, Q, nsamples, alpha, expectation_based)

    f_scipy = scipy_interface(loss, shape=(2 * nlayers,), jit=True, gradient=False)

    if init_params is None:
        params = backend.implicit_randn(shape=[2 * nlayers], stddev=0.5)
        # If init_params is not provided, initialize the parameters randomly.
    else:
        params = init_params
        # If init_params is provided, use the provided parameters.

    # Initialize the parameters for the ansatz circuit.
    params = backend.implicit_randn(shape=[2 * nlayers], stddev=0.5)

    r = optimize.minimize(
        f_scipy,
        params,
        method="COBYLA",
        callback=callback,
        options={"maxiter": maxiter},
    )
    # Perform the optimization using the COBYLA method from scipy.optimize.

    return r.x
    # Return the optimized parameters for the ansatz circuit.
