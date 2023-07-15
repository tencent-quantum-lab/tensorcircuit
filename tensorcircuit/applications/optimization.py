"""
modules for QUBO problems in QAOA
"""

from typing import List, Callable, Any, Optional, Tuple
from functools import partial

import numpy as np
import tensorflow as tf
import scipy.optimize as optimize

from ..cons import backend, get_backend
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
    :param ncircuits (optional): The number of circuits when using vectorized variational adjoint gradient. Default is 10.
    :param init_params (optional): The initial parameters for the ansatz circuit. Default is None, which initializes the parameters randomly.
    :paran mixer (optional): The mixer operator to use. Default is "X". The other options are "XY" and "ZZ".
    :param learning_rate (optional): The learning rate for the Adam optimizer. Default is 1e-2.
    :param callback (optional): A callback function that is executed during each iteration. Default is None.
    :param full_coupling (optional): A flag indicating whether to use all-to-all coupling in mixers. Default is False.
    :return params: The optimized parameters for the ansatz circuit.
    """
    if backend != get_backend("tensorflow"):
        raise ValueError("`QUBO_QAOA` is designed for tensorflow backend.")
        # Check if the backend is set to TensorFlow. Raise an error if it is not.

    pauli_terms, weights, offset = QUBO_to_Ising(Q)

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
        if vvag == True:
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

    for i in range(iterations):
        loss, grads = loss_val_grad(params)
        # Calculate the loss and gradients using the loss_val_grad_jit function.

        params = opt.update(grads, params)
        # Update the parameters using the optimizer and gradients.

        if callback is not None:
            callback(loss, params)
        # Execute the callback function with the current loss and parameters.

    return params
    # Return the optimized parameters for the ansatz circuit.


def cvar_value(r: List[float], p: List[float], percent: float) -> float:
    """
    Calculate the Conditional Value at Risk (CVaR) according to the measurement results.

    :param r: The results showing after measurements.
    :param p: Probabilities corresponding to each result.
    :param percent: The cut-off percentage of CVaR.
    :return: The calculated CVaR value.
    """
    sorted_indices = np.argsort(r)
    p = np.array(p)[sorted_indices]
    r = np.array(r)[sorted_indices]

    sump = 0.0  # The sum of probabilities.
    count = 0
    cvar_result = 0.0

    # Iterate over the sorted results and calculate CVaR.
    while sump < percent:
        if round(sump + p[count], 6) >= percent:
            # Add the remaining portion of the last result that exceeds the cut-off percentage.
            cvar_result += r[count] * (percent - sump)
            count += 1
            break
        else:
            # Add the entire result to the CVaR calculation.
            sump += p[count]
            cvar_result += r[count] * p[count]
            count += 1

    cvar_result /= percent
    return cvar_result


def cvar_from_circuit(
    circuit: Circuit, nsamples: int, Q: Tensor, alpha: float
) -> float:
    """
    Directly calculate the Conditional Value at Risk (CVaR) from a circuit.
    The CVaR depends on a bunch of measurements.

    :param circuit: The quantum circuit used to prepare the state.
    :param nsamples: The number of samples to take for measurements.
    :param Q: The Q-matrix representing the Quadratic Unconstrained Binary Optimization (QUBO) problem.
    :param alpha: The cut-off percentage for CVaR.
    :return: The calculated CVaR value.
    """
    s = circuit.state()
    results = measurement_results(
        s, counts=nsamples, format="count_dict_bin"
    )  # Get readouts from the measurements.
    results = {k: v / nsamples for k, v in results.items()}  # Normalize the results.
    values = []  # List to store the measurement values.
    probabilities = []  # List to store the corresponding probabilities.

    # Iterate over the measurement results and calculate the values and probabilities.
    for k, v in results.items():
        x = np.array([int(bit) for bit in k])
        values.append(np.dot(x, np.dot(Q, x)))
        probabilities.append(v)

    cvar_result = cvar_value(values, probabilities, alpha)
    # Calculate the CVaR using the cvar_value function.

    return cvar_result


def cvar_from_expectation(circuit: Circuit, Q: Tensor, alpha: float) -> float:
    """
    Calculate the Conditional Value at Risk (CVaR) from the expectation values of a quantum circuit.

    :param circuit: The quantum circuit.
    :param Q: The Q-matrix representing the Quadratic Unconstrained Binary Optimization (QUBO) problem.
    :param alpha: The cut-off percentage for CVaR.
    :return: The calculated CVaR value.
    """
    prob = circuit.probability()  # Get the probabilities of the circuit states.
    prob /= np.sum(prob)
    states = []

    # Generate all possible binary states based on the length of Q.
    for i in range(2 ** len(Q)):
        a = f"{bin(i)[2:]:0>{len(Q)}}"
        states.append(a)

    values = []
    for state in states:
        x = np.array([int(bit) for bit in state])
        values.append(np.dot(x, np.dot(Q, x)))
    # Calculate the values by taking the dot product of each state with the Q-matrix.

    cvar_result = cvar_value(values, prob, alpha)
    # Calculate the CVaR using the cvar_value function.

    return cvar_result


def cvar_loss(
    nlayers: int,
    Q: Tensor,
    nsamples: int,
    alpha: float,
    fake: bool,
    params: List[float],
) -> float:
    """
    Calculate the CVaR loss for a given QUBO problem using the QAOA ansatz.

    :param nlayers: The number of layers (depth) in the QAOA ansatz.
    :param Q: The Q-matrix representing the Quadratic Unconstrained Binary Optimization (QUBO) problem.
    :param nsamples: The number of samples to take for measurements in the CVaR calculation.
    :param alpha: The cut-off percentage for CVaR.
    :param fake: A flag indicating the type of CVaR ansatz (circuit-based or expectation-based).
    :param params: The parameters for the QAOA ansatz circuit.
    :return: The calculated CVaR loss.
    """
    pauli_terms, weights, offset = QUBO_to_Ising(Q)

    c = QAOA_ansatz_for_Ising(params, nlayers, pauli_terms, weights)
    # Generate the QAOA ansatz circuit for the given parameters.

    if fake == False:
        return cvar_from_circuit(c, nsamples, Q, alpha)
        # Calculate CVaR using circuit-based measurement results.
    elif fake == True:
        return cvar_from_expectation(c, Q, alpha)
        # Calculate CVaR using expectation values of the circuit.
    else:
        raise ValueError("Invalid CVaR ansatz type.")
        # Raise an error if an invalid CVaR ansatz type is provided.


def QUBO_QAOA_cvar(
    Q: Tensor,
    nlayers: int,
    alpha: int,
    nsamples: int = 1000,
    callback: Optional[Callable[[List[float], float], None]] = None,
    fake: bool = False,
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
    :param fake: A flag indicating the type of CVaR ansatz (circuit-based or expectation-based). Default is False.
    :param maxiter: The maximum number of iterations for the optimization. Default is 1000.
    :return: The optimized parameters for the ansatz circuit.
    """
    loss = partial(cvar_loss, nlayers, Q, nsamples, alpha, fake)

    f_scipy = scipy_interface(loss, shape=[2 * nlayers], jit=False, gradient=False)

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
        # bounds=[(0, (2 - np.mod(i, 2))*np.pi) for i in range(2*nlayers)]
    )
    # Perform the optimization using the COBYLA method from scipy.optimize.

    return r.x
    # Return the optimized parameters for the ansatz circuit.
