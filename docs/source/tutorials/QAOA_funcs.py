###
### functions for QAOA problems
###

from typing import List, Tuple, Callable, Any
import tensorcircuit as tc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
from functools import partial
import scipy.optimize as optimize

Array = any
Tensor = Any


# moved
def QUBO_to_Ising(Q: List[list]) -> Tuple[List[list], list, float]:
    """
    Cnvert the Q matrix into a the indication of pauli terms, the corresponding weights, and the offset.
    The outputs are used to construct an Ising Hamiltonian for QAOA.

    :param Q: The n-by-n square and symmetric Q-matrix.
    :return pauli_terms: A list of 0/1 series, where each element represents a Pauli term.
    A value of 1 indicates the presence of a Pauli-Z operator, while a value of 0 indicates its absence.
    :return weights: A list of weights corresponding to each Pauli term.
    :return offset: A float representing the offset term of the Ising Hamiltonian.
    """

    # input is n-by-n symmetric numpy array corresponding to Q-matrix
    # output is the components of Ising Hamiltonian

    n = Q.shape[0]

    # square matrix check
    if Q[0].shape[0] != n:
        raise ValueError("Matrix is not a square matrix.")

    offset = (
        np.triu(Q, 0).sum() / 2
    )  # Calculate the offset term of the Ising Hamiltonian
    pauli_terms = []  # List to store the Pauli terms
    weights = (
        -np.sum(Q, axis=1) / 2
    )  # Calculate the weights corresponding to each Pauli term

    for i in range(n):
        term = np.zeros(n)
        term[i] = 1
        pauli_terms.append(
            term.tolist()
        )  # Add a Pauli term corresponding to a single qubit

    for i in range(n - 1):
        for j in range(i + 1, n):
            term = np.zeros(n)
            term[i] = 1
            term[j] = 1
            pauli_terms.append(
                term.tolist()
            )  # Add a Pauli term corresponding to a two-qubit interaction

            weight = (
                Q[i][j] / 2
            )  # Calculate the weight for the two-qubit interaction term
            weights = np.concatenate(
                (weights, weight), axis=None
            )  # Add the weight to the weights list

    return pauli_terms, weights, offset


def Ising_loss(c: tc.Circuit, pauli_terms: List[list], weights: list) -> float:
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

    return K.real(loss)


def QAOA_loss(
    nlayers: int, pauli_terms: List[list], weights: list, params: list
) -> float:
    """
    computes the loss function for the Quantum Approximate Optimization Algorithm (QAOA) applied to the Ising model.

    :param nlayers: The number of layers in the QAOA ansatz.
    :param pauli_terms: A list of Pauli terms, where each term is represented as a list of 0/1 series.
    :param weights: A list of weights corresponding to each Pauli term.
    :param params: A list of parameter values used in the QAOA ansatz.
    :return: The computed loss value.
    """
    c = QAOA_ansatz_for_Ising(params, nlayers, pauli_terms, weights)
    # Obtain the quantum circuit using QAOA_from_Ising function

    return Ising_loss(c, pauli_terms, weights)
    # Compute the Ising loss using Ising_loss function on the obtained circuit


def QUBO_QAOA(
    Q: List[list],
    ansatz: Callable[[list, int, List[list], list], tc.Circuit],
    nlayers: int,
    iterations: int,
    vvag: bool = False,
    ncircuits: int = 10,
) -> list:
    """
    Performs the QAOA on a given QUBO problem.

    :param Q: The n-by-n square and symmetric Q-matrix representing the QUBO problem.
    :param ansatz: The ansatz function to be used for the QAOA.
    :param nlayers: The number of layers in the QAOA ansatz.
    :param iterations: The number of iterations to run the optimization.
    :param vvag (optional): A flag indicating whether to use vectorized variational adjoint gradient. Default is False.
    :param ncircuits (optional): The number of circuits when using vectorized variational adjoint gradient. Default is 10.
    :return params: The optimized parameters for the ansatz circuit.
    """
    try:
        K
    except NameError:
        print("select a backend and assign it to K.")

    pauli_terms, weights, offset = QUBO_to_Ising(Q)
    learning_rate = 1e-2

    loss_val_grad = K.value_and_grad(partial(ansatz, nlayers, pauli_terms, weights))
    params = K.implicit_randn(shape=[2 * nlayers], stddev=0.5)
    # Initialize the parameters for the ansatz circuit

    if vvag == True:
        loss_val_grad = tc.backend.vvag(loss_val_grad, argnums=0, vectorized_argnums=0)
        params = K.implicit_randn(shape=[ncircuits, 2 * nlayers], stddev=0.1)
        # Use vectorized variational adjoint gradient (vvag) if vvag flag is set to True

    loss_val_grad_jit = K.jit(loss_val_grad, static_argnums=(1, 2))

    opt = K.optimizer(tf.keras.optimizers.Adam(learning_rate))
    # Define the optimizer (Adam optimizer) with the specified learning rate

    for i in range(iterations):
        loss, grads = loss_val_grad_jit(params)
        # Calculate the loss and gradients using the loss_val_grad_jit function

        params = opt.update(grads, params)
        # Update the parameters using the optimizer and gradients

        if i % 100 == 0:  # print the cost every 100 iterations
            print(K.numpy(loss))

    return params


# calcelled
def print_result_prob(c: tc.Circuit, wrap: bool = False, reverse: bool = False) -> None:
    """
    Print the results and probabilities of a given quantum circuit.
    The default order is from the highest probability to the lowest one

    :param c: The quantum circuit to print the results and probabilities.
    :param wrap (optional): A flag indicating whether to wrap the output. Default is False.
    :param reverse (optional): A flag indicating whether to reverse the order of the output. Default is False.
    """
    try:
        K
    except NameError:
        print("select a backend and assign it to K.")

    states = []
    n_qubits = c._nqubits
    for i in range(2**n_qubits):
        a = f"{bin(i)[2:]:0>{n_qubits}}"
        states.append(a)
        # Generate all possible binary states for the given number of qubits

    probs = K.numpy(c.probability()).round(decimals=4)
    # Calculate the probabilities of each state using the circuit's probability method

    sorted_indices = np.argsort(probs)[::-1]
    if reverse == True:
        sorted_indices = sorted_indices[::-1]
    state_sorted = np.array(states)[sorted_indices]
    prob_sorted = np.array(probs)[sorted_indices]
    # Sort the states and probabilities in descending order based on the probabilities

    print("\n-------------------------------------")
    print("    selection\t  |\tprobability")
    print("-------------------------------------")
    if wrap == False:
        for i in range(len(states)):
            print("%10s\t  |\t  %.4f" % (state_sorted[i], prob_sorted[i]))
            # Print the sorted states and their corresponding probabilities
    elif wrap == True:
        for i in range(4):
            print("%10s\t  |\t  %.4f" % (state_sorted[i], prob_sorted[i]))
        print("               ... ...")
        for i in range(-4, -1):
            print("%10s\t  |\t  %.4f" % (state_sorted[i], prob_sorted[i]))
    print("-------------------------------------")


# calcelled
def print_result_cost(
    c: tc.Circuit, Q: List[list], wrap: bool = False, reverse: bool = False
) -> None:
    """
    Print the results and costs of a given quantum circuit.
    Specificly designed for the variational circuit.
    The default order is from the highest probability to the lowest one.

    :param c: The quantum circuit to print the results and probabilities.
    :param Q: The n-by-n square and symmetric Q-matrix representing the QUBO problem.
    :param wrap (optional): A flag indicating whether to wrap the output. Default is False.
    :param reverse (optional): A flag indicating whether to reverse the order of the output. Default is False.
    """
    cost_dict = {}
    states = []
    n_qubits = c._nqubits
    for i in range(2**n_qubits):
        a = f"{bin(i)[2:]:0>{n_qubits}}"
        states.append(a)
        # Generate all possible binary states for the given number of qubits
    for selection in states:
        x = np.array([int(bit) for bit in selection])
        cost_dict[selection] = np.dot(x, np.dot(Q, x))
    cost_sorted = dict(sorted(cost_dict.items(), key=lambda item: item[1]))
    if reverse == True:
        cost_sorted = dict(
            sorted(cost_dict.items(), key=lambda item: item[1], reverse=True)
        )
    num = 0
    print("\n-------------------------------------")
    print("    selection\t  |\t  cost")
    print("-------------------------------------")
    for k, v in cost_sorted.items():
        print("%10s\t  |\t%.4f" % (k, v))
        num += 1
        if (num >= 8) & (wrap == True):
            break
    print("-------------------------------------")


# cancelled
def print_Q_cost(Q: List[list], wrap: bool = False, reverse: bool = False) -> None:
    n_stocks = len(Q)
    states = []
    for i in range(2**n_stocks):
        a = f"{bin(i)[2:]:0>{n_stocks}}"
        n_ones = 0
        for j in a:
            if j == "1":
                n_ones += 1
        states.append(a)

    cost_dict = {}
    for selection in states:
        x = np.array([int(bit) for bit in selection])
        cost_dict[selection] = np.dot(x, np.dot(Q, x))
    cost_sorted = dict(sorted(cost_dict.items(), key=lambda item: item[1]))
    if reverse == True:
        cost_sorted = dict(
            sorted(cost_dict.items(), key=lambda item: item[1], reverse=True)
        )
    num = 0
    print("\n-------------------------------------")
    print("    selection\t  |\t  cost")
    print("-------------------------------------")
    for k, v in cost_sorted.items():
        print("%10s\t  |\t%.4f" % (k, v))
        num += 1
        if (num >= 8) & (wrap == True):
            break
    print("-------------------------------------")


# moved
class StockData:
    """
    convert real-world stock data to the inputs of QAOA.
    """

    def __init__(self, data: Tensor) -> None:
        """
        stock data object

        :param data: real-world stock data, in the form of several lists of daily price.
        """
        self.data = data  # add data
        self.n_stocks = len(data)  # num of stocks
        self.n_days = len(data[1])

        # check the number of days
        n_days = [len(i) for i in data]
        if max(n_days) != (sum(n_days) / len(n_days)):
            raise Exception("timespan of stocks should be the same")

        # calculate the daily percentage price change
        self.daily_change = []  # daily percentage price change
        for i in range(self.n_stocks):
            each_stcok = []
            for j in range(self.n_days - 1):
                each_stcok.append((data[i][j + 1] - data[i][j]) / data[i][j])
            self.daily_change.append(each_stcok)

    def get_return(self) -> Array:
        """
        :return ret: annualized return (mu)
        """
        ret = np.mean(self.daily_change, axis=1)
        return ret

    def get_covariance(self) -> Array:
        """
        :return cov: symmetric annualized covariance matrix (sigma)
        """
        return np.cov(self.daily_change)

    def get_pentalty(
        self, cov: Array, ret: Array, risk_pre: float, budget: int
    ) -> float:
        """
        calculate the pentalty using the method in https://link.springer.com/article/10.1007/s11128-022-03766-5
        brutal force is used

        :param cov: symmetrix annualized covariance matrix (sigma)
        :param ret: annualized return (ret)
        :param risk_pre: risk preference of the investor
        :param budge: the number of assets to be chosen for the portfolio
        """
        # get all fesible and unfeasible states
        self.f_state = []  # feasible states (num of '1's equal to budge)
        self.uf_state = []  # unfeasible states
        self.all_state = []
        for i in range(2**self.n_stocks):
            state = f"{bin(i)[2:]:0>{self.n_stocks}}"
            n_ones = 0
            for j in state:
                if j == "1":
                    n_ones += 1
            self.all_state.append(state)
            if n_ones == budget:
                self.f_state.append(state)
            else:
                self.uf_state.append(state)

        # determine the penalty factor
        mark = False
        penalty = 0  # initial value
        while mark == False:
            R = np.diag(ret)
            S = np.ones((self.n_stocks, self.n_stocks)) - 2 * budget * np.diag(
                np.ones(self.n_stocks)
            )
            Q = risk_pre * cov - R + penalty * S
            F = []
            for state in self.f_state:
                x = np.array([int(bit) for bit in state])
                F.append(np.dot(x, np.dot(Q, x)) + penalty * budget**2)
            Fmin = np.amin(F)
            Fbar = np.mean(F)
            F = []
            for state in self.uf_state:
                x = np.array([int(bit) for bit in state])
                F.append(np.dot(x, np.dot(Q, x)) + penalty * budget**2)
            Fmin_uf = np.amin(F)
            location = np.where(F == Fmin_uf)[0][0]
            if Fmin_uf < 0.5 * (Fmin + Fbar):
                n_ones = 0
                for j in self.uf_state[location]:
                    if j == "1":
                        n_ones += 1
                penalty += (0.5 * (Fmin + Fbar) - Fmin_uf) / (n_ones - budget) ** 2
                # mark = True
            else:
                mark = True  # ready to return the penalty
        return penalty


# moved
def QUBO_from_portfolio(cov: Array, mean: Array, q: float, B: int, t: float) -> Tensor:
    """
    convert portfolio parameters to a Q matrix
    :param cov: n-by-n covariance numpy array
    :param mean: numpy array of means
    :param q: the risk preference of investor
    :param B: budget
    :param t: penalty factor
    :return Q: n-by-n symmetric Q matrix
    """
    n = cov.shape[0]
    R = np.diag(mean)
    S = np.ones((n, n)) - 2 * B * np.diag(np.ones(n))

    Q = q * cov - R + t * S
    return Q


def print_output(c):
    n = c._nqubits
    N = 2**n
    # Calculate the total number of states based on the number of qubits

    x_label = r"$\left|{0:0" + str(n) + r"b}\right>$"
    labels = [x_label.format(i) for i in range(N)]
    # Generate labels for the x-axis representing the binary states

    plt.bar(range(N), c.probability())
    # Create a bar plot with the probabilities of each state

    plt.xticks(range(N), labels, rotation=70)
    # Set the x-axis ticks to the generated labels and rotate them for better visibility
