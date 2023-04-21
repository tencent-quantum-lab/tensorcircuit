"""
demo example of mipt in tc style
"""

from functools import partial
import time
import numpy as np
from scipy import stats
import tensorcircuit as tc

K = tc.set_backend("jax")
# tf backend is slow (at least on cpu)


@partial(K.jit, static_argnums=(2, 3, 4))
def circuit_output(random_matrix, status, n, d, p):
    """
    mipt circuit

    :param random_matrix: a float or complex tensor containing 4*4 random haar matrix wth size [d*n, 4, 4]
    :type random_matrix: _type_
    :param status: a int tensor with element in 0 or 1 or 2 (no meausrement) with size d*n
    :type status: _type_
    :param n: number of qubits
    :type n: _type_
    :param d: number of depth
    :type d: _type_
    :param p: measurement ratio
    :type p: float
    :return: output state
    """
    random_matrix = K.reshape(random_matrix, [d, n, 4, 4])
    status = K.reshape(status, [d, n])
    inputs = None
    for j in range(d):
        if inputs is None:
            c = tc.Circuit(n)
        else:
            c = tc.Circuit(n, inputs=inputs)
        for i in range(0, n, 2):
            c.unitary(i, (i + 1) % n, unitary=random_matrix[j, i])
        for i in range(1, n, 2):
            c.unitary(i, (i + 1) % n, unitary=random_matrix[j, i])
        inputs = c.state()
        c = tc.Circuit(n, inputs=inputs)
        for i in range(n):
            c.general_kraus(
                [
                    np.sqrt(p) * np.array([[1.0, 0], [0, 0]]),
                    np.sqrt(p) * np.array([[0, 0], [0, 1.0]]),
                    np.sqrt(1 - p) * np.eye(2),
                ],
                i,
                status=status[j, i],
            )
            inputs = c.state()
            c = tc.Circuit(n, inputs=inputs)
        inputs = c.state()
        inputs /= K.norm(inputs)
    return inputs


@partial(K.jit, static_argnums=(2, 3, 4))
def cals(random_matrix, status, n, d, p):
    state = circuit_output(random_matrix, status, n, d, p)
    rho = tc.quantum.reduced_density_matrix(state, cut=[i for i in range(n // 2)])
    return tc.quantum.entropy(rho), tc.quantum.renyi_entropy(rho, k=2)


if __name__ == "__main__":
    n = 12
    d = 12
    st = np.random.uniform(size=[d * n])
    ## assume all X gate instead
    rm = [stats.unitary_group.rvs(4) for _ in range(d * n)]
    rm = [r / np.linalg.det(r) for r in rm]
    rm = np.stack(rm)
    time0 = time.time()
    print(cals(rm, st, n, d, 0.1))
    time1 = time.time()
    st = np.random.uniform(size=[d * n])
    print(cals(rm, st, n, d, 0.1))
    time2 = time.time()
    print(f"compiling time {time1-time0}, running time {time2-time1}")
