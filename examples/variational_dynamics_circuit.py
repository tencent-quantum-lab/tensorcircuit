"""
Variational quantum simulation by directly contruct circuit for matrix elements
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorcircuit as tc

tc.set_backend("tensorflow")
tc.set_dtype("complex128")


# realize R gates in paper
def R_gate(k, c, ODE_theta):
    if door[k][0] == 0:
        c.rx(door[k][1] + 1, theta=ODE_theta[k])
    if door[k][0] == 1:
        c.ry(door[k][1] + 1, theta=ODE_theta[k])
    if door[k][0] == 2:
        c.rz(door[k][1] + 1, theta=ODE_theta[k])
    if door[k][0] == 3:
        c.rxx(door[k][1] + 1, door[k][2] + 1, theta=ODE_theta[k])
    if door[k][0] == 4:
        c.ryy(door[k][1] + 1, door[k][2] + 1, theta=ODE_theta[k])
    if door[k][0] == 5:
        c.rzz(door[k][1] + 1, door[k][2] + 1, theta=ODE_theta[k])
    if door[k][0] == 6:
        c.crx(door[k][1] + 1, door[k][2] + 1, theta=ODE_theta[k])
    if door[k][0] == 7:
        c.cry(door[k][1] + 1, door[k][2] + 1, theta=ODE_theta[k])
    if door[k][0] == 8:
        c.crz(door[k][1] + 1, door[k][2] + 1, theta=ODE_theta[k])


# realize U and H gates in paper
def U_H_gate(k, UHgate):
    if UHgate[k][0] == 0:
        gate_now = tc.gates.multicontrol_gate(
            np.kron(tc.gates._x_matrix, np.eye(2)), [1]
        )
    if UHgate[k][0] == 1:
        gate_now = tc.gates.multicontrol_gate(
            np.kron(tc.gates._y_matrix, np.eye(2)), [1]
        )
    if UHgate[k][0] == 2:
        gate_now = tc.gates.multicontrol_gate(
            np.kron(tc.gates._z_matrix, np.eye(2)), [1]
        )
    if UHgate[k][0] == 3:
        gate_now = tc.gates.multicontrol_gate(tc.gates._xx_matrix, [1])
    if UHgate[k][0] == 4:
        gate_now = tc.gates.multicontrol_gate(tc.gates._yy_matrix, [1])
    if UHgate[k][0] == 5:
        gate_now = tc.gates.multicontrol_gate(tc.gates._zz_matrix, [1])
    if UHgate[k][0] == 6:
        gate_now = tc.gates.multicontrol_gate(tc.gates._x_matrix, [1, 1])
    if UHgate[k][0] == 7:
        gate_now = tc.gates.multicontrol_gate(tc.gates._y_matrix, [1, 1])
    if UHgate[k][0] == 8:
        gate_now = tc.gates.multicontrol_gate(tc.gates._z_matrix, [1, 1])
    return gate_now.eval_matrix()


# use quantum circuit to calculate coefficient of variation A and C in paper
def Calculation_A(theta_x, is_k, is_q, ODE_theta):
    # mod: a in paper; theta_x: theta in paper; k, q: A[k, q] or C[k] qth term(k <= q)
    c = tc.Circuit(N + 1, inputs=np.kron([1, 1] / np.sqrt(2), state))
    c.rz(0, theta=-theta_x)
    for i in range(len(door)):
        c.conditional_gate(is_k[i], [np.eye(2), tc.gates._x_matrix], 0)
        c.conditional_gate(
            is_k[i],
            [np.eye(8), U_H_gate(i, door)],
            0,
            door[i][1] + 1,
            door[i][2] + 1,
        )
        c.conditional_gate(is_k[i], [np.eye(2), tc.gates._x_matrix], 0)
        c.conditional_gate(
            is_q[i],
            [np.eye(8), U_H_gate(i, door)],
            0,
            door[i][1] + 1,
            door[i][2] + 1,
        )
        R_gate(i, c, ODE_theta)
    pstar = c.expectation([np.array([[1, 1], [1, 1]]) / 2, [0]])
    return 2 * pstar - 1


Calculation_A_vmap = tc.backend.jit(
    tc.backend.vmap(Calculation_A, vectorized_argnums=[0, 1, 2])
)


def Calculation_C(theta_x, is_k, is_q, ODE_theta):
    # mod: a in paper; theta_x: theta in paper; k, q: A[k, q] or C[k] qth term
    c = tc.Circuit(N + 1, inputs=np.kron([1, 1] / np.sqrt(2), state))
    c.rz(0, theta=-theta_x)
    for i in range(len(door)):
        c.conditional_gate(is_k[i], [np.eye(2), tc.gates._x_matrix], 0)
        c.conditional_gate(
            is_k[i],
            [np.eye(8), U_H_gate(i, door)],
            0,
            door[i][1] + 1,
            door[i][2] + 1,
        )
        c.conditional_gate(is_k[i], [np.eye(2), tc.gates._x_matrix], 0)
        R_gate(i, c, ODE_theta)
    for i in range(len(h_door)):
        c.conditional_gate(
            is_q[i],
            [np.eye(8), U_H_gate(i, h_door)],
            0,
            h_door[i][1] + 1,
            h_door[i][2] + 1,
        )
    pstar = c.expectation([np.array([[1, 1], [1, 1]]) / 2, [0]])
    return 2 * pstar - 1


Calculation_C_vmap = tc.backend.jit(
    tc.backend.vmap(Calculation_C, vectorized_argnums=[0, 1, 2])
)


# use original quantum circuit simulate with c
@tc.backend.jit
def simulation(ODE_theta):
    c = tc.Circuit(N, inputs=state)
    for k in range(len(door)):
        if door[k][0] == 0:
            c.rx(door[k][1], theta=ODE_theta[k])
        if door[k][0] == 1:
            c.ry(door[k][1], theta=ODE_theta[k])
        if door[k][0] == 2:
            c.rz(door[k][1], theta=ODE_theta[k])
        if door[k][0] == 3:
            c.rxx(door[k][1], door[k][2], theta=ODE_theta[k])
        if door[k][0] == 4:
            c.ryy(door[k][1], door[k][2], theta=ODE_theta[k])
        if door[k][0] == 5:
            c.rzz(door[k][1], door[k][2], theta=ODE_theta[k])
        if door[k][0] == 6:
            c.crx(door[k][1], door[k][2], theta=ODE_theta[k])
        if door[k][0] == 7:
            c.cry(door[k][1], door[k][2], theta=ODE_theta[k])
        if door[k][0] == 8:
            c.crz(door[k][1], door[k][2], theta=ODE_theta[k])
    return tc.backend.real(c.expectation([tc.gates.x(), [1]]))


def numdiff(i):
    return (i + 1) % N


if __name__ == "__main__":
    # l: layers; h and J: coefficient of Hamiltonian;
    # L_var and L_num: results of variation method and numerical method
    N = 3
    l = 4
    J = 1 / 4
    dt = 0.005
    t = 1
    h = []
    L_var = []
    L_num = []
    x_value = []

    how_variation = 0  # 0:McLachlan; 1:time-dependent

    # the priciple correspond with all gates
    # the first term: 0rx,1ry,2rz,3rxx,4ryy,5rzz,6crx,7cry,8crz;
    # the second and the third term: num/ctrl+num
    # f: coefficient with simulation gates in paper
    door = []
    h_door = []
    f = []
    for k in range(l):
        for i in range(N):
            f.append(-0.5j)
            door.append([0, i, numdiff(i)])
        for i in range(N - 1):
            f.append(-0.5j)
            door.append([5, i, i + 1])
        for i in range(N - 1):
            f.append(-0.5j)
            door.append([3, i, i + 1])
    for i in range(N):
        h.append(1)
        h_door.append([0, i, numdiff(i)])
    for i in range(N - 1):
        h.append(J)
        h_door.append([5, i, i + 1])
    f = tf.constant(f, dtype="complex64")
    h = tf.constant(h, dtype="float32")

    # initial state
    state = np.zeros(1 << N)
    state[0] = 1

    # numerical realize H
    ls = []
    weight = []
    for q in range(len(h_door)):
        if h_door[q][0] == 0:
            r = [0 for _ in range(N)]
            r[h_door[q][1]] = 1
        if h_door[q][0] == 1:
            r = [0 for _ in range(N)]
            r[h_door[q][1]] = 2
        if h_door[q][0] == 2:
            r = [0 for _ in range(N)]
            r[h_door[q][1]] = 3
        if h_door[q][0] == 3:
            r = [0 for _ in range(N)]
            r[h_door[q][1]] = 1
            r[h_door[q][2]] = 1
        if h_door[q][0] == 4:
            r = [0 for _ in range(N)]
            r[h_door[q][1]] = 2
            r[h_door[q][2]] = 2
        if h_door[q][0] == 5:
            r = [0 for _ in range(N)]
            r[h_door[q][1]] = 3
            r[h_door[q][2]] = 3
        ls.append(r)
        weight.append(h[q])
    ls = tc.array_to_tensor(ls)
    weight = tc.array_to_tensor(weight)
    H = tc.quantum.PauliStringSum2Dense(ls, weight, numpy=False)

    # variation realize
    ODE_theta = tf.zeros(len(door), dtype="float64")

    a_batch_theta = []
    a_batch_is_k = []
    a_batch_is_q = []
    for k in range(len(door)):
        for q in range(len(door)):
            is_k = [0 for _ in range(len(door))]
            is_k[k] = 1
            is_q = [0 for _ in range(len(door))]
            is_q[q] = 1
            if how_variation == 0:
                a_batch_theta.append(np.angle(f[q]) - np.angle(f[k]))
            else:
                a_batch_theta.append(np.angle(f[q]) - np.angle(f[k]) - math.pi / 2)
            a_batch_is_k.append(is_k)
            a_batch_is_q.append(is_q)
    a_batch_theta = tc.array_to_tensor(a_batch_theta)
    a_batch_is_k = tf.constant(a_batch_is_k)
    a_batch_is_q = tf.constant(a_batch_is_q)

    c_batch_theta = []
    c_batch_is_k = []
    c_batch_is_q = []
    for k in range(len(door)):
        for q in range(len(h_door)):
            is_k = [0 for _ in range(len(door))]
            is_k[k] = 1
            is_q = [0 for _ in range(len(door))]
            is_q[q] = 1
            c_batch_is_k.append(is_k)
            c_batch_is_q.append(is_q)
            if how_variation == 0:
                c_batch_theta.append(np.angle(h[q]) - np.angle(f[k]) - math.pi / 2)
            else:
                c_batch_theta.append(np.angle(h[q]) - np.angle(f[k]) + math.pi)
    c_batch_theta = tc.array_to_tensor(c_batch_theta)
    c_batch_is_k = tf.constant(c_batch_is_k)
    c_batch_is_q = tf.constant(c_batch_is_q)

    for T in range(int(t / dt)):
        # calculate coefficient in paper

        vmap_result = Calculation_A_vmap(
            a_batch_theta, a_batch_is_k, a_batch_is_q, ODE_theta
        )
        A = tf.cast(
            tf.tensordot(tf.abs(f), tf.abs(f), 0), dtype="float64"
        ) * tf.reshape(
            tc.backend.cast(vmap_result, dtype="float64"), [len(door), len(door)]
        )

        vmap_result = Calculation_C_vmap(
            c_batch_theta, c_batch_is_k, c_batch_is_q, ODE_theta
        )
        C = tf.reduce_sum(
            tf.cast(tf.tensordot(tf.abs(f), tf.abs(h), 0), dtype="float64")
            * tf.reshape(
                tc.backend.cast(vmap_result, dtype="float64"), [len(door), len(h_door)]
            ),
            1,
        )

        # calculate parameter and its derivative
        A += np.eye(len(door)) * 1e-7
        ODE_dtheta = tc.backend.solve(A, C)
        ODE_theta += ODE_dtheta * dt

        # numerical results
        ep = np.array(tc.backend.expm(-1j * H * (T + 1) * dt)) @ state
        L_num.append(
            np.array(
                tc.backend.real(
                    tc.expectation([tc.gates.x(), [1]], ket=ep.astype("complex128"))
                )
            )
        )

        # variation results
        L_var.append(tc.backend.numpy(simulation(ODE_theta)).tolist())

        x_value.append(round((T + 1) * dt, 3))
        print("Now time:", x_value[T], "Loss:", L_num[T] - L_var[T])

    plt.plot(x_value, L_var, color="green")
    plt.plot(x_value, L_num, color="red")
    plt.show()
