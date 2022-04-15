"""
DQAS-style optimization for discrete Clifford type circuit
"""

import numpy as np
import tensorflow as tf

import tensorcircuit as tc

ctype, rtype = tc.set_dtype("complex64")
K = tc.set_backend("tensorflow")

n = 6
nlayers = 4


def ansatz(structureo, structuret, preprocess="direct"):
    c = tc.Circuit(n)
    if preprocess == "softmax":
        structureo = K.softmax(structureo, axis=-1)
        structuret = K.softmax(structuret, axis=-1)
    elif preprocess == "most":
        structureo = K.onehot(K.argmax(structureo, axis=-1), num=7)
        structuret = K.onehot(K.argmax(structuret, axis=-1), num=3)
    elif preprocess == "direct":
        pass

    structureo = K.cast(structureo, ctype)
    structuret = K.cast(structuret, ctype)

    structureo = tf.reshape(structureo, shape=[nlayers, n, 7])
    structuret = tf.reshape(structuret, shape=[nlayers, n, 3])

    for i in range(n):
        c.H(i)
    for j in range(nlayers):
        for i in range(n):
            c.unitary(
                i,
                unitary=structureo[j, i, 0] * tc.gates.i().tensor
                + structureo[j, i, 1] * tc.gates.x().tensor
                + structureo[j, i, 2] * tc.gates.y().tensor
                + structureo[j, i, 3] * tc.gates.z().tensor
                + structureo[j, i, 4] * tc.gates.h().tensor
                + structureo[j, i, 5] * tc.gates.s().tensor
                + structureo[j, i, 6] * tc.gates.sd().tensor,
            )
        for i in range(n - 1):
            c.unitary(
                i,
                i + 1,
                unitary=structuret[j, i, 0] * tc.gates.ii().tensor
                + structuret[j, i, 1] * tc.gates.cnot().tensor
                + structuret[j, i, 2] * tc.gates.cz().tensor,
            )
    # loss = K.real(
    #     sum(
    #         [c.expectation_ps(z=[i, i + 1]) for i in range(n - 1)]
    #         + [c.expectation_ps(x=[i]) for i in range(n)]
    #     )
    # )
    s = c.state()
    loss = -K.real(tc.quantum.entropy(tc.quantum.reduced_density_matrix(s, cut=n // 2)))
    return loss


def sampling_from_structure(structures, batch=1):
    ch = structures.shape[-1]
    prob = K.softmax(K.real(structures), axis=-1)
    prob = K.reshape(prob, [-1, ch])
    p = prob.shape[0]
    r = np.stack(
        np.array(
            [np.random.choice(ch, p=K.numpy(prob[i]), size=[batch]) for i in range(p)]
        )
    )
    return r.transpose()


@K.jit
def best_from_structure(structures):
    return K.argmax(structures, axis=-1)


def nmf_gradient(structures, oh):
    """
    compute the Monte Carlo gradient with respect of naive mean-field probabilistic model
    """
    choice = K.argmax(oh, axis=-1)
    prob = K.softmax(K.real(structures), axis=-1)
    indices = K.transpose(
        K.stack([K.cast(tf.range(structures.shape[0]), "int64"), choice])
    )
    prob = tf.gather_nd(prob, indices)
    prob = K.reshape(prob, [-1, 1])
    prob = K.tile(prob, [1, structures.shape[-1]])

    return K.real(
        tf.tensor_scatter_nd_add(
            tf.cast(-prob, dtype=ctype),
            indices,
            tf.ones([structures.shape[0]], dtype=ctype),
        )
    )


nmf_gradient_vmap = K.jit(K.vmap(nmf_gradient, vectorized_argnums=1))
vf = K.jit(K.vmap(ansatz, vectorized_argnums=(0, 1)), static_argnums=2)

so = K.implicit_randn([nlayers * n, 7], stddev=0.1)
st = K.implicit_randn([nlayers * n, 3], stddev=0.1)
verbose = False
epochs = 2000
debug_step = 50
batch = 256
lr = tf.keras.optimizers.schedules.ExponentialDecay(0.05, 600, 0.5)
structure_opt = tc.backend.optimizer(tf.keras.optimizers.Adam(lr))

avcost = 0
avcost2 = 0
for epoch in range(epochs):  # iteration to update strcuture param
    batched_stuctureo = K.onehot(
        sampling_from_structure(so, batch=batch),
        num=so.shape[-1],
    )
    batched_stucturet = K.onehot(
        sampling_from_structure(st, batch=batch),
        num=st.shape[-1],
    )
    vs = vf(batched_stuctureo, batched_stucturet, "direct")
    avcost = K.mean(vs)
    go = nmf_gradient_vmap(so, batched_stuctureo)  # \nabla lnp
    gt = nmf_gradient_vmap(st, batched_stucturet)  # \nabla lnp
    go = K.mean(K.reshape(vs - avcost2, [-1, 1, 1]) * go, axis=0)
    gt = K.mean(K.reshape(vs - avcost2, [-1, 1, 1]) * gt, axis=0)

    # go = [(vs[i] - avcost2) * go[i] for i in range(batch)]
    # gt = [(vs[i] - avcost2) * gt[i] for i in range(batch)]
    # go = tf.math.reduce_mean(go, axis=0)
    # gt = tf.math.reduce_mean(gt, axis=0)
    avcost2 = avcost

    [so, st] = structure_opt.update([go, gt], [so, st])
    # so -= K.reshape(K.mean(so, axis=-1), [-1, 1])
    # st -= K.reshape(K.mean(st, axis=-1), [-1, 1])
    if epoch % debug_step == 0 or epoch == epochs - 1:
        print("----------epoch %s-----------" % epoch)
        print(
            "batched average loss: ",
            np.mean(vs),
            "minimum candidate loss: ",
            np.min(vs),
        )
        if verbose:
            print(gt)
            print(st)
            print(
                "strcuture parameter: \n",
                so.numpy(),
                "\n",
                st.numpy(),
            )

        cand_preseto = best_from_structure(so)
        cand_presett = best_from_structure(st)
        print(
            K.reshape(cand_preseto, [nlayers, n]), K.reshape(cand_presett, [nlayers, n])
        )
        print("current recommendation loss: ", ansatz(so, st, "most"))
