"""
slicing the output wavefunction to save the memory in VQA context
"""

from itertools import product
import numpy as np
import tensorcircuit as tc

K = tc.set_backend("jax")


def circuit(param, n, nlayers):
    c = tc.Circuit(n)
    for i in range(n):
        c.h(i)
    c = tc.templates.blocks.example_block(c, param, nlayers)
    return c


def sliced_state(c, cut, mask):
    # mask = Tensor([0, 1, 0])
    # cut = [0, 1, 2]
    n = c._nqubits
    ncut = len(cut)
    end0 = tc.array_to_tensor(np.array([1.0, 0.0]))
    end1 = tc.array_to_tensor(np.array([0.0, 1.0]))
    ends = [tc.Gate(mask[i] * end1 + (1 - mask[i]) * end0) for i in range(ncut)]
    nodes, front = c._copy()
    for j, i in enumerate(cut):
        front[i] ^ ends[j][0]
    oeo = []
    for i in range(n):
        if i not in cut:
            oeo.append(front[i])
    ss = tc.contractor(nodes + ends, output_edge_order=oeo)
    return ss


def sliced_op(ps, cut, mask1, mask2):
    # ps: Tensor([0, 0, 1, 1])
    n = K.shape_tuple(ps)[-1]
    ncut = len(cut)
    end0 = tc.array_to_tensor(np.array([1.0, 0.0]))
    end1 = tc.array_to_tensor(np.array([0.0, 1.0]))
    endsr = [tc.Gate(mask1[i] * end1 + (1 - mask1[i]) * end0) for i in range(ncut)]
    endsl = [tc.Gate(mask2[i] * end1 + (1 - mask2[i]) * end0) for i in range(ncut)]

    structuresc = K.cast(ps, dtype="int32")
    structuresc = K.onehot(structuresc, num=4)
    structuresc = K.cast(structuresc, dtype=tc.dtypestr)
    obs = []
    for i in range(n):
        obs.append(
            tc.Gate(
                sum(
                    [
                        structuresc[i, k] * g.tensor
                        for k, g in enumerate(tc.gates.pauli_gates)
                    ]
                )
            )
        )
    for j, i in enumerate(cut):
        obs[i][0] ^ endsl[j][0]
        obs[i][1] ^ endsr[j][0]
    oeo = []
    for i in range(n):
        if i not in cut:
            oeo.append(obs[i][0])
    for i in range(n):
        if i not in cut:
            oeo.append(obs[i][1])
    return obs + endsl + endsr, oeo


def sliced_core(param, n, nlayers, ps, cut, mask1, mask2):
    # param, ps, mask1, mask2 are all tensor
    c = circuit(param, n, nlayers)
    ss = sliced_state(c, cut, mask1)
    ssc = sliced_state(c, cut, mask2)
    ssc, _ = tc.Circuit.copy([ssc], conj=True)
    op_nodes, op_edges = sliced_op(ps, cut, mask1, mask2)
    nodes = [ss] + ssc + op_nodes
    ssc = ssc[0]
    n = c._nqubits
    nleft = n - len(cut)
    for i in range(nleft):
        op_edges[i + nleft] ^ ss[i]
        op_edges[i] ^ ssc[i]
    scalar = tc.contractor(nodes)
    return K.real(scalar.tensor)


sliced_core_vvg = K.jit(
    K.vectorized_value_and_grad(sliced_core, argnums=0, vectorized_argnums=(5, 6)),
    static_argnums=(1, 2, 4),
)  # vmap version if memory is enough

sliced_core_vg = K.jit(
    K.value_and_grad(sliced_core, argnums=0),
    static_argnums=(1, 2, 4),
)  # nonvmap version is memory is tight and distrubution workload may be enabled


def sliced_expectation_and_grad(param, n, nlayers, ps, cut, is_vmap=True):
    pst = tc.array_to_tensor(ps)
    res = 0.0
    mask1s = []
    mask2s = []
    for mask1 in product(*[(0, 1) for _ in cut]):
        mask1t = tc.array_to_tensor(np.array(mask1))
        mask1s.append(mask1t)
        mask2 = list(mask1)
        for j, i in enumerate(cut):
            if ps[i] in [1, 2]:
                mask2[j] = 1 - mask1[j]
        mask2t = tc.array_to_tensor(np.array(mask2))
        mask2s.append(mask2t)
    if is_vmap:
        mask1s = K.stack(mask1s)
        mask2s = K.stack(mask2s)
        res = sliced_core_vvg(param, n, nlayers, pst, cut, mask1s, mask2s)
        res = list(res)
        res[0] = K.sum(res[0])
        res = tuple(res)
    else:
        # memory bounded
        # can modified to adpative pmap
        vs = 0.0
        gs = 0.0
        for i in range(len(mask1s)):
            mask1t = mask1s[i]
            mask2t = mask2s[i]
            v, g = sliced_core_vg(param, n, nlayers, pst, cut, mask1t, mask2t)
            vs += v
            gs += g
        res = (vs, gs)
    return res


def sliced_expectation_ref(c, ps, cut):
    """
    reference implementation
    """
    # ps: [0, 2, 1]
    res = 0.0
    for mask1 in product(*[(0, 1) for _ in cut]):
        mask1t = tc.array_to_tensor(np.array(mask1))
        ss = sliced_state(c, cut, mask1t)
        mask2 = list(mask1)
        for j, i in enumerate(cut):
            if ps[i] in [1, 2]:
                mask2[j] = 1 - mask1[j]
        mask2t = tc.array_to_tensor(np.array(mask2))
        ssc = sliced_state(c, cut, mask2t)
        ssc, _ = tc.Circuit.copy([ssc], conj=True)
        ps = tc.array_to_tensor(ps)
        op_nodes, op_edges = sliced_op(ps, cut, mask1t, mask2t)
        nodes = [ss] + ssc + op_nodes
        ssc = ssc[0]
        n = c._nqubits
        nleft = n - len(cut)
        for i in range(nleft):
            op_edges[i + nleft] ^ ss[i]
            op_edges[i] ^ ssc[i]
        scalar = tc.contractor(nodes)
        res += scalar.tensor
    return res


if __name__ == "__main__":
    n = 10
    nlayers = 5
    param = K.ones([n, 2 * nlayers], dtype="float32")
    cut = (0, 2, 5, 9)
    ops = [2, 0, 3, 1, 0, 0, 1, 2, 0, 1]
    ops_dict = tc.quantum.ps2xyz(ops)

    def trivial_core(param, n, nlayers):
        c = circuit(param, n, nlayers)
        return K.real(c.expectation_ps(**ops_dict))

    trivial_vg = K.jit(K.value_and_grad(trivial_core, argnums=0), static_argnums=(1, 2))

    print("reference impl")
    r0 = tc.utils.benchmark(trivial_vg, param, n, nlayers)
    print("vmapped slice")
    r1 = tc.utils.benchmark(
        sliced_expectation_and_grad, param, n, nlayers, ops, cut, True
    )
    print("naive for slice")
    r2 = tc.utils.benchmark(
        sliced_expectation_and_grad, param, n, nlayers, ops, cut, False
    )

    np.testing.assert_allclose(r0[0][0], r1[0][0], atol=1e-5)
    np.testing.assert_allclose(r2[0][0], r1[0][0], atol=1e-5)
    np.testing.assert_allclose(r0[0][1], r1[0][1], atol=1e-5)
    np.testing.assert_allclose(r2[0][1], r1[0][1], atol=1e-5)
