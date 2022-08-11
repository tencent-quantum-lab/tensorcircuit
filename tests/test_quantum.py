# pylint: disable=invalid-name

from functools import partial
import os
import sys

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf
import tensornetwork as tn

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc
from tensorcircuit import quantum as qu

# Note that the first version of this file is adpated from source code of tensornetwork: (Apache2)
# https://github.com/google/TensorNetwork/blob/master/tensornetwork/quantum/quantum_test.py

# tc.set_contractor("greedy")
atol = 1e-5  # relax jax 32 precision
decimal = 5


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_constructor(backend):
    psi_tensor = np.random.rand(2, 2)
    psi_node = tn.Node(psi_tensor)

    op = qu.quantum_constructor([psi_node[0]], [psi_node[1]])
    assert not op.is_scalar()
    assert not op.is_vector()
    assert not op.is_adjoint_vector()
    assert len(op.out_edges) == 1
    assert len(op.in_edges) == 1
    assert op.out_edges[0] is psi_node[0]
    assert op.in_edges[0] is psi_node[1]

    op = qu.quantum_constructor([psi_node[0], psi_node[1]], [])
    assert not op.is_scalar()
    assert op.is_vector()
    assert not op.is_adjoint_vector()
    assert len(op.out_edges) == 2
    assert len(op.in_edges) == 0
    assert op.out_edges[0] is psi_node[0]
    assert op.out_edges[1] is psi_node[1]

    op = qu.quantum_constructor([], [psi_node[0], psi_node[1]])
    assert not op.is_scalar()
    assert not op.is_vector()
    assert op.is_adjoint_vector()
    assert len(op.out_edges) == 0
    assert len(op.in_edges) == 2
    assert op.in_edges[0] is psi_node[0]
    assert op.in_edges[1] is psi_node[1]

    with pytest.raises(ValueError):
        op = qu.quantum_constructor([], [], [psi_node])

    _ = psi_node[0] ^ psi_node[1]
    op = qu.quantum_constructor([], [], [psi_node])
    assert op.is_scalar()
    assert not op.is_vector()
    assert not op.is_adjoint_vector()
    assert len(op.out_edges) == 0
    assert len(op.in_edges) == 0


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_checks(backend):
    node1 = tn.Node(np.random.rand(2, 2))
    node2 = tn.Node(np.random.rand(2, 2))
    _ = node1[1] ^ node2[0]

    # extra dangling edges must be explicitly ignored
    with pytest.raises(ValueError):
        _ = qu.QuVector([node1[0]])

    # correctly ignore the extra edge
    _ = qu.QuVector([node1[0]], ignore_edges=[node2[1]])

    # in/out edges must be dangling
    with pytest.raises(ValueError):
        _ = qu.QuVector([node1[0], node1[1], node2[1]])


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_from_tensor(backend):
    psi_tensor = np.random.rand(2, 2)

    op = qu.QuOperator.from_tensor(psi_tensor, [0], [1])
    assert not op.is_scalar()
    assert not op.is_vector()
    assert not op.is_adjoint_vector()
    np.testing.assert_almost_equal(op.eval(), psi_tensor, decimal=decimal)

    op = qu.QuVector.from_tensor(psi_tensor, [0, 1])
    assert not op.is_scalar()
    assert op.is_vector()
    assert not op.is_adjoint_vector()
    np.testing.assert_almost_equal(op.eval(), psi_tensor, decimal=decimal)

    op = qu.QuAdjointVector.from_tensor(psi_tensor, [0, 1])
    assert not op.is_scalar()
    assert not op.is_vector()
    assert op.is_adjoint_vector()
    np.testing.assert_almost_equal(op.eval(), psi_tensor, decimal=decimal)

    op = qu.QuScalar.from_tensor(1.0)
    assert op.is_scalar()
    assert not op.is_vector()
    assert not op.is_adjoint_vector()
    assert op.eval() == 1.0


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_identity(backend):
    E = qu.identity((2, 3, 4), dtype=np.float64)
    for n in E.nodes:
        assert isinstance(n, tn.CopyNode)
    twentyfour = E.trace()
    for n in twentyfour.nodes:
        assert isinstance(n, tn.CopyNode)
    assert twentyfour.eval() == 24

    tensor = np.random.rand(2, 2)
    psi = qu.QuVector.from_tensor(tensor)
    E = qu.identity((2, 2), dtype=np.float64)
    np.testing.assert_allclose((E @ psi).eval(), psi.eval(), atol=atol)

    np.testing.assert_allclose(
        (psi.adjoint() @ E @ psi).eval(), psi.norm().eval(), atol=atol
    )

    op = qu.QuOperator.from_tensor(tensor, [0], [1])
    op_I = op.tensor_product(E)
    op_times_4 = op_I.partial_trace([1, 2])
    np.testing.assert_allclose(op_times_4.eval(), 4 * op.eval(), atol=atol)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_tensor_product(backend):
    psi = qu.QuVector.from_tensor(np.random.rand(2, 2))
    psi_psi = psi.tensor_product(psi)
    assert len(psi_psi.subsystem_edges) == 4
    np.testing.assert_almost_equal(
        psi_psi.norm().eval(), psi.norm().eval() ** 2, decimal=decimal
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_matmul(backend):
    mat = np.random.rand(2, 2)
    op = qu.QuOperator.from_tensor(mat, [0], [1])
    res = (op @ op).eval()
    np.testing.assert_allclose(res, mat @ mat, atol=atol)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_mul(backend):
    mat = np.eye(2)
    scal = np.float64(0.5)
    op = qu.QuOperator.from_tensor(mat, [0], [1])
    scal_op = qu.QuScalar.from_tensor(scal)

    res = (op * scal_op).eval()
    np.testing.assert_allclose(res, mat * 0.5, atol=atol)

    res = (scal_op * op).eval()
    np.testing.assert_allclose(res, mat * 0.5, atol=atol)

    res = (scal_op * scal_op).eval()
    np.testing.assert_almost_equal(res, 0.25, decimal=decimal)

    res = (op * np.float64(0.5)).eval()
    np.testing.assert_allclose(res, mat * 0.5, atol=atol)

    res = (np.float64(0.5) * op).eval()
    np.testing.assert_allclose(res, mat * 0.5, atol=atol)

    with pytest.raises(ValueError):
        _ = op * op

    with pytest.raises(ValueError):
        _ = op * mat


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_expectations(backend):

    psi_tensor = np.random.rand(2, 2, 2) + 1.0j * np.random.rand(2, 2, 2)
    op_tensor = np.random.rand(2, 2) + 1.0j * np.random.rand(2, 2)

    psi = qu.QuVector.from_tensor(psi_tensor)
    op = qu.QuOperator.from_tensor(op_tensor, [0], [1])

    op_3 = op.tensor_product(qu.identity((2, 2), dtype=psi_tensor.dtype))
    res1 = (psi.adjoint() @ op_3 @ psi).eval()

    rho_1 = psi.reduced_density([1, 2])  # trace out sites 2 and 3
    res2 = (op @ rho_1).trace().eval()

    np.testing.assert_almost_equal(res1, res2, decimal=decimal)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_projector(backend):
    psi_tensor = np.random.rand(2, 2)
    psi_tensor /= np.linalg.norm(psi_tensor)
    psi = qu.QuVector.from_tensor(psi_tensor)
    P = psi.projector()
    np.testing.assert_allclose((P @ psi).eval(), psi_tensor, atol=atol)

    np.testing.assert_allclose((P @ P).eval(), P.eval(), atol=atol)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_nonsquare_quop(backend):
    op = qu.QuOperator.from_tensor(np.ones([2, 2, 2, 2, 2]), [0, 1, 2], [3, 4])
    op2 = qu.QuOperator.from_tensor(np.ones([2, 2, 2, 2, 2]), [0, 1], [2, 3, 4])
    np.testing.assert_allclose(
        (op @ op2).eval(), 4 * np.ones([2, 2, 2, 2, 2, 2]), atol=atol
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_expectation_local_tensor(backend):
    op = qu.QuOperator.from_local_tensor(
        np.array([[1.0, 0.0], [0.0, 1.0]]), space=[2, 2, 2, 2], loc=[1]
    )
    state = np.zeros([2, 2, 2, 2])
    state[0, 0, 0, 0] = 1.0
    psi = qu.QuVector.from_tensor(state)
    psi_d = psi.adjoint()
    np.testing.assert_allclose((psi_d @ op @ psi).eval(), 1.0, atol=atol)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_rm_state_vs_mps(backend):
    @partial(tc.backend.jit, jit_compile=False, static_argnums=(1, 2))
    def entanglement1(param, n, nlayers):
        c = tc.Circuit(n)
        c = tc.templates.blocks.example_block(c, param, nlayers)
        w = c.wavefunction()
        rm = qu.reduced_density_matrix(w, int(n / 2))
        return qu.entropy(rm)

    @partial(tc.backend.jit, jit_compile=False, static_argnums=(1, 2))
    def entanglement2(param, n, nlayers):
        c = tc.Circuit(n)
        c = tc.templates.blocks.example_block(c, param, nlayers)
        w = c.get_quvector()
        rm = w.reduced_density([i for i in range(int(n / 2))])
        return qu.entropy(rm)

    param = tc.backend.ones([6, 6])
    rm1 = entanglement1(param, 6, 3)
    rm2 = entanglement2(param, 6, 3)
    np.testing.assert_allclose(rm1, rm2, atol=atol)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_trace_product(backend):
    o = np.ones([2, 2])
    h = np.eye(2)
    np.testing.assert_allclose(qu.trace_product(o, h), 2, atol=atol)
    oq = qu.QuOperator.from_tensor(o)
    hq = qu.QuOperator.from_tensor(h)
    np.testing.assert_allclose(qu.trace_product(oq, hq), 2, atol=atol)
    np.testing.assert_allclose(qu.trace_product(oq, h), 2, atol=atol)
    np.testing.assert_allclose(qu.trace_product(o, hq), 2, atol=atol)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_free_energy(backend):
    rho = np.array([[1.0, 0], [0, 0]])
    h = np.array([[-1.0, 0], [0, 1]])
    np.testing.assert_allclose(qu.free_energy(rho, h, 0.5), -1, atol=atol)
    np.testing.assert_allclose(qu.renyi_free_energy(rho, h, 0.5), -1, atol=atol)
    hq = qu.QuOperator.from_tensor(h)
    np.testing.assert_allclose(qu.free_energy(rho, hq, 0.5), -1, atol=atol)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_measurement_counts(backend):
    state = np.ones([4])
    ct, cs = qu.measurement_counts(state, format="count_tuple")
    np.testing.assert_allclose(ct.shape[0], 4, atol=atol)
    np.testing.assert_allclose(tc.backend.sum(cs), 8192, atol=atol)
    state = np.ones([2, 2])
    ct, cs = qu.measurement_counts(state, format="count_tuple")
    np.testing.assert_allclose(ct.shape[0], 2, atol=atol)
    np.testing.assert_allclose(tc.backend.sum(cs), 8192, atol=atol)
    state = np.array([1.0, 1.0, 0, 0])
    print(qu.measurement_counts(state))


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_extract_from_measure(backend):
    np.testing.assert_allclose(
        qu.spin_by_basis(2, 1), np.array([1, -1, 1, -1]), atol=atol
    )
    state = tc.array_to_tensor(np.array([0.6, 0.4, 0, 0]))
    np.testing.assert_allclose(
        qu.correlation_from_counts([0, 1], state), 0.2, atol=atol
    )
    np.testing.assert_allclose(qu.correlation_from_counts([1], state), 0.2, atol=atol)

    samples_int = tc.array_to_tensor(np.array([0, 0, 3, 3, 3]), dtype="int32")
    r = qu.correlation_from_samples([0, 1], samples_int, n=2)
    np.testing.assert_allclose(r, 1, atol=1e-5)


def test_heisenberg_ham(tfb):
    g = tc.templates.graphs.Line1D(6)
    h = tc.quantum.heisenberg_hamiltonian(g, sparse=False)
    e, _ = tc.backend.eigh(h)
    np.testing.assert_allclose(e[0], -11.2111, atol=1e-4)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_reduced_density_from_density(backend):
    n = 6
    w = np.random.normal(size=[2**n]) + 1.0j * np.random.normal(size=[2**n])
    w /= np.linalg.norm(w)
    rho = np.reshape(w, [-1, 1]) @ np.reshape(np.conj(w), [1, -1])
    dm1 = tc.quantum.reduced_density_matrix(w, cut=[0, 2])
    dm2 = tc.quantum.reduced_density_matrix(rho, cut=[0, 2])
    np.testing.assert_allclose(dm1, dm2, atol=1e-5)

    # with p
    n = 5
    w = np.random.normal(size=[2**n]) + 1.0j * np.random.normal(size=[2**n])
    w /= np.linalg.norm(w)
    p = np.random.normal(size=[2**3])
    p = tc.backend.softmax(p)
    p = tc.backend.cast(p, "complex128")
    rho = np.reshape(w, [-1, 1]) @ np.reshape(np.conj(w), [1, -1])
    dm1 = tc.quantum.reduced_density_matrix(w, cut=[1, 2, 3], p=p)
    dm2 = tc.quantum.reduced_density_matrix(rho, cut=[1, 2, 3], p=p)
    np.testing.assert_allclose(dm1, dm2, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_mutual_information(backend):
    n = 5
    w = np.random.normal(size=[2**n]) + 1.0j * np.random.normal(size=[2**n])
    w /= np.linalg.norm(w)
    rho = np.reshape(w, [-1, 1]) @ np.reshape(np.conj(w), [1, -1])
    dm1 = tc.quantum.mutual_information(w, cut=[1, 2, 3])
    dm2 = tc.quantum.mutual_information(rho, cut=[1, 2, 3])
    np.testing.assert_allclose(dm1, dm2, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_expectation_quantum(backend):
    c = tc.Circuit(3)
    c.ry(0, theta=0.4)
    c.cnot(0, 1)
    exp1 = c.expectation([tc.gates.z(), [0]], [tc.gates.z(), [2]], reuse=False)
    qv = c.quvector()
    exp2 = tc.expectation([tc.gates.z(), [0]], [tc.gates.z(), [2]], ket=qv)
    np.testing.assert_allclose(exp1, exp2, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_tn2qop(backend):
    nwires = 6
    dtype = np.complex64
    # only obc is supported, even if you supply nwires Jx terms
    Jx = np.array([1.0 for _ in range(nwires - 1)])  # strength of xx interaction (OBC)
    Bz = np.array([-1.0 for _ in range(nwires)])  # strength of transverse field
    tn_mpo = tn.matrixproductstates.mpo.FiniteTFI(Jx, Bz, dtype=dtype)
    qu_mpo = tc.quantum.tn2qop(tn_mpo)
    h1 = qu_mpo.eval_matrix()
    g = tc.templates.graphs.Line1D(nwires, pbc=False)
    h2 = tc.quantum.heisenberg_hamiltonian(
        g, hzz=0, hxx=1, hyy=0, hz=1, hx=0, hy=0, sparse=False, numpy=True
    )
    np.testing.assert_allclose(h1, h2, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_qb2qop(backend):
    try:
        import quimb
    except ImportError:
        pytest.skip("quimb is not installed")
    nwires = 6
    qb_mpo = quimb.tensor.tensor_gen.MPO_ham_ising(nwires, 4, 2, cyclic=True)
    qu_mpo = tc.quantum.quimb2qop(qb_mpo)
    h1 = qu_mpo.eval_matrix()
    g = tc.templates.graphs.Line1D(nwires, pbc=True)
    h2 = tc.quantum.heisenberg_hamiltonian(
        g, hzz=1, hxx=0, hyy=0, hz=0, hx=-1, hy=0, sparse=False, numpy=True
    )
    np.testing.assert_allclose(h1, h2, atol=1e-5)

    # in out edge order test
    builder = quimb.tensor.tensor_gen.SpinHam()
    # new version quimb breaking API change: SpinHam1D -> SpinHam
    builder += 1, "Y"
    builder += 1, "X"
    H = builder.build_mpo(3)
    h = tc.quantum.quimb2qop(H)
    m1 = h.eval_matrix()
    g = tc.templates.graphs.Line1D(3, pbc=False)
    m2 = tc.quantum.heisenberg_hamiltonian(
        g, hzz=0, hxx=0, hyy=0, hz=0, hy=0.5, hx=0.5, sparse=False, numpy=True
    )
    np.testing.assert_allclose(m1, m2, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_counts_2(backend):
    z0 = tc.backend.convert_to_tensor(np.array([0.1, 0, -0.3, 0]))
    x, y = tc.quantum.count_d2s(z0)
    print(x, y)
    np.testing.assert_allclose(x, np.array([0, 2]))
    np.testing.assert_allclose(y, np.array([0.1, -0.3]))
    z = tc.quantum.count_s2d((x, y), 2)
    np.testing.assert_allclose(z, z0)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_measurement_results(backend):
    n = 4
    w = tc.backend.ones([2**n])
    r = tc.quantum.measurement_results(w, counts=9, format="sample_bin", jittable=True)
    assert tc.backend.shape_tuple(r) == (9, n)
    print(r)
    r = tc.quantum.measurement_results(w, counts=9, format="sample_int", jittable=True)
    assert tc.backend.shape_tuple(r) == (9,)
    print(r)
    for c in (9, -9):
        r = tc.quantum.measurement_results(
            w, counts=c, format="count_vector", jittable=True
        )
        assert tc.backend.shape_tuple(r) == (2**n,)
        print(r)
        r = tc.quantum.measurement_results(w, counts=c, format="count_tuple")
        print(r)
        r = tc.quantum.measurement_results(
            w, counts=c, format="count_dict_bin", jittable=True
        )
        print(r)
        r = tc.quantum.measurement_results(
            w, counts=c, format="count_dict_int", jittable=True
        )
        print(r)
