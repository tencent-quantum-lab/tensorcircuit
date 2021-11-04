import os
import sys
import numpy as np
import pytest
import tensornetwork as tn
from pytest_lazyfixture import lazy_fixture as lf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
from tensorcircuit import quantum as qu

# Note the first version of this file is adpated from source code of tensornetwork: (Apache2)
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
    E = qu.identity((2, 3, 4))
    for n in E.nodes:
        assert isinstance(n, tn.CopyNode)
    twentyfour = E.trace()
    for n in twentyfour.nodes:
        assert isinstance(n, tn.CopyNode)
    assert twentyfour.eval() == 24

    tensor = np.random.rand(2, 2)
    psi = qu.QuVector.from_tensor(tensor)
    E = qu.identity((2, 2))
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
