"""
customized ops for ML framework
"""
# pylint: disable=invalid-name

from typing import Any, Tuple, Sequence

import jax
import jax.numpy as jnp
import tensorflow as tf  # type: ignore

Array = Any  # jnp.array


@jax.custom_vjp  # type: ignore
def adaware_svd(A: Array) -> Any:
    return jnp.linalg.svd(A, full_matrices=False)


def _safe_reciprocal(x: Array, epsilon: float = 1e-15) -> Array:
    return x / (x * x + epsilon)


def jaxsvd_fwd(A: Array) -> Any:
    u, s, v = adaware_svd(A)
    return (u, s, v), (u, s, v)


def jaxsvd_bwd(r: Sequence[Array], tangents: Sequence[Array]) -> Tuple[Array]:
    # https://arxiv.org/pdf/1909.02659.pdf
    u, s, v = r
    du, ds, dv = tangents
    v = jnp.conj(jnp.transpose(v))
    dv = jnp.conj(jnp.transpose(dv))

    F = s * s - (s * s)[:, None]
    F = _safe_reciprocal(F) - jnp.diag(jnp.diag(_safe_reciprocal(F)))
    # note the double character of ``jnp.diag``` ...
    S = jnp.diag(s)
    dAs = jnp.conj(u) @ jnp.diag(ds) @ jnp.transpose(v)

    J = F * (jnp.transpose(u) @ du)
    dAu = jnp.conj(u) @ (J + jnp.transpose(jnp.conj(J))) @ S @ jnp.transpose(v)

    K = F * (jnp.transpose(v) @ dv)
    dAv = jnp.conj(u) @ S @ (K + jnp.conj(jnp.transpose(K))) @ jnp.transpose(v)

    Sinv = jnp.diag(_safe_reciprocal(s))
    L = jnp.diag(jnp.diag(jnp.transpose(v) @ dv)) @ Sinv
    dAc = 1 / 2.0 * jnp.conj(u) @ (jnp.conj(L) - L) @ jnp.transpose(v)

    dAu += (
        (jnp.eye(jnp.size(u[:, 1])) - jnp.conj(u) @ jnp.transpose(u))
        @ du
        @ Sinv
        @ jnp.transpose(v)
    )

    dAv += (
        jnp.conj(u)
        @ Sinv
        @ jnp.transpose(dv)
        @ (jnp.eye(jnp.size(v[:, 1])) - jnp.conj(v) @ jnp.transpose(v))
    )

    grad_a = dAv + dAu + dAs + dAc

    return (grad_a,)


adaware_svd.defvjp(jaxsvd_fwd, jaxsvd_bwd)

adaware_svd_jit = jax.jit(adaware_svd)


qr_epsilon = 1e-8


@jax.custom_vjp  # type: ignore
def adaware_qr(A: Array) -> Any:
    # q, r = jnp.linalg.qr(A)
    return jnp.linalg.qr(A)


def jaxqr_fwd(A: Array) -> Any:
    q, r = adaware_qr(A)
    return (q, r), (A, q, r)


def jaxqr_bwd(res: Sequence[Array], tangents: Sequence[Array]) -> Tuple[Array]:
    a, q, r = res
    dq, dr = tangents
    dq = dq.conj()
    dr = dr.conj()

    def _TriangularSolve(x: Array, r: Array) -> Array:
        return jax.scipy.linalg.solve_triangular(  # type: ignore
            r, x.T.conj(), lower=False, trans=0
        ).T.conj()

    def _QrGradSquareAndDeepMatrices(q: Array, r: Array, dq: Array, dr: Array) -> Array:
        # Modification begins
        rdiag = jnp.diag(r)
        rdiag = (jnp.abs(rdiag) < qr_epsilon) * qr_epsilon + (
            jnp.abs(rdiag) >= qr_epsilon
        ) * rdiag
        diag_indices = jnp.arange(min(r.shape[-2:]))
        r = r.at[diag_indices, diag_indices].set(rdiag)
        # Modification ends

        qdq = q.T.conj().dot(dq)
        qdq_ = qdq - qdq.T.conj()
        rdr = r.dot(dr.T.conj())
        rdr_ = rdr - rdr.T.conj()
        tril = jnp.tril(qdq_ + rdr_)

        grad_a = q.dot(dr + _TriangularSolve(tril, r))
        grad_b = _TriangularSolve(dq - q.dot(qdq), r)
        ret = grad_a + grad_b

        if jnp.iscomplexobj(q):
            m = rdr - qdq.T.conj()
            length = m.shape[0]
            eyem = jnp.zeros_like(m)
            eyem = eyem.at[jnp.arange(length), jnp.arange(length)].set(jnp.diag(m))
            correction = eyem - jnp.real(eyem)
            ret = ret + _TriangularSolve(q.dot(correction.T.conj()), r)

        return ret.conj()

    num_rows, num_cols = q.shape[-2], r.shape[-1]

    if num_rows >= num_cols:
        result = _QrGradSquareAndDeepMatrices(q, r, dq, dr)
        return (result,)

    y = a[..., :, num_rows:]
    u = r[..., :, :num_rows]
    dv = dr[..., :, num_rows:]
    du = dr[..., :, :num_rows]
    dy = q.dot(dv)
    dx = _QrGradSquareAndDeepMatrices(q, u, dq + y.dot(dv.T.conj()), du)
    result = jnp.concatenate([dx, dy], axis=-1)
    return (result,)


adaware_qr.defvjp(jaxqr_fwd, jaxqr_bwd)

adaware_qr_jit = jax.jit(adaware_qr)


def tfqr_grad(a: Array, q: Array, r: Array, dq: Array, dr: Array) -> Array:
    """Gradient for Qr."""

    if (
        r.shape.ndims is None
        or r.shape.as_list()[-2] is None
        or r.shape.as_list()[-1] is None
    ):
        raise NotImplementedError(
            "QrGrad not implemented with dynamic shapes. "
            f"Received r.shape: {r.shape}"
        )
    if (
        r.shape.dims[-2].value > r.shape.dims[-1].value
        and q.shape.dims[-2].value == q.shape.dims[-1].value
    ):
        raise NotImplementedError(
            "QrGrad not implemented when nrows > ncols "
            "and full_matrices is true. Received r.shape="
            f"{r.shape} with nrows={r.shape.dims[-2]}"
            f"and ncols={r.shape.dims[-1]}."
        )

    def _TriangularSolve(x: Array, r: Array) -> Array:
        """Equiv to matmul(x, adjoint(matrix_inverse(r))) if r is upper-tri."""
        return tf.linalg.adjoint(
            tf.linalg.triangular_solve(
                r, tf.linalg.adjoint(x), lower=False, adjoint=False
            )
        )

    def _QrGradSquareAndDeepMatrices(q: Array, r: Array, dq: Array, dr: Array) -> Array:
        """Gradient for matrix orders num_rows >= num_cols
        and full_matrices is false.
        """

        # Modification begins
        rdiag = tf.linalg.diag_part(r)
        small_indices = tf.where(tf.math.abs(rdiag) < qr_epsilon)
        length = tf.shape(small_indices)[0]
        newvalues = tf.ones((length,), dtype=rdiag.dtype) * qr_epsilon
        rdiag = tf.tensor_scatter_nd_update(rdiag, small_indices, newvalues)
        delta_r = tf.linalg.set_diag(r, rdiag) - r
        r = r + delta_r
        # delta_dq = math_ops.matmul(q, math_ops.matmul(dr, tf.linalg.adjoint(delta_r)))
        # dq = dq + delta_dq
        # Modification ends

        qdq = tf.matmul(q, dq, adjoint_a=True)
        qdq_ = qdq - tf.linalg.adjoint(qdq)
        rdr = tf.matmul(r, dr, adjoint_b=True)
        rdr_ = rdr - tf.linalg.adjoint(rdr)
        tril = tf.linalg.band_part(qdq_ + rdr_, -1, 0)

        grad_a = tf.matmul(q, dr + _TriangularSolve(tril, r))
        grad_b = _TriangularSolve(dq - tf.matmul(q, qdq), r)
        ret = grad_a + grad_b

        if q.dtype.is_complex:
            m = rdr - tf.linalg.adjoint(qdq)
            eyem = tf.linalg.set_diag(tf.zeros_like(m), tf.linalg.diag_part(m))
            correction = eyem - tf.cast(tf.math.real(eyem), q.dtype)
            ret = ret + _TriangularSolve(tf.matmul(q, tf.linalg.adjoint(correction)), r)

        return ret

    num_rows, num_cols = q.shape.dims[-2].value, r.shape.dims[-1]

    if num_rows >= num_cols:
        return _QrGradSquareAndDeepMatrices(q, r, dq, dr)

    y = a[..., :, num_rows:]
    u = r[..., :, :num_rows]
    dv = dr[..., :, num_rows:]
    du = dr[..., :, :num_rows]
    dy = tf.matmul(q, dv)
    dx = _QrGradSquareAndDeepMatrices(q, u, dq + tf.matmul(y, dv, adjoint_b=True), du)
    return tf.concat([dx, dy], axis=-1)


@tf.custom_gradient  # type: ignore
def tfqr(A: Array) -> Any:
    q, r = tf.linalg.qr(A)

    def grad(dq: Array, dr: Array) -> Any:
        return tfqr_grad(A, q, r, dq, dr)

    return (q, r), grad
