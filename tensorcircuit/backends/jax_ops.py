"""
Customized ops for ML framework
"""
# pylint: disable=invalid-name

from typing import Any, Tuple, Sequence

import jax
import jax.numpy as jnp

Array = Any  # jnp.array


@jax.custom_vjp
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
    m = u.shape[-2]
    n = v.shape[-2]

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

    grad_a = dAv + dAu + dAs + dAc

    if m > n:
        grad_a += (du - jnp.conj(u) @ jnp.transpose(u) @ du) @ Sinv @ jnp.transpose(v)
    elif m < n:
        grad_a += (
            jnp.conj(u)
            @ Sinv
            @ jnp.transpose(jnp.conj(dv))
            @ (jnp.eye(n) - jnp.conj(v) @ jnp.transpose(v))
        )
    # m=n do nothing

    return (grad_a,)


adaware_svd.defvjp(jaxsvd_fwd, jaxsvd_bwd)  # type: ignore

adaware_svd_jit = jax.jit(adaware_svd)


qr_epsilon = 1e-8


@jax.custom_vjp
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


adaware_qr.defvjp(jaxqr_fwd, jaxqr_bwd)  # type: ignore

adaware_qr_jit = jax.jit(adaware_qr)


@jax.custom_vjp
def adaware_eigh(A: Array) -> Array:
    return jnp.linalg.eigh(A)


def jaxeigh_fwd(A: Array) -> Array:
    e, v = jnp.linalg.eigh(A)
    return (e, v), (A, e, v)


def jaxeigh_bwd(r: Array, tangents: Array) -> Array:
    a, e, v = r
    de, dv = tangents
    eye_n = jnp.eye(a.shape[-1], dtype=a.dtype)
    f = _safe_reciprocal(e[..., jnp.newaxis, :] - e[..., jnp.newaxis] + eye_n) - eye_n
    middle = jnp.diag(de) + jnp.multiply(f, (v.T @ dv))
    grad_a = jnp.conj(v) @ middle @ v.T
    return (grad_a,)


# denegerate eigev values lead nan in eigh gradients, while tf has fixed that long ago

adaware_eigh.defvjp(jaxeigh_fwd, jaxeigh_bwd)
adaware_eigh_jit = jax.jit(adaware_eigh)
