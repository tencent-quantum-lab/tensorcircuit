"""
customized ops for ML framework
"""

from typing import Any, Tuple, Sequence

import jax
import jax.numpy as jnp

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
