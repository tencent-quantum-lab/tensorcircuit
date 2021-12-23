"""
experimental features
"""

from typing import Any, Callable

from .cons import backend

Tensor = Any


def qng(
    f: Callable[..., Tensor], eps: float = 1e-4, real_only: bool = True
) -> Callable[..., Tensor]:
    def wrapper(params: Tensor, **kws: Any) -> Tensor:
        @backend.jit  # type: ignore
        def vdot(i: Tensor, j: Tensor) -> Tensor:
            return backend.tensordot(backend.conj(i), j, 1)

        psi = f(params)
        jac = backend.jacfwd(f)(params)
        jac = backend.transpose(jac)

        def ij(i: Tensor, j: Tensor) -> Tensor:
            return vdot(i, j) - vdot(i, psi) * vdot(psi, j)

        vij = backend.vmap(ij, vectorized_argnums=0)
        vvij = backend.vmap(vij, vectorized_argnums=1)

        fim = vvij(jac, jac)
        # TODO(@refraction-ray): investigate more on
        # suitable hyperparameters and methods for regularization?
        fim += eps * backend.eye(fim.shape[0])
        fim = backend.real(fim)
        return fim

    return wrapper
