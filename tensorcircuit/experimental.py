"""
experimental features
"""

from functools import partial
from typing import Any, Callable, Optional, Sequence, Union

from .cons import backend

Tensor = Any


def adaptive_vmap(
    f: Callable[..., Any],
    vectorized_argnums: Union[int, Sequence[int]] = 0,
    chunk_size: Optional[int] = None,
) -> Callable[..., Any]:
    if chunk_size is None:
        return backend.vmap(f, vectorized_argnums)  # type: ignore

    if isinstance(vectorized_argnums, int):
        vectorized_argnums = (vectorized_argnums,)

    def wrapper(*args: Any, **kws: Any) -> Tensor:
        # only support `f` outputs a tensor
        s1, s2 = divmod(args[vectorized_argnums[0]].shape[0], chunk_size)  # type: ignore
        # repetition, rest
        reshape_args = []
        rest_args = []
        for i, arg in enumerate(args):
            if i in vectorized_argnums:  # type: ignore
                if s2 != 0:
                    arg_rest = arg[-s2:]
                    arg = arg[:-s2]
                arg = backend.reshape(
                    arg,
                    [s1, chunk_size] + list(backend.shape_tuple(arg))[1:],
                )

            else:
                arg_rest = arg
            reshape_args.append(arg)
            if s2 != 0:
                rest_args.append(arg_rest)
        _vmap = backend.vmap(f, vectorized_argnums)
        r = []
        for i in range(s1):
            # currently using naive python loop for simplicity
            nreshape_args = [
                a[i] if j in vectorized_argnums else a  # type: ignore
                for j, a in enumerate(reshape_args)
            ]
            r.append(_vmap(*nreshape_args, **kws))
        r = backend.stack(r)
        rshape = list(backend.shape_tuple(r))
        if len(rshape) == 2:
            nshape = [rshape[0] * rshape[1]]
        else:
            nshape = [rshape[0] * rshape[1], -1]
        r = backend.reshape(r, nshape)
        if s2 != 0:
            rest_r = _vmap(*rest_args, **kws)
            return backend.concat([r, rest_r])
        return r

    return wrapper


def qng(
    f: Callable[..., Tensor], kernel: str = "qng", postprocess: Optional[str] = "qng"
) -> Callable[..., Tensor]:
    def wrapper(params: Tensor, **kws: Any) -> Tensor:
        @backend.jit  # type: ignore
        def vdot(i: Tensor, j: Tensor) -> Tensor:
            return backend.tensordot(backend.conj(i), j, 1)

        psi = f(params)
        jac = backend.jacfwd(f)(params)
        jac = backend.transpose(jac)
        if kernel == "qng":

            def ij(i: Tensor, j: Tensor) -> Tensor:
                return vdot(i, j) - vdot(i, psi) * vdot(psi, j)

        elif kernel == "dynamics":

            def ij(i: Tensor, j: Tensor) -> Tensor:
                return vdot(i, j)

        vij = backend.vmap(ij, vectorized_argnums=0)
        vvij = backend.vmap(vij, vectorized_argnums=1)

        fim = vvij(jac, jac)
        # TODO(@refraction-ray): investigate more on
        # suitable hyperparameters and methods for regularization?
        if isinstance(postprocess, str):
            if postprocess == "qng":

                def _post_process(t: Tensor) -> Tensor:
                    eps = 1e-4
                    t += eps * backend.eye(t.shape[0])
                    t = backend.real(t)
                    return t

        elif postprocess is None:
            _post_process = lambda _: _
        else:
            _post_process = postprocess  # callable
        fim = _post_process(fim)
        return fim

    return wrapper


dynamics_matrix = partial(qng, kernel="dynamics", postprocess=None)


def dynamics_rhs(f: Callable[..., Any], h: Tensor) -> Callable[..., Any]:
    def wrapper(params: Tensor, **kws: Any) -> Tensor:
        def energy(params: Tensor) -> Tensor:
            w = f(params, **kws)
            wr = backend.stop_gradient(w)
            wl = backend.conj(w)
            wl = backend.reshape(wl, [1, -1])
            wr = backend.reshape(wr, [-1, 1])
            if not backend.is_sparse(h):
                e = wl @ h @ wr
            else:
                tmp = backend.sparse_dense_matmul(h, wr)
                e = wl @ tmp
            return backend.real(e)[0, 0]

        return backend.grad(energy)(params)

    return wrapper
