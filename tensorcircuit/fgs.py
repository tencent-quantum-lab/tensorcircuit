"""
Fermion Gaussian state simulator
"""
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import openfermion
except ModuleNotFoundError:
    pass

from .cons import backend, dtypestr
from .circuit import Circuit
from . import quantum

Tensor = Any


def onehot_matrix(i: int, j: int, N: int) -> Tensor:
    m = np.zeros([N, N])
    m[i, j] = 1
    m = backend.convert_to_tensor(m)
    m = backend.cast(m, dtypestr)
    return m


# TODO(@refraction-ray): efficiency benchmark with jit
# TODO(@refraction-ray): occupation number post-selection/measurement
# TODO(@refraction-ray): FGS mixed state support?
# TODO(@refraction-ray): overlap?


class FGSSimulator:
    r"""
    main refs: https://arxiv.org/pdf/2306.16595.pdf,
    https://arxiv.org/abs/2209.06945,
    https://scipost.org/SciPostPhysLectNotes.54/pdf

    convention:
    for Hamiltonian (c^dagger, c)H(c, c^\dagger)
    for correlation <(c, c^\dagger)(c^\dagger, c)>
    c' = \alpha^\dagger (c, c^\dagger)
    """

    def __init__(
        self,
        L: int,
        filled: Optional[List[int]] = None,
        alpha: Optional[Tensor] = None,
        hc: Optional[Tensor] = None,
        cmatrix: Optional[Tensor] = None,
    ):
        """
        _summary_

        :param L: system size
        :type L: int
        :param filled: the fermion site that is fully occupied, defaults to None
        :type filled: Optional[List[int]], optional
        :param cmatrix: only used for debug, defaults to None
        :type cmatrix: Optional[Tensor], optional
        """
        if filled is None:
            filled = []
        self.L = L
        if alpha is None:
            if hc is None:
                self.alpha = self.init_alpha(filled, L)
            else:
                _, _, self.alpha = self.fermion_diagonalization(hc, L)
        else:
            self.alpha = alpha
        self.wtransform = self.wmatrix(L)
        self.cmatrix = cmatrix

    @classmethod
    def fermion_diagonalization(
        cls, hc: Tensor, L: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        es, u = backend.eigh(hc)
        es = es[::-1]
        u = u[:, ::-1]
        alpha = u[:, :L]
        return es, u, alpha

    @classmethod
    def fermion_diagonalization_2(
        cls, hc: Tensor, L: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        w = cls.wmatrix(L)
        hm = 0.25 * w @ hc @ backend.adjoint(w)
        hm = backend.real((-1.0j) * hm)
        hd, om = backend.schur(hm, output="real")
        # order not kept
        # eps = 1e-10
        # idm = backend.convert_to_tensor(np.array([[1.0, 0], [0, 1.0]]))
        # idm = backend.cast(idm, dtypestr)
        # xm = backend.convert_to_tensor(np.array([[0, 1.0], [1.0, 0]]))
        # xm = backend.cast(xm, dtypestr)
        # for i in range(0, 2 * L, 2):
        #     (backend.sign(hd[i, i + 1] + eps) + 1) / 2 * idm - (
        #         backend.sign(hd[i, i + 1] + eps) - 1
        #     ) / 2 * xm
        # print(hd)

        es = backend.adjoint(w) @ (1.0j * hd) @ w
        u = 0.5 * backend.adjoint(w) @ backend.transpose(om) @ w
        alpha = backend.adjoint(u)[:, :L]
        # c' = u@c
        # e_k (c^\dagger_k c_k - c_k c^\dagger_k)
        return (es, u, alpha)

    @staticmethod
    def wmatrix(L: int) -> Tensor:
        w = np.zeros([2 * L, 2 * L], dtype=complex)
        for i in range(2 * L):
            if i % 2 == 1:
                w[i, (i - 1) // 2] = 1.0j
                w[i, (i - 1) // 2 + L] = -1.0j
            else:
                w[i, i // 2] = 1
                w[i, i // 2 + L] = 1
        return backend.convert_to_tensor(w)

    @staticmethod
    def init_alpha(filled: List[int], L: int) -> Tensor:
        alpha = np.zeros([2 * L, L])
        for i in range(L):
            if i not in filled:
                alpha[i, i] = 1
            else:
                alpha[i + L, i] = 1
        alpha = backend.convert_to_tensor(alpha)
        alpha = backend.cast(alpha, dtypestr)
        return alpha

    def get_alpha(self) -> Tensor:
        return self.alpha

    def get_cmatrix(self) -> Tensor:
        if self.cmatrix is not None:
            return self.cmatrix
        return self.alpha @ backend.adjoint(self.alpha)

    def get_reduced_cmatrix(self, subsystems_to_trace_out: List[int]) -> Tensor:
        m = self.get_cmatrix()
        if subsystems_to_trace_out is None:
            subsystems_to_trace_out = []
        keep = [i for i in range(self.L) if i not in subsystems_to_trace_out]
        keep += [i + self.L for i in range(self.L) if i not in subsystems_to_trace_out]
        keep = backend.convert_to_tensor(keep)

        def slice_(a: Tensor) -> Tensor:
            return backend.gather1d(a, keep)

        slice_ = backend.vmap(slice_)
        m = backend.gather1d(slice_(m), keep)
        return m

    def renyi_entropy(self, n: int, subsystems_to_trace_out: List[int]) -> Tensor:
        """
        compute renyi_entropy of order ``n`` for the fermion state

        :param n: _description_
        :type n: int
        :param subsystems_to_trace_out: system sites to be traced out
        :type subsystems_to_trace_out: List[int]
        :return: _description_
        :rtype: Tensor
        """
        m = self.get_reduced_cmatrix(subsystems_to_trace_out)
        lbd, _ = backend.eigh(m)
        lbd = backend.real(lbd)
        lbd = backend.relu(lbd)
        eps = 1e-6

        entropy = backend.sum(backend.log(lbd**n + (1 - lbd) ** n + eps))
        s = 1 / (2 * (1 - n)) * entropy
        return s

    def entropy(self, subsystems_to_trace_out: Optional[List[int]] = None) -> Tensor:
        """
        compute von Neumann entropy for the fermion state

        :param subsystems_to_trace_out: _description_, defaults to None
        :type subsystems_to_trace_out: Optional[List[int]], optional
        :return: _description_
        :rtype: Tensor
        """
        m = self.get_reduced_cmatrix(subsystems_to_trace_out)  # type: ignore
        lbd, _ = backend.eigh(m)
        lbd = backend.real(lbd)
        lbd = backend.relu(lbd)
        #         lbd /= backend.sum(lbd)
        eps = 1e-6
        entropy = -backend.sum(
            lbd * backend.log(lbd + eps) + (1 - lbd) * backend.log(1 - lbd + eps)
        )
        return entropy / 2

    def evol_hamiltonian(self, h: Tensor) -> None:
        r"""
        Evolve as :math:`e^{-i/2 \hat{h}}`

        :param h: _description_
        :type h: Tensor
        """
        # e^{-i/2 H}
        h = backend.cast(h, dtype=dtypestr)
        self.alpha = backend.expm(-1.0j * h) @ self.alpha

    def evol_ihamiltonian(self, h: Tensor) -> None:
        r"""
        Evolve as :math:`e^{-1/2 \hat{h}}`

        :param h: _description_
        :type h: Tensor
        """
        # e^{-1/2 H}
        h = backend.cast(h, dtype=dtypestr)
        self.alpha = backend.expm(h) @ self.alpha
        self.orthogonal()

    def orthogonal(self) -> None:
        q, _ = backend.qr(self.alpha)
        self.alpha = q

    @staticmethod
    def hopping(chi: Tensor, i: int, j: int, L: int) -> Tensor:
        # chi * ci dagger cj + hc.
        chi = backend.convert_to_tensor(chi)
        chi = backend.cast(chi, dtypestr)
        m = chi / 2 * onehot_matrix(i, j, 2 * L)
        m += -chi / 2 * onehot_matrix(j + L, i + L, 2 * L)
        m += backend.adjoint(m)
        return m

    def evol_hp(self, i: int, j: int, chi: Tensor = 0) -> None:
        r"""
        The evolve Hamiltonian is :math:`chi c_i^\dagger c_j +h.c.`

        :param i: _description_
        :type i: int
        :param j: _description_
        :type j: int
        :param chi: _description_, defaults to 0
        :type chi: Tensor, optional
        """
        self.evol_hamiltonian(self.hopping(chi, i, j, self.L))

    @staticmethod
    def chemical_potential(chi: Tensor, i: int, L: int) -> Tensor:
        chi = backend.convert_to_tensor(chi)
        chi = backend.cast(chi, dtypestr)
        m = chi / 2 * onehot_matrix(i, i, 2 * L)
        m += -chi / 2 * onehot_matrix(i + L, i + L, 2 * L)
        return m

    @staticmethod
    def sc_pairing(chi: Tensor, i: int, j: int, L: int) -> Tensor:
        chi = backend.convert_to_tensor(chi)
        chi = backend.cast(chi, dtypestr)
        m = chi / 2 * onehot_matrix(i, j + L, 2 * L)
        m += -chi / 2 * onehot_matrix(j, i + L, 2 * L)
        m += backend.adjoint(m)
        return m

    def evol_sp(self, i: int, j: int, chi: Tensor = 0) -> None:
        r"""
        The evolve Hamiltonian is :math:`chi c_i^\dagger c_j^\dagger +h.c.`


        :param i: _description_
        :type i: int
        :param j: _description_
        :type j: int
        :param chi: _description_, defaults to 0
        :type chi: Tensor, optional
        """
        self.evol_hamiltonian(self.sc_pairing(chi, i, j, self.L))

    def evol_cp(self, i: int, chi: Tensor = 0) -> None:
        r"""
        The evolve Hamiltonian is :math:`chi c_i^\dagger c_i`

        :param i: _description_
        :type i: int
        :param chi: _description_, defaults to 0
        :type chi: Tensor, optional
        """
        self.evol_hamiltonian(self.chemical_potential(chi, i, self.L))

    def evol_icp(self, i: int, chi: Tensor = 0) -> None:
        r"""
        The evolve Hamiltonian is :math:`chi c_i^\dagger c_i` with :math:`\exp^{-H/2}`

        :param i: _description_
        :type i: int
        :param chi: _description_, defaults to 0
        :type chi: Tensor, optional
        """
        self.evol_ihamiltonian(self.chemical_potential(chi, i, self.L))

    def get_bogoliubov_uv(self) -> Tuple[Tensor, Tensor]:
        return (
            backend.gather1d(
                self.alpha, backend.convert_to_tensor([i for i in range(self.L)])
            ),
            backend.gather1d(
                self.alpha,
                backend.convert_to_tensor([i + self.L for i in range(self.L)]),
            ),
        )

    def get_cmatrix_majorana(self) -> Tensor:
        c = self.get_cmatrix()
        return self.wtransform @ c @ backend.adjoint(self.wtransform)

    def get_covariance_matrix(self) -> Tensor:
        m = self.get_cmatrix_majorana()
        return -1.0j * (2 * m - backend.eye(self.L * 2))

    # def product(self, other):
    #     # self@other
    #     gamma1 = self.get_covariance_matrix()
    #     gamma2 = other.get_covariance_matrix()
    #     den = backend.inv(1 + gamma1 @ gamma2)
    #     idm = backend.eye(2 * self.L)
    #     covm = idm - (idm - gamma2) @ den @ (idm - gamma1)
    #     cm = (1.0j * covm + idm) / 2
    #     cmatrix = backend.adjoint(self.wtransform) @ cm @ self.wtransform * 0.25
    #     return type(self)(self.L, cmatrix=cmatrix)

    # def overlap(self, other):
    #     # ?
    #     u, v = self.get_bogoliubov_uv()
    #     u1, v1 = other.get_bogoliubov_uv()
    #     return backend.det(
    #         backend.adjoint(u1) @ u + backend.adjoint(v1) @ v
    #     ) * backend.det(backend.adjoint(v1) @ v), backend.det(
    #         backend.adjoint(u1) @ u + backend.adjoint(v1) @ v
    #     ) * backend.det(
    #         backend.adjoint(u1) @ u
    #     )


class FGSTestSimulator:
    def __init__(
        self, L: int, filled: Optional[List[int]] = None, hc: Optional[Tensor] = None
    ):
        if filled is None:
            filled = []
        self.L = L
        self.state = self.init_state(filled, L)
        if hc is not None:
            self.state = self.fermion_diagonalization(hc, L)

    @staticmethod
    def init_state(filled: List[int], L: int) -> Tensor:
        c = Circuit(L)
        for i in filled:
            c.x(i)  # type: ignore
        return c.state()

    @classmethod
    def fermion_diagonalization(cls, hc: Tensor, L: int) -> Tensor:
        h = cls.get_hmatrix(hc, L)
        _, u = np.linalg.eigh(h)
        return u[:, 0]

    @staticmethod
    def get_hmatrix(hc: Tensor, L: int) -> Tensor:
        hm = np.zeros([2**L, 2**L], dtype=complex)
        for i in range(L):
            for j in range(L):
                op = openfermion.FermionOperator(f"{str(i)}^ {str(j)}")
                hm += (
                    hc[i, j] * openfermion.get_sparse_operator(op, n_qubits=L).todense()
                )

        for i in range(L, 2 * L):
            for j in range(L):
                op = openfermion.FermionOperator(f"{str(i-L)} {str(j)}")
                hm += (
                    hc[i, j] * openfermion.get_sparse_operator(op, n_qubits=L).todense()
                )

        for i in range(L):
            for j in range(L, 2 * L):
                op = openfermion.FermionOperator(f"{str(i)}^ {str(j-L)}^")
                hm += (
                    hc[i, j] * openfermion.get_sparse_operator(op, n_qubits=L).todense()
                )

        for i in range(L, 2 * L):
            for j in range(L, 2 * L):
                op = openfermion.FermionOperator(f"{str(i-L)} {str(j-L)}^")
                hm += (
                    hc[i, j] * openfermion.get_sparse_operator(op, n_qubits=L).todense()
                )

        return hm

    @staticmethod
    def hopping_jw(chi: Tensor, i: int, j: int, L: int) -> Tensor:
        op = chi * openfermion.FermionOperator(f"{str(i)}^ {str(j)}") + np.conj(
            chi
        ) * openfermion.FermionOperator(f"{str(j)}^ {str(i)}")
        sop = openfermion.transforms.jordan_wigner(op)
        m = openfermion.get_sparse_operator(sop, n_qubits=L).todense()
        return m

    @staticmethod
    def chemical_potential_jw(chi: Tensor, i: int, L: int) -> Tensor:
        op = chi * openfermion.FermionOperator(f"{str(i)}^ {str(i)}")
        sop = openfermion.transforms.jordan_wigner(op)
        m = openfermion.get_sparse_operator(sop, n_qubits=L).todense()
        return m

    def evol_hamiltonian(self, h: Tensor) -> None:
        self.state = backend.expm(-1 / 2 * 1.0j * h) @ backend.reshape(
            self.state, [-1, 1]
        )
        self.state = backend.reshape(self.state, [-1])

    def evol_ihamiltonian(self, h: Tensor) -> None:
        self.state = backend.expm(-1 / 2 * h) @ backend.reshape(self.state, [-1, 1])
        self.state = backend.reshape(self.state, [-1])
        self.orthogonal()

    def evol_hp(self, i: int, j: int, chi: Tensor = 0) -> None:
        self.evol_hamiltonian(self.hopping_jw(chi, i, j, self.L))

    def evol_cp(self, i: int, chi: Tensor = 0) -> None:
        self.evol_hamiltonian(self.chemical_potential_jw(chi, i, self.L))

    def evol_icp(self, i: int, chi: Tensor = 0) -> None:
        self.evol_ihamiltonian(self.chemical_potential_jw(chi, i, self.L))

    @staticmethod
    def sc_pairing_jw(chi: Tensor, i: int, j: int, L: int) -> Tensor:
        op = chi * openfermion.FermionOperator(f"{str(i)}^ {str(j)}^") + np.conj(
            chi
        ) * openfermion.FermionOperator(f"{str(j)} {str(i)}")
        sop = openfermion.transforms.jordan_wigner(op)
        m = openfermion.get_sparse_operator(sop, n_qubits=L).todense()
        return m

    def evol_sp(self, i: int, j: int, chi: Tensor = 0) -> None:
        self.evol_hamiltonian(self.sc_pairing_jw(chi, i, j, self.L))

    def orthogonal(self) -> None:
        self.state /= backend.norm(self.state)

    def get_cmatrix(self) -> Tensor:
        alpha1_jw = self.state
        cmatrix = np.zeros([2 * self.L, 2 * self.L], dtype=complex)
        for i in range(self.L):
            for j in range(self.L):
                op = openfermion.FermionOperator(f"{str(i)} {str(j)}^")
                m = openfermion.get_sparse_operator(op, n_qubits=self.L).todense()
                cmatrix[i, j] = backend.item(
                    (
                        backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                        @ m
                        @ backend.reshape(alpha1_jw, [-1, 1])
                    )[0, 0]
                )
        for i in range(self.L, 2 * self.L):
            for j in range(self.L):
                op = openfermion.FermionOperator(f"{str(i-self.L)}^ {str(j)}^")
                m = openfermion.get_sparse_operator(op, n_qubits=self.L).todense()
                cmatrix[i, j] = backend.item(
                    (
                        backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                        @ m
                        @ backend.reshape(alpha1_jw, [-1, 1])
                    )[0, 0]
                )
        for i in range(self.L):
            for j in range(self.L, 2 * self.L):
                op = openfermion.FermionOperator(f"{str(i)} {str(j-self.L)}")
                m = openfermion.get_sparse_operator(op, n_qubits=self.L).todense()
                cmatrix[i, j] = backend.item(
                    (
                        backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                        @ m
                        @ backend.reshape(alpha1_jw, [-1, 1])
                    )[0, 0]
                )
        for i in range(self.L, 2 * self.L):
            for j in range(self.L, 2 * self.L):
                op = openfermion.FermionOperator(f"{str(i-self.L)}^ {str(j-self.L)}")
                m = openfermion.get_sparse_operator(op, n_qubits=self.L).todense()
                cmatrix[i, j] = backend.item(
                    (
                        backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                        @ m
                        @ backend.reshape(alpha1_jw, [-1, 1])
                    )[0, 0]
                )
        return cmatrix

    def get_cmatrix_majorana(self) -> Tensor:
        alpha1_jw = self.state
        cmatrix = np.zeros([2 * self.L, 2 * self.L], dtype=complex)
        for i in range(2 * self.L):
            for j in range(2 * self.L):
                op = openfermion.MajoranaOperator((i,)) * openfermion.MajoranaOperator(
                    (j,)
                )
                if j % 2 + i % 2 == 1:
                    op *= -1
                # convention diff in jordan wigner
                # c\dagger = X\pm iY

                op = openfermion.jordan_wigner(op)
                m = openfermion.get_sparse_operator(op, n_qubits=self.L).todense()
                cmatrix[i, j] = backend.item(
                    (
                        backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                        @ m
                        @ backend.reshape(alpha1_jw, [-1, 1])
                    )[0, 0]
                )

        return cmatrix

    def entropy(self, subsystems_to_trace_out: Optional[List[int]] = None) -> Tensor:
        rm = quantum.reduced_density_matrix(self.state, subsystems_to_trace_out)  # type: ignore
        return quantum.entropy(rm)

    def renyi_entropy(self, n: int, subsystems_to_trace_out: List[int]) -> Tensor:
        rm = quantum.reduced_density_matrix(self.state, subsystems_to_trace_out)
        return quantum.renyi_entropy(rm, n)

    def overlap(self, other: "FGSTestSimulator") -> Tensor:
        return backend.tensordot(backend.conj(self.state), other.state, 1)

    def get_dm(self) -> Tensor:
        s = backend.reshape(self.state, [-1, 1])
        return s @ backend.adjoint(s)

    def product(self, other: "FGSTestSimulator") -> Tensor:
        rho1 = self.get_dm()
        rho2 = other.get_dm()
        rho = rho1 @ rho2
        rho /= backend.trace(rho)
        return rho
