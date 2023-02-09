"""
FiniteMPS from tensornetwork with bug fixed
"""
# pylint: disable=invalid-name

from typing import Any, Optional, List, Sequence

import numpy as np
from tensornetwork.linalg.node_linalg import conj
import tensornetwork as tn
import tensornetwork.ncon_interface as ncon
from tensornetwork.network_components import Node


from .cons import backend

Tensor = Any


class FiniteMPS(tn.FiniteMPS):  # type: ignore
    center_position: Optional[int]

    # TODO(@SUSYUSTC): Maybe more functions can be put here to disentangle with circuits
    def apply_two_site_gate(
        self,
        gate: Tensor,
        site1: int,
        site2: int,
        max_singular_values: Optional[int] = None,
        max_truncation_err: Optional[float] = None,
        center_position: Optional[int] = None,
        relative: bool = False,
    ) -> Tensor:
        """
        Apply a two-site gate to an MPS. This routine will in general destroy
        any canonical form of the state. If a canonical form is needed, the user
        can restore it using `FiniteMPS.position`.

        :param gate: A two-body gate.
        :type gate: Tensor
        :param site1: The first site where the gate acts.
        :type site1: int
        :param site2: The second site where the gate acts.
        :type site2: int
        :param max_singular_values: The maximum number of singular values to keep.
        :type max_singular_values: Optional[float], optional
        :param max_truncation_err: The maximum allowed truncation error.
        :type max_truncation_err: Optional[float], optional
        :param center_position: An optional value to choose the MPS tensor at
            `center_position` to be isometric after the application of the gate.
            Defaults to `site1`. If the MPS is canonical (i.e.`BaseMPS.center_position != None`),
            and if the orthogonality center
            coincides with either `site1` or `site2`,  the orthogonality center will
            be shifted to `center_position` (`site1` by default).
            If the orthogonality center does not coincide with `(site1, site2)` then
            `MPS.center_position` is set to `None`.
        :type center_position: Optional[int],optional
        :param relative: Multiply `max_truncation_err` with the largest singular value.
        :type relative: bool
        :raises ValueError: "rank of gate is {} but has to be 4", "site1 = {} is not between 0 <= site < N - 1 = {}",
            "site2 = {} is not between 1 <= site < N = {}","Found site2 ={}, site1={}. Only nearest
            neighbor gates are currently supported",
            "f center_position = {center_position} not  f in {(site1, site2)} ", or
            "center_position = {}, but gate is applied at sites {}, {}. Truncation should only be done if the gate
            is applied at the center position of the MPS."
        :return: A scalar tensor containing the truncated weight of the truncation.
        :rtype: Tensor
        """
        # Note: google/tensornetwork implementation has a center position bug, which is fixed here
        if len(gate.shape) != 4:
            raise ValueError(
                "rank of gate is {} but has to be 4".format(len(gate.shape))
            )
        if site1 < 0 or site1 >= len(self) - 1:
            raise ValueError(
                "site1 = {} is not between 0 <= site < N - 1 = {}".format(
                    site1, len(self)
                )
            )
        if site2 < 1 or site2 >= len(self):
            raise ValueError(
                "site2 = {} is not between 1 <= site < N = {}".format(site2, len(self))
            )
        if site2 <= site1:
            raise ValueError(
                "site2 = {} has to be larger than site2 = {}".format(site2, site1)
            )
        if site2 != site1 + 1:
            raise ValueError(
                "Found site2 ={}, site1={}. Only nearest "
                "neighbor gates are currently"
                "supported".format(site2, site1)
            )

        if center_position is not None and center_position not in (site1, site2):
            raise ValueError(
                f"center_position = {center_position} not " f"in {(site1, site2)} "
            )

        if (max_singular_values or max_truncation_err) and self.center_position not in (
            site1,
            site2,
        ):
            raise ValueError(
                "center_position = {}, but gate is applied at sites {}, {}. "
                "Truncation should only be done if the gate "
                "is applied at the center position of the MPS".format(
                    self.center_position, site1, site2
                )
            )

        use_svd = (max_truncation_err is not None) or (max_singular_values is not None)
        gate = self.backend.convert_to_tensor(gate)
        tensor = ncon.ncon(
            [self.tensors[site1], self.tensors[site2], gate],
            [[-1, 1, 2], [2, 3, -4], [-2, -3, 1, 3]],
            backend=self.backend,
        )

        def set_center_position(site: int) -> None:
            if self.center_position is not None:
                if self.center_position in (site1, site2):
                    assert site in (site1, site2)
                    self.center_position = site
                else:
                    self.center_position = None

        if center_position is None:
            center_position = site1

        if use_svd:
            U, S, V, tw = self.backend.svd(
                tensor,
                pivot_axis=2,
                max_singular_values=max_singular_values,
                max_truncation_error=max_truncation_err,
                relative=relative,
            )
            # Note: fix the center position bug here
            if center_position == site2:
                left_tensor = U
                right_tensor = ncon.ncon(
                    [self.backend.diagflat(S), V],
                    [[-1, 1], [1, -2, -3]],
                    backend=self.backend,
                )
                set_center_position(site2)
            else:
                left_tensor = ncon.ncon(
                    [U, self.backend.diagflat(S)],
                    [[-1, -2, 1], [1, -3]],
                    backend=self.backend,
                )
                right_tensor = V
                set_center_position(site1)

        else:
            tw = self.backend.zeros(1, dtype=self.dtype)
            if center_position == site1:
                R, Q = self.backend.rq(tensor, pivot_axis=2)
                left_tensor = R
                right_tensor = Q
                set_center_position(site1)
            else:
                Q, R = self.backend.qr(tensor, pivot_axis=2)
                left_tensor = Q
                right_tensor = R
                set_center_position(site2)

        self.tensors[site1] = left_tensor
        self.tensors[site2] = right_tensor
        return tw

    def copy(self) -> "FiniteMPS":
        tensors = [backend.copy(item) for item in self.tensors]
        result = FiniteMPS(tensors, backend=self.backend, canonicalize=False)
        result.center_position = self.center_position
        return result

    def conj(self) -> "FiniteMPS":
        tensors = [backend.copy(backend.conj(item)) for item in self.tensors]
        result = FiniteMPS(tensors, backend=self.backend, canonicalize=False)
        result.center_position = self.center_position
        return result

    def measure_local_operator(
        self, ops: List[Tensor], sites: Sequence[int]
    ) -> List[Tensor]:
        """
        Measure the expectation value of local operators `ops` site `sites`.

        :param ops: A list Tensors of rank 2; the local operators to be measured.
        :type ops: List[Tensor]
        :param sites: Sites where `ops` act.
        :type sites: Sequence[int]
        :returns: measurements :math:`\\langle` `ops[n]`:math:`\\rangle` for n in `sites`
        :rtype: List[Tensor]
        """
        # Note: google/tensornetwork implementation returns floating numbers which cannot be differentiated or jitted
        if not len(ops) == len(sites):
            raise ValueError("measure_1site_ops: len(ops) has to be len(sites)!")
        right_envs = self.right_envs(sites)
        left_envs = self.left_envs(sites)
        res = []
        for n, site in enumerate(sites):
            O = Node(ops[n], backend=self.backend)
            R = Node(right_envs[site], backend=self.backend)
            L = Node(left_envs[site], backend=self.backend)
            A = Node(self.tensors[site], backend=self.backend)
            conj_A = conj(A)
            O[1] ^ A[1]
            O[0] ^ conj_A[1]
            R[0] ^ A[2]
            R[1] ^ conj_A[2]
            L[0] ^ A[0]
            L[1] ^ conj_A[0]
            result = L @ A @ O @ conj_A @ R
            res.append(result.tensor)
        return res

    def measure_two_body_correlator(
        self, op1: Tensor, op2: Tensor, site1: int, sites2: Sequence[int]
    ) -> List[Tensor]:
        """
        Compute the correlator
        :math:`\\langle` `op1[site1], op2[s]`:math:`\\rangle`
        between `site1` and all sites `s` in `sites2`. If `s == site1`,
        `op2[s]` will be applied first.

        :param op1: Tensor of rank 2; the local operator at `site1`.
        :type op1: Tensor
        :param op2: Tensor of rank 2; the local operator at `sites2`.
        :type op2: Tensor
        :param site1: The site where `op1`  acts
        :type site1: int
        :param sites2: Sites where operator `op2` acts.
        :type sites2: Sequence[int]
        :returns: Correlator :math:`\\langle` `op1[site1], op2[s]`:math:`\\rangle` for `s` :math:`\\in` `sites2`.
        :rtype: List[Tensor]
        """
        # Note: google/tensornetwork implementation returns floating numbers which cannot be differentiated or jitted
        N = len(self)
        if site1 < 0:
            raise ValueError(
                "Site site1 out of range: {} not between 0 <= site < N = {}.".format(
                    site1, N
                )
            )
        sites2 = np.array(sites2)  # type: ignore

        # we break the computation into two parts:
        # first we get all correlators <op2(site2) op1(site1)> with site2 < site1
        # then all correlators <op1(site1) op2(site2)> with site2 >= site1

        # get all sites smaller than site1
        left_sites = np.sort(sites2[sites2 < site1])  # type: ignore
        # get all sites larger than site1
        right_sites = np.sort(sites2[sites2 > site1])  # type: ignore
        # compute all neccessary right reduced
        # density matrices in one go. This is
        # more efficient than calling right_envs
        # for each site individually
        rs = self.right_envs(np.append(site1, np.mod(right_sites, N)).astype(np.int64))
        ls = self.left_envs(np.append(np.mod(left_sites, N), site1).astype(np.int64))

        c = []
        if len(left_sites) > 0:
            A = Node(self.tensors[site1], backend=self.backend)
            O1 = Node(op1, backend=self.backend)
            conj_A = conj(A)
            R = Node(rs[site1], backend=self.backend)
            R[0] ^ A[2]
            R[1] ^ conj_A[2]
            A[1] ^ O1[1]
            conj_A[1] ^ O1[0]
            R = ((R @ A) @ O1) @ conj_A
            n1 = np.min(left_sites)
            #          -- A--------
            #             |        |
            # compute   op1(site1) |
            #             |        |
            #          -- A*-------
            # and evolve it to the left by contracting tensors at site2 < site1
            # if site2 is in `sites2`, calculate the observable
            #
            #  ---A--........-- A--------
            # |   |             |        |
            # |  op2(site2)    op1(site1)|
            # |   |             |        |
            #  ---A--........-- A*-------

            for n in range(site1 - 1, n1 - 1, -1):
                if n in left_sites:
                    A = Node(self.tensors[n % N], backend=self.backend)
                    conj_A = conj(A)
                    O2 = Node(op2, backend=self.backend)
                    L = Node(ls[n % N], backend=self.backend)
                    L[0] ^ A[0]
                    L[1] ^ conj_A[0]
                    O2[0] ^ conj_A[1]
                    O2[1] ^ A[1]
                    R[0] ^ A[2]
                    R[1] ^ conj_A[2]

                    res = (((L @ A) @ O2) @ conj_A) @ R
                    c.append(res.tensor)
                if n > n1:
                    R = Node(
                        self.apply_transfer_operator(n % N, "right", R.tensor),
                        backend=self.backend,
                    )

            c = list(reversed(c))

        # compute <op2(site1)op1(site1)>
        if site1 in sites2:
            O1 = Node(op1, backend=self.backend)
            O2 = Node(op2, backend=self.backend)
            L = Node(ls[site1], backend=self.backend)
            R = Node(rs[site1], backend=self.backend)
            A = Node(self.tensors[site1], backend=self.backend)
            conj_A = conj(A)

            O1[1] ^ O2[0]
            L[0] ^ A[0]
            L[1] ^ conj_A[0]
            R[0] ^ A[2]
            R[1] ^ conj_A[2]
            A[1] ^ O2[1]
            conj_A[1] ^ O1[0]
            O = O1 @ O2
            res = (((L @ A) @ O) @ conj_A) @ R
            c.append(res.tensor)

        # compute <op1(site1) op2(site2)> for site1 < site2
        if len(right_sites) > 0:
            A = Node(self.tensors[site1], backend=self.backend)
            conj_A = conj(A)
            L = Node(ls[site1], backend=self.backend)
            O1 = Node(op1, backend=self.backend)
            L[0] ^ A[0]
            L[1] ^ conj_A[0]
            A[1] ^ O1[1]
            conj_A[1] ^ O1[0]
            L = L @ A @ O1 @ conj_A
            n2 = np.max(right_sites)
            #          -- A--
            #         |   |
            # compute | op1(site1)
            #         |   |
            #          -- A*--
            # and evolve it to the right by contracting tensors at site2 > site1
            # if site2 is in `sites2`, calculate the observable
            #
            #  ---A--........-- A--------
            # |   |             |        |
            # |  op1(site1)    op2(site2)|
            # |   |             |        |
            #  ---A--........-- A*-------
            for n in range(site1 + 1, n2 + 1):
                if n in right_sites:
                    R = Node(rs[n % N], backend=self.backend)
                    A = Node(self.tensors[n % N], backend=self.backend)
                    conj_A = conj(A)
                    O2 = Node(op2, backend=self.backend)
                    A[0] ^ L[0]
                    conj_A[0] ^ L[1]
                    O2[0] ^ conj_A[1]
                    O2[1] ^ A[1]
                    R[0] ^ A[2]
                    R[1] ^ conj_A[2]
                    res = L @ A @ O2 @ conj_A @ R
                    c.append(res.tensor)

                if n < n2:
                    L = Node(
                        self.apply_transfer_operator(n % N, "left", L.tensor),
                        backend=self.backend,
                    )

        return c
