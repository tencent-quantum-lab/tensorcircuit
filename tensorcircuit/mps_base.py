from typing import Any, Optional
import tensornetwork as tn  # type: ignore
import tensornetwork.ncon_interface as ncon  # type: ignore
from .cons import backend, npdtype

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
        """Apply a two-site gate to an MPS. This routine will in general destroy
        any canonical form of the state. If a canonical form is needed, the user
        can restore it using `FiniteMPS.position`.

        Args:
          gate: A two-body gate.
          site1: The first site where the gate acts.
          site2: The second site where the gate acts.
          max_singular_values: The maximum number of singular values to keep.
          max_truncation_err: The maximum allowed truncation error.
          center_position: An optional value to choose the MPS tensor at
            `center_position` to be isometric after the application of the gate.
            Defaults to `site1`. If the MPS is canonical (i.e.
            `BaseMPS.center_position != None`), and if the orthogonality center
            coincides with either `site1` or `site2`,  the orthogonality center will
            be shifted to `center_position` (`site1` by default). If the
            orthogonality center does not coincide with `(site1, site2)` then
            `MPS.center_position` is set to `None`.
          relative: Multiply `max_truncation_err` with the largest singular value.

        Returns:
          `Tensor`: A scalar tensor containing the truncated weight of the
            truncation.
        """
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
            # fix the center position bug here
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
