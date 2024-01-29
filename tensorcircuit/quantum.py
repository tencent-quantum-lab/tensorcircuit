"""
Quantum state and operator class backend by tensornetwork

:IMPORT:

.. code-block:: python

    import tensorcircuit.quantum as qu
"""

# pylint: disable=invalid-name

import os
from functools import reduce, partial
import logging
from operator import or_, mul, matmul
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from tensornetwork.network_components import AbstractNode, Node, Edge, connect
from tensornetwork.network_components import CopyNode
from tensornetwork.network_operations import get_all_nodes, copy, reachable
from tensornetwork.network_operations import get_subgraph_dangling, remove_node

try:
    import tensorflow as tf
except ImportError:
    pass

from .cons import backend, contractor, dtypestr, npdtype, rdtypestr
from .backends import get_backend
from .utils import is_m1mac, arg_alias
from .gates import Gate

Tensor = Any
Graph = Any

logger = logging.getLogger(__name__)

# Note the first version of part of this file is adapted from source code of tensornetwork: (Apache2)
# https://github.com/google/TensorNetwork/blob/master/tensornetwork/quantum/quantum.py
# For the reason of adoption instead of direct import: see https://github.com/google/TensorNetwork/issues/950


# general conventions left (first) out, right (then) in


def quantum_constructor(
    out_edges: Sequence[Edge],
    in_edges: Sequence[Edge],
    ref_nodes: Optional[Collection[AbstractNode]] = None,
    ignore_edges: Optional[Collection[Edge]] = None,
) -> "QuOperator":
    """
    Constructs an appropriately specialized QuOperator.
    If there are no edges, creates a QuScalar. If the are only output (input)
    edges, creates a QuVector (QuAdjointVector). Otherwise creates a QuOperator.

    :Example:

    .. code-block:: python

        def show_attributes(op):
            print(f"op.is_scalar() \t\t-> {op.is_scalar()}")
            print(f"op.is_vector() \t\t-> {op.is_vector()}")
            print(f"op.is_adjoint_vector() \t-> {op.is_adjoint_vector()}")
            print(f"len(op.out_edges) \t-> {len(op.out_edges)}")
            print(f"len(op.in_edges) \t-> {len(op.in_edges)}")

    >>> psi_node = tn.Node(np.random.rand(2, 2))
    >>>
    >>> op = qu.quantum_constructor([psi_node[0]], [psi_node[1]])
    >>> show_attributes(op)
    op.is_scalar()          -> False
    op.is_vector()          -> False
    op.is_adjoint_vector()  -> False
    len(op.out_edges)       -> 1
    len(op.in_edges)        -> 1
    >>> # psi_node[0] -> op.out_edges[0]
    >>> # psi_node[1] -> op.in_edges[0]

    >>> op = qu.quantum_constructor([psi_node[0], psi_node[1]], [])
    >>> show_attributes(op)
    op.is_scalar()          -> False
    op.is_vector()          -> True
    op.is_adjoint_vector()  -> False
    len(op.out_edges)       -> 2
    len(op.in_edges)        -> 0
    >>> # psi_node[0] -> op.out_edges[0]
    >>> # psi_node[1] -> op.out_edges[1]

    >>> op = qu.quantum_constructor([], [psi_node[0], psi_node[1]])
    >>> show_attributes(op)
    op.is_scalar()          -> False
    op.is_vector()          -> False
    op.is_adjoint_vector()  -> True
    len(op.out_edges)       -> 0
    len(op.in_edges)        -> 2
    >>> # psi_node[0] -> op.in_edges[0]
    >>> # psi_node[1] -> op.in_edges[1]

    :param out_edges: A list of output edges.
    :type out_edges: Sequence[Edge]
    :param in_edges: A list of input edges.
    :type in_edges: Sequence[Edge]
    :param ref_nodes: Reference nodes for the tensor network (needed if there is a.
        scalar component).
    :type ref_nodes: Optional[Collection[AbstractNode]], optional
    :param ignore_edges: Edges to ignore when checking the dimensionality of the
        tensor network.
    :type ignore_edges: Optional[Collection[Edge]], optional
    :return: The new created QuOperator object.
    :rtype: QuOperator
    """
    if len(out_edges) == 0 and len(in_edges) == 0:
        return QuScalar(ref_nodes, ignore_edges)  # type: ignore
    if len(out_edges) == 0:
        return QuAdjointVector(in_edges, ref_nodes, ignore_edges)
    if len(in_edges) == 0:
        return QuVector(out_edges, ref_nodes, ignore_edges)
    return QuOperator(out_edges, in_edges, ref_nodes, ignore_edges)


def identity(
    space: Sequence[int],
    dtype: Any = None,
) -> "QuOperator":
    """
    Construct a 'QuOperator' representing the identity on a given space.
    Internally, this is done by constructing 'CopyNode's for each edge, with
    dimension according to 'space'.

    :Example:

    >>> E = qu.identity((2, 3, 4))
    >>> float(E.trace().eval())
    24.0

    >>> tensor = np.random.rand(2, 2)
    >>> psi = qu.QuVector.from_tensor(tensor)
    >>> E = qu.identity((2, 2))
    >>> psi.eval()
    array([[0.03964233, 0.99298281],
           [0.38564989, 0.00950596]])
    >>> (E @ psi).eval()
    array([[0.03964233, 0.99298281],
           [0.38564989, 0.00950596]])
    >>>
    >>> (psi.adjoint() @ E @ psi).eval()
    array(1.13640257)
    >>> psi.norm().eval()
    array(1.13640257)

    :param space: A sequence of integers for the dimensions of the tensor product
        factors of the space (the edges in the tensor network).
    :type space: Sequence[int]
    :param dtype: The data type by np.* (for conversion to dense). defaults None to tc dtype.
    :type dtype: Any type
    :return: The desired identity operator.
    :rtype: QuOperator
    """
    if dtype is None:
        dtype = npdtype
    nodes = [CopyNode(2, d, dtype=dtype) for d in space]
    out_edges = [n[0] for n in nodes]
    in_edges = [n[1] for n in nodes]
    return quantum_constructor(out_edges, in_edges)


def check_spaces(edges_1: Sequence[Edge], edges_2: Sequence[Edge]) -> None:
    """
    Check the vector spaces represented by two lists of edges are compatible.
    The number of edges must be the same and the dimensions of each pair of edges
    must match. Otherwise, an exception is raised.

    :param edges_1: List of edges representing a many-body Hilbert space.
    :type edges_1: Sequence[Edge]
    :param edges_2: List of edges representing a many-body Hilbert space.
    :type edges_2: Sequence[Edge]

    :raises ValueError: Hilbert-space mismatch: "Cannot connect {} subsystems with {} subsystems", or
        "Input dimension {} != output dimension {}."
    """
    if len(edges_1) != len(edges_2):
        raise ValueError(
            "Hilbert-space mismatch: Cannot connect {} subsystems "
            "with {} subsystems.".format(len(edges_1), len(edges_2))
        )

    for i, (e1, e2) in enumerate(zip(edges_1, edges_2)):
        if e1.dimension != e2.dimension:
            raise ValueError(
                "Hilbert-space mismatch on subsystems {}: Input "
                "dimension {} != output dimension {}.".format(
                    i, e1.dimension, e2.dimension
                )
            )


def eliminate_identities(nodes: Collection[AbstractNode]) -> Tuple[dict, dict]:  # type: ignore
    """
    Eliminates any connected CopyNodes that are identity matrices.
    This will modify the network represented by `nodes`.
    Only identities that are connected to other nodes are eliminated.

    :param nodes: Collection of nodes to search.
    :type nodes: Collection[AbstractNode]
    :return: The Dictionary mapping remaining Nodes to any replacements, Dictionary specifying all dangling-edge
        replacements.
    :rtype: Dict[Union[CopyNode, AbstractNode], Union[Node, AbstractNode]], Dict[Edge, Edge]
    """
    nodes_dict = {}
    dangling_edges_dict = {}
    for n in nodes:
        if (
            isinstance(n, CopyNode)
            and n.get_rank() == 2
            and not (n[0].is_dangling() and n[1].is_dangling())
        ):
            old_edges = [n[0], n[1]]
            _, new_edges = remove_node(n)
            if 0 in new_edges and 1 in new_edges:
                e = connect(new_edges[0], new_edges[1])
            elif 0 in new_edges:  # 1 was dangling
                dangling_edges_dict[old_edges[1]] = new_edges[0]
            elif 1 in new_edges:  # 0 was dangling
                dangling_edges_dict[old_edges[0]] = new_edges[1]
            else:
                # Trace of identity, so replace with a scalar node!
                d = n.get_dimension(0)
                # NOTE: Assume CopyNodes have numpy dtypes.
                nodes_dict[n] = Node(np.array(d, dtype=n.dtype))
        else:
            for e in n.get_all_dangling():
                dangling_edges_dict[e] = e
            nodes_dict[n] = n

    return nodes_dict, dangling_edges_dict


class QuOperator:
    """
    Represents a linear operator via a tensor network.
    To interpret a tensor network as a linear operator, some of the dangling
    edges must be designated as `out_edges` (output edges) and the rest as
    `in_edges` (input edges).
    Considered as a matrix, the `out_edges` represent the row index and the
    `in_edges` represent the column index.
    The (right) action of the operator on another then consists of connecting
    the `in_edges` of the first operator to the `out_edges` of the second.
    Can be used to do simple linear algebra with tensor networks.
    """

    __array_priority__ = 100.0  # for correct __rmul__ with scalar ndarrays

    def __init__(
        self,
        out_edges: Sequence[Edge],
        in_edges: Sequence[Edge],
        ref_nodes: Optional[Collection[AbstractNode]] = None,
        ignore_edges: Optional[Collection[Edge]] = None,
    ) -> None:
        """
        Creates a new `QuOperator` from a tensor network.
        This encapsulates an existing tensor network, interpreting it as a linear
        operator.
        The network is checked for consistency: All dangling edges must either be
        in `out_edges`, `in_edges`, or `ignore_edges`.

        :param out_edges: The edges of the network to be used as the output edges.
        :type out_edges: Sequence[Edge]
        :param in_edges: The edges of the network to be used as the input edges.
        :type in_edges: Sequence[Edge]
        :param ref_nodes: Nodes used to refer to parts of the tensor network that are
            not connected to any input or output edges (for example: a scalar
            factor).
        :type ref_nodes: Optional[Collection[AbstractNode]], optional
        :param ignore_edges: Optional collection of dangling edges to ignore when
            performing consistency checks.
        :type ignore_edges: Optional[Collection[Edge]], optional
        :raises ValueError: At least one reference node is required to specify a scalar. None provided!
        """
        # TODO: Decide whether the user must also supply all nodes involved.
        #       This would enable extra error checking and is probably clearer
        #       than `ref_nodes`.
        if len(in_edges) == 0 and len(out_edges) == 0 and not ref_nodes:
            raise ValueError(
                "At least one reference node is required to specify a "
                "scalar. None provided!"
            )
        self.out_edges = list(out_edges)
        self.in_edges = list(in_edges)
        self.ignore_edges = set(ignore_edges) if ignore_edges else set()
        self.ref_nodes = set(ref_nodes) if ref_nodes else set()
        self.check_network()

    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
        out_axes: Optional[Sequence[int]] = None,
        in_axes: Optional[Sequence[int]] = None,
    ) -> "QuOperator":
        r"""
        Construct a `QuOperator` directly from a single tensor.
        This first wraps the tensor in a `Node`, then constructs the `QuOperator`
        from that `Node`.

        :Example:

        .. code-block:: python

            def show_attributes(op):
                print(f"op.is_scalar() \t\t-> {op.is_scalar()}")
                print(f"op.is_vector() \t\t-> {op.is_vector()}")
                print(f"op.is_adjoint_vector() \t-> {op.is_adjoint_vector()}")
                print(f"op.eval() \n{op.eval()}")

        >>> psi_tensor = np.random.rand(2, 2)
        >>> psi_tensor
        array([[0.27260127, 0.91401091],
               [0.06490953, 0.38653646]])
        >>> op = qu.QuOperator.from_tensor(psi_tensor, out_axes=[0], in_axes=[1])
        >>> show_attributes(op)
        op.is_scalar()          -> False
        op.is_vector()          -> False
        op.is_adjoint_vector()  -> False
        op.eval()
        [[0.27260127 0.91401091]
         [0.06490953 0.38653646]]

        :param tensor: The tensor.
        :type tensor: Tensor
        :param out_axes: The axis indices of `tensor` to use as `out_edges`.
        :type out_axes: Optional[Sequence[int]], optional
        :param in_axes: The axis indices of `tensor` to use as `in_edges`.
        :type in_axes: Optional[Sequence[int]], optional
        :return: The new operator.
        :rtype: QuOperator
        """
        nlegs = len(tensor.shape)
        if (out_axes is None) and (in_axes is None):
            out_axes = [i for i in range(int(nlegs / 2))]
            in_axes = [i for i in range(int(nlegs / 2), nlegs)]
        elif out_axes is None:
            out_axes = [i for i in range(nlegs) if i not in in_axes]  # type: ignore
        elif in_axes is None:
            in_axes = [i for i in range(nlegs) if i not in out_axes]
        n = Node(tensor)
        out_edges = [n[i] for i in out_axes]
        in_edges = [n[i] for i in in_axes]  # type: ignore
        return cls(out_edges, in_edges)

    @classmethod
    def from_local_tensor(
        cls,
        tensor: Tensor,
        space: Sequence[int],
        loc: Sequence[int],
        out_axes: Optional[Sequence[int]] = None,
        in_axes: Optional[Sequence[int]] = None,
    ) -> "QuOperator":
        nlegs = len(tensor.shape)
        if (out_axes is None) and (in_axes is None):
            out_axes = [i for i in range(int(nlegs / 2))]
            in_axes = [i for i in range(int(nlegs / 2), nlegs)]
        elif out_axes is None:
            out_axes = [i for i in range(nlegs) if i not in in_axes]  # type: ignore
        elif in_axes is None:
            in_axes = [i for i in range(nlegs) if i not in out_axes]
        localn = Node(tensor)
        out_edges = [localn[i] for i in out_axes]
        in_edges = [localn[i] for i in in_axes]  # type: ignore
        id_nodes = [
            CopyNode(2, d, dtype=npdtype) for i, d in enumerate(space) if i not in loc
        ]
        for n in id_nodes:
            out_edges.append(n[0])
            in_edges.append(n[1])

        return cls(out_edges, in_edges)

    @property
    def nodes(self) -> Set[AbstractNode]:
        """All tensor-network nodes involved in the operator."""
        return reachable(get_all_nodes(self.out_edges + self.in_edges) | self.ref_nodes)  # type: ignore

    @property
    def in_space(self) -> List[int]:
        return [e.dimension for e in self.in_edges]

    @property
    def out_space(self) -> List[int]:
        return [e.dimension for e in self.out_edges]

    def is_scalar(self) -> bool:
        """
        Returns a bool indicating if QuOperator is a scalar.
        Examples can be found in the `QuOperator.from_tensor`.
        """
        return len(self.out_edges) == 0 and len(self.in_edges) == 0

    def is_vector(self) -> bool:
        """
        Returns a bool indicating if QuOperator is a vector.
        Examples can be found in the `QuOperator.from_tensor`.
        """
        return len(self.out_edges) > 0 and len(self.in_edges) == 0

    def is_adjoint_vector(self) -> bool:
        """
        Returns a bool indicating if QuOperator is an adjoint vector.
        Examples can be found in the `QuOperator.from_tensor`.
        """
        return len(self.out_edges) == 0 and len(self.in_edges) > 0

    def check_network(self) -> None:
        """
        Check that the network has the expected dimensionality.
        This checks that all input and output edges are dangling and that
        there are no other dangling edges (except any specified in
        `ignore_edges`). If not, an exception is raised.
        """
        for i, e in enumerate(self.out_edges):
            if not e.is_dangling():
                raise ValueError("Output edge {} is not dangling!".format(i))
        for i, e in enumerate(self.in_edges):
            if not e.is_dangling():
                raise ValueError("Input edge {} is not dangling!".format(i))
        for e in self.ignore_edges:
            if not e.is_dangling():
                raise ValueError(
                    "ignore_edges contains non-dangling edge: {}".format(str(e))
                )

        known_edges = set(self.in_edges) | set(self.out_edges) | self.ignore_edges
        all_dangling_edges = get_subgraph_dangling(self.nodes)
        if known_edges != all_dangling_edges:
            raise ValueError(
                "The network includes unexpected dangling edges (that "
                "are not members of ignore_edges)."
            )

    def adjoint(self) -> "QuOperator":
        """
        The adjoint of the operator.
        This creates a new `QuOperator` with complex-conjugate copies of all
        tensors in the network and with the input and output edges switched.

        :return: The adjoint of the operator.
        :rtype: QuOperator
        """
        nodes_dict, edge_dict = copy(self.nodes, True)
        out_edges = [edge_dict[e] for e in self.in_edges]
        in_edges = [edge_dict[e] for e in self.out_edges]
        ref_nodes = [nodes_dict[n] for n in self.ref_nodes]
        ignore_edges = [edge_dict[e] for e in self.ignore_edges]
        return quantum_constructor(out_edges, in_edges, ref_nodes, ignore_edges)

    def copy(self) -> "QuOperator":
        """
        The deep copy of the operator.

        :return: The new copy of the operator.
        :rtype: QuOperator
        """
        nodes_dict, edge_dict = copy(self.nodes, False)
        out_edges = [edge_dict[e] for e in self.out_edges]
        in_edges = [edge_dict[e] for e in self.in_edges]
        ref_nodes = [nodes_dict[n] for n in self.ref_nodes]
        ignore_edges = [edge_dict[e] for e in self.ignore_edges]
        return quantum_constructor(out_edges, in_edges, ref_nodes, ignore_edges)

    def trace(self) -> "QuOperator":
        """The trace of the operator."""
        return self.partial_trace(range(len(self.in_edges)))

    def norm(self) -> "QuOperator":
        """
        The norm of the operator.
        This is the 2-norm (also known as the Frobenius or Hilbert-Schmidt
        norm).
        """
        return (self.adjoint() @ self).trace()

    def partial_trace(self, subsystems_to_trace_out: Collection[int]) -> "QuOperator":
        """
        The partial trace of the operator.
        Subsystems to trace out are supplied as indices, so that dangling edges
        are connected to each other as:
        `out_edges[i] ^ in_edges[i] for i in subsystems_to_trace_out`
        This does not modify the original network. The original ordering of the
        remaining subsystems is maintained.

        :param subsystems_to_trace_out: Indices of subsystems to trace out.
        :type subsystems_to_trace_out: Collection[int]
        :return: A new QuOperator or QuScalar representing the result.
        :rtype: QuOperator
        """
        out_edges_trace = [self.out_edges[i] for i in subsystems_to_trace_out]
        in_edges_trace = [self.in_edges[i] for i in subsystems_to_trace_out]

        check_spaces(in_edges_trace, out_edges_trace)

        nodes_dict, edge_dict = copy(self.nodes, False)
        for e1, e2 in zip(out_edges_trace, in_edges_trace):
            edge_dict[e1] = edge_dict[e1] ^ edge_dict[e2]

        # get leftover edges in the original order
        out_edges_trace = set(out_edges_trace)  # type: ignore
        in_edges_trace = set(in_edges_trace)  # type: ignore
        out_edges = [edge_dict[e] for e in self.out_edges if e not in out_edges_trace]
        in_edges = [edge_dict[e] for e in self.in_edges if e not in in_edges_trace]
        ref_nodes = [n for _, n in nodes_dict.items()]
        ignore_edges = [edge_dict[e] for e in self.ignore_edges]

        return quantum_constructor(out_edges, in_edges, ref_nodes, ignore_edges)

    def __matmul__(self, other: Union["QuOperator", Tensor]) -> "QuOperator":
        """
        The action of this operator on another.
        Given `QuOperator`s `A` and `B`, produces a new `QuOperator` for `A @ B`,
        where `A @ B` means: "the action of A, as a linear operator, on B".
        Under the hood, this produces copies of the tensor networks defining `A`
        and `B` and then connects the copies by hooking up the `in_edges` of
        `A.copy()` to the `out_edges` of `B.copy()`.
        """
        if not isinstance(other, QuOperator):
            other = self.from_tensor(other)
        check_spaces(self.in_edges, other.out_edges)

        # Copy all nodes involved in the two operators.
        # We must do this separately for self and other, in case self and other
        # are defined via the same network components (e.g. if self === other).
        nodes_dict1, edges_dict1 = copy(self.nodes, False)
        nodes_dict2, edges_dict2 = copy(other.nodes, False)

        # connect edges to create network for the result
        for e1, e2 in zip(self.in_edges, other.out_edges):
            _ = edges_dict1[e1] ^ edges_dict2[e2]

        in_edges = [edges_dict2[e] for e in other.in_edges]
        out_edges = [edges_dict1[e] for e in self.out_edges]
        ref_nodes = [n for _, n in nodes_dict1.items()] + [
            n for _, n in nodes_dict2.items()
        ]
        ignore_edges = [edges_dict1[e] for e in self.ignore_edges] + [
            edges_dict2[e] for e in other.ignore_edges
        ]

        return quantum_constructor(out_edges, in_edges, ref_nodes, ignore_edges)

    def __rmatmul__(self, other: Union["QuOperator", Tensor]) -> "QuOperator":
        return self.__matmul__(other)

    def __mul__(self, other: Union["QuOperator", AbstractNode, Tensor]) -> "QuOperator":
        """
        Scalar multiplication of operators.
        Given two operators `A` and `B`, one of the which is a scalar (it has no
        input or output edges), `A * B` produces a new operator representing the
        scalar multiplication of `A` and `B`.
        For convenience, one of `A` or `B` may be a number or scalar-valued tensor
        or `Node` (it will automatically be wrapped in a `QuScalar`).
        Note: This is a special case of `tensor_product()`.
        """
        if not isinstance(other, QuOperator):
            if isinstance(other, AbstractNode):
                node = other
            else:
                node = Node(other)
            if node.shape:
                raise ValueError(
                    "Cannot perform elementwise multiplication by a "
                    "non-scalar tensor."
                )
            other = QuScalar([node])

        if self.is_scalar() or other.is_scalar():
            return self.tensor_product(other)

        raise ValueError(
            "Elementwise multiplication is only supported if at "
            "least one of the arguments is a scalar."
        )

    def __rmul__(
        self, other: Union["QuOperator", AbstractNode, Tensor]
    ) -> "QuOperator":
        """
        Scalar multiplication of operators.
        See `.__mul__()`.
        """
        return self.__mul__(other)

    def tensor_product(self, other: "QuOperator") -> "QuOperator":
        """
        Tensor product with another operator.
        Given two operators `A` and `B`, produces a new operator `AB` representing
        :math:`A ⊗ B`. The `out_edges` (`in_edges`) of `AB` is simply the
        concatenation of the `out_edges` (`in_edges`) of `A.copy()` with that of
        `B.copy()`:
        `new_out_edges = [*out_edges_A_copy, *out_edges_B_copy]`
        `new_in_edges = [*in_edges_A_copy, *in_edges_B_copy]`

        :Example:

        >>> psi = qu.QuVector.from_tensor(np.random.rand(2, 2))
        >>> psi_psi = psi.tensor_product(psi)
        >>> len(psi_psi.subsystem_edges)
        4
        >>> float(psi_psi.norm().eval())
        2.9887872748523585
        >>> psi.norm().eval() ** 2
        2.9887872748523585

        :param other: The other operator (`B`).
        :type other: QuOperator
        :return: The result (`AB`).
        :rtype: QuOperator
        """
        nodes_dict1, edges_dict1 = copy(self.nodes, False)
        nodes_dict2, edges_dict2 = copy(other.nodes, False)

        in_edges = [edges_dict1[e] for e in self.in_edges] + [
            edges_dict2[e] for e in other.in_edges
        ]
        out_edges = [edges_dict1[e] for e in self.out_edges] + [
            edges_dict2[e] for e in other.out_edges
        ]
        ref_nodes = [n for _, n in nodes_dict1.items()] + [
            n for _, n in nodes_dict2.items()
        ]
        ignore_edges = [edges_dict1[e] for e in self.ignore_edges] + [
            edges_dict2[e] for e in other.ignore_edges
        ]

        return quantum_constructor(out_edges, in_edges, ref_nodes, ignore_edges)

    def __or__(self, other: "QuOperator") -> "QuOperator":
        """
        Tensor product of operators.
        Given two operators `A` and `B`, `A | B` produces a new operator representing the
        tensor product of `A` and `B`.
        """
        return self.tensor_product(other)

    def contract(
        self,
        final_edge_order: Optional[Sequence[Edge]] = None,
    ) -> "QuOperator":
        """
        Contract the tensor network in place.
        This modifies the tensor network representation of the operator (or vector,
        or scalar), reducing it to a single tensor, without changing the value.

        :param final_edge_order: Manually specify the axis ordering of the final tensor.
        :type final_edge_order: Optional[Sequence[Edge]], optional
        :return: The present object.
        :rtype: QuOperator
        """
        nodes_dict, dangling_edges_dict = eliminate_identities(self.nodes)
        self.in_edges = [dangling_edges_dict[e] for e in self.in_edges]
        self.out_edges = [dangling_edges_dict[e] for e in self.out_edges]
        self.ignore_edges = set(dangling_edges_dict[e] for e in self.ignore_edges)
        self.ref_nodes = set(nodes_dict[n] for n in self.ref_nodes if n in nodes_dict)
        self.check_network()
        if final_edge_order:
            final_edge_order = [dangling_edges_dict[e] for e in final_edge_order]
            self.ref_nodes = set(
                [contractor(self.nodes, output_edge_order=final_edge_order)]
            )
        else:
            self.ref_nodes = set([contractor(self.nodes, ignore_edge_order=True)])
        return self

    def eval(
        self,
        final_edge_order: Optional[Sequence[Edge]] = None,
    ) -> Tensor:
        """
        Contracts the tensor network in place and returns the final tensor.
        Note that this modifies the tensor network representing the operator.
        The default ordering for the axes of the final tensor is:
        `*out_edges, *in_edges`.
        If there are any "ignored" edges, their axes come first:
        `*ignored_edges, *out_edges, *in_edges`.

        :param final_edge_order: Manually specify the axis ordering of the final tensor.
            The default ordering is determined by `out_edges` and `in_edges` (see above).
        :type final_edge_order: Optional[Sequence[Edge]], optional
        :raises ValueError: Node count '{}' > 1 after contraction!
        :return: The final tensor representing the operator.
        :rtype: Tensor
        """
        if not final_edge_order:
            final_edge_order = list(self.ignore_edges) + self.out_edges + self.in_edges
        self.contract(final_edge_order)
        nodes = self.nodes
        if len(nodes) != 1:
            raise ValueError(
                "Node count '{}' > 1 after contraction!".format(len(nodes))
            )
        return list(nodes)[0].tensor

    def eval_matrix(self, final_edge_order: Optional[Sequence[Edge]] = None) -> Tensor:
        r"""
        Contracts the tensor network in place and returns the final tensor
        in two dimentional matrix.
        The default ordering for the axes of the final tensor is:
        (:math:`\prod` dimension of out_edges, :math:`\prod` dimension of in_edges)

        :param final_edge_order: Manually specify the axis ordering of the final tensor.
            The default ordering is determined by `out_edges` and `in_edges` (see above).
        :type final_edge_order: Optional[Sequence[Edge]], optional
        :raises ValueError: Node count '{}' > 1 after contraction!
        :return: The two-dimentional tensor representing the operator.
        :rtype: Tensor
        """
        t = self.eval(final_edge_order)
        shape1 = reduce(mul, [e.dimension for e in self.out_edges] + [1])
        shape2 = reduce(mul, [e.dimension for e in self.in_edges] + [1])
        return backend.reshape(t, [shape1, shape2])


class QuVector(QuOperator):
    """Represents a (column) vector via a tensor network."""

    def __init__(
        self,
        subsystem_edges: Sequence[Edge],
        ref_nodes: Optional[Collection[AbstractNode]] = None,
        ignore_edges: Optional[Collection[Edge]] = None,
    ) -> None:
        """
        Constructs a new `QuVector` from a tensor network.
        This encapsulates an existing tensor network, interpreting it as a (column) vector.

        :param subsystem_edges: The edges of the network to be used as the output edges.
        :type subsystem_edges: Sequence[Edge]
        :param ref_nodes: Nodes used to refer to parts of the tensor network that are
            not connected to any input or output edges (for example: a scalar factor).
        :type ref_nodes: Optional[Collection[AbstractNode]], optional
        :param ignore_edges: Optional collection of edges to ignore when performing consistency checks.
        :type ignore_edges: Optional[Collection[Edge]], optional
        """
        super().__init__(subsystem_edges, [], ref_nodes, ignore_edges)

    @classmethod
    def from_tensor(  # type: ignore
        cls,
        tensor: Tensor,
        subsystem_axes: Optional[Sequence[int]] = None,
    ) -> "QuVector":
        r"""
        Construct a `QuVector` directly from a single tensor.
        This first wraps the tensor in a `Node`, then constructs the `QuVector`
        from that `Node`.

        :Example:

        .. code-block:: python

            def show_attributes(op):
                print(f"op.is_scalar() \t\t-> {op.is_scalar()}")
                print(f"op.is_vector() \t\t-> {op.is_vector()}")
                print(f"op.is_adjoint_vector() \t-> {op.is_adjoint_vector()}")
                print(f"op.eval() \n{op.eval()}")

        >>> psi_tensor = np.random.rand(2, 2)
        >>> psi_tensor
        array([[0.27260127, 0.91401091],
               [0.06490953, 0.38653646]])
        >>> op = qu.QuVector.from_tensor(psi_tensor, [0, 1])
        >>> show_attributes(op)
        op.is_scalar()          -> False
        op.is_vector()          -> True
        op.is_adjoint_vector()  -> False
        op.eval()
        [[0.27260127 0.91401091]
         [0.06490953 0.38653646]]

        :param tensor: The tensor for constructing a "QuVector".
        :type tensor: Tensor
        :param subsystem_axes: Sequence of integer indices specifying the order in which
            to interpret the axes as subsystems (output edges). If not specified,
            the axes are taken in ascending order.
        :type subsystem_axes: Optional[Sequence[int]], optional
        :return: The new constructed QuVector from the given tensor.
        :rtype: QuVector
        """
        n = Node(tensor)
        if subsystem_axes is not None:
            subsystem_edges = [n[i] for i in subsystem_axes]
        else:
            subsystem_edges = n.get_all_edges()
        return cls(subsystem_edges)

    @property
    def subsystem_edges(self) -> List[Edge]:
        return self.out_edges

    @property
    def space(self) -> List[int]:
        return self.out_space

    def projector(self) -> "QuOperator":
        """
        The projector of the operator.
        The operator, as a linear operator, on the adjoint of the operator.

        Set :math:`A` is the operator in matrix form, then the projector of operator is defined as: :math:`A A^\\dagger`

        :return: The projector of the operator.
        :rtype: QuOperator
        """
        return self @ self.adjoint()

    def reduced_density(self, subsystems_to_trace_out: Collection[int]) -> "QuOperator":
        """
        The reduced density of the operator.

        Set :math:`A` is the matrix of the operator, then the reduced density is defined as:

        .. math::

            \\mathrm{Tr}_{subsystems}(A A^\\dagger)

        Firstly, take the projector of the operator, then trace out the subsystems
        to trace out are supplied as indices, so that dangling edges are connected
        to each other as:
        `out_edges[i] ^ in_edges[i] for i in subsystems_to_trace_out`
        This does not modify the original network. The original ordering of the
        remaining subsystems is maintained.

        :param subsystems_to_trace_out: Indices of subsystems to trace out.
        :type subsystems_to_trace_out: Collection[int]
        :return: The QuOperator of the reduced density of the operator with given subsystems.
        :rtype: QuOperator
        """
        rho = self.projector()
        return rho.partial_trace(subsystems_to_trace_out)


class QuAdjointVector(QuOperator):
    """Represents an adjoint (row) vector via a tensor network."""

    def __init__(
        self,
        subsystem_edges: Sequence[Edge],
        ref_nodes: Optional[Collection[AbstractNode]] = None,
        ignore_edges: Optional[Collection[Edge]] = None,
    ) -> None:
        """
        Constructs a new `QuAdjointVector` from a tensor network.
        This encapsulates an existing tensor network, interpreting it as an adjoint
        vector (row vector).

        :param subsystem_edges: The edges of the network to be used as the input edges.
        :type subsystem_edges: Sequence[Edge]
        :param ref_nodes: Nodes used to refer to parts of the tensor network that are
            not connected to any input or output edges (for example: a scalar factor).
        :type ref_nodes: Optional[Collection[AbstractNode]], optional
        :param ignore_edges: Optional collection of edges to ignore when performing consistency checks.
        :type ignore_edges: Optional[Collection[Edge]], optional
        """
        super().__init__([], subsystem_edges, ref_nodes, ignore_edges)

    @classmethod
    def from_tensor(  # type: ignore
        cls,
        tensor: Tensor,
        subsystem_axes: Optional[Sequence[int]] = None,
    ) -> "QuAdjointVector":
        r"""
        Construct a `QuAdjointVector` directly from a single tensor.
        This first wraps the tensor in a `Node`, then constructs the `QuAdjointVector` from that `Node`.

        :Example:

        .. code-block:: python

            def show_attributes(op):
                print(f"op.is_scalar() \t\t-> {op.is_scalar()}")
                print(f"op.is_vector() \t\t-> {op.is_vector()}")
                print(f"op.is_adjoint_vector() \t-> {op.is_adjoint_vector()}")
                print(f"op.eval() \n{op.eval()}")

        >>> psi_tensor = np.random.rand(2, 2)
        >>> psi_tensor
        array([[0.27260127, 0.91401091],
               [0.06490953, 0.38653646]])
        >>> op = qu.QuAdjointVector.from_tensor(psi_tensor, [0, 1])
        >>> show_attributes(op)
        op.is_scalar()          -> False
        op.is_vector()          -> False
        op.is_adjoint_vector()  -> True
        op.eval()
        [[0.27260127 0.91401091]
         [0.06490953 0.38653646]]

        :param tensor: The tensor for constructing an QuAdjointVector.
        :type tensor: Tensor
        :param subsystem_axes: Sequence of integer indices specifying the order in which
            to interpret the axes as subsystems (input edges). If not specified,
            the axes are taken in ascending order.
        :type subsystem_axes: Optional[Sequence[int]], optional
        :return: The new constructed QuAdjointVector give from the given tensor.
        :rtype: QuAdjointVector
        """
        n = Node(tensor)
        if subsystem_axes is not None:
            subsystem_edges = [n[i] for i in subsystem_axes]
        else:
            subsystem_edges = n.get_all_edges()
        return cls(subsystem_edges)

    @property
    def subsystem_edges(self) -> List[Edge]:
        return self.in_edges

    @property
    def space(self) -> List[int]:
        return self.in_space

    def projector(self) -> "QuOperator":
        """
        The projector of the operator.
        The operator, as a linear operator, on the adjoint of the operator.

        Set :math:`A` is the operator in matrix form, then the projector of operator is defined as: :math:`A^\\dagger A`

        :return: The projector of the operator.
        :rtype: QuOperator
        """
        return self.adjoint() @ self

    def reduced_density(self, subsystems_to_trace_out: Collection[int]) -> "QuOperator":
        """
        The reduced density of the operator.

        Set :math:`A` is the matrix of the operator, then the reduced density is defined as:

        .. math::

            \\mathrm{Tr}_{subsystems}(A^\\dagger A)

        Firstly, take the projector of the operator, then trace out the subsystems
        to trace out are supplied as indices, so that dangling edges are connected
        to each other as:
        `out_edges[i] ^ in_edges[i] for i in subsystems_to_trace_out`
        This does not modify the original network. The original ordering of the
        remaining subsystems is maintained.

        :param subsystems_to_trace_out: Indices of subsystems to trace out.
        :type subsystems_to_trace_out: Collection[int]
        :return: The QuOperator of the reduced density of the operator with given subsystems.
        :rtype: QuOperator
        """
        rho = self.projector()
        return rho.partial_trace(subsystems_to_trace_out)


class QuScalar(QuOperator):
    """Represents a scalar via a tensor network."""

    def __init__(
        self,
        ref_nodes: Collection[AbstractNode],
        ignore_edges: Optional[Collection[Edge]] = None,
    ) -> None:
        """
        Constructs a new `QuScalar` from a tensor network.
        This encapsulates an existing tensor network, interpreting it as a scalar.

        :param ref_nodes: Nodes used to refer to the tensor network (need not be
            exhaustive - one node from each disconnected subnetwork is sufficient).
        :type ref_nodes: Collection[AbstractNode]
        :param ignore_edges: Optional collection of edges to ignore when performing consistency checks.
        :type ignore_edges: Optional[Collection[Edge]], optional
        """
        super().__init__([], [], ref_nodes, ignore_edges)

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> "QuScalar":  # type: ignore
        r"""
        Construct a `QuScalar` directly from a single tensor.
        This first wraps the tensor in a `Node`, then constructs the `QuScalar` from that `Node`.

        :Example:

        .. code-block:: python

            def show_attributes(op):
                print(f"op.is_scalar() \t\t-> {op.is_scalar()}")
                print(f"op.is_vector() \t\t-> {op.is_vector()}")
                print(f"op.is_adjoint_vector() \t-> {op.is_adjoint_vector()}")
                print(f"op.eval() \n{op.eval()}")

        >>> op = qu.QuScalar.from_tensor(1.0)
        >>> show_attributes(op)
        op.is_scalar()          -> True
        op.is_vector()          -> False
        op.is_adjoint_vector()  -> False
        op.eval()
        1.0

        :param tensor: The tensor for constructing a new QuScalar.
        :type tensor: Tensor
        :return: The new constructed QuScalar from the given tensor.
        :rtype: QuScalar
        """
        n = Node(tensor)
        return cls(set([n]))


def ps2xyz(ps: List[int]) -> Dict[str, List[int]]:
    """
    pauli string list to xyz dict

    # ps2xyz([1, 2, 2, 0]) = {"x": [0], "y": [1, 2], "z": []}

    :param ps: _description_
    :type ps: List[int]
    :return: _description_
    :rtype: Dict[str, List[int]]
    """
    xyz: Dict[str, List[int]] = {"x": [], "y": [], "z": []}
    for i, j in enumerate(ps):
        if j == 1:
            xyz["x"].append(i)
        if j == 2:
            xyz["y"].append(i)
        if j == 3:
            xyz["z"].append(i)
    return xyz


def xyz2ps(xyz: Dict[str, List[int]], n: Optional[int] = None) -> List[int]:
    """
    xyz dict to pauli string list

    :param xyz: _description_
    :type xyz: Dict[str, List[int]]
    :param n: _description_, defaults to None
    :type n: Optional[int], optional
    :return: _description_
    :rtype: List[int]
    """
    if n is None:
        n = max(xyz.get("x", []) + xyz.get("y", []) + xyz.get("z", [])) + 1
    ps = [0 for _ in range(n)]
    for i in range(n):
        if i in xyz.get("x", []):
            ps[i] = 1
        elif i in xyz.get("y", []):
            ps[i] = 2
        elif i in xyz.get("z", []):
            ps[i] = 3
    return ps


def generate_local_hamiltonian(
    *hlist: Sequence[Tensor], matrix_form: bool = True
) -> Union[QuOperator, Tensor]:
    """
    Generate a local Hamiltonian operator based on the given sequence of Tensor.
    Note: further jit is recommended.
    For large Hilbert space, sparse Hamiltonian is recommended

    :param hlist: A sequence of Tensor.
    :type hlist: Sequence[Tensor]
    :param matrix_form: Return Hamiltonian operator in form of matrix, defaults to True.
    :type matrix_form: bool, optional
    :return: The Hamiltonian operator in form of QuOperator or matrix.
    :rtype: Union[QuOperator, Tensor]
    """
    hlist = [backend.cast(h, dtype=dtypestr) for h in hlist]  # type: ignore
    hop_list = [QuOperator.from_tensor(h) for h in hlist]
    hop = reduce(or_, hop_list)
    if matrix_form:
        tensor = hop.eval_matrix()
        return tensor
    return hop


def tn2qop(tn_mpo: Any) -> QuOperator:
    """
    Convert MPO in TensorNetwork package to QuOperator.

    :param tn_mpo: MPO in the form of TensorNetwork package
    :type tn_mpo: ``tn.matrixproductstates.mpo.*``
    :return: MPO in the form of QuOperator
    :rtype: QuOperator
    """
    tn_mpo = tn_mpo.tensors
    nwires = len(tn_mpo)
    mpo = []
    for i in range(nwires):
        mpo.append(Node(tn_mpo[i]))

    for i in range(nwires - 1):
        connect(mpo[i][1], mpo[i + 1][0])
    # TODO(@refraction-ray): whether in and out edge is in the correct order require further check
    qop = quantum_constructor(
        [mpo[i][-1] for i in range(nwires)],  # out_edges
        [mpo[i][-2] for i in range(nwires)],  # in_edges
        [],
        [mpo[0][0], mpo[-1][1]],  # ignore_edges
    )
    return qop


def quimb2qop(qb_mpo: Any) -> QuOperator:
    """
    Convert MPO in Quimb package to QuOperator.

    :param tn_mpo: MPO in the form of Quimb package
    :type tn_mpo: ``quimb.tensor.tensor_gen.*``
    :return: MPO in the form of QuOperator
    :rtype: QuOperator
    """
    qb_mpo = qb_mpo.tensors
    nwires = len(qb_mpo)
    assert nwires >= 3, "number of tensors must be larger than 2"
    mpo = []
    edges = []
    for i in range(nwires):
        mpo.append(Node(qb_mpo[i].data))
        for j, ind in enumerate(qb_mpo[i].inds):
            edges.append((i, j, ind))

    out_edges = []
    in_edges = []

    for i, e1 in enumerate(edges):
        if e1[2].startswith("k"):
            out_edges.append(mpo[e1[0]][e1[1]])
        elif e1[2].startswith("b"):
            in_edges.append(mpo[e1[0]][e1[1]])
        else:
            for j, e2 in enumerate(edges[i + 1 :]):
                if e2[2] == e1[2]:
                    connect(mpo[e1[0]][e1[1]], mpo[e2[0]][e2[1]])

    qop = quantum_constructor(
        out_edges,  # out_edges
        in_edges,  # in_edges
        [],
        [],  # ignore_edges
    )
    return qop


try:

    def _id(x: Any) -> Any:
        return x

    if is_m1mac():
        compiled_jit = _id
    else:
        compiled_jit = partial(get_backend("tensorflow").jit, jit_compile=True)

    def heisenberg_hamiltonian(
        g: Graph,
        hzz: float = 1.0,
        hxx: float = 1.0,
        hyy: float = 1.0,
        hz: float = 0.0,
        hx: float = 0.0,
        hy: float = 0.0,
        sparse: bool = True,
        numpy: bool = False,
    ) -> Tensor:
        """
        Generate Heisenberg Hamiltonian with possible external fields.
        Currently requires tensorflow installed

        :Example:

        >>> g = tc.templates.graphs.Line1D(6)
        >>> h = qu.heisenberg_hamiltonian(g, sparse=False)
        >>> tc.backend.eigh(h)[0][:10]
        array([-11.2111025,  -8.4721365,  -8.472136 ,  -8.472136 ,  -6.       ,
                -5.123106 ,  -5.123106 ,  -5.1231055,  -5.1231055,  -5.1231055],
            dtype=float32)

        :param g: input circuit graph
        :type g: Graph
        :param hzz: zz coupling, default is 1.0
        :type hzz: float
        :param hxx: xx coupling, default is 1.0
        :type hxx: float
        :param hyy: yy coupling, default is 1.0
        :type hyy: float
        :param hz: External field on z direction, default is 0.0
        :type hz: float
        :param hx: External field on y direction, default is 0.0
        :type hx: float
        :param hy: External field on x direction, default is 0.0
        :type hy: float
        :param sparse: Whether to return sparse Hamiltonian operator, default is True.
        :type sparse: bool, defalts True
        :param numpy: whether return the matrix in numpy or tensorflow form
        :type numpy: bool, defaults False,

        :return: Hamiltonian measurements
        :rtype: Tensor
        """
        n = len(g.nodes)
        ls = []
        weight = []
        for e in g.edges:
            if hzz != 0:
                r = [0 for _ in range(n)]
                r[e[0]] = 3
                r[e[1]] = 3
                ls.append(r)
                weight.append(hzz)
            if hxx != 0:
                r = [0 for _ in range(n)]
                r[e[0]] = 1
                r[e[1]] = 1
                ls.append(r)
                weight.append(hxx)
            if hyy != 0:
                r = [0 for _ in range(n)]
                r[e[0]] = 2
                r[e[1]] = 2
                ls.append(r)
                weight.append(hyy)
        for node in g.nodes:
            if hz != 0:
                r = [0 for _ in range(n)]
                r[node] = 3
                ls.append(r)
                weight.append(hz)
            if hx != 0:
                r = [0 for _ in range(n)]
                r[node] = 1
                ls.append(r)
                weight.append(hx)
            if hy != 0:
                r = [0 for _ in range(n)]
                r[node] = 2
                ls.append(r)
                weight.append(hy)
        ls = tf.constant(ls)
        weight = tf.constant(weight)
        ls = get_backend("tensorflow").cast(ls, dtypestr)
        weight = get_backend("tensorflow").cast(weight, dtypestr)
        if sparse:
            r = PauliStringSum2COO_numpy(ls, weight)
            if numpy:
                return r
            return backend.coo_sparse_matrix_from_numpy(r)
        return PauliStringSum2Dense(ls, weight, numpy=numpy)

    def PauliStringSum2Dense(
        ls: Sequence[Sequence[int]],
        weight: Optional[Sequence[float]] = None,
        numpy: bool = False,
    ) -> Tensor:
        """
        Generate dense matrix from Pauli string sum.
        Currently requires tensorflow installed.


        :param ls: 2D Tensor, each row is for a Pauli string,
            e.g. [1, 0, 0, 3, 2] is for :math:`X_0Z_3Y_4`
        :type ls: Sequence[Sequence[int]]
        :param weight: 1D Tensor, each element corresponds the weight for each Pauli string
            defaults to None (all Pauli strings weight 1.0)
        :type weight: Optional[Sequence[float]], optional
        :param numpy: default False. If True, return numpy coo
            else return backend compatible sparse tensor
        :type numpy: bool
        :return: the tensorflow dense matrix
        :rtype: Tensor
        """
        sparsem = PauliStringSum2COO_numpy(ls, weight)
        if numpy:
            return sparsem.todense()
        sparsem = backend.coo_sparse_matrix_from_numpy(sparsem)
        densem = backend.to_dense(sparsem)
        return densem

    # already implemented as backend method
    #
    # def _tf2numpy_sparse(a: Tensor) -> Tensor:
    #     return get_backend("numpy").coo_sparse_matrix(
    #         indices=a.indices,
    #         values=a.values,
    #         shape=a.get_shape(),
    #     )

    # def _numpy2tf_sparse(a: Tensor) -> Tensor:
    #     return get_backend("tensorflow").coo_sparse_matrix(
    #         indices=np.array([a.row, a.col]).T,
    #         values=a.data,
    #         shape=a.shape,
    #     )

    def PauliStringSum2COO(
        ls: Sequence[Sequence[int]],
        weight: Optional[Sequence[float]] = None,
        numpy: bool = False,
    ) -> Tensor:
        """
        Generate sparse tensor from Pauli string sum.
        Currently requires tensorflow installed

        :param ls: 2D Tensor, each row is for a Pauli string,
            e.g. [1, 0, 0, 3, 2] is for :math:`X_0Z_3Y_4`
        :type ls: Sequence[Sequence[int]]
        :param weight: 1D Tensor, each element corresponds the weight for each Pauli string
            defaults to None (all Pauli strings weight 1.0)
        :type weight: Optional[Sequence[float]], optional
        :param numpy: default False. If True, return numpy coo
            else return backend compatible sparse tensor
        :type numpy: bool
        :return: the scipy coo sparse matrix
        :rtype: Tensor
        """
        # numpy version is 3* faster!

        nterms = len(ls)
        # n = len(ls[0])
        # s = 0b1 << n
        if weight is None:
            weight = [1.0 for _ in range(nterms)]
        if not (isinstance(weight, tf.Tensor) or isinstance(weight, tf.Variable)):
            weight = tf.constant(weight, dtype=getattr(tf, dtypestr))
        # rsparse = get_backend("numpy").coo_sparse_matrix(
        #     indices=np.array([[0, 0]], dtype=np.int64),
        #     values=np.array([0.0], dtype=getattr(np, dtypestr)),
        #     shape=(s, s),
        # )
        rsparses = [
            get_backend("tensorflow").numpy(PauliString2COO(ls[i], weight[i]))  # type: ignore
            for i in range(nterms)
        ]
        rsparse = _dc_sum(rsparses)
        # auto transformed into csr format!!

        # for i in range(nterms):
        #     rsparse += get_backend("tensorflow").numpy(PauliString2COO(ls[i], weight[i]))  # type: ignore
        rsparse = rsparse.tocoo()
        if numpy:
            return rsparse
        return backend.coo_sparse_matrix_from_numpy(rsparse)

    def _dc_sum(l: List[Any]) -> Any:
        """
        For the sparse sum, the speed is determined by the non zero terms,
        so the DC way to do the sum can indeed bring some speed advantage (several times)

        :param l: _description_
        :type l: List[Any]
        :return: _description_
        :rtype: Any
        """
        n = len(l)
        if n > 2:
            return _dc_sum(l[: n // 2]) + _dc_sum(l[n // 2 :])
        else:
            return sum(l)

    PauliStringSum2COO_numpy = partial(PauliStringSum2COO, numpy=True)

    def PauliStringSum2COO_tf(
        ls: Sequence[Sequence[int]], weight: Optional[Sequence[float]] = None
    ) -> Tensor:
        """
        Generate tensorflow sparse matrix from Pauli string sum

        :param ls: 2D Tensor, each row is for a Pauli string,
            e.g. [1, 0, 0, 3, 2] is for :math:`X_0Z_3Y_4`
        :type ls: Sequence[Sequence[int]]
        :param weight: 1D Tensor, each element corresponds the weight for each Pauli string
            defaults to None (all Pauli strings weight 1.0)
        :type weight: Optional[Sequence[float]], optional
        :return: the tensorflow coo sparse matrix
        :rtype: Tensor
        """
        nterms = len(ls)
        n = len(ls[0])
        s = 0b1 << n
        if weight is None:
            weight = [1.0 for _ in range(nterms)]
        if not (isinstance(weight, tf.Tensor) or isinstance(weight, tf.Variable)):
            weight = tf.constant(weight, dtype=getattr(tf, dtypestr))
        rsparse = tf.SparseTensor(
            indices=tf.constant([[0, 0]], dtype=tf.int64),
            values=tf.constant([0.0], dtype=weight.dtype),  # type: ignore
            dense_shape=(s, s),
        )
        for i in range(nterms):
            rsparse = tf.sparse.add(rsparse, PauliString2COO(ls[i], weight[i]))  # type: ignore
            # very slow sparse.add?
        return rsparse

    @compiled_jit
    def PauliString2COO(l: Sequence[int], weight: Optional[float] = None) -> Tensor:
        """
        Generate tensorflow sparse matrix from Pauli string sum

        :param l: 1D Tensor representing for a Pauli string,
            e.g. [1, 0, 0, 3, 2] is for :math:`X_0Z_3Y_4`
        :type l: Sequence[int]
        :param weight: the weight for the Pauli string
            defaults to None (all Pauli strings weight 1.0)
        :type weight: Optional[float], optional
        :return: the tensorflow sparse matrix
        :rtype: Tensor
        """
        n = len(l)
        one = tf.constant(0b1, dtype=tf.int64)
        idx_x = tf.constant(0b0, dtype=tf.int64)
        idx_y = tf.constant(0b0, dtype=tf.int64)
        idx_z = tf.constant(0b0, dtype=tf.int64)
        i = tf.constant(0, dtype=tf.int64)
        for j in l:
            # i, j from enumerate is python, non jittable when cond using tensor
            if j == 1:  # xi
                idx_x += tf.bitwise.left_shift(one, n - i - 1)
            elif j == 2:  # yi
                idx_y += tf.bitwise.left_shift(one, n - i - 1)
            elif j == 3:  # zi
                idx_z += tf.bitwise.left_shift(one, n - i - 1)
            i += 1

        if weight is None:
            weight = tf.constant(1.0, dtype=tf.complex64)
        return ps2coo_core(idx_x, idx_y, idx_z, weight, n)

    @compiled_jit
    def ps2coo_core(
        idx_x: Tensor, idx_y: Tensor, idx_z: Tensor, weight: Tensor, nqubits: int
    ) -> Tuple[Tensor, Tensor]:
        dtype = weight.dtype
        s = 0b1 << nqubits
        idx1 = tf.cast(tf.range(s), dtype=tf.int64)
        idx2 = (idx1 ^ idx_x) ^ (idx_y)
        indices = tf.transpose(tf.stack([idx1, idx2]))
        tmp = idx1 & (idx_y | idx_z)
        e = idx1 * 0
        ny = 0
        for i in range(nqubits):
            # if tmp[i] is power of 2 (non zero), then e[i] = 1
            e ^= tf.bitwise.right_shift(tmp, i) & 0b1
            # how many 1 contained in idx_y
            ny += tf.bitwise.right_shift(idx_y, i) & 0b1
        ny = tf.math.mod(ny, 4)
        values = (
            tf.cast((1 - 2 * e), dtype)
            * tf.math.pow(tf.constant(-1.0j, dtype=dtype), tf.cast(ny, dtype))
            * weight
        )
        return tf.SparseTensor(indices=indices, values=values, dense_shape=(s, s))  # type: ignore

except (NameError, ImportError):
    logger.info(
        "tensorflow is not installed, and sparse Hamiltonian generation utilities are disabled"
    )

# some quantum quatities below


def op2tensor(
    fn: Callable[..., Any], op_argnums: Union[int, Sequence[int]] = 0
) -> Callable[..., Any]:
    from .interfaces.tensortrans import args_to_tensor

    return args_to_tensor(fn, op_argnums, qop_to_tensor=True, cast_dtype=False)


# def op2tensor(
#     fn: Callable[..., Any], op_argnums: Union[int, Sequence[int]] = 0
# ) -> Callable[..., Any]:
#     if isinstance(op_argnums, int):
#         op_argnums = [op_argnums]

#     @wraps(fn)
#     def wrapper(*args: Any, **kwargs: Any) -> Any:
#         nargs = list(args)
#         for i in op_argnums:  # type: ignore
#             if isinstance(args[i], QuOperator):
#                 nargs[i] = args[i].copy().eval_matrix()
#         out = fn(*nargs, **kwargs)
#         return out

#     return wrapper


@op2tensor
def entropy(rho: Union[Tensor, QuOperator], eps: Optional[float] = None) -> Tensor:
    """
    Compute the entropy from the given density matrix ``rho``.

    :Example:

    .. code-block:: python

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

    >>> param = tc.backend.ones([6, 6])
    >>> tc.backend.trace(param)
    >>> entanglement1(param, 6, 3)
    1.3132654
    >>> entanglement2(param, 6, 3)
    1.3132653

    :param rho: The density matrix in form of Tensor or QuOperator.
    :type rho: Union[Tensor, QuOperator]
    :param eps: Epsilon, default is 1e-12.
    :type eps: float
    :return: Entropy on the given density matrix.
    :rtype: Tensor
    """
    eps_env = os.environ.get("TC_QUANTUM_ENTROPY_EPS")
    if eps is None and eps_env is None:
        eps = 1e-12
    elif eps is None and eps_env is not None:
        eps = 10 ** (-int(eps_env))
    rho += eps * backend.cast(backend.eye(rho.shape[-1]), rho.dtype)  # type: ignore
    lbd = backend.real(backend.eigh(rho)[0])
    lbd = backend.relu(lbd)
    lbd /= backend.sum(lbd)
    # we need the matrix anyway for AD.
    entropy = -backend.sum(lbd * backend.log(lbd + eps))
    return backend.real(entropy)


def trace_product(*o: Union[Tensor, QuOperator]) -> Tensor:
    """
    Compute the trace of several inputs ``o`` as tensor or ``QuOperator``.

    .. math ::

        \\operatorname{Tr}(\\prod_i O_i)

    :Example:

    >>> o = np.ones([2, 2])
    >>> h = np.eye(2)
    >>> qu.trace_product(o, h)
    2.0
    >>> oq = qu.QuOperator.from_tensor(o)
    >>> hq = qu.QuOperator.from_tensor(h)
    >>> qu.trace_product(oq, hq)
    array([[2.]])
    >>> qu.trace_product(oq, h)
    array([[2.]])
    >>> qu.trace_product(o, hq)
    array([[2.]])

    :return: The trace of several inputs.
    :rtype: Tensor
    """
    prod = reduce(matmul, o)
    if isinstance(prod, QuOperator):
        return prod.trace().eval_matrix()
    return backend.trace(prod)


@op2tensor
def entanglement_entropy(state: Tensor, cut: Union[int, List[int]]) -> Tensor:
    rho = reduced_density_matrix(state, cut)
    return entropy(rho)


def reduced_wavefunction(
    state: Tensor, cut: List[int], measure: Optional[List[int]] = None
) -> Tensor:
    """
    Compute the reduced wavefunction from the quantum state ``state``.
    The fixed measure result is guaranteed by users,
    otherwise final normalization may required in the return

    :param state: _description_
    :type state: Tensor
    :param cut: the list of position for qubit to be reduced
    :type cut: List[int]
    :param measure: the fixed results of given qubits in the same shape list as ``cut``
    :type measure: List[int]
    :return: _description_
    :rtype: Tensor
    """
    if measure is None:
        measure = [0 for _ in cut]
    s = backend.reshape2(state)
    n = len(backend.shape_tuple(s))
    s_node = Gate(s)
    end_nodes = []
    for c, m in zip(cut, measure):
        rt = backend.cast(backend.convert_to_tensor(1 - m), dtypestr) * backend.cast(
            backend.convert_to_tensor(np.array([1.0, 0.0])), dtypestr
        ) + backend.cast(backend.convert_to_tensor(m), dtypestr) * backend.cast(
            backend.convert_to_tensor(np.array([0.0, 1.0])), dtypestr
        )
        end_node = Gate(rt)
        end_nodes.append(end_node)
        s_node[c] ^ end_node[0]
    new_node = contractor(
        [s_node] + end_nodes,
        output_edge_order=[s_node[i] for i in range(n) if i not in cut],
    )
    return backend.reshape(new_node.tensor, [-1])


def reduced_density_matrix(
    state: Union[Tensor, QuOperator],
    cut: Union[int, List[int]],
    p: Optional[Tensor] = None,
) -> Union[Tensor, QuOperator]:
    """
    Compute the reduced density matrix from the quantum state ``state``.

    :param state: The quantum state in form of Tensor or QuOperator.
    :type state: Union[Tensor, QuOperator]
    :param cut: the index list that is traced out, if cut is a int,
        it indicates [0, cut] as the traced out region
    :type cut: Union[int, List[int]]
    :param p: probability decoration, default is None.
    :type p: Optional[Tensor]
    :return: The reduced density matrix.
    :rtype: Union[Tensor, QuOperator]
    """
    if isinstance(cut, list) or isinstance(cut, tuple) or isinstance(cut, set):
        traceout = list(cut)
    else:
        traceout = [i for i in range(cut)]
    if isinstance(state, QuOperator):
        if p is not None:
            raise NotImplementedError(
                "p arguments is not supported when state is a `QuOperator`"
            )
        return state.partial_trace(traceout)
    if len(state.shape) == 2 and state.shape[0] == state.shape[1]:
        # density operator
        freedomexp = backend.sizen(state)
        # traceout = sorted(traceout)[::-1]
        freedom = int(np.log2(freedomexp) / 2)
        # traceout2 = [i + freedom for i in traceout]
        left = traceout + [i for i in range(freedom) if i not in traceout]
        right = [i + freedom for i in left]
        rho = backend.reshape(state, [2 for _ in range(2 * freedom)])
        rho = backend.transpose(rho, perm=left + right)
        rho = backend.reshape(
            rho,
            [
                2 ** len(traceout),
                2 ** (freedom - len(traceout)),
                2 ** len(traceout),
                2 ** (freedom - len(traceout)),
            ],
        )
        if p is None:
            # for i, (tr, tr2) in enumerate(zip(traceout, traceout2)):
            #     rho = backend.trace(rho, axis1=tr, axis2=tr2 - i)
            # correct but tf trace fail to support so much dimension with tf.einsum

            rho = backend.trace(rho, axis1=0, axis2=2)
        else:
            p = backend.reshape(p, [-1])
            rho = backend.einsum("a,aiaj->ij", p, rho)
        rho = backend.reshape(
            rho, [2 ** (freedom - len(traceout)), 2 ** (freedom - len(traceout))]
        )
        rho /= backend.trace(rho)

    else:
        w = state / backend.norm(state)
        freedomexp = backend.sizen(state)
        freedom = int(np.log(freedomexp) / np.log(2))
        perm = [i for i in range(freedom) if i not in traceout]
        perm = perm + traceout
        w = backend.reshape(w, [2 for _ in range(freedom)])
        w = backend.transpose(w, perm=perm)
        w = backend.reshape(w, [-1, 2 ** len(traceout)])
        if p is None:
            rho = w @ backend.adjoint(w)
        else:
            rho = w @ backend.diagflat(p) @ backend.adjoint(w)
            rho /= backend.trace(rho)
    return rho


def free_energy(
    rho: Union[Tensor, QuOperator],
    h: Union[Tensor, QuOperator],
    beta: float = 1,
    eps: float = 1e-12,
) -> Tensor:
    """
    Compute the free energy of the given density matrix.

    :Example:

    >>> rho = np.array([[1.0, 0], [0, 0]])
    >>> h = np.array([[-1.0, 0], [0, 1]])
    >>> qu.free_energy(rho, h, 0.5)
    -0.9999999999979998
    >>> hq = qu.QuOperator.from_tensor(h)
    >>> qu.free_energy(rho, hq, 0.5)
    array([[-1.]])

    :param rho: The density matrix in form of Tensor or QuOperator.
    :type rho: Union[Tensor, QuOperator]
    :param h: Hamiltonian operator in form of Tensor or QuOperator.
    :type h: Union[Tensor, QuOperator]
    :param beta: Constant for the optimization, default is 1.
    :type beta: float, optional
    :param eps: Epsilon, default is 1e-12.
    :type eps: float, optional

    :return: The free energy of the given density matrix with the Hamiltonian operator.
    :rtype: Tensor
    """
    energy = backend.real(trace_product(rho, h))
    s = entropy(rho, eps)
    return backend.real(energy - s / beta)


def renyi_entropy(rho: Union[Tensor, QuOperator], k: int = 2) -> Tensor:
    """
    Compute the Rényi entropy of order :math:`k` by given density matrix.

    :param rho: The density matrix in form of Tensor or QuOperator.
    :type rho: Union[Tensor, QuOperator]
    :param k: The order of Rényi entropy, default is 2.
    :type k: int, optional
    :return: The :math:`k` th order of Rényi entropy.
    :rtype: Tensor
    """
    s = 1 / (1 - k) * backend.real(backend.log(trace_product(*[rho for _ in range(k)])))
    return s


def renyi_free_energy(
    rho: Union[Tensor, QuOperator],
    h: Union[Tensor, QuOperator],
    beta: float = 1,
    k: int = 2,
) -> Tensor:
    """
    Compute the Rényi free energy of the corresponding density matrix and Hamiltonian.

    :Example:

    >>> rho = np.array([[1.0, 0], [0, 0]])
    >>> h = np.array([[-1.0, 0], [0, 1]])
    >>> qu.renyi_free_energy(rho, h, 0.5)
    -1.0
    >>> qu.free_energy(rho, h, 0.5)
    -0.9999999999979998

    :param rho: The density matrix in form of Tensor or QuOperator.
    :type rho: Union[Tensor, QuOperator]
    :param h: Hamiltonian operator in form of Tensor or QuOperator.
    :type h: Union[Tensor, QuOperator]
    :param beta: Constant for the optimization, default is 1.
    :type beta: float, optional
    :param k: The order of Rényi entropy, default is 2.
    :type k: int, optional
    :return: The :math:`k` th order of Rényi entropy.
    :rtype: Tensor
    """
    energy = backend.real(trace_product(rho, h))
    s = renyi_entropy(rho, k)
    return backend.real(energy - s / beta)


def taylorlnm(x: Tensor, k: int) -> Tensor:
    """
    Taylor expansion of :math:`ln(x+1)`.

    :param x: The density matrix in form of Tensor.
    :type x: Tensor
    :param k: The :math:`k` th order, default is 2.
    :type k: int, optional
    :return: The :math:`k` th order of Taylor expansion of :math:`ln(x+1)`.
    :rtype: Tensor
    """
    dtype = x.dtype
    s = x.shape[-1]
    y = 1 / k * (-1) ** (k + 1) * backend.eye(s, dtype=dtype)
    for i in reversed(range(k)):
        y = y @ x
        if i > 0:
            y += 1 / (i) * (-1) ** (i + 1) * backend.eye(s, dtype=dtype)
    return y


def truncated_free_energy(
    rho: Tensor, h: Tensor, beta: float = 1, k: int = 2
) -> Tensor:
    """
    Compute the truncated free energy from the given density matrix ``rho``.

    :param rho: The density matrix in form of Tensor.
    :type rho: Tensor
    :param h: Hamiltonian operator in form of Tensor.
    :type h: Tensor
    :param beta: Constant for the optimization, default is 1.
    :type beta: float, optional
    :param k: The :math:`k` th order, defaults to 2
    :type k: int, optional
    :return: The :math:`k` th order of the truncated free energy.
    :rtype: Tensor
    """
    dtype = rho.dtype
    s = rho.shape[-1]
    tyexpand = rho @ taylorlnm(rho - backend.eye(s, dtype=dtype), k - 1)
    renyi = -backend.real(backend.trace(tyexpand))
    energy = backend.real(trace_product(rho, h))
    return energy - renyi / beta


@op2tensor
def partial_transpose(rho: Tensor, transposed_sites: List[int]) -> Tensor:
    """
    _summary_

    :param rho: density matrix
    :type rho: Tensor
    :param transposed_sites: sites int list to be transposed
    :type transposed_sites: List[int]
    :return: _description_
    :rtype: Tensor
    """
    rho = backend.reshape2(rho)
    rho_node = Gate(rho)
    n = len(rho.shape) // 2
    left_edges = []
    right_edges = []
    for i in range(n):
        if i not in transposed_sites:
            left_edges.append(rho_node[i])
            right_edges.append(rho_node[i + n])
        else:
            left_edges.append(rho_node[i + n])
            right_edges.append(rho_node[i])
    rhot_op = QuOperator(out_edges=left_edges, in_edges=right_edges)
    rhot = rhot_op.eval_matrix()
    return rhot


@op2tensor
def entanglement_negativity(rho: Tensor, transposed_sites: List[int]) -> Tensor:
    """
    _summary_

    :param rho: _description_
    :type rho: Tensor
    :param transposed_sites: _description_
    :type transposed_sites: List[int]
    :return: _description_
    :rtype: Tensor
    """
    rhot = partial_transpose(rho, transposed_sites)
    es = backend.eigvalsh(rhot)
    rhot_m = backend.sum(backend.abs(es))
    return (backend.log(rhot_m) - 1.0) / 2.0


@op2tensor
def log_negativity(rho: Tensor, transposed_sites: List[int], base: str = "e") -> Tensor:
    """
    _summary_

    :param rho: _description_
    :type rho: Tensor
    :param transposed_sites: _description_
    :type transposed_sites: List[int]
    :param base: whether use 2 based log or e based log, defaults to "e"
    :type base: str, optional
    :return: _description_
    :rtype: Tensor
    """
    rhot = partial_transpose(rho, transposed_sites)
    es = backend.eigvalsh(rhot)
    rhot_m = backend.sum(backend.abs(es))
    een = backend.log(rhot_m)
    if base in ["2", 2]:
        return een / backend.cast(backend.log(2.0), rdtypestr)
    return een


@partial(op2tensor, op_argnums=(0, 1))
def trace_distance(rho: Tensor, rho0: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Compute the trace distance between two density matrix ``rho`` and ``rho2``.

    :param rho: The density matrix in form of Tensor.
    :type rho: Tensor
    :param rho0: The density matrix in form of Tensor.
    :type rho0: Tensor
    :param eps: Epsilon, defaults to 1e-12
    :type eps: float, optional
    :return: The trace distance between two density matrix ``rho`` and ``rho2``.
    :rtype: Tensor
    """
    d2 = rho - rho0
    d2 = backend.adjoint(d2) @ d2
    lbds = backend.real(backend.eigh(d2)[0])
    lbds = backend.relu(lbds)
    return 0.5 * backend.sum(backend.sqrt(lbds + eps))


@partial(op2tensor, op_argnums=(0, 1))
def fidelity(rho: Tensor, rho0: Tensor) -> Tensor:
    """
    Return fidelity scalar between two states rho and rho0.

    .. math::

        \\operatorname{Tr}(\\sqrt{\\sqrt{rho} rho_0 \\sqrt{rho}})

    :param rho: The density matrix in form of Tensor.
    :type rho: Tensor
    :param rho0: The density matrix in form of Tensor.
    :type rho0: Tensor
    :return: The sqrtm of a Hermitian matrix ``a``.
    :rtype: Tensor
    """
    rhosqrt = backend.sqrtmh(rho)
    return backend.real(backend.trace(backend.sqrtmh(rhosqrt @ rho0 @ rhosqrt)) ** 2)


@op2tensor
def gibbs_state(h: Tensor, beta: float = 1) -> Tensor:
    """
    Compute the Gibbs state of the given Hamiltonian operator ``h``.

    :param h: Hamiltonian operator in form of Tensor.
    :type h: Tensor
    :param beta: Constant for the optimization, default is 1.
    :type beta: float, optional
    :return: The Gibbs state of ``h`` with the given ``beta``.
    :rtype: Tensor
    """
    rho = backend.expm(-beta * h)
    rho /= backend.trace(rho)
    return rho


@op2tensor
def double_state(h: Tensor, beta: float = 1) -> Tensor:
    """
    Compute the double state of the given Hamiltonian operator ``h``.

    :param h: Hamiltonian operator in form of Tensor.
    :type h: Tensor
    :param beta: Constant for the optimization, default is 1.
    :type beta: float, optional
    :return: The double state of ``h`` with the given ``beta``.
    :rtype: Tensor
    """
    rho = backend.expm(-beta / 2 * h)
    state = backend.reshape(rho, [-1])
    norm = backend.norm(state)
    return state / norm


@op2tensor
def mutual_information(s: Tensor, cut: Union[int, List[int]]) -> Tensor:
    """
    Mutual information between AB subsystem described by ``cut``.

    :param s: The density matrix in form of Tensor.
    :type s: Tensor
    :param cut: The AB subsystem.
    :type cut: Union[int, List[int]]
    :return: The mutual information between AB subsystem described by ``cut``.
    :rtype: Tensor
    """
    if isinstance(cut, list) or isinstance(cut, tuple) or isinstance(cut, set):
        traceout = list(cut)
    else:
        traceout = [i for i in range(cut)]

    if len(s.shape) == 2 and s.shape[0] == s.shape[1]:
        # mixed state
        n = int(np.log2(backend.sizen(s)) / 2)
        hab = entropy(s)

        # subsystem a
        rhoa = reduced_density_matrix(s, traceout)
        ha = entropy(rhoa)

        # need subsystem b as well
        other = tuple(i for i in range(n) if i not in traceout)
        rhob = reduced_density_matrix(s, other)  # type: ignore
        hb = entropy(rhob)

    # pure system
    else:
        hab = 0.0
        rhoa = reduced_density_matrix(s, traceout)
        ha = hb = entropy(rhoa)

    return ha + hb - hab


# measurement results and transformations and correlations below


def count_s2d(srepr: Tuple[Tensor, Tensor], n: int) -> Tensor:
    """
    measurement shots results, sparse tuple representation to dense representation
    count_vector to count_tuple

    :param srepr: [description]
    :type srepr: Tuple[Tensor, Tensor]
    :param n: number of qubits
    :type n: int
    :return: [description]
    :rtype: Tensor
    """
    return backend.scatter(
        backend.cast(backend.zeros([2**n]), srepr[1].dtype),
        backend.reshape(srepr[0], [-1, 1]),
        srepr[1],
    )


counts_v2t = count_s2d


def count_d2s(drepr: Tensor, eps: float = 1e-7) -> Tuple[Tensor, Tensor]:
    """
    measurement shots results, dense representation to sparse tuple representation
    non-jittable due to the non fixed return shape
    count_tuple to count_vector

    :Example:

    >>> tc.quantum.counts_d2s(np.array([0.1, 0, -0.3, 0.2]))
    (array([0, 2, 3]), array([ 0.1, -0.3,  0.2]))

    :param drepr: [description]
    :type drepr: Tensor
    :param eps: cutoff to determine nonzero elements, defaults to 1e-7
    :type eps: float, optional
    :return: [description]
    :rtype: Tuple[Tensor, Tensor]
    """
    # unjittable since the return shape is not fixed
    # tf.boolean_mask is better but no jax counterpart
    xl = []
    cl = []
    for i, j in enumerate(drepr):
        if backend.abs(j) > eps:
            xl.append(i)
            cl.append(j)
    xl = backend.convert_to_tensor(xl)
    cl = backend.stack(cl)
    return xl, cl


count_t2v = count_d2s


def sample_int2bin(sample: Tensor, n: int) -> Tensor:
    """
    int sample to bin sample

    :param sample: in shape [trials] of int elements in the range [0, 2**n)
    :type sample: Tensor
    :param n: number of qubits
    :type n: int
    :return: in shape [trials, n] of element (0, 1)
    :rtype: Tensor
    """
    confg = backend.mod(
        backend.right_shift(sample[..., None], backend.reverse(backend.arange(n))),
        2,
    )
    return confg


def sample_bin2int(sample: Tensor, n: int) -> Tensor:
    """
    bin sample to int sample

    :param sample: in shape [trials, n] of elements (0, 1)
    :type sample: Tensor
    :param n: number of qubits
    :type n: int
    :return: in shape [trials]
    :rtype: Tensor
    """
    power = backend.convert_to_tensor([2**j for j in reversed(range(n))])
    return backend.sum(sample * power, axis=-1)


def sample2count(
    sample: Tensor, n: int, jittable: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    sample_int to count_tuple

    :param sample: _description_
    :type sample: Tensor
    :param n: _description_
    :type n: int
    :param jittable: _description_, defaults to True
    :type jittable: bool, optional
    :return: _description_
    :rtype: Tuple[Tensor, Tensor]
    """
    d = 2**n
    if not jittable:
        results = backend.unique_with_counts(sample)  # non-jittable
    else:  # jax specified
        results = backend.unique_with_counts(sample, size=d, fill_value=-1)
    return results


def count_vector2dict(count: Tensor, n: int, key: str = "bin") -> Dict[Any, int]:
    """
    convert_vector to count_dict_bin or count_dict_int

    :param count: tensor in shape [2**n]
    :type count: Tensor
    :param n: number of qubits
    :type n: int
    :param key: can be "int" or "bin", defaults to "bin"
    :type key: str, optional
    :return: _description_
    :rtype: _type_
    """
    from .interfaces import which_backend

    b = which_backend(count)
    d = {i: b.numpy(count[i]).item() for i in range(2**n)}
    if key == "int":
        return d
    else:
        dn = {}
        for k, v in d.items():
            kn = str(bin(k))[2:].zfill(n)
            dn[kn] = v
        return dn


def count_tuple2dict(
    count: Tuple[Tensor, Tensor], n: int, key: str = "bin"
) -> Dict[Any, int]:
    """
    count_tuple to count_dict_bin or count_dict_int

    :param count: count_tuple format
    :type count: Tuple[Tensor, Tensor]
    :param n: number of qubits
    :type n: int
    :param key: can be "int" or "bin", defaults to "bin"
    :type key: str, optional
    :return: count_dict
    :rtype: _type_
    """
    d = {
        backend.numpy(i).item(): backend.numpy(j).item()
        for i, j in zip(count[0], count[1])
        if i >= 0
    }
    if key == "int":
        return d
    else:
        dn = {}
        for k, v in d.items():
            kn = str(bin(k))[2:].zfill(n)
            dn[kn] = v
        return dn


@partial(arg_alias, alias_dict={"counts": ["shots"], "format": ["format_"]})
def measurement_counts(
    state: Tensor,
    counts: Optional[int] = 8192,
    format: str = "count_vector",
    is_prob: bool = False,
    random_generator: Optional[Any] = None,
    status: Optional[Tensor] = None,
    jittable: bool = False,
) -> Any:
    """
    Simulate the measuring of each qubit of ``p`` in the computational basis,
    thus producing output like that of ``qiskit``.

    Six formats of measurement counts results:

    "sample_int": # np.array([0, 0])

    "sample_bin": # [np.array([1, 0]), np.array([1, 0])]

    "count_vector": # np.array([2, 0, 0, 0])

    "count_tuple": # (np.array([0]), np.array([2]))

    "count_dict_bin": # {"00": 2, "01": 0, "10": 0, "11": 0}

    "count_dict_int": # {0: 2, 1: 0, 2: 0, 3: 0}

    :Example:

    >>> n = 4
    >>> w = tc.backend.ones([2**n])
    >>> tc.quantum.measurement_results(w, counts=3, format="sample_bin", jittable=True)
    array([[0, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 1]])
    >>> tc.quantum.measurement_results(w, counts=3, format="sample_int", jittable=True)
    array([ 7, 15, 11])
    >>> tc.quantum.measurement_results(w, counts=3, format="count_vector", jittable=True)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1])
    >>> tc.quantum.measurement_results(w, counts=3, format="count_tuple")
    (array([1, 2, 8]), array([1, 1, 1]))
    >>> tc.quantum.measurement_results(w, counts=3, format="count_dict_bin")
    {'0001': 1, '0011': 1, '1101': 1}
    >>> tc.quantum.measurement_results(w, counts=3, format="count_dict_int")
    {3: 1, 6: 2}

    :param state: The quantum state, assumed to be normalized, as either a ket or density operator.
    :type state: Tensor
    :param counts: The number of counts to perform.
    :type counts: int
    :param format: defaults to be "direct", see supported format above
    :type format: str
    :param is_prob: if True, the `state` is directly regarded as a probability list,
        defaults to be False
    :type is_prob: bool
    :param random_generator: random_generator, defaults to None
    :type random_generator: Optional[Any]
    :param status: external randomness given by tensor uniformly from [0, 1],
        if set, can overwrite random_generator
    :type status: Optional[Tensor]
    :param jittable: if True, jax backend try using a jittable count, defaults to False
    :type jittable: bool
    :return: The counts for each bit string measured.
    :rtype: Tuple[]
    """
    if is_prob:
        pi = state / backend.sum(state)
    else:
        if len(state.shape) == 2:
            state /= backend.trace(state)
            pi = backend.abs(backend.diagonal(state))
        else:
            state /= backend.norm(state)
            pi = backend.real(backend.conj(state) * state)
        pi = backend.reshape(pi, [-1])
    d = int(backend.shape_tuple(pi)[0])
    n = int(np.log(d) / np.log(2) + 1e-8)
    if (counts is None) or counts <= 0:
        if format == "count_vector":
            return pi
        elif format == "count_tuple":
            return count_d2s(pi)
        elif format == "count_dict_bin":
            return count_vector2dict(pi, n, key="bin")
        elif format == "count_dict_int":
            return count_vector2dict(pi, n, key="int")
        else:
            raise ValueError(
                "unsupported format %s for analytical measurement" % format
            )
    else:
        raw_counts = backend.probability_sample(
            counts, pi, status=status, g=random_generator
        )
        # if random_generator is None:
        # raw_counts = backend.implicit_randc(drange, shape=counts, p=pi)
        # else:
        # raw_counts = backend.stateful_randc(
        # random_generator, a=drange, shape=counts, p=pi
        # )
        return sample2all(raw_counts, n, format=format, jittable=jittable)


measurement_results = measurement_counts


@partial(arg_alias, alias_dict={"format": ["format_"]})
def sample2all(
    sample: Tensor, n: int, format: str = "count_vector", jittable: bool = False
) -> Any:
    """
    transform ``sample_int`` or ``sample_bin`` form results to other forms specified by ``format``

    :param sample: measurement shots results in ``sample_int`` or ``sample_bin`` format
    :type sample: Tensor
    :param n: number of qubits
    :type n: int
    :param format: see the doc in the doc in :py:meth:`tensorcircuit.quantum.measurement_results`,
        defaults to "count_vector"
    :type format: str, optional
    :param jittable: only applicable to count transformation in jax backend, defaults to False
    :type jittable: bool, optional
    :return: measurement results specified as ``format``
    :rtype: Any
    """
    if len(backend.shape_tuple(sample)) == 1:
        sample_int = sample
        sample_bin = sample_int2bin(sample, n)
    elif len(backend.shape_tuple(sample)) == 2:
        sample_int = sample_bin2int(sample, n)
        sample_bin = sample
    else:
        raise ValueError("unrecognized tensor shape for sample")
    if format == "sample_int":
        return sample_int
    elif format == "sample_bin":
        return sample_bin
    else:
        count_tuple = sample2count(sample_int, n, jittable)
        if format == "count_tuple":
            return count_tuple
        elif format == "count_vector":
            return count_s2d(count_tuple, n)
        elif format == "count_dict_bin":
            return count_tuple2dict(count_tuple, n, key="bin")
        elif format == "count_dict_int":
            return count_tuple2dict(count_tuple, n, key="int")
        else:
            raise ValueError(
                "unsupported format %s for finite shots measurement" % format
            )


def spin_by_basis(n: int, m: int, elements: Tuple[int, int] = (1, -1)) -> Tensor:
    """
    Generate all n-bitstrings as an array, each row is a bitstring basis.
    Return m-th col.

    :Example:

    >>> qu.spin_by_basis(2, 1)
    array([ 1, -1,  1, -1])

    :param n: length of a bitstring
    :type n: int
    :param m: m<n,
    :type m: int
    :param elements: the binary elements to generate, default is (1, -1).
    :type elements: Tuple[int, int], optional
    :return: The value for the m-th position in bitstring when going through
        all bitstring basis.
    :rtype: Tensor
    """
    s = backend.tile(
        backend.cast(
            backend.convert_to_tensor(np.array([[elements[0]], [elements[1]]])), "int32"
        ),
        [2**m, int(2 ** (n - m - 1))],
    )
    return backend.reshape(s, [-1])


def correlation_from_samples(index: Sequence[int], results: Tensor, n: int) -> Tensor:
    r"""
    Compute :math:`\prod_{i\in \\text{index}} s_i (s=\pm 1)`,
    Results is in the format of "sample_int" or "sample_bin"

    :param index: list of int, indicating the position in the bitstring
    :type index: Sequence[int]
    :param results: sample tensor
    :type results: Tensor
    :param n: number of qubits
    :type n: int
    :return: Correlation expectation from measurement shots
    :rtype: Tensor
    """
    if len(backend.shape_tuple(results)) == 1:
        results = sample_int2bin(results, n)
    results = 1 - results * 2
    r = results[:, index[0]]
    for i in index[1:]:
        r *= results[:, i]
    r = backend.cast(r, rdtypestr)
    return backend.mean(r)


def correlation_from_counts(index: Sequence[int], results: Tensor) -> Tensor:
    r"""
    Compute :math:`\prod_{i\in \\text{index}} s_i`,
    where the probability for each bitstring is given as a vector ``results``.
    Results is in the format of "count_vector"

    :Example:

    >>> prob = tc.array_to_tensor(np.array([0.6, 0.4, 0, 0]))
    >>> qu.correlation_from_counts([0, 1], prob)
    (0.20000002+0j)
    >>> qu.correlation_from_counts([1], prob)
    (0.20000002+0j)

    :param index: list of int, indicating the position in the bitstring
    :type index: Sequence[int]
    :param results: probability vector of shape 2^n
    :type results: Tensor
    :return: Correlation expectation from measurement shots.
    :rtype: Tensor
    """
    results = backend.reshape(results, [-1])
    results = backend.cast(results, rdtypestr)
    results /= backend.sum(results)
    n = int(np.log(results.shape[0]) / np.log(2))
    for i in index:
        results = results * backend.cast(spin_by_basis(n, i), results.dtype)
    return backend.sum(results)


# @op2tensor
# def purify(rho):
#     """
#     Take state rho and purify it into a wavefunction of squared dimension.
#     """
#     d = rho.shape[0]
#     evals, vs = backend.eigh(rho)
#     evals = backend.relu(evals)
#     psi = np.zeros(shape=(d ** 2, 1), dtype=complex)
#     for i, lbd in enumerate(lbd):
#         psi += lbd * kron(vs[:, [i]], basis_vec(i, d))
#     return psi
