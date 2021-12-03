"""
shortcuts for measurement patterns on circuit
"""
# circuit in, circuit out

from typing import Any, Optional, Sequence, Tuple

Circuit = Any  # we don't use the real circuit class as too many mypy complains emerge


def Bell_pair_block(
    c: Circuit, links: Optional[Sequence[Tuple[int, int]]] = None
) -> Circuit:
    # from |00> return |01>-|10>
    n = c._nqubits
    if links is None:
        links = [(i, i + 1) for i in range(0, n - 1, 2)]
    for a, b in links:
        c.X(a)
        c.H(a)
        c.cnot(a, b)
        c.X(b)
    return c
