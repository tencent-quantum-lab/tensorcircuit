import os
import sys

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import numpy as np
import tensornetwork as tn
from tensorcircuit import simplify


def test_infer_shape():
    a = tn.Node(np.ones([2, 3, 5]))
    b = tn.Node(np.ones([3, 5, 7]))
    a[1] ^ b[0]
    a[2] ^ b[1]
    assert simplify.infer_new_shape(a, b) == ((2, 7), (2, 3, 5), (3, 5, 7))


def test_rank_simplify():
    a = tn.Node(np.ones([2, 2]), name="a")
    b = tn.Node(np.ones([2, 2]), name="b")
    c = tn.Node(np.ones([2, 2, 2, 2]), name="c")
    d = tn.Node(np.ones([2, 2, 2, 2, 2, 2]), name="d")
    e = tn.Node(np.ones([2, 2]), name="e")

    a[1] ^ c[0]
    b[1] ^ c[1]
    c[2] ^ d[0]
    c[3] ^ d[1]
    d[4] ^ e[0]

    nodes = simplify._full_rank_simplify([a, b, c, d, e])
    assert nodes[0].shape == tuple([2 for _ in range(6)])
    assert len(nodes) == 1

    f = tn.Node(np.ones([2, 2]), name="f")
    g = tn.Node(np.ones([2, 2, 2, 2]), name="g")
    h = tn.Node(np.ones([2, 2, 2, 2]), name="h")

    f[1] ^ g[0]
    g[2] ^ h[1]

    nodes = simplify._full_rank_simplify([f, g, h])
    assert len(nodes) == 2
