"""
Very simple transformations that qiskit may even fail or hard to control
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from copy import copy

import numpy as np

from ..abstractcircuit import AbstractCircuit
from ..cons import backend
from ..quantum import QuOperator
from .. import gates
from ..utils import is_sequence


def replace_r(circuit: AbstractCircuit, **kws: Any) -> AbstractCircuit:
    qir = circuit.to_qir()
    c: Any = type(circuit)(**circuit.circuit_param)
    for d in qir:
        if "parameters" not in d:
            c.apply_general_gate_delayed(d["gatef"], d["name"], mpo=d["mpo"])(
                c, *d["index"], split=d["split"]
            )
        else:
            if d["gatef"].n == "rx":
                c.h(*d["index"])
                c.rz(*d["index"], theta=d["parameters"].get("theta", 0.0))
                c.h(*d["index"])

            elif d["gatef"].n == "ry":
                c.sd(*d["index"])
                c.h(*d["index"])
                c.rz(*d["index"], theta=d["parameters"].get("theta", 0.0))
                c.h(*d["index"])
                c.s(*d["index"])

            elif d["gatef"].n == "rzz":
                c.cx(*d["index"])
                c.rz(d["index"][1], theta=d["parameters"].get("theta", 0.0))
                c.cx(*d["index"])

            elif d["gatef"].n == "rxx":
                c.h(d["index"][0])
                c.h(d["index"][1])
                c.cx(*d["index"])
                c.rz(d["index"][1], theta=d["parameters"].get("theta", 0.0))
                c.cx(*d["index"])
                c.h(d["index"][0])
                c.h(d["index"][1])

            elif d["gatef"].n == "ryy":
                c.sd(d["index"][0])
                c.sd(d["index"][1])
                c.h(d["index"][0])
                c.h(d["index"][1])
                c.cx(*d["index"])
                c.rz(d["index"][1], theta=d["parameters"].get("theta", 0.0))
                c.cx(*d["index"])
                c.h(d["index"][0])
                c.h(d["index"][1])
                c.s(d["index"][0])
                c.s(d["index"][1])

            else:
                c.apply_general_variable_gate_delayed(
                    d["gatef"], d["name"], mpo=d["mpo"]
                )(c, *d["index"], **d["parameters"], split=d["split"])

    return c  # type: ignore


def replace_u(circuit: AbstractCircuit, **kws: Any) -> AbstractCircuit:
    qir = circuit.to_qir()
    c: Any = type(circuit)(**circuit.circuit_param)
    for d in qir:
        if "parameters" not in d:
            c.apply_general_gate_delayed(d["gatef"], d["name"], mpo=d["mpo"])(
                c, *d["index"], split=d["split"]
            )
        else:
            if d["gatef"].n == "u":
                c.rz(*d["index"], theta=d["parameters"].get("lbd", 0) - np.pi / 2)
                c.h(*d["index"])
                c.rz(*d["index"], theta=d["parameters"].get("theta", 0))
                c.h(*d["index"])
                c.rz(*d["index"], theta=np.pi / 2 + d["parameters"].get("phi", 0))
            else:
                c.apply_general_variable_gate_delayed(
                    d["gatef"], d["name"], mpo=d["mpo"]
                )(c, *d["index"], **d["parameters"], split=d["split"])

    return c  # type: ignore


def _get_matrix(qir_item: Dict[str, Any]) -> Any:
    if "gate" in qir_item:
        op = qir_item["gate"]
    else:
        op = qir_item["gatef"](**qir_item["parameters"])
    if isinstance(op, QuOperator):
        m = backend.numpy(op.eval_matrix())
    else:
        m = backend.numpy(backend.reshapem(op.tensor))
    return m


def prune(
    circuit: Union[AbstractCircuit, List[Dict[str, Any]]],
    rtol: float = 1e-3,
    atol: float = 1e-3,
    **kws: Any
) -> Any:
    if isinstance(circuit, list):
        qir = circuit
        output = "qir"
    else:
        qir = circuit.to_qir()
        output = "tc"
    if output in ["tc", "circuit"]:
        c: Any = type(circuit)(**circuit.circuit_param)  # type: ignore
        for d in qir:
            m = _get_matrix(d)

            if not np.allclose(
                m / (m[0, 0] + 1e-8), np.eye(m.shape[0]), rtol=rtol, atol=atol
            ):
                # upto a phase
                if "parameters" not in d:
                    c.apply_general_gate_delayed(d["gatef"], d["name"], mpo=d["mpo"])(
                        c, *d["index"], split=d["split"]
                    )
                else:
                    c.apply_general_variable_gate_delayed(
                        d["gatef"], d["name"], mpo=d["mpo"]
                    )(c, *d["index"], **d["parameters"], split=d["split"])
        return c

    elif output in ["qir"]:
        nqir = []
        for d in qir:
            m = _get_matrix(d)

            if not np.allclose(
                m / (m[0, 0] + 1e-8), np.eye(m.shape[0]), rtol=rtol, atol=atol
            ):
                nqir.append(d)
                # upto a phase

        return nqir


# upto global phase
default_merge_rules = {
    ("s", "s"): "z",
    ("sd", "sd"): "z",
    ("t", "t"): "s",
    ("td", "td"): "sd",
    ("x", "y"): "z",
    ("y", "x"): "z",
    ("x", "z"): "y",
    ("z", "x"): "y",
    ("z", "y"): "x",
    ("y", "z"): "x",
    ("x", "x"): "i",
    ("y", "y"): "i",
    ("z", "z"): "i",
    ("h", "h"): "i",
    ("rz", "rz"): "rz",
    ("rx", "rx"): "rx",
    ("ry", "ry"): "rx",
    ("rzz", "rzz"): "rzz",
    ("rxx", "rxx"): "rxx",
    ("ryy", "ryy"): "ryy",
    ("crz", "crz"): "crz",
    ("crx", "crx"): "crx",
    ("cry", "cry"): "crx",
    ("cnot", "cnot"): "i",
    ("cz", "cz"): "i",
    ("cy", "cy"): "i",
}


def _find_next(qir: List[Dict[str, Any]], i: int) -> Optional[int]:
    # if qir[i] is None:
    # return None
    index = qir[i]["index"]
    for j, item in enumerate(qir[i + 1 :]):
        # if item is not None:
        if item["index"] == index:
            return j + i + 1
        for ind in item["index"]:
            if ind in index:
                return None
    return None


def _get_theta(qir_item: Dict[str, Any]) -> float:
    theta = qir_item["parameters"].get("theta", 0.0)
    if is_sequence(theta) and len(theta) == 1:
        return theta[0]  # type: ignore
    return theta  # type: ignore


def _merge(
    qir: List[Dict[str, Any]], rules: Dict[Tuple[str, ...], str]
) -> Tuple[List[Dict[str, Any]], bool]:
    i = 0
    flg = False
    while i < len(qir) - 1:
        j = _find_next(qir, i)

        if j is not None:
            if (qir[i]["gatef"].n, qir[j]["gatef"].n) in rules:
                nn = rules[(qir[i]["gatef"].n, qir[j]["gatef"].n)]
                if nn == "i":
                    del qir[i]
                    del qir[j - 1]
                    # qir[i] = None
                    # qir[j] = None
                else:
                    param = {}
                    if nn.startswith("r") or nn.startswith("cr"):
                        param = {"theta": _get_theta(qir[i]) + _get_theta(qir[j])}
                    qir[i] = {
                        "gatef": getattr(gates, nn),
                        "name": nn,
                        "mpo": False,
                        "split": False,
                        "parameters": param,
                        "index": qir[i]["index"],
                    }
                    del qir[j]
                    # qir[j] = None
                flg = True
                # return qir, flg
            elif (
                qir[i]["gatef"].n == qir[j]["gatef"].n + "d"
                or qir[i]["gatef"].n + "d" == qir[j]["gatef"].n
            ):
                del qir[i]
                del qir[j - 1]
                # qir[i] = None
                # qir[j] = None
                flg = True
                # return qir, True
        i += 1
    return qir, flg


def merge(
    circuit: Union[AbstractCircuit, List[Dict[str, Any]]],
    rules: Optional[Dict[Tuple[str, ...], str]] = None,
    **kws: Any
) -> Any:
    merge_rules = copy(default_merge_rules)
    if rules is not None:
        merge_rules.update(rules)  # type: ignore
    if isinstance(circuit, list):
        qir = circuit
        output = "qir"
    else:
        qir = circuit.to_qir()
        output = "tc"
    flg = True
    while flg:
        qir, flg = _merge(qir, merge_rules)  # type: ignore
    if output in ["qir"]:
        return qir
    elif output in ["tc", "circuit"]:
        c: Any = type(circuit).from_qir(qir, circuit.circuit_param)  # type: ignore
        return c


def simple_compile(
    circuit: Any,
    info: Optional[Dict[str, Any]] = None,
    output: str = "tc",
    compiled_options: Optional[Dict[str, Any]] = None,
) -> Any:
    if compiled_options is None:
        compiled_options = {}
    len0 = len(circuit.to_qir())
    for d in circuit._extra_qir:
        if d["pos"] < len0 - 1:
            raise ValueError(
                "TC's simple compiler doesn't support measurement/reset instructions \
                in the middle of the circuit"
            )

    c = replace_r(circuit, **compiled_options)
    c = replace_u(c, **compiled_options)
    qir = c.to_qir()
    len0 = len(qir)
    qir = merge(qir, **compiled_options)
    qir = prune(qir, **compiled_options)
    len1 = len(qir)
    while len1 != len0:
        len0 = len1
        qir = merge(qir, **compiled_options)
        if len(qir) == len0:
            break
        qir = prune(qir, **compiled_options)
        len1 = len(qir)

    c = type(circuit).from_qir(qir, circuit.circuit_param)

    for d in circuit._extra_qir:
        d["pos"] = len1
        c._extra_qir.append(d)
    return (c, info)
