"""
quantum error mitigation functionalities
"""

from typing import Any, List, Union, Optional, Dict, Tuple, Callable, Sequence
from itertools import product
import functools
from functools import partial
import collections
import operator
from random import choice
import logging

logger = logging.getLogger(__name__)

import numpy as np

try:
    from mitiq import zne, ddd
    from mitiq.zne.scaling import fold_gates_at_random

    zne_option = zne
    dd_option = ddd
except ModuleNotFoundError:
    logger.warning("mitiq is not installed, please ``pip install mitiq`` first")
    zne_option = None
    dd_option = None

from ... import Circuit
from ... import backend, gates
from ...compiler import simple_compiler

Gate = gates.Gate


def apply_zne(
    circuit: Any,
    executor: Callable[[Union[Any, Sequence[Any]]], Any],
    factory: Optional[Any],
    scale_noise: Optional[Callable[[Any, float], Any]] = None,
    num_to_average: int = 1,
    **kws: Any,
) -> Any:
    """
    Apply zero-noise extrapolation (ZNE) and return the mitigated results.

    :param circuit: The aim circuit.
    :type circuit: Any
    :param executor: A executor that executes a single circuit or a batch of circuits and return results.
    :type executor: Callable[[Union[Any, Sequence[Any]]], Any]
    :param factory: Determines the extropolation method.
    :type factory: Optional[Factory]
    :param scale_noise: The scaling function for the aim circuit, defaults to fold_gates_at_random
    :type scale_noise: Callable[[Any, float], Any], optional
    :param num_to_average: Number of times expectation values are computed by
            the executor, average each point, defaults to 1.
    :type num_to_average: int, optional
    :return: Mitigated average value by ZNE.
    :rtype: float
    """
    if scale_noise is None:
        scale_noise = fold_gates_at_random

    def executortc(c):  # type: ignore
        c = Circuit.from_qiskit(c, c.num_qubits)
        return executor(c)

    circuit = circuit.to_qiskit(enable_instruction=True)
    result = zne.execute_with_zne(
        circuit=circuit,
        executor=executortc,
        factory=factory,
        scale_noise=scale_noise,
        num_to_average=num_to_average,
        **kws,
    )
    return result


def prune_ddcircuit(c: Any, qlist: List[int]) -> Any:
    """
    Discard DD sequence on idle qubits and Discard identity gate
    (no identity/idle gate on device now) filled in DD sequence.

    :param c: circuit
    :type c: Any
    :param qlist: qubit list to apply DD sequence
    :type qlist: list
    :return: new circuit
    :rtype: Any
    """
    qir = c.to_qir()
    cnew = Circuit(c.circuit_param["nqubits"])
    for d in qir:
        if d["index"][0] in qlist:
            if_iden = np.sum(abs(np.array([[1, 0], [0, 1]]) - d["gate"].get_tensor()))
            if if_iden > 1e-4:
                if "parameters" not in d:
                    cnew.apply_general_gate_delayed(d["gatef"], d["name"])(
                        cnew, *d["index"]
                    )
                else:
                    cnew.apply_general_variable_gate_delayed(d["gatef"], d["name"])(
                        cnew, *d["index"], **d["parameters"]
                    )
    return cnew


def used_qubits(c: Any) -> List[int]:
    """
    Create a qubit list that includes all qubits having gate manipulation.

    :param c: a circuit
    :type c: Any
    :return: qubit list
    :rtype: List
    """
    qir = c.to_qir()
    qlist = []
    for d in qir:
        for i in range(len(d["index"])):
            if d["index"][i] not in qlist:
                qlist.append(d["index"][i])
    return qlist


def add_dd(c: Any, rule: Callable[[int], Any]) -> Any:
    """
    Add DD sequence to A circuit

    :param c: circuit
    :type c: Any
    :param rule: The rule to conduct the DD sequence
    :type rule:  Callable[[int], Any]
    :return: new circuit
    :rtype: Any
    """
    nqubit = c.circuit_param["nqubits"]
    input_circuit = c.to_qiskit()
    circuit_dd = dd_option.insert_ddd_sequences(input_circuit, rule=rule)
    circuit_dd = Circuit.from_qiskit(circuit_dd, nqubit)
    return circuit_dd


def apply_dd(
    circuit: Any,
    executor: Callable[[Any], Any],
    rule: Union[Callable[[int], Any], List[str]],
    rule_args: Optional[Dict[str, Any]] = None,
    num_trials: int = 1,
    full_output: bool = False,
    ignore_idle_qubit: bool = True,
    fulldd: bool = False,
    iscount: bool = False,
) -> Union[
    float, Tuple[float, List[Any]], Dict[str, float], Tuple[Dict[str, float], List[Any]]
]:
    """
    Apply dynamic decoupling (DD) and return the mitigated results.


    :param circuit: The aim circuit.
    :type circuit: Any
    :param executor: A executor that executes a circuit and return results.
    :type executor: Callable[[Any], Any]
    :param rule: The rule to construct DD sequence, can use default rule "dd_option.rules.xx"
    or custom rule "['X','X']"
    :type rule: Union[Callable[[int], Any], List[str]]
    :param rule_args:An optional dictionary of keyword arguments for ``rule``, defaults to {}.
    :type rule_args: Dict[str, Any], optional
    :param num_trials: The number of independent experiments to average over, defaults to 1
    :type num_trials: int, optional
    :param full_output: If ``False`` only the mitigated expectation value is
        returned. If ``True`` a dictionary containing all DD data is
        returned too, defaults to False
    :type full_output: bool, optional
    :param ig_idle_qubit: ignore the DD sequences that added to unused qubits, defaults to True
    :type ig_idle_qubit: bool, optional
    :param fulldd: dd sequence full fill the idle circuits, defaults to False
    :type fulldd: bool, optional
    :param iscount: whether the output is bit string, defaults to False
    :type iscount: bool, optional
    :return: mitigated expectation value or mitigated expectation value and DD circuit information
    :rtype: Union[float, Tuple[float, Dict[str, Any]]]
    """
    if rule_args is None:
        rule_args = {}

    def dd_rule(slack_length: int, spacing: int = -1) -> Any:
        """
        Set DD rule.

        :param slack_length: Length of idle window to fill. Automatically calculated for a circuit.
        :type slack_length: int
        :param spacing: How many identity spacing gates to apply between dynamical
              decoupling gates, defaults to -1
        :type spacing: int, optional
        """
        dd_sequence = dd_option.rules.general_rule(
            slack_length=slack_length,
            spacing=spacing,
            gates=gates,
        )
        return dd_sequence

    if isinstance(rule, list):
        import cirq

        gates = []
        for i in rule:
            gates.append(getattr(cirq, i))
        rule = dd_rule

    if ignore_idle_qubit is True:
        qlist = used_qubits(circuit)
    else:
        qlist = list(range(circuit.circuit_param["nqubits"]))

    rule_partial = partial(
        rule, **rule_args
    )  # rule_args influence the finall DD sequence
    c2 = circuit
    c3 = add_dd(c2, rule_partial)
    c3 = prune_ddcircuit(c3, qlist)
    if fulldd is True:
        while c2.to_openqasm() != c3.to_openqasm():
            c2 = c3
            c3 = add_dd(c2, rule_partial)
            c3 = prune_ddcircuit(c3, qlist)

    exp = []
    for i in range(num_trials):  # type: ignore
        exp.append(executor(c3))

    if iscount is True:
        sumcount = dict(
            functools.reduce(operator.add, map(collections.Counter, exp))  # type:ignore
        )  # type:ignore
        result = {k: sumcount[k] / num_trials for k in sumcount.keys()}
    else:
        result = np.mean(exp)  # type:ignore

    if full_output is True:
        result = [result, c3]  # type:ignore

    # def executortc(c):
    #     c = Circuit.from_qiskit(c, c.num_qubits)
    #     c = prune_ddcircuit(c,qlist)
    #     return executor(c)
    # circuit = circuit.to_qiskit()
    # result = ddd.execute_with_ddd(
    #     circuit=circuit,
    #     executor=executortc,
    #     rule=rule,
    #     rule_args=rule_args,
    #     num_trials=num_trials,
    #     full_output=full_output,
    # )

    return result  # type:ignore


def rc_candidates(gate: Gate) -> List[Any]:
    pauli = [m.tensor for m in gates.pauli_gates]
    if isinstance(gate, gates.Gate):
        gate = gate.tensor
    gatem = backend.reshapem(gate)
    r = []
    for combo in product(*[range(4) for _ in range(4)]):
        i = (
            np.kron(pauli[combo[0]], pauli[combo[1]])
            @ gatem
            @ np.kron(pauli[combo[2]], pauli[combo[3]])
        )
        if np.allclose(i, gatem, atol=1e-4):
            r.append(combo)
        elif np.allclose(i, -gatem, atol=1e-4):
            r.append(combo)
    return r


def _apply_gate(c: Any, i: int, j: int) -> Any:
    if i == 0:
        c.i(j)
    elif i == 1:
        c.x(j)
    elif i == 2:
        c.y(j)
    elif i == 3:
        c.z(j)
    return c


candidate_dict = {}  # type: ignore


def rc_circuit(c: Any) -> Any:
    qir = c.to_qir()
    cnew = Circuit(c.circuit_param["nqubits"])
    for d in qir:
        if len(d["index"]) == 2:
            if d["gate"].name in candidate_dict:
                rc_cand = candidate_dict[d["gate"].name]
                rc_list = choice(rc_cand)
            else:
                rc_cand = rc_candidates(d["gate"])
                rc_list = choice(rc_cand)
                candidate_dict[d["gate"].name] = rc_cand

            cnew = _apply_gate(cnew, rc_list[0], d["index"][0])
            cnew = _apply_gate(cnew, rc_list[1], d["index"][1])
            if "parameters" not in d:
                cnew.apply_general_gate_delayed(d["gatef"], d["name"])(
                    cnew, *d["index"]
                )
            else:
                cnew.apply_general_variable_gate_delayed(d["gatef"], d["name"])(
                    cnew, *d["index"], **d["parameters"]
                )
            cnew = _apply_gate(cnew, rc_list[2], d["index"][0])
            cnew = _apply_gate(cnew, rc_list[3], d["index"][1])
        else:
            if "parameters" not in d:
                cnew.apply_general_gate_delayed(d["gatef"], d["name"])(
                    cnew, *d["index"]
                )
            else:
                cnew.apply_general_variable_gate_delayed(d["gatef"], d["name"])(
                    cnew, *d["index"], **d["parameters"]
                )
    return cnew


def apply_rc(
    circuit: Any,
    executor: Callable[[Any], Any],
    num_to_average: int = 1,
    simplify: bool = True,
    iscount: bool = False,
    **kws: Any,
) -> Tuple[float, List[Any]]:
    """
    Apply Randomized Compiling or Pauli twirling on two-qubit gates.

    :param circuit: Input circuit
    :type circuit: Any
    :param executor: A executor that executes a circuit and return results.
    :type executor: Callable[[Any], Any]
    :param num_to_average: Number of circuits for RC, defaults to 1
    :type num_to_average: int, optional
    :param simplify: Whether simplify the circuits by merging single qubit gates, defaults to True
    :type simplify: bool, optional
    :param iscount: whether the output is bit string, defaults to False
    :type iscount: bool, optional
    :return: Mitigated results by RC
    :rtype: float
    """
    exp = []
    circuit_list = []
    for _ in range(num_to_average):
        circuit1 = rc_circuit(circuit)

        if simplify is True:
            circuit1 = prune_ddcircuit(
                circuit1, qlist=list(range(circuit1.circuit_param["nqubits"]))
            )
            circuit1 = simple_compiler.merge(circuit1)

        exp.append(executor(circuit1))
        circuit_list.append(circuit1)

    if iscount is True:
        sumcount = dict(
            functools.reduce(operator.add, map(collections.Counter, exp))  # type:ignore
        )  # type:ignore
        result = {k: sumcount[k] / num_to_average for k in sumcount.keys()}
    else:
        result = np.mean(exp)  # type:ignore

    return result, circuit_list  # type:ignore


# TODO(yuqinchen) add executor with batch ability
