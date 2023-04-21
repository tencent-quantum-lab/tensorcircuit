"""
General Noise Model Construction.
"""
import logging
from functools import partial
from typing import Any, Sequence, Optional, List, Dict, Tuple, Callable, Union

import tensornetwork as tn

from .abstractcircuit import AbstractCircuit
from . import gates
from . import Circuit, DMCircuit
from .cons import backend
from .channels import KrausList

Gate = gates.Gate
Tensor = Any
logger = logging.getLogger(__name__)


class NoiseConf:
    """
    ``Noise Configuration`` class.

    .. code-block:: python

        error1 = tc.channels.generaldepolarizingchannel(0.1, 1)
        error2 = tc.channels.thermalrelaxationchannel(300, 400, 100, "ByChoi", 0)
        readout_error = [[0.9, 0.75], [0.4, 0.7]]

        noise_conf = NoiseConf()
        noise_conf.add_noise("x", error1)
        noise_conf.add_noise("h", [error1, error2], [[0], [1]])
        noise_conf.add_noise("readout", readout_error)
    """

    def __init__(self) -> None:
        """
        Establish a noise configuration.
        """
        # description, condition and Kraus operators
        self.nc: List[Tuple[Any, Callable[[Dict[str, Any]], bool], KrausList]] = []
        self.has_quantum = False
        self.readout_error = None

    def add_noise(
        self,
        gate_name: str,
        kraus: Union[KrausList, Sequence[KrausList]],
        qubit: Optional[Sequence[Any]] = None,
    ) -> None:
        """
        Add noise channels on specific gates and specific qubits in form of Kraus operators.

        :param gate_name: noisy gate
        :type gate_name: str
        :param kraus: noise channel
        :type kraus: Sequence[Gate]
        :param qubit: the list of noisy qubit, defaults to None, indicating applying the noise channel on all qubits
        :type qubit: Optional[Sequence[Any]], optional
        """
        if gate_name == "readout":
            assert qubit is None
            self.readout_error = kraus  # type:ignore
            return

        gate_name = AbstractCircuit.standardize_gate(gate_name)
        description: Any

        if qubit is None:
            description = gate_name

            def condition(d: Dict[str, Any]) -> bool:
                return str(d["gatef"].n) == gate_name

            if not isinstance(kraus, KrausList):
                assert len(kraus) == 1
                krauslist = kraus[0]
            else:
                krauslist = kraus
            self.nc.append((description, condition, krauslist))
        else:
            for idx, krauslist in zip(qubit, kraus):
                description = (gate_name, idx)

                # https://stackoverflow.com/questions/1107210/python-create-function-in-a-loop-capturing-the-loop-variable
                def condition(  # type:ignore
                    d: Dict[str, Any], _idx: Sequence[Any]
                ) -> bool:
                    # avoid bad black style because the long is too long
                    b1 = d["gatef"].n == gate_name
                    b2 = tuple(d["index"]) == tuple(_idx)
                    return b1 and b2

                condition = partial(condition, _idx=idx)
                self.nc.append((description, condition, krauslist))

        self.has_quantum = True

    def add_noise_by_condition(
        self,
        condition: Callable[[Dict[str, Any]], bool],
        kraus: KrausList,
        name: Optional[Any] = "custom",
    ) -> None:
        """
        Add noise based on specified condition

        :param condition: a function to decide if the noise should be added to the qir.
        :type condition: Callable[[Dict[str, Any]], bool]
        :param kraus: the error channel
        :type kraus: KrausList
        :param name: the name of the condition. A metadata that does not affect the numerics.
        :type name: Any
        """
        self.nc.append((name, condition, kraus))

    def channel_count(self, c: Circuit) -> int:
        """
        Count the total number of channels in a given circuit

        :param c: the circuit to be counted
        :type c: Circuit
        :return: the count
        :rtype: int
        """
        count = 0
        for d in c.to_qir():
            for _, condition, _ in self.nc:
                if condition(d):
                    count += 1
        return count


def apply_qir_with_noise(
    c: Any,
    qir: List[Dict[str, Any]],
    noise_conf: NoiseConf,
    status: Optional[Tensor] = None,
) -> Any:
    """

    :param c: A newly defined circuit
    :type c: AbstractCircuit
    :param qir: The qir of the clean circuit
    :type qir: List[Dict[str, Any]]
    :param noise_conf: Noise Configuration
    :type noise_conf: NoiseConf
    :param status: The status for Monte Carlo sampling, defaults to None
    :type status: 1D Tensor, optional
    :return: A newly constructed circuit with noise
    :rtype: AbstractCircuit
    """
    quantum_index = 0
    for d in qir:
        d["name"] = AbstractCircuit.standardize_gate(d["name"])

        if "parameters" not in d:  # paramized gate
            c.apply_general_gate_delayed(d["gatef"], d["name"], d["mpo"])(
                c, *d["index"]
            )
        else:
            c.apply_general_variable_gate_delayed(d["gatef"], d["name"], d["mpo"])(
                c, *d["index"], **d["parameters"]
            )

        if isinstance(c, DMCircuit):
            for _, condition, krauslist in noise_conf.nc:
                if condition(d):
                    c.general_kraus(krauslist, *d["index"])
        else:
            for _, condition, krauslist in noise_conf.nc:
                if condition(d):
                    if krauslist.is_unitary:
                        c.unitary_kraus(
                            krauslist,
                            *d["index"],
                            status=status[quantum_index],  #  type: ignore
                        )
                    else:
                        c.general_kraus(
                            krauslist,
                            *d["index"],
                            status=status[quantum_index],  #  type: ignore
                        )
                    quantum_index += 1

    return c


def circuit_with_noise(
    c: AbstractCircuit, noise_conf: NoiseConf, status: Optional[Tensor] = None
) -> Any:
    """Noisify a clean circuit.

    :param c: A clean circuit
    :type c: AbstractCircuit
    :param noise_conf: Noise Configuration
    :type noise_conf: NoiseConf
    :param status: The status for Monte Carlo sampling, defaults to None
    :type status: 1D Tensor, optional
    :return: A newly constructed circuit with noise
    :rtype: AbstractCircuit
    """
    qir = c.to_qir()
    cnew: AbstractCircuit
    if isinstance(c, DMCircuit):
        cnew = DMCircuit(**c.circuit_param)
    else:
        cnew = Circuit(**c.circuit_param)
    cnew = apply_qir_with_noise(cnew, qir, noise_conf, status)
    return cnew


def sample_expectation_ps_noisfy(
    c: Any,
    x: Optional[Sequence[int]] = None,
    y: Optional[Sequence[int]] = None,
    z: Optional[Sequence[int]] = None,
    noise_conf: Optional[NoiseConf] = None,
    nmc: int = 1000,
    shots: Optional[int] = None,
    statusc: Optional[Tensor] = None,
    status: Optional[Tensor] = None,
    **kws: Any,
) -> Tensor:
    """
    Calculate sample_expectation_ps with noise configuration.

    :param c: The clean circuit
    :type c: Any
    :param x: sites to apply X gate, defaults to None
    :type x: Optional[Sequence[int]], optional
    :param y: sites to apply Y gate, defaults to None
    :type y: Optional[Sequence[int]], optional
    :param z: sites to apply Z gate, defaults to None
    :type z: Optional[Sequence[int]], optional
    :param noise_conf: Noise Configuration, defaults to None
    :type noise_conf: Optional[NoiseConf], optional
    :param nmc: repetition time for Monte Carlo sampling  for noisfy calculation, defaults to 1000
    :type nmc: int, optional
    :param shots: number of measurement shots, defaults to None, indicating analytical result
    :type shots: Optional[int], optional
    :param statusc: external randomness given by tensor uniformly from [0, 1], defaults to None,
        used for noisfy circuit sampling
    :type statusc: Optional[Tensor], optional
    :param status: external randomness given by tensor uniformly from [0, 1], defaults to None,
        used for measurement sampling
    :type status: Optional[Tensor], optional
    :return: sample expectation value with noise
    :rtype: Tensor
    """

    if noise_conf is None:
        noise_conf = NoiseConf()

    readout_error = noise_conf.readout_error

    if noise_conf.has_quantum:
        # density matrix
        if isinstance(c, DMCircuit):
            cnoise = circuit_with_noise(c, noise_conf)
            return cnoise.sample_expectation_ps(
                x=x, y=y, z=z, shots=shots, status=status, readout_error=readout_error
            )

        # monte carlo
        else:

            def mcsim(statusc: Optional[Tensor], status: Optional[Tensor]) -> Tensor:
                cnoise = circuit_with_noise(c, noise_conf, statusc)  #  type: ignore
                return cnoise.sample_expectation_ps(
                    x=x,
                    y=y,
                    z=z,
                    shots=shots,
                    status=status,
                    readout_error=readout_error,
                )

            mcsim_vmap = backend.vmap(mcsim, vectorized_argnums=(0, 1))
            if statusc is None:
                num_quantum = noise_conf.channel_count(c)
                statusc = backend.implicit_randu([nmc, num_quantum])

            if status is None:
                if shots is None:
                    status = backend.implicit_randu([nmc, 1])
                else:
                    status = backend.implicit_randu([nmc, shots])

            value = backend.mean(mcsim_vmap(statusc, status))
            return value

    else:
        value = c.sample_expectation_ps(
            x=x, y=y, z=z, shots=shots, status=status, readout_error=readout_error
        )
        return value


def expectation_noisfy(
    c: Any,
    *ops: Tuple[tn.Node, List[int]],
    noise_conf: Optional[NoiseConf] = None,
    nmc: int = 1000,
    status: Optional[Tensor] = None,
    **kws: Any,
) -> Tensor:
    """
    Calculate expectation value with noise configuration.

    :param c: The clean circuit
    :type c: Any
    :param noise_conf: Noise Configuration, defaults to None
    :type noise_conf: Optional[NoiseConf], optional
    :param nmc: repetition time for Monte Carlo sampling for noisfy calculation, defaults to 1000
    :type nmc: int, optional
    :param status: external randomness given by tensor uniformly from [0, 1], defaults to None,
        used for noisfy circuit sampling
    :type status: Optional[Tensor], optional
    :return: expectation value with noise
    :rtype: Tensor
    """

    if noise_conf is None:
        noise_conf = NoiseConf()

    if noise_conf.readout_error is not None:
        logger.warning("expectation_ps_noisfy can't support readout error.")

    if noise_conf.has_quantum:
        # density matrix
        if isinstance(c, DMCircuit):
            cnoise = circuit_with_noise(c, noise_conf)
            return cnoise.expectation(*ops, **kws)

        # monte carlo
        else:

            def mcsim(status: Optional[Tensor]) -> Tensor:
                cnoise = circuit_with_noise(c, noise_conf, status)  #  type: ignore
                return cnoise.expectation(*ops, **kws)

            mcsim_vmap = backend.vmap(mcsim, vectorized_argnums=0)
            if status is None:
                num_quantum = noise_conf.channel_count(c)
                status = backend.implicit_randu([nmc, num_quantum])

            value = backend.mean(mcsim_vmap(status))

            return value

    else:
        return c.expectation(*ops, **kws)
