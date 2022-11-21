"""
General Noise Model Construction.
"""
import logging
from typing import Any, Sequence, Optional, List, Dict

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
        self.nc = {}  # type: ignore
        self.has_quantum = False
        self.has_readout = False

    def add_noise(
        self,
        gate_name: str,
        kraus: Sequence[KrausList],
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
        if gate_name not in self.nc:
            qubit_kraus = {}
        else:
            qubit_kraus = self.nc[gate_name]

        if qubit is None:
            qubit_kraus["Default"] = kraus
        else:
            for i in range(len(qubit)):
                qubit_kraus[tuple(qubit[i])] = kraus[i]
        self.nc[gate_name] = qubit_kraus

        if gate_name == "readout":
            self.has_readout = True
        else:
            self.has_quantum = True


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
        if "parameters" not in d:  # paramized gate
            c.apply_general_gate_delayed(d["gatef"], d["name"])(c, *d["index"])
        else:
            c.apply_general_variable_gate_delayed(d["gatef"], d["name"])(
                c, *d["index"], **d["parameters"]
            )

        if isinstance(c, DMCircuit):
            if d["name"] in noise_conf.nc:
                if (
                    "Default" in noise_conf.nc[d["name"]]
                    or d["index"] in noise_conf.nc[d["name"]]
                ):

                    if "Default" in noise_conf.nc[d["name"]]:
                        noise_kraus = noise_conf.nc[d["name"]]["Default"]
                    if d["index"] in noise_conf.nc[d["name"]]:
                        noise_kraus = noise_conf.nc[d["name"]][d["index"]]

                    c.general_kraus(noise_kraus, *d["index"])

        else:
            if d["name"] in noise_conf.nc:
                if (
                    "Default" in noise_conf.nc[d["name"]]
                    or d["index"] in noise_conf.nc[d["name"]]
                ):

                    if "Default" in noise_conf.nc[d["name"]]:
                        noise_kraus = noise_conf.nc[d["name"]]["Default"]
                    if d["index"] in noise_conf.nc[d["name"]]:
                        noise_kraus = noise_conf.nc[d["name"]][d["index"]]

                    if noise_kraus.is_unitary is True:
                        c.unitary_kraus(
                            noise_kraus,
                            *d["index"],
                            status=status[quantum_index]  #  type: ignore
                        )
                    else:
                        c.general_kraus(
                            noise_kraus,
                            *d["index"],
                            status=status[quantum_index]  #  type: ignore
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
        cnew = DMCircuit(c._nqubits)
    else:
        cnew = Circuit(c._nqubits)
    cnew = apply_qir_with_noise(cnew, qir, noise_conf, status)
    return cnew


def expectation_ps_noisfy(
    c: Any,
    x: Optional[Sequence[int]] = None,
    y: Optional[Sequence[int]] = None,
    z: Optional[Sequence[int]] = None,
    noise_conf: Optional[NoiseConf] = None,
    nmc: int = 1000,
    status: Optional[Tensor] = None,
) -> Tensor:

    if noise_conf is None:
        noise_conf = NoiseConf()
    else:
        pass

    num_quantum = c.gate_count(list(noise_conf.nc.keys()))

    if noise_conf.has_readout is True:
        logger.warning("expectation_ps_noisfy can't support readout error.")
    else:
        pass

    if noise_conf.has_quantum is True:

        # density matrix
        if isinstance(c, DMCircuit):
            cnoise = circuit_with_noise(c, noise_conf)
            return cnoise.expectation_ps(x=x, y=y, z=z)

        # monte carlo
        else:

            def mcsim(status: Optional[Tensor]) -> Tensor:
                cnoise = circuit_with_noise(c, noise_conf, status)  #  type: ignore
                return cnoise.expectation_ps(x=x, y=y, z=z)

            mcsim_vmap = backend.vmap(mcsim, vectorized_argnums=0)
            if status is None:
                status = backend.implicit_randu([nmc, num_quantum])
            else:
                pass
            value = backend.mean(mcsim_vmap(status))

            return value

    else:
        return c.expectation_ps(x=x, y=y, z=z)


def sample_expectation_ps_noisfy(
    c: Any,
    x: Optional[Sequence[int]] = None,
    y: Optional[Sequence[int]] = None,
    z: Optional[Sequence[int]] = None,
    noise_conf: Optional[NoiseConf] = None,
    nmc: int = 1000,
    shots: Optional[int] = None,
    status: Optional[Tensor] = None,
) -> Tensor:

    if noise_conf is None:
        noise_conf = NoiseConf()
    else:
        pass

    num_quantum = c.gate_count(list(noise_conf.nc.keys()))

    if noise_conf.has_readout is True:
        readout_error = noise_conf.nc["readout"]["Default"]
    else:
        readout_error = None

    if noise_conf.has_quantum is True:

        # density matrix
        if isinstance(c, DMCircuit):
            cnoise = circuit_with_noise(c, noise_conf)  #  type: ignore
            return cnoise.sample_expectation_ps(
                x=x, y=y, z=z, shots=shots, readout_error=readout_error
            )

        # monte carlo
        else:

            def mcsim(status: Optional[Tensor]) -> Tensor:
                cnoise = circuit_with_noise(c, noise_conf, status)  #  type: ignore
                return cnoise.sample_expectation_ps(
                    x=x, y=y, z=z, shots=shots, readout_error=readout_error
                )

            mcsim_vmap = backend.vmap(mcsim, vectorized_argnums=0)
            if status is None:
                status = backend.implicit_randu([nmc, num_quantum])
            else:
                pass
            value = backend.mean(mcsim_vmap(status))
            return value

    else:
        value = c.sample_expectation_ps(
            x=x, y=y, z=z, shots=shots, readout_error=readout_error
        )
        return value
