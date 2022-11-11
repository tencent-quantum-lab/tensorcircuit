"""
General Noise Model Construction.
"""
import logging
from tensorcircuit.abstractcircuit import AbstractCircuit
from . import Circuit, DMCircuit
from .cons import backend

logger = logging.getLogger(__name__)


class NoiseConf:
    def __init__(self) -> None:  # type: ignore
        self.nc = {}  # type: ignore
        self.has_quantum = False
        self.has_readout = False

    def add_noise(self, gate_name, kraus, qubit=None):  # type: ignore
        if gate_name not in self.nc:
            qubit_kraus = {}
        else:
            qubit_kraus = self.nc[gate_name]

        if qubit == None:
            qubit_kraus["Default"]= kraus
        else: 
            for i in range(len(qubit)):
                qubit_kraus[tuple(qubit[i])]= kraus[i]
        self.nc[gate_name] = qubit_kraus


        if gate_name == "readout":
            self.has_readout = True
        else:
            self.has_quantum = True

       


def apply_qir_with_noise(c, qir, noise_conf, status=None):  # type: ignore

    quantum_index = 0
    for d in qir:
        print("gate:",d["index"],d["name"])
        if "parameters" not in d:  # paramized gate
            c.apply_general_gate_delayed(d["gatef"], d["name"])(  # type: ignore
                c, *d["index"]  # type: ignore
            )
        else:
            c.apply_general_variable_gate_delayed(d["gatef"], d["name"])(  # type: ignore
                c, *d["index"], **d["parameters"]  # type: ignore
            )

        if isinstance(c, DMCircuit):
            if d["name"] in noise_conf.nc:
                if "Default" in noise_conf.nc[d["name"]] or d["index"] in noise_conf.nc[d["name"]]:
                    print("add:", d["index"], d["name"])
                    if "Default" in noise_conf.nc[d["name"]]:
                        noise_kraus = noise_conf.nc[d["name"]]["Default"]
                    if d["index"] in noise_conf.nc[d["name"]]:
                        noise_kraus = noise_conf.nc[d["name"]][d["index"]]

                    c.general_kraus(noise_kraus, *d["index"])

        else:
            if d["name"] in noise_conf.nc:
                if "Default" in noise_conf.nc[d["name"]] or d["index"] in noise_conf.nc[d["name"]]:
                    print("add:", d["index"], d["name"])

                    if "Default" in noise_conf.nc[d["name"]]:
                        noise_kraus = noise_conf.nc[d["name"]]["Default"]
                    if d["index"] in noise_conf.nc[d["name"]]:
                        noise_kraus = noise_conf.nc[d["name"]][d["index"]]
        
                    if noise_kraus.is_unitary is True:
                        c.unitary_kraus(
                            noise_kraus,
                            *d["index"],
                            status=status[quantum_index]
                        )
                    else:
                        c.general_kraus(
                            noise_kraus,
                            *d["index"],
                            status=status[quantum_index]
                        )
                    quantum_index += 1

    return c



def circuit_with_noise(c, noise_conf, status=None):  # type: ignore
    qir = c.to_qir()
    cnew: AbstractCircuit
    if isinstance(c, DMCircuit):
        cnew = DMCircuit(c._nqubits)
    else:
        cnew = Circuit(c._nqubits)
    cnew = apply_qir_with_noise(cnew, qir, noise_conf, status)
    return cnew




def expectation_ps_noisfy(c, x=None, y=None, z=None, noise_conf=None, nmc=1000, status=None):  # type: ignore

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

            def mcsim(status):  # type: ignore
                cnoise = circuit_with_noise(c, noise_conf, status)
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


def sample_expectation_ps_noisfy(  # type: ignore
    c, x=None, y=None, z=None, noise_conf=None, nmc=1000, shots=None, status=None
):

    if noise_conf is None:
        noise_conf = NoiseConf()
    else:
        pass

    num_quantum = c.gate_count(list(noise_conf.nc.keys()))

    if noise_conf.has_readout is True:
        readout_error = noise_conf.nc["readout"]
    else:
        readout_error = None

    if noise_conf.has_quantum is True:

        # density matrix
        if isinstance(c, DMCircuit):
            cnoise = circuit_with_noise(c, noise_conf)
            return cnoise.sample_expectation_ps(
                x=x, y=y, z=z, shots=shots, readout_error=readout_error
            )

        # monte carlo
        else:

            def mcsim(status):  # type: ignore
                cnoise = circuit_with_noise(c, noise_conf, status)
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
