from tensorcircuit.tensorcircuit.abstractcircuit import AbstractCircuit
from . import Circuit, DMCircuit
from .cons import backend


class NoiseConf:
    def __init__(self) -> None:  # type: ignore
        self.nc = {}  # type: ignore
        self.quantum = False
        self.readout = False
        self.num_quantum = 0

    def add_noise(self, gate_name, kraus):  # type: ignore
        self.nc[gate_name] = kraus
        if gate_name == "readout":
            self.readout = True
        else:
            self.quantum = True
            self.num_quantum += 1


def apply_qir(c, qir, noise_conf, randx=None):  # type: ignore

    quantum_index = 0
    for d in qir:

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
                c.general_kraus(noise_conf.nc[d["name"]], *d["index"])

        else:
            if d["name"] in noise_conf.nc:
                quantum_index += 1
                if noise_conf.nc[d["name"]].is_unitary is True:
                    c.unitary_kraus(
                        noise_conf.nc[d["name"]],
                        *d["index"],
                        status=randx[quantum_index - 1]
                    )
                else:
                    c.general_kraus(
                        noise_conf.nc[d["name"]],
                        *d["index"],
                        status=randx[quantum_index - 1]
                    )

    return c


def circuit_with_noise(c, noise_conf, randx=None):  # type: ignore
    qir = c.to_qir()
    cnew: AbstractCircuit
    if isinstance(c, DMCircuit):
        cnew = DMCircuit(c._nqubits)
    else:
        cnew = Circuit(c._nqubits)
    cnew = apply_qir(cnew, qir, noise_conf, randx)
    return cnew


def expectation_ps_noisfy(c, x=None, y=None, z=None, noise_conf=NoiseConf(), nmc=1000):  # type: ignore

    if noise_conf.readout is True:
        raise ValueError("expectation_ps_noisfy can't support readout error.")

    else:
        if noise_conf.quantum is True:

            # density matrix
            if isinstance(c, DMCircuit):
                cnoise = circuit_with_noise(c, noise_conf)
                return cnoise.expectation_ps(x=x, y=y, z=z)

            # monte carlo
            else:

                def mcsim(randx):  # type: ignore
                    cnoise = circuit_with_noise(c, noise_conf, randx)
                    return cnoise.expectation_ps(x=x, y=y, z=z)

                mcsim_vmap = backend.vmap(mcsim, vectorized_argnums=0)
                randx = backend.implicit_randu([nmc, noise_conf.num_quantum])
                value = backend.mean(mcsim_vmap(randx))

                return value

        else:
            return c.expectation_ps(x=x, y=y, z=z)


def sample_expectation_ps_noisfy(  # type: ignore
    c, x=None, y=None, z=None, noise_conf=NoiseConf(), nmc=1000, shots=None
):

    if noise_conf.readout is True:
        readout_error = noise_conf.nc["readout"]
    else:
        readout_error = None

    if noise_conf.quantum is True:

        # density matrix
        if isinstance(c, DMCircuit):
            cnoise = circuit_with_noise(c, noise_conf)
            return cnoise.sample_expectation_ps(
                x=x, y=y, z=z, shots=shots, readout_error=readout_error
            )

        # monte carlo
        else:

            def mcsim(randx):  # type: ignore
                cnoise = circuit_with_noise(c, noise_conf, randx)
                return cnoise.sample_expectation_ps(
                    x=x, y=y, z=z, shots=shots, readout_error=readout_error
                )

            mcsim_vmap = backend.vmap(mcsim, vectorized_argnums=0)
            randx = backend.implicit_randu([nmc, noise_conf.num_quantum])
            value = backend.mean(mcsim_vmap(randx))
            return value

    else:
        value = c.sample_expectation_ps(
            x=x, y=y, z=z, shots=shots, readout_error=readout_error
        )
        return value
