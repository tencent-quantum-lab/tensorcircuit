import tensorcircuit as tc

tc.set_backend("tensorflow")
nwires, nlayers = 6, 3


def vqe_forward(param):
    c = tc.Circuit(nwires)
    for i in range(nwires):
        c.H(i)
    for j in range(nlayers):
        for i in range(nwires - 1):
            c.exp1(i, i + 1, theta=param[2 * j, i], unitary=tc.gates._zz_matrix)
        for i in range(nwires):
            c.rx(i, theta=param[2 * j + 1, i])
    e = sum(
        [-1.0 * c.expectation_ps(x=[i]) for i in range(nwires)]
        + [1.0 * c.expectation_ps(z=[i, i + 1]) for i in range(nwires - 1)]
    )
    return e


tc_vg = tc.backend.jit(tc.backend.value_and_grad(vqe_forward))
param = tc.backend.cast(tc.backend.randn([2 * nlayers, nwires]), "complex64")
print(tc_vg(param))
