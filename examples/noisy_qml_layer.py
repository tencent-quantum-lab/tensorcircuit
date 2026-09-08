import tensorcircuit as tc

K = tc.set_backend("jax")

n = 3
p = 0.1


def f(state, noise, weights):
        c = tc.Circuit(n, inputs=state)
        for i in range(n):
            c.rz(i, theta=weights[i])
        for i in range(n):
            c.depolarizing(i, px=p, py=p, pz=p, status=noise[i])
        return K.real(c.expectation_ps(x=[0]))

torchb = tc.get_backend("pytorch")

torch_layer = tc.TorchLayer(f,[n],use_vmap=True,vectorized_argnums=[0, 1],)
state = torchb.ones([2, 2**n]) / 2 ** (n / 2)
noise = torchb.implicit_randu([2, n])
torch_output = torch_layer(state, noise)
print("TorchLayer output:")
print(torch_output)

def f_keras(inputs, weights):
    state = inputs["state"]
    noise = inputs["noise"]

    return f(state, noise, weights)


keras_layer = tc.KerasLayer(f_keras, [n])

state = K.ones([2, 2**n]) / 2 ** (n / 2)
noise = K.implicit_randu([2, n])
keras_output = keras_layer({"state": state,"noise": noise,})
print("KerasLayer output:")
print(keras_output)