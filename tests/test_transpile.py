import tensorcircuit as tc


def test_qis_to_cirq():
    # TODO
    c = tc.Circuit(5)
    c.rx(0, theta=0.1)
    c.ry(1, theta=0.2)
    pass
