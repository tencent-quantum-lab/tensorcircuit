API comparison on the same standard variational circuit evaluation task demonstrating the advantage of TensorCircuit API design.

* QML subtask refers building a keras model of quantum circuit.

* VQE subtask refers getting energy and circuit gradients.

| # Lines (# Packages) | TensorFlow Quantum | Pennylane | TensorCircuit |
| :------------------: | :----------------: | :-------: | :-----------: |
|     QML subtask      |       32 (5)       |  18 (2)   |    16 (1)     |
|     VQE subtask      |       47 (5)       |  29 (2)   |    20 (1)     |

