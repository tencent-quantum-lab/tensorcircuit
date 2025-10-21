import sys
import os
import matplotlib.pyplot as plt

# Add the directory containing your module to Python's search path
module_path = ".."
sys.path.insert(0, module_path)

import tensorcircuit as tc
from tensorcircuit.cloud.apis import submit_task, get_device, set_provider, set_token
from library.adder import gen_adder_circ
from library.aqft import gen_aqft_circ
from library.bv import gen_bv_circ
from library.grover import gen_grover_circ
from library.qft_adder import qft_adder
from library.uccsd  import gen_uccsd
from library.hwea  import gen_hwea_circ

set_token(os.getenv("TOKEN"))
shots = 8192
mit = tc.results.rem.ReadoutMit('tianji_s2?o=7') 
mit.cals_from_system(13, shots, method='local')
set_provider("tencent")
d = get_device("tianji_s2")
submit_task(
    circuit=gen_adder_circ(3),
    shots=shots,
    device=d
)
submit_task(
    circuit=gen_grover_circ(3),
    shots=shots,
    device=d
)
# submit_task(
#     circuit=dynamics_circuit(4, 1.0, 0.1, 3),
#     shots=shots,
#     device=get_device("tianji_s2"),
# )
submit_task(
    circuit=gen_bv_circ(9, 4),
    shots=shots,
    device=d
)
submit_task(
    circuit=gen_uccsd(4),
    shots=shots,
    device=d
)
qc = qft_adder(3, 2, 5)
print(qc)
submit_task(
    circuit=qc,
    shots=shots,
    device=d
)
submit_task(
    circuit=gen_hwea_circ(4, 2),
    shots=shots,
    device=d
)   
submit_task(
    circuit=gen_aqft_circ(4),
    shots=shots, 
    device=d
)
print("✅ TEST FILE DONE")