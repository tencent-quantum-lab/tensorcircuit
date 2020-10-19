import sys

sys.path.insert(0, "../")

import numpy as np
import tensorflow as tf
import tensorcircuit as tc
from tensorcircuit.applications.dqas import (
    parallel_qaoa_train,
    single_generator,
    set_op_pool,
)
from tensorcircuit.applications.layers import *
from tensorcircuit.applications.graphdata import get_graph

tc.set_backend("tensorflow")

set_op_pool([Hlayer, rxlayer, rylayer, rzlayer, xxlayer, yylayer, zzlayer])
if __name__ == "__main__":
    parallel_qaoa_train(
        [0, 6, 1, 6, 1],
        single_generator(get_graph("8B")),
        tries=4,
        cores=2,
        batch=1,
        epochs=5,
        scale=0.8,
    )
