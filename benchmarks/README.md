# benchmark4tc

`cd scripts`

`python benchmark.py -n [# of Qubits] -nlayer [# of QC layers] -nitrs [# of max iterations] -nbatch [# of batch for QML task] -t [time limitation] -gpu [0 for no gpu and 1 for gpu enabled] -tcbackend [jax or tensorflow]`

then a `.json` file will be created in data folder which contains the information of benchmarking parameters and results.

Since tensorcircuit may be installed in a local dir, you may have to firstly set in terminal: `export PYTHONPATH=/abs/path/for/tc`.
