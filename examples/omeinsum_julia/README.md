# Use OMEinsum in TensorCircuit

This example introduces how to use OMEinsum, a julia-based einsum package, to contract a circuit in TensorCircuit.

We provide two solutions 

* use subprocess to call a stand-alone julia script (recommended)
* use juliacall to integrate julia script into python (seems to be more elegant, but not recommended)

## Subprocess solution (Recommended)

This solution calls a stand-alone julia script `omeinsum.jl` for tensor network contraction. 

### Setup

* Step 1: install julia, see https://julialang.org/download/. Please install julia >= 1.8.5, the 1.6.7 LTS version raises: `Error in python: free(): invalid pointer`
* Step 2: add julia path to the PATH env variable so that we can find it
* Step 3: install julia package `OMEinsum`, `ArgParse` and `JSON`, this example was tested with OMEinsum v0.7.2, ArgParse v1.1.4 and JSON v0.21.3. See https://docs.julialang.org/en/v1/stdlib/Pkg/ for more details on julia's package manager

### How to run

Run
`JULIA_NUM_THREADS=N python omeinsum_contractor_subprocess.py`. The env variable `JULIA_NUM_THREADS=N` will be passed to the julia script, so that you can enjoy the accelaration brought by julia multi-threading.


## JuliaCall solution (Not Recommended)

JuliaCall seems to be a more elegant solution because all related code are integrated into a single python script.
However, in order to use julia multi-threading in juliacall, we have to turn off julia GC at the risk of OOM. See see https://github.com/cjdoris/PythonCall.jl/issues/219 for more details.


### Setup

* Step 1: install julia, see https://julialang.org/download/. Please install julia >= 1.8.5, the 1.6.7 LTS version raises: `Error in python: free(): invalid pointer`
* Step 2: add julia path to the PATH env variable so that juliacall can find it
* Step 3: install juliacall via `pip install juliacall`, this example was tested with juliacall 0.9.9
* Step 4: install julia package `OMEinsum`, this example was tested with OMEinsum v0.7.2, see https://docs.julialang.org/en/v1/stdlib/Pkg/ for more details on julia's package manager
* Step 5: for julia multi-threading, set env variable `PYTHON_JULIACALL_THREADS=<N|auto>`. 

### How to run

Run
`PYTHON_JULIACALL_THREADS=<N|auto> python omeinsum_contractor_juliacall.py`.


