# Use OMEinsum in TensorCircuit

This example introduces how to use OMEinsum, a julia-based einsum package, to contract a circuit in TensorCircuit.

We provide two solutions:

* use subprocess to call a stand-alone julia script (recommended)
* use juliacall to integrate julia script into python (seems to be more elegant, but not recommended)

We highly recommend to use the first solution based on subprocess, not only due to its compatibility to julia multi-threading, but also because the experimental KaHyPar-based initialization is developed based on it.

## Experiments

We test contractors from OMEinsum on Google random circuits ([available online](https://datadryad.org/stash/dataset/doi:10.5061/dryad.k6t1rj8)) and compare with the cotengra contractor. 
For circuits only differ in PRNG seed number (which means with the same tensor network structure, but different tenser entries), we choose the one with the largest seed. For example, we benchmark `circuit_n12_m14_s9_e6_pEFGH.qsim`, but skip
circuits like `circuit_n12_m14_s0_e6_pEFGH.qsim`. 
We list experimental results in [benchmark_results.csv](benchmark_results.csv).


Specifically, we test the following three methods:

* **cotengra**: the HyperOptimizer from cotengra. More formally:
```python
opt = ctg.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    parallel=True,
    minimize="flops",
    max_repeats=1024,
    progbar=False,
)
```
* **treesa_greedy**: TreeSA contractor with greedy initialization. More formally:
```python
opt = OMEinsumTreeSAOptimizerSubprocess(
    sc_target=60, sc_weight=0.0, rw_weight=0.0, ntrials=16, niters=64
)
```
* **treesa_kahypar**: TreeSA contractor with kahypar initialization, where the betas hyperparameters (initial temperature for SA) are [suggested by the author of OMEimsum](https://github.com/TensorBFS/OMEinsumContractionOrders.jl/issues/35#issuecomment-1397117653). More formally:
```python
opt = OMEinsumTreeSAOptimizerSubprocess(
    sc_target=60,
    sc_weight=0.0,
    rw_weight=0.0,
    kahypar_init=True,
    ntrials=16,
    niters=64,
    betas=(10, 0.01, 40),
)
```

The above three `opt` are passed to TensorCircuit for contraction, respectively:
```python
tc.set_contractor(
    "custom",
    optimizer=opt,
    preprocessing=True,
    contraction_info=True,
    debug_level=2,
)
```

We compute
```python
c.expectation_ps(z=[0], reuse=False)
```

Both OMEimsum and cotengra are able to optimize a weighted average of `log10[FLOPs]`, `log2[SIZE]` and `log2[WRITE]`.
However, OMEimsum and cotengra have different weight coefficient, which makes fair comparison difficult. 
Thus we force each method to purely optimized `FLOPs`, but we do collect all contraction information in the table, including 
`log10[FLOPs]`, `log2[SIZE]`, `log2[WRITE]`, `PathFindingTime`.

For three circuits, namely `circuit_patch_n46_m14_s19_e21_pEFGH`, `circuit_patch_n44_m14_s19_e21_pEFGH` and `circuit_n42_m14_s9_e0_pEFGH`, we meet [errors in OMEinsum](https://github.com/TensorBFS/OMEinsumContractionOrders.jl/issues/35#issuecomment-1405236778), and there results are set to empty in [benchmark_results.csv](benchmark_results.csv).


## Details about subprocess and JuliaCall solutions
### Subprocess solution (Recommended)

This solution calls a stand-alone julia script [omeinsum.jl](omeinsum.jl) for tensor network contraction. 

#### Setup

* Step 1: install julia, see https://julialang.org/download/. Please install julia >= 1.8.5, the 1.6.7 LTS version raises: `Error in python: free(): invalid pointer`
* Step 2: add julia path to the PATH env variable so that we can find it
* Step 3: install julia package `OMEinsum`, `ArgParse`, `JSON` and `KaHyPar`, this example was tested with OMEinsum v0.7.2, ArgParse v1.1.4, JSON v0.21.3 and KaHyPar v0.3.0. See https://docs.julialang.org/en/v1/stdlib/Pkg/ for more details on julia's package manager

#### How to run

Run
`JULIA_NUM_THREADS=N python omeinsum_contractor_subprocess.py`. The env variable `JULIA_NUM_THREADS=N` will be passed to the julia script, so that you can enjoy the accelaration brought by julia multi-threading.


#### KaHyPar initialization

The choice of initial status plays an important role in simulated annealing.
In a [discussion with the author of OMEinsum](https://github.com/TensorBFS/OMEinsumContractionOrders.jl/issues/35), we
found that there was a way to run TreeSA with initialzier other than greedy or random. We demo how KaHyPar can be used to produce the initial status of simulated annealing. Although we haven't seen significant improvement by using KaHyPar initialization, we believe it is a interesting topic to explore.

### JuliaCall solution (Not Recommended)

JuliaCall seems to be a more elegant solution because all related code are integrated into a single python script.
However, in order to use julia multi-threading in juliacall, we have to turn off julia GC at the risk of OOM. See see [this issue](https://github.com/cjdoris/PythonCall.jl/issues/219) for more details.


#### Setup

* Step 1: install julia (say, from [here](https://julialang.org/download/)). Please install julia >= 1.8.5, the 1.6.7 LTS version raises: `Error in python: free(): invalid pointer`
* Step 2: add julia path to the PATH env variable so that juliacall can find it
* Step 3: install juliacall via `pip install juliacall`, this example was tested with juliacall 0.9.9
* Step 4: install julia package `OMEinsum`, this example was tested with OMEinsum v0.7.2, see [here](https://docs.julialang.org/en/v1/stdlib/Pkg/) for more details on julia's package manager
* Step 5: for julia multi-threading, set env variable `PYTHON_JULIACALL_THREADS=<N|auto>`. 

#### How to run

Run
`PYTHON_JULIACALL_THREADS=<N|auto> python omeinsum_contractor_juliacall.py`.




