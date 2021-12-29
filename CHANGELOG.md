# Change Log

## Unreleased

### Added

- add `solve` method on backends to solve linear equations

- add full quantum natural gradient examples and `qng` method in experimental module

- add `concat` method to backends

- add `stop_gradient` method to backends

- add `has_aux` arg on `vvag` method

- add `imag` method on backends

### Changed

- contractor, dtype and backend set are default to return objects, `with tc.runtime_backend("jax") as K` or `K = tc.set_backend("jax")` could work

- change `perfect_sampling` to use `measure_jit` behind the scene

- `anygate` automatically reshape the unitary input to 2-d leg for users' good

- `quantum.renyi_entropy` computation with correct prefactor

### Fixed

- make four algorithms for quantum natural gradient consistent and correct

- torch `real` is now a real

## 0.0.211223

### Added

- add `quantum.heisenberg_hamiltonian` for hamiltonian generation shortcut

- add `has_aux` parameter in backend methods `grad` and `value_and_grad`, the semantic syntax is the same as jax

- add `optimizer` class on tensorflow and jax backend, so that a minimal and unified backend agnostic optimizer interface is provided

- add `quantum.mutual_information`, add support on mixed state for `quantum.reduced_density_matrix`

- add `jvp` methods for tensorflow, jax, torch backends, and ensure pytree support in `jvp` and `vjp` interfaces for tensorflow and jax backends; also ensure complex support for `jvp` and `vjp`

- add `jacfwd` and `jacrev` for backend methods (experimental API, may have bugs and subject to changes)

### Fixed

- fix `matmul` bug on tensornetwork tensorflow backend

### Changed

- delete `qcode` IR for `Circuit`, use `qir` instead (breaking changes)

- basic circuit running is ok on pytorch backend with some complex support fixing

## 0.0.211216

### Added

- add `get_random_state` and `random_split` methods to backends

- add qir representation of circuit, `c.to_qir()` and `Circuit.from_qir()` methods

- fine-grained control on backend, dtype and contractor setup: `tc.set_function_backend()` for function level decorator and `tc.runtime_backend()` as with context manager

- add `state_centric` decorator in `tc.templates.blocks` to transform circuit-to-circuit funtion to state-to-state function

- add `interfaces.scipy_optimize_interface` to transform quantum function into `scipy.optimize.minimize` campatible form

### Fixed

- avoid error on watch non `tf.Tensor` in tensorflow backend grad method

- circuit preprocessing simplification with only single qubit gates

- avoid the bug when random from jax backend with jitted function

- refresh the state cache in Circuit when new gate is applied

### Changed

- refactor `tc.gates` (breaking API on `rgate` -> `r_gate`, `iswapgate` -> `iswap_gate`)
