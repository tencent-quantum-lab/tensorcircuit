# Change Log

## Unreleased

### Added

- add `quantum.heisenberg_hamiltonian` for hamiltonian generation shortcut

- add `has_aux` parameter in backend methods `grad` and `value_and_grad`, the semantic syntax is the same as jax

- add `optimizer` class on tensorflow and jax backend, so that a minimal and unified backend agnostic optimizer interface is provided

- add `quantum.mutual_information`, add support on mixed state for `quantum.reduced_density_matrix`

- add `jvp` methods for tensorflow, jax, torch backends, and ensure pytree support in `jvp` and `vjp` interfaces for tensorflow and jax backends; also ensure complex support for `jvp` and `vjp`

- add `jacfwd` and `jacrev` for backend methods (experimental API, may have bugs and subject to changes)

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
