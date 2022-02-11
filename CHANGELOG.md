# Change Log

## Unreleased

### Added

- new universal contraction analyse tools and pseudo contraction rehearsals for debug

## 0.0.220126

### Added

- add `td` and `sd` gates for dagger version of T gate and S gate

- add `argmax` and `argmin` as backend methods

- add `expectation_before` methods for `tc.Circuit` for further manipulation on the tensornetwork

### Changed

- refined repr for `tc.gates.Gate`

- expectation API now supports int index besides list indexes

### Fixed

- make consistent `Gate` return for channels

- fixed bug on list optimizer for contraction

- stability for QR operator in terms of automatic differentiation

## 0.0.220118

### Added

- add `hessian` method on backends

- add further automatic pipelines for visualization by generating pdf or images

- add `reshape2` method on backends as a short cut to reshape a tensor with all legs 2-d

- add `reshapem` method on backends to reshape any tensor as a square matrix

- add `controlled` and `ocontrolled` API to generate more gates

- add `crx`, `cry`, `crz` gate on `Circuit`

- add `__repr__` and `__str__` for backend object

- `tc.expectation` now support ket arg as quvector form

### Fixed

- `sizen` correctly returns 1 for tensor of no shape

- fixed `convert_to_tensor` bug in numpy backend in TensorNetwork

- `any_gate` also support Gate format instead of matrix

- `matrix_for_gate` works now for backends more than numpy

### Changed

- `expectation` API now also accepts plain tensor instead of `tc.Gate`.

- `DMCircuit` and `DMCircuit2` are all pointing the efficent implementations (breaking changes)

## 0.0.220106

### Added

- add `solve` method on backends to solve linear equations

- add full quantum natural gradient examples and `qng` method in experimental module

- add `concat` method to backends

- add `stop_gradient` method to backends

- add `has_aux` arg on `vvag` method

- add `imag` method on backends

- add `Circuit.vis_tex` interface that returns the quantikz circuit latex

### Changed

- contractor, dtype and backend set are default to return objects, `with tc.runtime_backend("jax") as K` or `K = tc.set_backend("jax")` could work

- change `perfect_sampling` to use `measure_jit` behind the scene

- `anygate` automatically reshape the unitary input to 2-d leg for users' good

- `quantum.renyi_entropy` computation with correct prefactor

- `Circuit` gate can provided other names by name attr

- `example_block` support param auto reshape for users' good

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
