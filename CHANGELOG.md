# Change Log

## Unreleased

### Added

- add `get_random_state` and `random_split` methods to backends

- add qir representation of circuit, `c.to_qir()` and `Circuit.from_qir()` methods

### Fixed

- avoid error on watch non `tf.Tensor` in tensorflow backend grad method

- circuit preprocessing simplification with only single qubit gates

- avoid the bug when random from jax backend with jitted function

### Changed

- refactor `tc.gates` (breaking API on `rgate` -> `r_gate`, `iswapgate` -> `iswap_gate`)
