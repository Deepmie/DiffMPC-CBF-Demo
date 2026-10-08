# Modernization notes

This source tree is a compatibility update of qpth 0.0.18 for modern NumPy
2.x and PyTorch 2.x environments.

## Main changes

- Removed the `numpy<2` package restriction and declared NumPy 2.x support.
- Added an explicit PyTorch dependency and modern PEP 517 build metadata.
- Replaced legacy single-problem linear algebra (`potrf`, `potrs`, `gesv`)
  with wrappers around the maintained `torch.linalg` batched implementation.
- Reworked `SpQPFunction` so it no longer subclasses `autograd.Function` as an
  instance or depends on the removed `torch.spbqrfactsolve` API.
- Replaced legacy sparse tensor constructors with `torch.sparse_coo_tensor`.
- Made the CVXPY backend optional and lazy-loaded.
- Replaced `.data`, `Variable`, device-unsafe empty tensors, and unsafe NumPy
  conversion patterns in maintained Python code.
- Added modern pytest coverage and GitHub Actions CI.

## Compatibility target

The code is intended to be compatible with the current NumPy 2.x / PyTorch
2.x API family, including the stable NumPy 2.5 and PyTorch 2.14 lines current
when this update was prepared.

Local validation in the provided build environment used:

- Python 3.13.5
- NumPy 2.3.5
- PyTorch 2.10.0+cpu

Validation performed:

- `python -m compileall`
- 6 pytest regression tests (forward, backward, batched, equality constraints,
  float32, sparse compatibility, NumPy conversion)
- wheel build with local PEP 517 tooling
- installation of the built wheel into an isolated target directory
- forward/backward smoke test from the installed wheel

The CI workflow installs current dependency releases, so a GitHub run can be
used to continuously verify future NumPy/PyTorch updates.
