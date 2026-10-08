# qpth • [![Build Status][travis-image]][travis] [![PyPi][pypi-image]][pypi] [![License][license-image]][license]

[travis-image]: https://travis-ci.org/locuslab/qpth.png?branch=master
[travis]: http://travis-ci.org/locuslab/qpth

[pypi-image]: https://img.shields.io/pypi/v/qpth.svg
[pypi]: https://pypi.python.org/pypi/qpth

[license-image]: http://img.shields.io/badge/license-Apache--2-blue.svg?style=flat
[license]: LICENSE

*A fast and differentiable QP solver for PyTorch.
Crafted by [Brandon Amos](http://bamos.github.io) and
[J. Zico Kolter](http://zicokolter.com).*

---

+ [More details are available on our project website here](http://locuslab.github.io/qpth)

## Modern NumPy / PyTorch compatibility

This fork has been updated for current NumPy 2.x and PyTorch 2.x releases.
The main dense PDIPM solver uses `torch.linalg` APIs and works with ordinary
CPU/CUDA tensors.  The historical sparse API is kept through `SpQPFunction`,
but its removed `torch.spbqrfactsolve` dependency has been replaced by a
COO-to-dense compatibility path that remains differentiable with respect to
sparse values.

### Installation

```bash
pip install .
```

The native PyTorch solver does not require CVXPY.  Install the optional
reference backend with:

```bash
pip install '.[cvxpy]'
```

Run the compatibility tests with:

```bash
pip install '.[test]'
pytest -q
```

The package metadata accepts NumPy `>=2.0,<3` and PyTorch `>=2.0`, and the
compatibility update was written against the modern `torch.linalg` API used by
current PyTorch releases.
