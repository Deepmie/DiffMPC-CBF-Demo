"""Solver backends used by qpth.

The CVXPY backend is intentionally not imported eagerly.  This keeps the
native PyTorch PDIPM solver usable without installing the optional CVXPY
stack and avoids importing compiled NumPy extensions unless they are needed.
"""

__all__ = ["cvxpy", "pdipm"]
