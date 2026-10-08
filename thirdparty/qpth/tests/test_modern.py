"""Regression tests for modern NumPy / PyTorch compatibility."""

import numpy as np
import torch

from qpth.qp import QPFunction, SpQPFunction
from qpth.util import to_np


def _empty_eq(dtype, device="cpu"):
    return (
        torch.empty((0, 2), dtype=dtype, device=device),
        torch.empty((0,), dtype=dtype, device=device),
    )


def test_dense_unbatched_forward_backward():
    dtype = torch.float64
    Q = torch.eye(2, dtype=dtype)
    p = torch.tensor([-1.0, -2.0], dtype=dtype, requires_grad=True)
    G = -torch.eye(2, dtype=dtype)
    h = torch.zeros(2, dtype=dtype)
    A, b = _empty_eq(dtype)

    x = QPFunction()(Q, p, G, h, A, b)
    torch.testing.assert_close(x, torch.tensor([[1.0, 2.0]], dtype=dtype), atol=1e-6, rtol=1e-6)

    x.sum().backward()
    torch.testing.assert_close(p.grad, torch.tensor([-1.0, -1.0], dtype=dtype), atol=1e-5, rtol=1e-5)


def test_dense_batched_shared_parameters():
    dtype = torch.float64
    Q = torch.eye(2, dtype=dtype)
    p = torch.tensor([[-1.0, -2.0], [-2.0, -1.0]], dtype=dtype, requires_grad=True)
    G = -torch.eye(2, dtype=dtype)
    h = torch.zeros(2, dtype=dtype)
    A, b = _empty_eq(dtype)

    x = QPFunction()(Q, p, G, h, A, b)
    expected = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=dtype)
    torch.testing.assert_close(x, expected, atol=1e-6, rtol=1e-6)

    x.square().sum().backward()
    assert p.grad is not None
    assert torch.isfinite(p.grad).all()


def test_equality_constraint():
    dtype = torch.float64
    Q = torch.eye(2, dtype=dtype)
    p = torch.tensor([-1.0, -1.0], dtype=dtype)
    G = -torch.eye(2, dtype=dtype)
    h = torch.zeros(2, dtype=dtype)
    A = torch.ones((1, 2), dtype=dtype)
    b = torch.ones(1, dtype=dtype)

    x = QPFunction(verbose=-1)(Q, p, G, h, A, b)
    torch.testing.assert_close(x, torch.full((1, 2), 0.5, dtype=dtype), atol=1e-6, rtol=1e-6)


def test_float32_support():
    dtype = torch.float32
    Q = torch.eye(2, dtype=dtype)
    p = torch.tensor([-1.0, -2.0], dtype=dtype)
    G = -torch.eye(2, dtype=dtype)
    h = torch.zeros(2, dtype=dtype)
    A, b = _empty_eq(dtype)

    x = QPFunction(verbose=-1)(Q, p, G, h, A, b)
    torch.testing.assert_close(x, torch.tensor([[1.0, 2.0]], dtype=dtype), atol=2e-4, rtol=2e-4)


def test_sparse_compatibility_wrapper_and_gradients():
    dtype = torch.float64
    Qi = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    Gi = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    Ai = torch.empty((2, 0), dtype=torch.long)

    Qv = torch.ones((1, 2), dtype=dtype, requires_grad=True)
    Gv = torch.full((1, 2), -1.0, dtype=dtype, requires_grad=True)
    Av = torch.empty((1, 0), dtype=dtype, requires_grad=True)
    p = torch.tensor([[-1.0, -2.0]], dtype=dtype, requires_grad=True)
    h = torch.zeros((1, 2), dtype=dtype)
    b = torch.empty((1, 0), dtype=dtype)

    x = SpQPFunction(Qi, (2, 2), Gi, (2, 2), Ai, (0, 2), verbose=-1)(
        Qv, p, Gv, h, Av, b
    )
    torch.testing.assert_close(x, torch.tensor([[1.0, 2.0]], dtype=dtype), atol=1e-6, rtol=1e-6)

    x.sum().backward()
    assert Qv.grad is not None and torch.isfinite(Qv.grad).all()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert Gv.grad is not None and torch.isfinite(Gv.grad).all()


def test_numpy_2_conversion_helper():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    out = to_np(x)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, np.array([1.0, 2.0], dtype=out.dtype))
