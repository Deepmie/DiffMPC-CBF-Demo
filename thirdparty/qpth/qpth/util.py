import numpy as np
import torch


def print_header(msg):
    print("===>", msg)


def to_np(t):
    if t is None:
        return None
    if t.numel() == 0:
        return np.array([])
    return t.detach().cpu().numpy()


def bger(x, y):
    """Batched outer product."""
    return x.unsqueeze(2).bmm(y.unsqueeze(1))


def get_sizes(G, A=None):
    if G.dim() == 2:
        nineq, nz = G.shape
        nBatch = 1
    elif G.dim() == 3:
        nBatch, nineq, nz = G.shape
    else:
        raise RuntimeError(f"G must be 2-D or 3-D, got shape {tuple(G.shape)}")

    if A is not None:
        neq = A.size(1) if A.numel() > 0 and A.dim() == 3 else (
            A.size(0) if A.numel() > 0 and A.dim() == 2 else 0
        )
    else:
        neq = None
    return nineq, nz, neq, nBatch


def bdiag(d):
    """Create a batch of diagonal matrices from a (batch, n) tensor."""
    return torch.diag_embed(d)


def expandParam(X, nBatch, nDim):
    if X.ndim in (0, nDim) or X.numel() == 0:
        return X, False
    if X.ndim == nDim - 1:
        return X.unsqueeze(0).expand(*([nBatch] + list(X.shape))), True
    raise RuntimeError(
        f"Unexpected number of dimensions: expected {nDim - 1} or {nDim}, got {X.ndim}."
    )


def extract_nBatch(Q, p, G, h, A, b):
    dims = (3, 2, 3, 2, 3, 2)
    for param, dim in zip((Q, p, G, h, A, b), dims):
        if param.ndim == dim:
            return param.size(0)
    return 1
