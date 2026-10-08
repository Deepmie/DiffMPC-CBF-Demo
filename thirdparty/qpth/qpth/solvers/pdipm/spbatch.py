"""Compatibility implementation of the old sparse batched backend.

Older qpth releases depended on ``torch.spbqrfactsolve``, an internal sparse
CUDA routine that is no longer part of modern PyTorch.  This module preserves
the public ``forward`` entry point by densifying the COO matrices and using the
maintained dense PDIPM solver.  The user-facing ``qpth.qp.SpQPFunction`` uses
the same strategy and remains differentiable with respect to sparse values.
"""

import torch

from . import batch as pdipm_b


def _dense(indices, values, size):
    indices = indices.to(device=values.device, dtype=torch.long)
    return torch.stack(
        [
            torch.sparse_coo_tensor(
                indices,
                row,
                size=torch.Size(size),
                dtype=values.dtype,
                device=values.device,
            ).coalesce().to_dense()
            for row in values
        ],
        dim=0,
    )


def forward(
    Qi,
    Qv,
    Qsz,
    p,
    Gi,
    Gv,
    Gsz,
    h,
    Ai,
    Av,
    Asz,
    b,
    eps=1e-12,
    verbose=0,
    notImprovedLim=3,
    maxIter=20,
):
    Q = _dense(Qi, Qv, Qsz)
    G = _dense(Gi, Gv, Gsz)
    A = _dense(Ai, Av, Asz)
    Q_LU, S_LU, R = pdipm_b.pre_factor_kkt(Q, G, A)
    return pdipm_b.forward(
        Q,
        p,
        G,
        h,
        A,
        b,
        Q_LU,
        S_LU,
        R,
        eps=eps,
        verbose=verbose,
        notImprovedLim=notImprovedLim,
        maxIter=maxIter,
    )


def cat_kkt(*args, **kwargs):
    raise RuntimeError(
        "qpth's legacy sparse QR KKT internals were removed because modern PyTorch "
        "no longer provides torch.spbqrfactsolve. Use SpQPFunction or spbatch.forward instead."
    )


def solve_kkt(*args, **kwargs):
    raise RuntimeError(
        "qpth's legacy sparse QR KKT internals were removed because modern PyTorch "
        "no longer provides torch.spbqrfactsolve. Use SpQPFunction or the dense PDIPM backend."
    )
