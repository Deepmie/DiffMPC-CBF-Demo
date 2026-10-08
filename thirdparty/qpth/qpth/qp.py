from enum import Enum

import torch
from torch.autograd import Function

from .util import bger, expandParam, extract_nBatch
from .solvers.pdipm import batch as pdipm_b


class QPSolvers(Enum):
    PDIPM_BATCHED = 1
    CVXPY = 2


def QPFunction(
    eps=1e-12,
    verbose=0,
    notImprovedLim=3,
    maxIter=20,
    solver=QPSolvers.PDIPM_BATCHED,
    check_Q_spd=True,
):
    class QPFunctionFn(Function):
        @staticmethod
        def forward(ctx, Q_, p_, G_, h_, A_, b_):
            r"""Solve a batch of differentiable quadratic programs.

            Each problem has the form

                z* = argmin_z  1/2 z^T Q z + p^T z
                     s.t.      G z <= h
                               A z  = b

            Parameters may be batched or shared across a batch.  Empty A/b
            tensors represent the absence of equality constraints.
            """
            nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_)
            Q, _ = expandParam(Q_, nBatch, 3)
            p, _ = expandParam(p_, nBatch, 2)
            G, _ = expandParam(G_, nBatch, 3)
            h, _ = expandParam(h_, nBatch, 2)
            A, _ = expandParam(A_, nBatch, 3)
            b, _ = expandParam(b_, nBatch, 2)

            if check_Q_spd:
                _, info = torch.linalg.cholesky_ex(Q)
                if torch.any(info != 0):
                    raise RuntimeError("Q is not SPD.")

            if G.ndim != 3:
                raise RuntimeError("G must contain at least one inequality constraint.")
            _, nineq, nz = G.shape
            neq = A.size(1) if A.numel() > 0 else 0
            if nineq <= 0:
                raise RuntimeError("The PDIPM backend requires at least one inequality constraint.")
            ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz

            if solver == QPSolvers.PDIPM_BATCHED:
                ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A)
                zhats, ctx.nus, ctx.lams, ctx.slacks = pdipm_b.forward(
                    Q,
                    p,
                    G,
                    h,
                    A,
                    b,
                    ctx.Q_LU,
                    ctx.S_LU,
                    ctx.R,
                    eps,
                    verbose,
                    notImprovedLim,
                    maxIter,
                )
            elif solver == QPSolvers.CVXPY:
                from .solvers import cvxpy as cvxpy_solver

                vals = Q.new_empty(nBatch)
                zhats = Q.new_empty((nBatch, nz))
                lams = Q.new_empty((nBatch, nineq))
                nus = Q.new_empty((nBatch, neq)) if neq > 0 else Q.new_empty((nBatch, 0))
                slacks = Q.new_empty((nBatch, nineq))

                for i in range(nBatch):
                    Ai, bi = (A[i], b[i]) if neq > 0 else (None, None)
                    args = tuple(
                        x.detach().cpu().numpy() if x is not None else None
                        for x in (Q[i], p[i], G[i], h[i], Ai, bi)
                    )
                    value, zhati, nui, lami, si = cvxpy_solver.forward_single_np(*args)
                    vals[i] = value
                    zhats[i].copy_(torch.as_tensor(zhati, device=Q.device, dtype=Q.dtype))
                    lams[i].copy_(torch.as_tensor(lami, device=Q.device, dtype=Q.dtype))
                    slacks[i].copy_(torch.as_tensor(si, device=Q.device, dtype=Q.dtype))
                    if neq > 0:
                        nus[i].copy_(torch.as_tensor(nui, device=Q.device, dtype=Q.dtype))

                ctx.vals = vals
                ctx.lams = lams
                ctx.nus = nus
                ctx.slacks = slacks
            else:
                raise ValueError(f"Unknown QP solver: {solver!r}")

            ctx.save_for_backward(zhats, Q_, p_, G_, h_, A_, b_)
            return zhats, ctx.nus, ctx.lams

        @staticmethod
        def backward(ctx, dl_dzhat, dl_dnu, dl_dlam):
            zhats, Q, p, G, h, A, b = ctx.saved_tensors
            nBatch = extract_nBatch(Q, p, G, h, A, b)
            Q, Q_e = expandParam(Q, nBatch, 3)
            p, p_e = expandParam(p, nBatch, 2)
            G, G_e = expandParam(G, nBatch, 3)
            h, h_e = expandParam(h, nBatch, 2)
            A, A_e = expandParam(A, nBatch, 3)
            b, b_e = expandParam(b, nBatch, 2)

            neq, nineq = ctx.neq, ctx.nineq
            if solver == QPSolvers.CVXPY:
                ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A)

            d = torch.clamp(ctx.lams, min=1e-8) / torch.clamp(ctx.slacks, min=1e-8)
            pdipm_b.factor_kkt(ctx.S_LU, ctx.R, d)

            zeros_ineq = G.new_zeros((nBatch, nineq))
            zeros_eq = G.new_zeros((nBatch, neq)) if neq > 0 else None
            dx, _, dlam, dnu = pdipm_b.solve_kkt(
                ctx.Q_LU,
                d,
                G,
                A,
                ctx.S_LU,
                dl_dzhat,
                zeros_ineq,
                zeros_ineq,
                zeros_eq,
            )

            dps = dx
            dGs = bger(dlam, zhats) + bger(ctx.lams, dx)
            if G_e:
                dGs = dGs.mean(0)
            dhs = -dlam
            if h_e:
                dhs = dhs.mean(0)

            if neq > 0:
                dAs = bger(dnu, zhats) + bger(ctx.nus, dx)
                dbs = -dnu
                if A_e:
                    dAs = dAs.mean(0)
                if b_e:
                    dbs = dbs.mean(0)
            else:
                dAs, dbs = None, None

            dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
            if Q_e:
                dQs = dQs.mean(0)
            if p_e:
                dps = dps.mean(0)

            return dQs, dps, dGs, dhs, dAs, dbs

    return QPFunctionFn.apply


def _sparse_values_to_dense(indices, values, size):
    """Convert shared COO indices + batched values to a dense batch.

    torch.sparse_coo_tensor -> to_dense is differentiable with respect to the
    values, so gradients still flow back to Qv/Gv/Av.
    """
    indices = indices.to(device=values.device, dtype=torch.long)
    if values.ndim == 1:
        return torch.sparse_coo_tensor(
            indices, values, size=size, dtype=values.dtype, device=values.device
        ).coalesce().to_dense()
    return torch.stack(
        [
            torch.sparse_coo_tensor(
                indices, v, size=size, dtype=values.dtype, device=values.device
            ).coalesce().to_dense()
            for v in values
        ],
        dim=0,
    )


class SpQPFunction:
    """Compatibility wrapper for the historical sparse qpth API.

    Old qpth subclassed ``torch.autograd.Function`` as an instance and relied
    on the removed ``torch.spbqrfactsolve`` CUDA API.  Modern PyTorch no longer
    supports that path.  This wrapper keeps the same public call syntax while
    reconstructing dense tensors and delegating to the maintained batched
    PDIPM implementation.  Gradients with respect to sparse values are kept.
    """

    def __init__(
        self,
        Qi,
        Qsz,
        Gi,
        Gsz,
        Ai,
        Asz,
        eps=1e-12,
        verbose=0,
        notImprovedLim=3,
        maxIter=20,
        check_Q_spd=True,
    ):
        self.Qi, self.Qsz = Qi, torch.Size(Qsz)
        self.Gi, self.Gsz = Gi, torch.Size(Gsz)
        self.Ai, self.Asz = Ai, torch.Size(Asz)
        self.qp = QPFunction(
            eps=eps,
            verbose=verbose,
            notImprovedLim=notImprovedLim,
            maxIter=maxIter,
            solver=QPSolvers.PDIPM_BATCHED,
            check_Q_spd=check_Q_spd,
        )

    def __call__(self, Qv, p, Gv, h, Av, b):
        Q = _sparse_values_to_dense(self.Qi, Qv, self.Qsz)
        G = _sparse_values_to_dense(self.Gi, Gv, self.Gsz)
        A = _sparse_values_to_dense(self.Ai, Av, self.Asz)
        return self.qp(Q, p, G, h, A, b)
