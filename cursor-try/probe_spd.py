"""Probe why QPFunction reports Q is not SPD in spnn_ex."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "spnn_ex"))

import torch
from qpth.util import expandParam, extract_nBatch
from qpth.qp import QPFunction as OrigQPFunction

import diffmpc_cbf.diffmpc as dm
from system import SpnnCost, SpnnDynamic
from diffmpc_cbf import DiffMPC
from diffmpc_cbf.lqr import DiffMPC as iLQRDiffMPC


_call = 0


def QPFunction(eps=1e-12, verbose=0, maxIter=20, **kwargs):
    real = OrigQPFunction(eps=eps, verbose=verbose, maxIter=maxIter, **kwargs)

    def apply(Q, p, G, h, A, b):
        global _call
        _call += 1
        print(f"\n=== QP call {_call} ===")
        print("shapes", Q.shape, p.shape, G.shape, h.shape, A.shape, b.shape)
        print("dtypes", Q.dtype, p.dtype)
        print("Q.requires_grad", Q.requires_grad, "Q.is_contiguous", Q.is_contiguous())
        print("sym err", (Q - Q.T).abs().max().item())
        eigs = torch.linalg.eigvalsh(Q)
        print("eig_min/max", eigs.min().item(), eigs.max().item())

        nBatch = extract_nBatch(Q, p, G, h, A, b)
        Qb, _ = expandParam(Q, nBatch, 3)
        print("nBatch", nBatch, "Qb.shape", Qb.shape, "Qb.contiguous", Qb.is_contiguous(), "stride", Qb.stride())
        _, info_b = torch.linalg.cholesky_ex(Qb)
        _, info_c = torch.linalg.cholesky_ex(Qb.contiguous())
        _, info_q = torch.linalg.cholesky_ex(Q)
        print("info expand/contig/raw", info_b, info_c, info_q)

        # Match qpth check exactly
        _, info = torch.linalg.cholesky_ex(Qb)
        print("torch.any(info!=0)", torch.any(info != 0).item(), "info", info)

        try:
            return real(Q, p, G, h, A, b)
        except Exception as e:
            print("OrigQPFunction raised:", e)
            # Show block eigenvalues of Q
            ntau, T, nx = 3, 5, 2
            for t in range(T):
                blk = Q[t * ntau : (t + 1) * ntau, t * ntau : (t + 1) * ntau]
                print(f" block[{t}] eigs", torch.linalg.eigvalsh(blk).tolist())
            term = Q[T * ntau : T * ntau + nx, T * ntau : T * ntau + nx]
            print(" terminal eigs", torch.linalg.eigvalsh(term).tolist())
            raise

    return apply


dm.QPFunction = QPFunction

torch.manual_seed(1)
nx, nu, T, N = 2, 1, 5, 50
cost = SpnnCost(nx, nu, T)
dynamic = SpnnDynamic(nx, nu, T, 0.1)
umin = torch.tensor([-10.0])
umax = torch.tensor([10.0])
xref = torch.tensor([[1.0], [0.0]], requires_grad=True)
mpc = DiffMPC(nx, nu, T, 1, cost, dynamic, umin, umax)
_ = iLQRDiffMPC(nx, nu, T, 1, cost, dynamic, umin, umax)
x0 = torch.tensor([[0.0], [0.0]])
xs = torch.zeros([N, nx])
xs[0] = x0.flatten()
tau = mpc.step(xs[0], {"cost": {"xref": xref}})
print("SUCCESS", tau.u.flatten())
