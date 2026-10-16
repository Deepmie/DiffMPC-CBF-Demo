"""Locate where the revised DiffMPCBF hits 'Q is not SPD' and test which factor causes it."""
import sys, os, io, contextlib, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'spnn_ex'))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')
import numpy as np
import torch
from diffmpc_cbf import DiffMPCBF
from env import Env
from check_cbf_v2 import make, P, HessFixMPCBF, hval, nx, nu, T, dt, xref, obs, quiet

LOG = []


def logged(cls, eps_override=None):
    class L(cls):
        def _build_sqp_params(self, x0, _tau, dveq, dvneq, params):
            Q, p, G, h, A, b = super()._build_sqp_params(x0, _tau, dveq, dvneq, params)
            if eps_override is not None:
                e = torch.linalg.eigvalsh(Q).min()
                if e < eps_override: Q = Q + (eps_override - e)*torch.eye(Q.shape[0])
            ev = torch.linalg.eigvalsh(Q)
            chol_ok = torch.linalg.cholesky_ex(Q)[1].item() == 0
            LOG.append(dict(eig_min=ev.min().item(), eig_max=ev.max().item(), chol=chol_ok,
                            nu_cbf=dvneq[:T].abs().max().item(), lam=dveq.abs().max().item(),
                            tau_max=_tau.tau.abs().max().item() if hasattr(_tau, 'tau') else float('nan')))
            return Q, p, G, h, A, b
    return L


def run(cls, tag, N=20, eps_override=None):
    LOG.clear()
    mpc, cost, dyn, bar = make(logged(cls, eps_override))
    x0 = torch.tensor([[0.], [0.]])
    env = Env(x0.numpy(), xref.numpy(), obs, delta_t=dt)
    x = x0.clone(); hs = []
    for t in range(1, N):
        n0 = len(LOG)
        try:
            with quiet():
                tau = mpc.step(x, P())
        except RuntimeError as e:
            print(f'[{tag}] FAILED at MPC step {t} ({e}); x={np.round(x.flatten().numpy(),4).tolist()}, h={hval(bar, x):.4f}')
            for k, r in enumerate(LOG[n0:]):
                print(f'     sqp it {k:2d}: eig_min={r["eig_min"]:.2e} eig_max={r["eig_max"]:.2e} chol_ok={r["chol"]} '
                      f'|nu_cbf|={r["nu_cbf"]:.2e} |lam|={r["lam"]:.2e} |tau|={r["tau_max"]:.2e}')
            return
        x = torch.tensor(env.step(tau.u[0].flatten().detach().numpy()))
        hs.append(hval(bar, x))
    print(f'[{tag}] ok, final x={np.round(x.flatten().numpy(),4).tolist()}, min h={min(hs):.4f}')
    print('   h:', np.round(hs, 3).tolist())


if __name__ == '__main__':
    run(DiffMPCBF, 'user, eps=1e-6')
    run(DiffMPCBF, 'user, eps=1e-4', eps_override=1e-4)
    run(HessFixMPCBF, 'Hessian fixed, eps=1e-6')
    run(HessFixMPCBF, 'Hessian fixed, eps=1e-4', eps_override=1e-4)
