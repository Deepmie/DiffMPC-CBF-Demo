"""Is the SQP fixed point converged, and are FD gradients stable? (user code untouched)"""
import sys, os, copy, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'spnn_ex'))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')
import numpy as np
import torch
from diffmpc_cbf import DiffMPCBF, Tau
from qpth.qp import QPFunction
from check_cbf_v2 import make, P, eps4, hval, quiet, T

STATES = {'far from obstacle': [0.0, -0.3], 'near obstacle': [0.1978, 0.3321]}


def last_qp_info(mpc, x):
    """Re-solve the final QP at tau_star and report step size / multipliers of the CBF rows."""
    p = P(); p['dynamic'] = {}
    with torch.no_grad():
        Q, q, G, h, A, b = mpc._build_sqp_params(x, mpc._tau, torch.zeros((T+1)*2), torch.zeros(T*3), p)
        with quiet():
            z, lam, nu = QPFunction(eps=1e-12, verbose=0, maxIter=20)(Q, q, G, h, A, b)
    slack = (h - G @ z.flatten())[:T]
    return z.abs().max().item(), nu.flatten()[:T], slack


def u0_of(mpc, x, base_tau):
    m = copy.deepcopy(mpc); m._tau = copy.deepcopy(base_tau)
    with quiet(), torch.no_grad():
        return m.step(x.clone(), P()).u[0].flatten()[0].item()


for name, s in STATES.items():
    torch.manual_seed(1)
    mpc, cost, dyn, bar = make(eps4(DiffMPCBF))
    x = torch.tensor(s).reshape(2, 1)
    for _ in range(3):                       # warm start a few times
        with quiet(): mpc.step(x.clone(), P())
    base = copy.deepcopy(mpc._tau)
    zmax, nu_cbf, slack = last_qp_info(mpc, x)
    print(f'=== {name}: x={s}, h={hval(bar, x):.4f} ===')
    print(f'   |delta tau| of final QP at tau_star = {zmax:.2e}   (should be ~0 if SQP converged)')
    print(f'   CBF multipliers = {np.round(nu_cbf.numpy(), 4).tolist()}')
    print(f'   CBF slacks      = {np.round(slack.numpy(), 5).tolist()}')
    ua = [u0_of(mpc, x, base) for _ in range(2)]
    print(f'   repeat u0: {ua}')
    params = list(cost.context_network.parameters())
    flat = torch.cat([p.detach().flatten() for p in params])
    for i in (8, 56):
        fds = []
        for e in (3e-2, 1e-2, 3e-3, 1e-3):
            v = []
            for sg in (1, -1):
                f = flat.clone(); f[i] += sg*e
                torch.nn.utils.vector_to_parameters(f, params)
                v.append(u0_of(mpc, x, base))
            torch.nn.utils.vector_to_parameters(flat, params)
            fds.append((v[0]-v[1])/(2*e))
        m = copy.deepcopy(mpc); m._cost = cost; m._tau = copy.deepcopy(base)
        with quiet(): out = m.step(x.clone(), P()).u[0].flatten()[0]
        g = torch.autograd.grad(out, params)
        ga = torch.cat([gg.flatten() for gg in g])[i].item()
        print(f'   param {i}: autograd={ga:+.5f}  FD(eps=3e-2,1e-2,3e-3,1e-3)={np.round(fds, 5).tolist()}')
