"""
Check the revised DiffMPCBF (user code untouched).
  Part 1: closed loop with user's DiffMPCBF + user's Env (same as test_sqp_withcbf.py, without show()).
  Part 2: Env.show() smoke test (Agg backend).
  Part 3: at a state where the CBF is active, compare du0/dtheta (cost network) from
          autograd vs central finite differences, for the user's Hessian and a corrected Hessian.
"""
import sys, os, copy, io, contextlib, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'spnn_ex'))
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import numpy as np
import torch
from diffmpc_cbf import DiffMPCBF
from system import SpnnCost, SpnnDynamic, SpnnBarrier
from env import Env

nx, nu, nb, T, dt = 2, 1, 1, 5, 0.1
umin, umax = torch.tensor([-10.0]), torch.tensor([10.0])
xref = torch.tensor([[1.], [0.]])
obs = [np.array([0.5, 0.5, 0.3, 0.02])]
gamma = 0.5
quiet = lambda: contextlib.redirect_stdout(io.StringIO())


class HessFixMPCBF(DiffMPCBF):
    """User's _build_sqp_params + correction of the CBF Lagrangian Hessian:
    L = f + nu^T g,  g_t = -h(x_{t+1}) + (1-gamma) h(x_t)
      block t   : +nu (1-gamma) H(x_t)      (user has -nu (1-gamma) H(x_t))
      block t+1 : -nu H(x_{t+1})            (missing in user's code)
    """
    def _build_sqp_params(self, x0, _tau, dveq, dvneq, params):
        Hl, Jl, G, h, A, b = super()._build_sqp_params(x0, _tau, dveq, dvneq, params)
        # undo user's eigen shift is not possible exactly, so rebuild the correction on top and re-shift
        nx, ntau, V, T = self._nx, self._ntau, self._var_dims, self._T
        bp = params['barrier']; gm = bp['gamma']
        cx = torch.concat([torch.eye(nx), torch.zeros([self._nu, nx])], dim=0)
        sel = lambda t: cx if t < T else torch.eye(nx)
        blk = lambda t: slice(t*ntau, min((t+1)*ntau, V))
        D = torch.zeros([V, V])
        for t in range(T):
            nu_t = dvneq[t*self._nb:(t+1)*self._nb].reshape(-1, 1, 1)
            H0 = self._barrier.forward(_tau.get_state(t), {**bp, 't': t}, order=2)
            H1 = self._barrier.forward(_tau.get_state(t+1), {**bp, 't': t+1}, order=2)
            D[blk(t), blk(t)] += (2 * nu_t * (1-gm) * sel(t) @ H0 @ sel(t).T).sum(0)   # flip sign
            D[blk(t+1), blk(t+1)] += (-nu_t * sel(t+1) @ H1 @ sel(t+1).T).sum(0)        # add missing
        Hl = Hl + D
        Hl = 0.5*(Hl + Hl.T)
        e = torch.linalg.eigvalsh(Hl).min()
        if e < 1e-6: Hl += (1e-6 - e)*torch.eye(V)
        return Hl, Jl, G, h, A, b


def make(cls):
    torch.manual_seed(1)
    cost = SpnnCost(nx, nu, T); dyn = SpnnDynamic(nx, nu, T, dt); bar = SpnnBarrier(nx, nu, T)
    mpc = cls(nx=nx, nu=nu, nb=nb, T=T, batch_size=1, cost=cost, dynamic=dyn, barrier=bar, umin=umin, umax=umax)
    return mpc, cost, dyn, bar


def P():
    return {'cost': {'xref': xref}, 'barrier': {'obs': obs, 'gamma': gamma}}


def hval(bar, x):
    return bar.forward(torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(2, 1), {'obs': obs}).item()


def closed_loop(cls, N=20):
    mpc, cost, dyn, bar = make(cls)
    x0 = torch.tensor([[0.], [0.]])
    env = Env(x0.numpy(), xref.numpy(), obs, delta_t=dt)
    x = x0.clone(); hs = []; states = [x0.flatten().numpy().copy()]
    for t in range(1, N):
        with quiet():
            tau = mpc.step(x, P())
        x = torch.tensor(env.step(tau.u[0].flatten().detach().numpy()))
        states.append(x.flatten().numpy().copy()); hs.append(hval(bar, x))
    print(f'   x dtype fed back to mpc.step: {x.dtype}')
    print(f'   final x = {np.round(states[-1], 4).tolist()},  min h = {min(hs):.4f}')
    print('   h(x_t):', np.round(hs, 3).tolist())
    return mpc, env, states


def grad_check(cls, x_test, tag):
    mpc, cost, dyn, bar = make(cls)
    with quiet():                                     # warm start like closed loop
        mpc.step(x_test.clone(), P())
    base_tau = copy.deepcopy(mpc._tau)
    params = list(cost.context_network.parameters())

    def u0(m):
        with quiet():
            return m.step(x_test.clone(), P()).u[0].flatten()[0]

    m = copy.deepcopy(mpc); m._cost = cost; m._tau = copy.deepcopy(base_tau)
    out = u0(m)
    g_ad = torch.autograd.grad(out, params, allow_unused=True)
    g_ad = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(g_ad, params)])

    flat = torch.cat([p.detach().flatten() for p in params])
    idx = torch.argsort(g_ad.abs(), descending=True)[:6]
    eps = 1e-2
    rows = []
    for i in idx.tolist():
        vals = []
        for s in (+1, -1):
            f = flat.clone(); f[i] += s*eps
            torch.nn.utils.vector_to_parameters(f, params)
            mm = copy.deepcopy(mpc); mm._tau = copy.deepcopy(base_tau)
            with torch.no_grad(): vals.append(u0(mm).item())
        torch.nn.utils.vector_to_parameters(flat, params)
        rows.append((i, g_ad[i].item(), (vals[0]-vals[1])/(2*eps)))
    ad = np.array([r[1] for r in rows]); fd = np.array([r[2] for r in rows])
    print(f'[{tag}] u0={out.item():+.4f}')
    for i, a, f in rows: print(f'   param {i:3d}: autograd={a:+.5f}  FD={f:+.5f}')
    print(f'   rel err = {np.linalg.norm(ad-fd)/max(np.linalg.norm(fd),1e-12):.3e}')


def eps4(cls):
    class E(cls):
        def _build_sqp_params(self, *a):
            Q, p, G, h, A, b = super()._build_sqp_params(*a)
            e = torch.linalg.eigvalsh(Q).min()
            if e < 1e-4: Q = Q + (1e-4 - e)*torch.eye(Q.shape[0])
            return Q, p, G, h, A, b
    return E


if __name__ == '__main__':
    print('=== Part 1: closed loop, user DiffMPCBF (eps shift 1e-4, otherwise unchanged) ===')
    mpc, env, states = closed_loop(eps4(DiffMPCBF))

    print('=== Part 2: Env.show() ===')
    try:
        env.show(); print('   ok')
    except Exception as e:
        print(f'   {type(e).__name__}: {e}')

    # pick the state with smallest h along the user trajectory (CBF most likely active)
    bar = SpnnBarrier(nx, nu, T)
    hs = [hval(bar, s) for s in states]
    for k in (8, 12):
        x_test = torch.tensor(states[k], dtype=torch.float32).reshape(2, 1)
        print(f'=== Part 3: gradient check at x={np.round(states[k],4).tolist()} (h={hs[k]:.4f}) ===')
        grad_check(eps4(DiffMPCBF), x_test, 'user Hessian')
        grad_check(eps4(HessFixMPCBF), x_test, 'corrected Hessian')
