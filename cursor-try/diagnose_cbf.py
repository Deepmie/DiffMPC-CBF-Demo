"""
Diagnose DiffMPCBF infeasibility in examples/spnn_ex/test_sqp_withcbf.py
(user code is NOT modified; fixes are applied via subclasses here).
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'spnn_ex'))
warnings.filterwarnings('ignore')
import io, contextlib
import numpy as np
import torch
from diffmpc_cbf import DiffMPCBF
from qpth.qp import QPFunction
from system import SpnnCost, SpnnDynamic, SpnnBarrier

torch.manual_seed(1)
nx, nu, nb, T, dt = 2, 1, 1, 5, 0.1
umin, umax = torch.tensor([-10.0]), torch.tensor([10.0])
xref = torch.tensor([[1.], [0.]])
obs = [np.array([0.5, 0.5, 0.3, 0.02])]
gamma = 0.5


class BarrierPV(SpnnBarrier):
    """(p - o0)^2 + (v - o1)^2 - r^2  (the user's version uses pt twice)"""
    def _define_barrier_function(self, xt, params):
        pt, vt = xt
        return torch.stack([(pt - o[0])**2 + (vt - o[1])**2 - o[2]**2 for o in params.get('obs')])


class FixedMPCBF(DiffMPCBF):
    """Same as user's _build_sqp_params, with: G=-Jg/h=g, sum(dim=0), full CBF Lagrangian Hessian."""
    def _build_sqp_params(self, x0, _tau, dveq, dvneq, params):
        nx, nu, nb, T, ntau, V = self._nx, self._nu, self._nb, self._T, self._ntau, self._var_dims
        Jl = torch.zeros([V]); Hl = torch.zeros([V, V])
        Jh = torch.zeros([(T+1)*nx, V]); h = torch.zeros([(T+1)*nx])
        Jg_cbf = torch.zeros([T*nb, V]); g_cbf = torch.zeros([T*nb])
        Jg_b = torch.zeros([T*2*nu, V]); g_b = torch.zeros([T*2*nu])
        cx = torch.concat([torch.eye(nx), torch.zeros([nu, nx])], dim=0)
        cu = torch.concat([torch.zeros([nx, nu]), torch.eye(nu)], dim=0)
        bp = params.get('barrier'); gm = bp.get('gamma')
        sel = lambda t: cx if t < T else torch.eye(nx)
        blk = lambda t: slice(t*ntau, min((t+1)*ntau, V))
        for t in range(T+1):
            cp = {**params.get('cost'), 't': t}
            if t < T:
                Jl[blk(t)] = self._cost.get_stage_cost(_tau[t], cp, order=1).flatten()
                Hl[blk(t), blk(t)] += self._cost.get_stage_cost(_tau[t], cp, order=2) + \
                    (dveq[(t+1)*nx:(t+2)*nx].reshape(-1, 1, 1) *
                     self._dynamic.forward(_tau[t], {**params.get('dynamic'), 't': t}, order=2)).sum(dim=0)
            else:
                Jl[blk(T)] = self._cost.get_terminal_cost(_tau.get_state(T), cp, order=1).flatten()
                Hl[blk(T), blk(T)] += self._cost.get_terminal_cost(_tau.get_state(T), cp, order=2)
            Jh[t*nx:(t+1)*nx, blk(t)] = -sel(t).T
            if t < T:
                Jh[(t+1)*nx:(t+2)*nx, blk(t)] = self._dynamic.forward(_tau[t], {**params.get('dynamic'), 't': t}, order=1)
            prev = x0 if t == 0 else self._dynamic.forward(_tau[t-1], {**params.get('dynamic'), 't': t-1})
            h[t*nx:(t+1)*nx] = (prev - _tau.get_state(t)).flatten()
            if t < T:
                nu_t = dvneq[t*nb:(t+1)*nb].reshape(-1, 1, 1)
                b0 = {**bp, 't': t}; b1 = {**bp, 't': t+1}
                Jg_cbf[t*nb:(t+1)*nb, blk(t)] = -(1-gm)*self._barrier.forward(_tau.get_state(t), b0, order=1) @ cx.T
                Jg_cbf[t*nb:(t+1)*nb, blk(t+1)] = self._barrier.forward(_tau.get_state(t+1), b1, order=1) @ sel(t+1).T
                g_cbf[t*nb:(t+1)*nb] = self._barrier.forward(_tau.get_state(t+1), b1).flatten() - \
                    (1-gm)*self._barrier.forward(_tau.get_state(t), b0).flatten()
                # L = f + nu^T(-g): block t gets +nu(1-g)H(x_t), block t+1 gets -nu H(x_{t+1})
                Hl[blk(t), blk(t)] += (nu_t * (1-gm) * sel(t) @ self._barrier.forward(_tau.get_state(t), b0, order=2) @ sel(t).T).sum(dim=0)
                Hl[blk(t+1), blk(t+1)] += (-nu_t * sel(t+1) @ self._barrier.forward(_tau.get_state(t+1), b1, order=2) @ sel(t+1).T).sum(dim=0)
                Jg_b[t*2*nu:(t+1)*2*nu, blk(t)] = torch.concat([-cu.T, cu.T], dim=0)
                g_b[t*2*nu:(t+1)*2*nu] = torch.concat([self._umax-_tau.get_control(t).flatten(), _tau.get_control(t).flatten()-self._umin])
        Hl = 0.5*(Hl + Hl.T)
        e = torch.linalg.eigvalsh(Hl).min()
        if e < 1e-4: Hl += (1e-4 - e)*torch.eye(V)
        Jg = torch.concat([Jg_cbf, Jg_b]); g = torch.concat([g_cbf, g_b])
        return Hl, Jl, -Jg, g, Jh, -h          # qpth: G z <= h  <=>  g + Jg z >= 0


def make(cls, barrier_cls):
    torch.manual_seed(1)
    cost = SpnnCost(nx, nu, T); dyn = SpnnDynamic(nx, nu, T, dt); bar = barrier_cls(nx, nu, T)
    return cls(nx=nx, nu=nu, nb=nb, T=T, batch_size=1, cost=cost, dynamic=dyn, barrier=bar, umin=umin, umax=umax), dyn


def params():
    return {'cost': {'xref': xref}, 'barrier': {'obs': obs, 'gamma': gamma}}


def part1_first_qp():
    print('=== Part 1: first QP of first MPC step (user code) ===')
    mpc, dyn = make(DiffMPCBF, SpnnBarrier)
    x0 = torch.zeros(2, 1); p = params(); p['dynamic'] = {}
    mpc._tau.set_state_init(x0.flatten())
    for t in range(T):
        mpc._tau.set_state(dyn.forward(mpc._tau[t], {'t': t}).flatten(), t+1)
    Q, q, G, hh, A, b = mpc._build_sqp_params(x0, mpc._tau, mpc._get_init_dveq(), mpc._get_init_dvneq(), p)
    nb_rows = T*nb
    print('control-bound rows of (G, h) for t=0 (qpth: G z <= h):')
    for r in range(nb_rows, nb_rows+2):
        j = int(torch.nonzero(G[r]).flatten()[0])
        print(f'   {G[r, j].item():+.0f} * du0 <= {hh[r].item():+.2f}')
    print('   => du0 >= 10 AND du0 <= -10  -> QP infeasible by construction')
    with contextlib.redirect_stdout(io.StringIO()):
        z, lam, nu_ = QPFunction(eps=1e-12, verbose=0, maxIter=20)(Q, q, G, hh, A, b)
    z = z.flatten()
    print(f'   qpth returned: |z|max={z.abs().max().item():.3e}, max(Gz-h)={(G@z-hh).max().item():.3e}, '
          f'|nu|max={nu_.abs().max().item():.3e}')
    print('   the old DiffMPC used G=[+I;-I] with h=[umax-u; u-umin] (correct); DiffMPCBF returns (Jg, -g) (flipped)\n')


def run_closed_loop(cls, barrier_cls, tag, N=50):
    mpc, dyn = make(cls, barrier_cls)
    xs = torch.zeros(N, nx); hs = []; status = 'ok'
    bar = barrier_cls(nx, nu, T)
    for t in range(1, N):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                tau = mpc.step(xs[t-1], params())
        except Exception as e:
            status = f'FAILED at step {t}: {e}'; xs = xs[:t]; break
        with torch.no_grad():
            xs[t] = dyn.forward((xs[t-1], tau.u[0].detach()), {'t': 0}).flatten()
        hs.append(bar.forward(xs[t].unsqueeze(-1), {'obs': obs}).item())
    print(f'[{tag}] {status}')
    if hs:
        print(f'   final x = {xs[-1].tolist()},  min h(x) = {min(hs):.4f}  (h<0 means inside obstacle)')
        print('   p trajectory:', np.round(xs[:, 0].numpy(), 3)[::5].tolist())
    return xs


if __name__ == '__main__':
    part1_first_qp()
    print('=== Part 2: closed loop ===')
    run_closed_loop(FixedMPCBF, SpnnBarrier, 'sign/Hessian fixed, barrier as written (pt twice)')
    run_closed_loop(FixedMPCBF, BarrierPV, 'sign/Hessian fixed, barrier (p,v) circle')
