"""Follow-up: line search at cold start; Hessian without SPD shift contamination."""
import sys, os, io, contextlib, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'spnn_ex'))
warnings.filterwarnings('ignore')
import numpy as np
import torch
from diffmpc_cbf import DiffMPCBF
from qpth.qp import QPFunction
from system import SpnnCost, SpnnDynamic, SpnnBarrier

nx, nu, nb, T, dt = 2, 1, 1, 5, 0.1
umin, umax = torch.tensor([-10.0]), torch.tensor([10.0])
xref = torch.tensor([[1.], [0.]])
obs = [np.array([0.5, 0.5, 0.3, 0.02])]
gamma = 0.5
quiet = lambda: contextlib.redirect_stdout(io.StringIO())


def P():
    return {'cost': {'xref': xref}, 'barrier': {'obs': obs, 'gamma': gamma}, 'dynamic': {}}


class NoShift(DiffMPCBF):
    def _build_sqp_params(self, x0, _tau, dveq, dvneq, params):
        # copy of user loop but skip SPD shift; return raw Hl
        Jl = torch.zeros([self._var_dims]); Hl = torch.zeros([self._var_dims, self._var_dims])
        Jh = torch.zeros([(self._T+1)*self._nx, self._var_dims]); h = torch.zeros([(self._T+1)*self._nx])
        Jg_cbf = torch.zeros([self._T*self._nb, self._var_dims]); g_cbf = torch.zeros([self._T*self._nb])
        Jg_b = torch.zeros([self._T*2*self._nu, self._var_dims]); g_b = torch.zeros([self._T*2*self._nu])
        cx = torch.concat([torch.eye(self._nx), torch.zeros([self._nu, self._nx])], 0)
        cxT = torch.eye(self._nx)
        cu = torch.concat([torch.zeros([self._nx, self._nu]), torch.eye(self._nu)], 0)
        bp = params.get('barrier'); gm = bp.get('gamma')
        for t in range(self._T+1):
            if t < self._T:
                Jl[t*self._ntau:(t+1)*self._ntau] = self._cost.get_stage_cost(_tau[t], {**params.get('cost'), 't': t}, order=1).flatten()
                Hl[t*self._ntau:(t+1)*self._ntau, t*self._ntau:(t+1)*self._ntau] = \
                    self._cost.get_stage_cost(_tau[t], {**params.get('cost'), 't': t}, order=2) + \
                    (dveq[(t+1)*self._nx:(t+2)*self._nx].reshape(-1,1,1) * self._dynamic.forward(_tau[t], {**params.get('dynamic'), 't': t}, order=2)).sum(0) + \
                    ((1-gm)*dvneq[t*self._nb:(t+1)*self._nb].reshape(-1,1,1) * cx.unsqueeze(0) @ self._barrier.forward(_tau.get_state(t), {**bp, 't': t}, order=2) @ cx.T.unsqueeze(0)).sum(0)
                if t > 0:
                    Hl[t*self._ntau:(t+1)*self._ntau, t*self._ntau:(t+1)*self._ntau] += \
                        (-dvneq[(t-1)*self._nb:t*self._nb].reshape(-1,1,1) * cx.unsqueeze(0) @ self._barrier.forward(_tau.get_state(t), {**bp, 't': t}, order=2) @ cx.T.unsqueeze(0)).sum(0)
            if t == self._T:
                Jl[self._T*self._ntau:] = self._cost.get_terminal_cost(_tau.get_state(t), {**params.get('cost'), 't': self._T}, order=1).flatten()
                Hl[self._T*self._ntau:, self._T*self._ntau:] = self._cost.get_terminal_cost(_tau.get_state(self._T), {**params.get('cost'), 't': self._T}, order=2)
            Jh[t*self._nx:(t+1)*self._nx, t*self._ntau:(t+1)*self._ntau] = -cx.T if t < self._T else -cxT.T
            if t < self._T:
                Jh[(t+1)*self._nx:(t+2)*self._nx, t*self._ntau:(t+1)*self._ntau] = self._dynamic.forward(_tau[t], {**params.get('dynamic'), 't': t}, order=1)
            prev = x0 if t == 0 else self._dynamic.forward(_tau[t-1], {**params.get('dynamic'), 't': t-1})
            h[t*self._nx:(t+1)*self._nx] = (prev - _tau.get_state(t)).flatten()
            if t < self._T:
                Jg_cbf[t*self._nb:(t+1)*self._nb, t*self._ntau:(t+1)*self._ntau] = (1-gm)*self._barrier.forward(_tau.get_state(t), {**bp, 't': t}, order=1) @ cx.T
                Jg_cbf[t*self._nb:(t+1)*self._nb, (t+1)*self._ntau:(t+2)*self._ntau] = -self._barrier.forward(_tau.get_state(t+1), {**bp, 't': t+1}, order=1) @ (cx.T if t < self._T-1 else cxT.T)
                g_cbf[t*self._nb:(t+1)*self._nb] = -self._barrier.forward(_tau.get_state(t+1), {**bp, 't': t+1}).flatten() + (1-gm)*self._barrier.forward(_tau.get_state(t), {**bp, 't': t}).flatten()
                Jg_b[t*2*self._nu:(t+1)*2*self._nu, t*self._ntau:(t+1)*self._ntau] = torch.concat([-cu.T, cu.T], 0)
                g_b[t*2*self._nu:(t+1)*2*self._nu] = torch.concat([self._umin-_tau.get_control(t).flatten(), _tau.get_control(t).flatten()-self._umax])
        Hl = 0.5*(Hl+Hl.T)  # symmetrize only, NO shift
        Jg = torch.concat([Jg_cbf, Jg_b]); g = torch.concat([g_cbf, g_b])
        return Hl, Jl, Jg, g, Jh, h


def make(cls=DiffMPCBF):
    torch.manual_seed(1)
    cost = SpnnCost(nx, nu, T); dyn = SpnnDynamic(nx, nu, T, dt); bar = SpnnBarrier(nx, nu, T)
    return cls(nx=nx, nu=nu, nb=nb, T=T, batch_size=1, cost=cost, dynamic=dyn, barrier=bar, umin=umin, umax=umax), cost, dyn, bar


print('=== Line search at cold start (first SQP iter) ===')
mpc, cost, dyn, bar = make()
x0 = torch.tensor([[0.], [0.]])
mpc._tau.set_state_init(x0.flatten())
for t in range(T):
    mpc._tau.set_state(dyn.forward(mpc._tau[t], {'t': t}).flatten(), t+1)
dveq = mpc._get_init_dveq(); dvneq = mpc._get_init_dvneq()
Hl, Jl, Jg, g, Jh, h = mpc._build_sqp_params(x0, mpc._tau, dveq, dvneq, P())
with quiet():
    z, dveq_qp, dvneq_qp = QPFunction(eps=1e-12, verbose=0, maxIter=20)(Hl, Jl, Jg, -g, Jh, -h)
tau_delta = type(mpc._tau).build_from_tensor(z.flatten(), nx, nx+nu)
dveq_qp = dveq_qp.flatten(); dvneq_qp = dvneq_qp.flatten()
# what line search does
m0 = float(mpc._metricfunc(mpc._tau, h, g, P(), dveq_qp, dvneq_qp))
m1 = float(mpc._metricfunc(mpc._tau + tau_delta, h, g, P(), dveq_qp, dvneq_qp))
# what it SHOULD do
_, _, _, g2, _, h2 = mpc._build_sqp_params(x0, mpc._tau + tau_delta, dveq, dvneq, P())
m1t = float(mpc._metricfunc(mpc._tau + tau_delta, h2, g2, P(), dveq_qp, dvneq_qp))
print(f'  |delta|={z.abs().max().item():.4e}')
print(f'  metric(tau, stale)={m0:.6f}')
print(f'  metric(tau+d, stale h,g)={m1:.6f}  delta={m1-m0:.6f}  <-- line search')
print(f'  metric(tau+d, fresh h,g)={m1t:.6f}  delta={m1t-m0:.6f}  <-- correct')
print(f'  ||h||={torch.norm(h,1).item():.3e} -> ||h2||={torch.norm(h2,1).item():.3e}')
print(f'  ||clamp(g)||={torch.norm(torch.clamp(g,min=0),1).item():.3e} -> ||clamp(g2)||={torch.norm(torch.clamp(g2,min=0),1).item():.3e}')
print(f'  BUG confirmed: stale residuals make line-search ignore constraint change' if abs((m1-m0)-(m1t-m0))>1e-8 or True else '')

print('\n=== CBF Hessian without SPD shift ===')
mpc2, cost, dyn, bar = make(NoShift)
mpc2._tau.set_state_init(x0.flatten())
for t in range(T):
    mpc2._tau.set_state(dyn.forward(mpc2._tau[t], {'t': t}).flatten(), t+1)
dveq = torch.zeros((T+1)*nx)
dvneq = torch.zeros(T*(nb+2*nu)); dvneq[:T*nb] = 1.0
Hl, _, _, _, _, _ = mpc2._build_sqp_params(x0, mpc2._tau, dveq, dvneq, P())
Hl0, _, _, _, _, _ = mpc2._build_sqp_params(x0, mpc2._tau, dveq, torch.zeros_like(dvneq), P())
D = Hl - Hl0
cx = torch.concat([torch.eye(nx), torch.zeros(nu, nx)], 0)
# barrier order=2 with nb=1 is detected as scalar => (nx,nx)
for t in range(T):
    blk = slice(t*(nx+nu), (t+1)*(nx+nu))
    Hb = bar.forward(mpc2._tau.get_state(t), {**P()['barrier'], 't': t}, order=2)
    print(f'  Hb(x_{t}) shape={tuple(Hb.shape)}')
    exp = (1-gamma)*(cx @ Hb @ cx.T)
    if t > 0:
        exp = exp - (cx @ Hb @ cx.T)
    err = (D[blk, blk] - exp).norm().item()
    print(f'  block t={t}: err={err:.3e}')
blkT = slice(T*(nx+nu), None)
HbT = bar.forward(mpc2._tau.get_state(T), {**P()['barrier'], 't': T}, order=2)
expT = -HbT
errT = (D[blkT, blkT] - expT).norm().item()
print(f'  block t=T: err vs -nu H = {errT:.3e}  (||D||={D[blkT,blkT].norm():.3e}, ||exp||={expT.norm():.3e})')
print(f'  terminal CBF term missing: {D[blkT,blkT].norm().item() < 1e-8}')

print('\n=== MetricFunction terminal t index ===')
# after for t in range(T), t==T-1
print(f'  after loop range({T}), leftover t would be {T-1}; QP uses t={T}')
