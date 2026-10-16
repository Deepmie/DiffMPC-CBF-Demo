"""Final check of DiffMPCBF after user's latest fixes (do not modify user code)."""
import sys, os, io, contextlib, copy, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'spnn_ex'))
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import numpy as np
import torch
from diffmpc_cbf import DiffMPCBF, DiffMPC
from qpth.qp import QPFunction
from system import SpnnCost, SpnnDynamic, SpnnBarrier
from env import Env

nx, nu, nb, T, dt = 2, 1, 1, 5, 0.1
umin, umax = torch.tensor([-10.0]), torch.tensor([10.0])
xref = torch.tensor([[1.], [0.]])
obs = [np.array([0.5, 0.5, 0.3, 0.02])]
gamma = 0.5
quiet = lambda: contextlib.redirect_stdout(io.StringIO())


def P():
    return {'cost': {'xref': xref}, 'barrier': {'obs': obs, 'gamma': gamma}, 'dynamic': {}}


def make():
    torch.manual_seed(1)
    cost = SpnnCost(nx, nu, T); dyn = SpnnDynamic(nx, nu, T, dt); bar = SpnnBarrier(nx, nu, T)
    mpc = DiffMPCBF(nx=nx, nu=nu, nb=nb, T=T, batch_size=1, cost=cost, dynamic=dyn,
                    barrier=bar, umin=umin, umax=umax)
    return mpc, cost, dyn, bar


def hval(bar, x):
    return bar.forward(torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(2, 1), {'obs': obs}).item()


print('=== 0. DiffMPC API ===')
try:
    c = SpnnCost(nx, nu, T); d = SpnnDynamic(nx, nu, T, dt)
    DiffMPC(nx=nx, nu=nu, T=T, batch_size=1, cost=c, dynamic=d, umin=umin, umax=umax)
    print('  DiffMPC init OK')
except Exception as e:
    print(f'  DiffMPC still broken: {type(e).__name__}: {e}')

print('\n=== 1. Terminal CBF Hessian ===')
mpc, cost, dyn, bar = make()
x0 = torch.tensor([[0.], [0.]])
mpc._tau.set_state_init(x0.flatten())
for t in range(T):
    mpc._tau.set_state(dyn.forward(mpc._tau[t], {'t': t}).flatten(), t+1)
dveq = torch.zeros((T+1)*nx)
dvneq = torch.zeros(T*(nb+2*nu)); dvneq[(T-1)*nb:T*nb] = 3.0
Hl, _, _, _, _, _ = mpc._build_sqp_params(x0, mpc._tau, dveq, dvneq, P())
# rebuild without shift by differencing vs nu=0 after undoing shift is messy; compare structure:
# expected terminal = Hcost - 3 * Hb
Hcost = cost.get_terminal_cost(mpc._tau.get_state(T), {**P()['cost'], 't': T}, order=2)
Hb = bar.forward(mpc._tau.get_state(T), {**P()['barrier'], 't': T}, order=2)
if Hb.ndim == 3: Hb = Hb[0]
expected = Hcost - 3.0 * Hb
actual = Hl[T*(nx+nu):, T*(nx+nu):]
# remove possible SPD shift: if eig shift applied uniformly, subtract identity*(eps-eigmin)
# better: call NoShift via manual expected and check before-shift by reconstructing from formula
# Use that shift is c*I: (actual - expected) should be c*I if only shift differs
diff = actual - expected
# check if diff is scalar * I
off = diff - torch.diag(torch.diag(diff))
diag = torch.diag(diff)
print(f'  ||actual - expected||={diff.norm().item():.3e}')
print(f'  offdiag of (actual-expected)={off.norm().item():.3e}, diag={diag.tolist()}')
if off.norm().item() < 1e-5 and (diag - diag[0]).abs().max().item() < 1e-5:
    print('  OK: terminal CBF term present (diff is pure SPD shift c*I)')
elif diff.norm().item() < 1e-5:
    print('  OK: exact match')
else:
    print('  FAIL: terminal Hessian still wrong')

# also verify slice sizes
V = T*(nx+nu)+nx
print(f'  terminal block shape={tuple(actual.shape)} (expect {(nx,nx)}), var_dims={V}')

print('\n=== 2. Stage CBF Hessian (subtract nu=0, account for shift) ===')
dvneq1 = torch.zeros(T*(nb+2*nu)); dvneq1[:T*nb] = 1.0
Hl1, _, _, _, _, _ = mpc._build_sqp_params(x0, mpc._tau, dveq, dvneq1, P())
Hl0, _, _, _, _, _ = mpc._build_sqp_params(x0, mpc._tau, dveq, torch.zeros_like(dvneq1), P())
# Can't subtract after independent shifts. Rebuild expected raw and compare via
# (Hl1 - c1 I) - (Hl0 - c0 I) vs expected D_cbf by checking on a mask away from identity
cx = torch.concat([torch.eye(nx), torch.zeros(nu, nx)], 0)
ok = True
for t in range(T+1):
    if t < T:
        blk = slice(t*(nx+nu), (t+1)*(nx+nu))
        Hb = bar.forward(mpc._tau.get_state(t), {**P()['barrier'], 't': t}, order=2)
        if Hb.ndim == 3: Hb = Hb[0]
        exp = (1-gamma)*(cx @ Hb @ cx.T)
        if t > 0: exp = exp - (cx @ Hb @ cx.T)
    else:
        blk = slice(T*(nx+nu), None)
        Hb = bar.forward(mpc._tau.get_state(T), {**P()['barrier'], 't': T}, order=2)
        if Hb.ndim == 3: Hb = Hb[0]
        exp = -Hb
    # extract relative: D = Hl1[blk]-Hl0[blk]; if both shifted by different c*I, D - exp ≈ (c1-c0)I on overlapping dims
    D = Hl1[blk, blk] - Hl0[blk, blk]
    rem = D - exp
    # project out multiple of I
    n = rem.shape[0]
    c = rem.diag().mean().item()
    rem2 = rem - c*torch.eye(n)
    err = rem2.norm().item()
    print(f'  block t={t}: err_after_removing_cI={err:.3e}')
    if err > 1e-4: ok = False
print('  stage+terminal Hessian OK' if ok else '  Hessian issues remain')

print('\n=== 3. Line search recomputes residuals ===')
mpc2, cost, dyn, bar = make()
mpc2._tau.set_state_init(x0.flatten())
for t in range(T):
    mpc2._tau.set_state(dyn.forward(mpc2._tau[t], {'t': t}).flatten(), t+1)
dveq = mpc2._get_init_dveq(); dvneq = mpc2._get_init_dvneq()
Hl, Jl, Jg, g, Jh, h = mpc2._build_sqp_params(x0, mpc2._tau, dveq, dvneq, P())
with quiet():
    z, dveq_qp, dvneq_qp = QPFunction(eps=1e-12, verbose=0, maxIter=20)(Hl, Jl, Jg, -g, Jh, -h)
tau_delta = type(mpc2._tau).build_from_tensor(z.flatten(), nx, nx+nu)
dveq_qp, dvneq_qp = dveq_qp.flatten(), dvneq_qp.flatten()
args = (x0, cost, dyn, bar, umin, umax, P(), dveq_qp, dvneq_qp)
m0 = float(mpc2._metricfunc(mpc2._tau, *args))
m1 = float(mpc2._metricfunc(mpc2._tau + tau_delta, *args))
print(f'  |delta|={z.abs().max().item():.4e}')
print(f'  metric(tau)={m0:.6f}, metric(tau+d)={m1:.6f}, delta={m1-m0:.6f}')
print('  OK: MetricFunction rebuilds h,g from tau' if m0 != m1 or z.abs().max() < 1e-8 else '  (values differ as expected)')

print('\n=== 4. Closed loop N=80 ===')
mpc3, cost, dyn, bar = make()
env = Env(x0.flatten().numpy(), xref.flatten().numpy(), obs, delta_t=dt)
x = x0.clone(); hs = []
failed = None
for t in range(1, 80):
    try:
        with quiet():
            tau = mpc3.step(x, P())
    except Exception as e:
        failed = (t, e); break
    x = torch.tensor(env.step(tau.u[0].flatten().detach().numpy()))
    hs.append(hval(bar, x))
if failed:
    print(f'  FAILED at {failed[0]}: {failed[1]}')
else:
    xs = np.stack(env._xs)
    print(f'  final x={np.round(xs[-1],4).tolist()}, min h={min(hs):.4f}')
    print(f'  p[::10]={np.round(xs[::10,0],3).tolist()}')

print('\n=== 5. Env.show ylim ===')
src = open('examples/spnn_ex/env.py').read()
print('  set_ylim present' if 'set_ylim' in src else '  still missing set_ylim (set_xlim twice)')
try:
    env.show(); print('  show() ran')
except Exception as e:
    print(f'  show error: {e}')

print('\n=== 6. Grad far from obstacle ===')
mpc4, cost, dyn, bar = make()
x = torch.tensor([[0.], [-0.3]])
for _ in range(2):
    with quiet(): mpc4.step(x.clone(), P())
base = copy.deepcopy(mpc4._tau)
params = list(cost.context_network.parameters())
flat = torch.cat([p.detach().flatten() for p in params])
m = copy.deepcopy(mpc4); m._cost = cost; m._tau = copy.deepcopy(base)
with quiet():
    out = m.step(x.clone(), P()).u[0].flatten()[0]
g_ad = torch.autograd.grad(out, params)
g_ad = torch.cat([g.flatten() for g in g_ad])
idx = torch.argsort(g_ad.abs(), descending=True)[:4]
rows = []
for i in idx.tolist():
    vals = []
    for sg in (1, -1):
        f = flat.clone(); f[i] += sg*1e-2
        torch.nn.utils.vector_to_parameters(f, params)
        mm = copy.deepcopy(mpc4); mm._cost = cost; mm._tau = copy.deepcopy(base)
        with quiet(), torch.no_grad():
            vals.append(mm.step(x.clone(), P()).u[0].flatten()[0].item())
    torch.nn.utils.vector_to_parameters(flat, params)
    rows.append((g_ad[i].item(), (vals[0]-vals[1])/2e-2))
ad = np.array([r[0] for r in rows]); fd = np.array([r[1] for r in rows])
rel = np.linalg.norm(ad-fd)/max(np.linalg.norm(fd), 1e-12)
print(f'  u0={out.item():+.4f}, rel_err={rel:.3e}')
for a, f in rows:
    print(f'    AD={a:+.5f} FD={f:+.5f}')
