"""
Re-check revised DiffMPCBF / MetricFunction (user code NOT modified).
"""
import sys, os, io, contextlib, copy, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'spnn_ex'))
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import numpy as np
import torch
from torch import Tensor
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
    cost = SpnnCost(nx, nu, T)
    dyn = SpnnDynamic(nx, nu, T, dt)
    bar = SpnnBarrier(nx, nu, T)
    mpc = DiffMPCBF(nx=nx, nu=nu, nb=nb, T=T, batch_size=1, cost=cost, dynamic=dyn,
                    barrier=bar, umin=umin, umax=umax)
    return mpc, cost, dyn, bar


def hval(bar, x):
    return bar.forward(torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(2, 1), {'obs': obs}).item()


def part0_diffmpc_api():
    print('=== Part 0: DiffMPC vs new MetricFunction API ===')
    try:
        cost = SpnnCost(nx, nu, T); dyn = SpnnDynamic(nx, nu, T, dt)
        DiffMPC(nx=nx, nu=nu, T=T, batch_size=1, cost=cost, dynamic=dyn, umin=umin, umax=umax)
        print('   DiffMPC init: ok (unexpected)')
    except TypeError as e:
        print(f'   DiffMPC init TypeError: {e}')
    except Exception as e:
        print(f'   DiffMPC init {type(e).__name__}: {e}')


def part1_closed_loop(N=80):
    print(f'=== Part 1: closed loop N={N} ===')
    mpc, cost, dyn, bar = make()
    x0 = torch.tensor([[0.], [0.]])
    env = Env(x0.flatten().numpy(), xref.flatten().numpy(), obs, delta_t=dt)
    x = x0.clone(); hs = []; us = []
    for t in range(1, N):
        try:
            with quiet():
                tau = mpc.step(x, P())
        except Exception as e:
            print(f'   FAILED at step {t}: {e}; x={np.round(x.flatten().numpy(),4).tolist()}')
            return None, None, None
        u0 = tau.u[0].flatten().detach().numpy()
        us.append(float(u0[0]))
        x = torch.tensor(env.step(u0))
        hs.append(hval(bar, x))
    xs = np.stack(env._xs)
    print(f'   final x={np.round(xs[-1],4).tolist()}, min h={min(hs):.4f}, |u|max={max(abs(u) for u in us):.3f}')
    print(f'   p[::10]={np.round(xs[::10,0],3).tolist()}')
    print(f'   h[::10]={np.round(hs[::10],4).tolist()}')
    return mpc, env, xs


def part2_line_search_bug():
    print('=== Part 2: line search uses stale (h,g) ===')
    mpc, cost, dyn, bar = make()
    x0 = torch.tensor([[0.], [0.]])
    with quiet():
        mpc.step(x0, P())  # warm
    # rebuild one SQP step manually at current tau
    dveq = mpc._get_init_dveq(); dvneq = mpc._get_init_dvneq()
    Hl, Jl, Jg, g, Jh, h = mpc._build_sqp_params(x0, mpc._tau, dveq, dvneq, P())
    with quiet():
        z, _, _ = QPFunction(eps=1e-12, verbose=0, maxIter=20)(Hl, Jl, Jg, -g, Jh, -h)
    tau_delta = type(mpc._tau).build_from_tensor(z.flatten(), nx, nx+nu)
    m0 = mpc._metricfunc(mpc._tau, h, g, P(), dveq, dvneq)
    m1 = mpc._metricfunc(mpc._tau + 1.0*tau_delta, h, g, P(), dveq, dvneq)
    # recompute true residuals at trial point
    Hl2, Jl2, Jg2, g2, Jh2, h2 = mpc._build_sqp_params(x0, mpc._tau + 1.0*tau_delta, dveq, dvneq, P())
    m1_true = mpc._metricfunc(mpc._tau + 1.0*tau_delta, h2, g2, P(), dveq, dvneq)
    m0f, m1f, m1tf = float(m0), float(m1), float(m1_true)
    print(f'   metric(tau, h,g)={m0f:.6f}')
    print(f'   metric(tau+d, SAME h,g)={m1f:.6f}   # what line search uses')
    print(f'   metric(tau+d, recomputed h,g)={m1tf:.6f}')
    cost0 = float(sum(mpc._cost.get_stage_cost(mpc._tau[t], {**P()['cost'], 't': t}) for t in range(T))
                  + mpc._cost.get_terminal_cost(mpc._tau.get_state(T), {**P()['cost'], 't': T}))
    cost1 = float(sum(mpc._cost.get_stage_cost((mpc._tau+tau_delta)[t], {**P()['cost'], 't': t}) for t in range(T))
                  + mpc._cost.get_terminal_cost((mpc._tau+tau_delta).get_state(T), {**P()['cost'], 't': T}))
    print(f'   delta_metric with stale h,g = {m1f-m0f:.6f}; delta_cost alone = {cost1-cost0:.6f}')
    print(f'   => line search effectively only compares cost (residuals cancel).')
    print(f'   ||h||={torch.linalg.norm(h,1).item():.4e}, ||clamp(g,0)||={torch.linalg.norm(torch.clamp(g,min=0.),1).item():.4e}')
    print(f'   ||h2||={torch.linalg.norm(h2,1).item():.4e}, ||clamp(g2,0)||={torch.linalg.norm(torch.clamp(g2,min=0.),1).item():.4e}')


def part3_terminal_hessian():
    print('=== Part 3: terminal CBF Hessian contribution ===')
    mpc, cost, dyn, bar = make()
    x0 = torch.tensor([[0.], [0.]])
    # force nonzero last CBF multiplier
    dveq = torch.zeros((T+1)*nx)
    dvneq = torch.zeros(T*(nb+2*nu)); dvneq[(T-1)*nb:T*nb] = 3.0  # nu_{T-1}=3
    # init tau by rollout
    mpc._tau.set_state_init(x0.flatten())
    for t in range(T):
        mpc._tau.set_state(dyn.forward(mpc._tau[t], {'t': t}).flatten(), t+1)
    Hl, _, _, _, _, _ = mpc._build_sqp_params(x0, mpc._tau, dveq, dvneq, P())
    # expected terminal block: terminal cost Hess - nu_{T-1} * H_barrier(x_T)
    Hterm_cost = cost.get_terminal_cost(mpc._tau.get_state(T), {**P()['cost'], 't': T}, order=2)
    Hb = bar.forward(mpc._tau.get_state(T), {**P()['barrier'], 't': T}, order=2)  # (nb, nx, nx)
    expected = Hterm_cost + (-3.0 * Hb[0])  # nb=1
    actual = Hl[T*(nx+nu):, T*(nx+nu):]
    print(f'   ||actual - terminal_cost_only|| = {(actual - Hterm_cost).norm().item():.4e}')
    print(f'   ||actual - expected(cost - nu H_b)|| = {(actual - expected).norm().item():.4e}')
    if (actual - Hterm_cost).norm().item() < 1e-8:
        print('   BUG: terminal block has NO CBF Hessian term (-nu_{T-1} H(x_T)).')
    else:
        print('   terminal CBF Hessian present (ok).')


def part4_stage_hessian_signs():
    print('=== Part 4: stage CBF Hessian signs vs L=f+nu^T g ===')
    # g_t = -h(x_{t+1})+(1-g)h(x_t)  =>  d2L/dx_t^2 += nu_t(1-g)Hb(x_t)
    #                                 d2L/dx_{t+1}^2 += -nu_t Hb(x_{t+1})
    mpc, cost, dyn, bar = make()
    x0 = torch.tensor([[0.], [0.]])
    mpc._tau.set_state_init(x0.flatten())
    for t in range(T):
        mpc._tau.set_state(dyn.forward(mpc._tau[t], {'t': t}).flatten(), t+1)
    dveq = torch.zeros((T+1)*nx)
    dvneq = torch.zeros(T*(nb+2*nu))
    # only CBF multipliers, set all to 1
    for t in range(T):
        dvneq[t*nb:(t+1)*nb] = 1.0
    Hl, _, _, _, _, _ = mpc._build_sqp_params(x0, mpc._tau, dveq, dvneq, P())
    # subtract cost+dyn (dyn hess with dveq=0 is zero) to isolate CBF part
    Hl0, _, _, _, _, _ = mpc._build_sqp_params(x0, mpc._tau, dveq, torch.zeros_like(dvneq), P())
    D = Hl - Hl0
    cx = torch.concat([torch.eye(nx), torch.zeros(nu, nx)], 0)
    ok = True
    for t in range(T):
        blk = slice(t*(nx+nu), (t+1)*(nx+nu))
        Hb = bar.forward(mpc._tau.get_state(t), {**P()['barrier'], 't': t}, order=2)[0]
        # contribution at block t: nu_t(1-g)Hb + (if t>0) -nu_{t-1} Hb
        exp = (1-gamma)*1.0 * (cx @ Hb @ cx.T)
        if t > 0:
            exp = exp + (-1.0) * (cx @ Hb @ cx.T)
        err = (D[blk, blk] - exp).norm().item()
        print(f'   block t={t}: ||D - expected||={err:.3e}')
        if err > 1e-5: ok = False
    # terminal
    blkT = slice(T*(nx+nu), None)
    HbT = bar.forward(mpc._tau.get_state(T), {**P()['barrier'], 't': T}, order=2)[0]
    expT = -1.0 * HbT
    errT = (D[blkT, blkT] - expT).norm().item()
    print(f'   block t=T: ||D - expected(-nu H)||={errT:.3e}  (expect fail if missing)')
    if errT > 1e-5: ok = False
    print('   stage blocks OK' if ok else '   issues remain (see above)')


def part5_grad(x_list):
    print('=== Part 5: gradient AD vs FD ===')
    for s in x_list:
        mpc, cost, dyn, bar = make()
        x = torch.tensor(s, dtype=torch.float32).reshape(2, 1)
        for _ in range(2):
            with quiet(): mpc.step(x.clone(), P())
        base = copy.deepcopy(mpc._tau)
        params = list(cost.context_network.parameters())
        flat = torch.cat([p.detach().flatten() for p in params])

        def u0():
            m = copy.deepcopy(mpc); m._cost = cost; m._tau = copy.deepcopy(base)
            with quiet():
                return m.step(x.clone(), P()).u[0].flatten()[0]

        out = u0()
        g_ad = torch.autograd.grad(out, params, allow_unused=True)
        g_ad = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(g_ad, params)])
        idx = torch.argsort(g_ad.abs(), descending=True)[:4]
        rows = []
        eps = 1e-2
        for i in idx.tolist():
            vals = []
            for sg in (+1, -1):
                f = flat.clone(); f[i] += sg*eps
                torch.nn.utils.vector_to_parameters(f, params)
                with torch.no_grad():
                    m = copy.deepcopy(mpc); m._cost = cost; m._tau = copy.deepcopy(base)
                    with quiet():
                        vals.append(m.step(x.clone(), P()).u[0].flatten()[0].item())
            torch.nn.utils.vector_to_parameters(flat, params)
            rows.append((i, g_ad[i].item(), (vals[0]-vals[1])/(2*eps)))
        ad = np.array([r[1] for r in rows]); fd = np.array([r[2] for r in rows])
        rel = np.linalg.norm(ad-fd)/max(np.linalg.norm(fd), 1e-12)
        print(f'   x={s}, h={hval(bar,x):.4f}, u0={out.item():+.4f}, rel_err={rel:.3e}')
        for i, a, f in rows:
            print(f'      p{i}: AD={a:+.5f} FD={f:+.5f}')


def part6_env_show(env):
    print('=== Part 6: Env.show() ===')
    try:
        env.show()
        print('   ok (but check set_xlim called twice — ylim missing in source)')
    except Exception as e:
        print(f'   {type(e).__name__}: {e}')


if __name__ == '__main__':
    part0_diffmpc_api()
    part2_line_search_bug()
    part3_terminal_hessian()
    part4_stage_hessian_signs()
    mpc, env, xs = part1_closed_loop(80)
    if env is not None:
        part6_env_show(env)
    part5_grad([[0.0, -0.3], [0.2, 0.33]])
