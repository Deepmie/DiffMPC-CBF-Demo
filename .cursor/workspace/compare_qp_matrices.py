"""Compare QP matrices from the current vectorized builder against a v2-style loop.

The loop uses the same cost / dynamic / barrier modules and the same zeta, duals,
and parameters. Only the assembly follows ``_versions/v2/diffmpc_cbf_v2.py``.
Terminal cost in v2 writes the state block only; the value plugged in here is the
current unified cost at that time, so a terminal-block mismatch is an assembly
difference rather than a different network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "tvto"))
sys.path.insert(0, str(ROOT / "src"))

from diffmpc_cbf import DiffMPCBF
from env import Env
from system import TvToBarrier, TvToCost, TvToDynamic

ATOL = 1e-8
NX, NU, NB = 4, 2, 2
T = 15
GAMMA = 0.2
RHO1, RHO2 = 1e2, 1e3
DEVICE = torch.device("cpu")
DTYPE = torch.float64


def _stat(a: Tensor, b: Tensor) -> tuple[float, float, int]:
    diff = (a - b).abs()
    return diff.max().item(), diff.mean().item(), int((diff > ATOL).sum().item())


def _fmt(name: str, a: Tensor, b: Tensor) -> str:
    mx, mean, n = _stat(a, b)
    flag = "MATCH" if mx <= ATOL else "DIFF "
    return f"  {flag}  {name:22s}  max {mx:.3e}   mean {mean:.3e}   n>{ATOL:.0e}: {n}/{a.numel()}"


def _time_index(n: int, start: int = 0) -> Tensor:
    return torch.arange(start, start + n, device=DEVICE, dtype=torch.int)


def _with_t(params: dict, t: Tensor) -> dict:
    return {**params, "t": t}


def _cx(ntau: int, nx: int) -> Tensor:
    return torch.eye(ntau, nx, device=DEVICE, dtype=DTYPE)


def _cu(nx: int, nu: int, ntau: int) -> Tensor:
    return torch.cat([torch.zeros(nx, nu, device=DEVICE, dtype=DTYPE), torch.eye(nu, device=DEVICE, dtype=DTYPE)], dim=0)


def _oracles(cost, dynamic, barrier, zeta: Tensor, params: dict):
    """Batched derivatives aligned with time, the quantities the v2 loop consumes."""
    ctpm, dypm, brpm = params["cost"], params["dynamic"], params["barrier"]
    ntau = NX + NU
    t_full = _time_index(T + 1)
    t_curr = _time_index(T)
    tau = zeta[:, :ntau]
    x = zeta[:, :NX]
    cost_g = cost.get_cost(tau, _with_t(ctpm, t_full), order=1).squeeze(1)
    cost_h = cost.get_cost(tau, _with_t(ctpm, t_full), order=2).squeeze(1)
    dyn_f = dynamic.forward(tau[:T], _with_t(dypm, t_curr), order=0)
    dyn_j = dynamic.forward(tau[:T], _with_t(dypm, t_curr), order=1)
    dyn_h = dynamic.forward(tau[:T], _with_t(dypm, t_curr), order=2)
    bar_f = barrier.forward(x, _with_t(brpm, t_full), order=0)
    bar_j = barrier.forward(x, _with_t(brpm, t_full), order=1)
    bar_h = barrier.forward(x, _with_t(brpm, t_full), order=2)
    # Arguments actually used by the current zero-order / barrier-Hessian code.
    dyn_f_next = dynamic.forward(tau[1:], _with_t(dypm, t_curr), order=0)
    bar_f_shift = barrier.forward(x[1:], _with_t(brpm, t_curr), order=0)
    bar_h_shift = barrier.forward(x[1:], _with_t(brpm, t_curr), order=2)
    return {
        "cost_g": cost_g,
        "cost_h": cost_h,
        "dyn_f": dyn_f,
        "dyn_j": dyn_j,
        "dyn_h": dyn_h,
        "bar_f": bar_f,
        "bar_j": bar_j,
        "bar_h": bar_h,
        "dyn_f_next": dyn_f_next,
        "bar_f_shift": bar_f_shift,
        "bar_h_shift": bar_h_shift,
    }


def _current_barrier_coeff(dvneq: Tensor, gamma: Tensor) -> Tensor:
    return torch.cat(
        [
            ((1 - gamma) * dvneq[0, :NB]).unsqueeze(0),
            -dvneq[: T - 1, :NB] + (1 - gamma) * dvneq[1:T, :NB],
            (-dvneq[T - 1, :NB]).unsqueeze(0),
        ],
        dim=0,
    )


def _v2_barrier_coeff(dvneq: Tensor, gamma: Tensor) -> Tensor:
    """Coeff of ∇²h(x_t) added into Q[t]. t = T drops λ_T, which v2 would index out of range."""
    coeff = torch.zeros(T + 1, NB, device=DEVICE, dtype=DTYPE)
    coeff[0] = (1 - gamma) * dvneq[0, :NB]
    for t in range(1, T):
        coeff[t] = -dvneq[t - 1, :NB] + (1 - gamma) * dvneq[t, :NB]
    coeff[T] = -dvneq[T - 1, :NB]
    return coeff


def _map_barrier_hess(hess_nb: Tensor, cx: Tensor) -> Tensor:
    # hess_nb: (..., nb, nx, nx) -> (..., nb, ntau, ntau)
    return torch.einsum("ai,tbij,cj->tbac", cx, hess_nb, cx)


def assemble_v2(zeta, x0, dveq, dvneq, params, umin, umax, ora, mode: str) -> dict[str, Tensor]:
    ntau, nv = NX + NU, NX + NU + NB
    ng, nd = 2 * (NB + NU), NB + 2 * NU
    gamma = params["barrier"]["gamma"]
    cx, cu = _cx(ntau, NX), _cu(NX, NU, ntau)
    Q = torch.zeros(T + 1, nv, T + 1, nv, device=DEVICE, dtype=DTYPE)
    p = torch.zeros(T + 1, nv, device=DEVICE, dtype=DTYPE)
    A = torch.zeros(T + 1, NX, T + 1, nv, device=DEVICE, dtype=DTYPE)
    b = torch.zeros(T + 1, NX, device=DEVICE, dtype=DTYPE)
    G = torch.zeros(T, ng, T + 1, nv, device=DEVICE, dtype=DTYPE)
    q = torch.zeros(T, ng, device=DEVICE, dtype=DTYPE)
    x0f = x0.flatten()

    for t in range(T + 1):
        xt, ut = zeta[t, :NX], zeta[t, NX:ntau]
        dym = x0f if t == 0 else ora["dyn_f"][t - 1]
        b[t] = xt - dym
        if t < T:
            q[t, :NB] = ora["bar_f"][t + 1] - (1 - gamma) * ora["bar_f"][t]
            q[t, NB:nd] = torch.cat([ut - umin, umax - ut], dim=0)
    slack = zeta[:T, ntau:]
    q[:, :NB] = q[:, :NB] + slack
    q[:, nd:] = slack

    for t in range(T + 1):
        if t < T:
            p[t, :ntau] = ora["cost_g"][t]
            A[t + 1, :, t, :ntau] = ora["dyn_j"][t]
            G[t, :NB, t, :ntau] = (1 - gamma) * (ora["bar_j"][t] @ cx.T)
            G[t, :NB, t + 1, :ntau] = -(ora["bar_j"][t + 1] @ cx.T)
            G[t, NB:nd, t, :ntau] = torch.cat([-cu.T, cu.T], dim=0)
        else:
            p[t, :NX] = ora["cost_g"][t, :NX]
        A[t, :, t, :ntau] = -cx.T
    p[:T, ntau:] = RHO1 + RHO2 * zeta[:T, ntau:]
    eye_s = torch.eye(T * NB, device=DEVICE, dtype=DTYPE).view(T, NB, T, NB)
    G[:, :NB, :T, ntau:] = -eye_s
    G[:, nd:, :T, ntau:] = -eye_s

    for t in range(T + 1):
        if t < T:
            Q[t, :ntau, t, :ntau] = ora["cost_h"][t]
            if mode == "full":
                Q[t, :ntau, t, :ntau] = Q[t, :ntau, t, :ntau] + (dveq[t + 1].reshape(-1, 1, 1) * ora["dyn_h"][t]).sum(0)
        else:
            Q[t, :NX, t, :NX] = ora["cost_h"][t, :NX, :NX]
            Q[t, NX:ntau, t, NX:ntau] = 1e-3 * torch.eye(NU, device=DEVICE, dtype=DTYPE)
    if mode == "full":
        coeff = _v2_barrier_coeff(dvneq, gamma)
        mapped = _map_barrier_hess(ora["bar_h"], cx)
        extra = (coeff.unsqueeze(-1).unsqueeze(-1) * mapped).sum(1)
        for t in range(T + 1):
            Q[t, :ntau, t, :ntau] = Q[t, :ntau, t, :ntau] + extra[t]
    Q[:T, ntau:, :T, ntau:] = RHO2 * eye_s
    return {"Q": Q, "p": p, "A": A, "b": b, "G": G, "q": q}


def assemble_current_reading(zeta, x0, dveq, dvneq, params, umin, umax, ora, mode: str) -> dict[str, Tensor]:
    """Explicit loop of the indexing in the current vectorized updates."""
    ntau, nv = NX + NU, NX + NU + NB
    ng, nd = 2 * (NB + NU), NB + 2 * NU
    gamma = params["barrier"]["gamma"]
    cx, cu = _cx(ntau, NX), _cu(NX, NU, ntau)
    Q = torch.zeros(T + 1, nv, T + 1, nv, device=DEVICE, dtype=DTYPE)
    p = torch.zeros(T + 1, nv, device=DEVICE, dtype=DTYPE)
    A = torch.zeros(T + 1, NX, T + 1, nv, device=DEVICE, dtype=DTYPE)
    b = torch.zeros(T + 1, NX, device=DEVICE, dtype=DTYPE)
    G = torch.zeros(T, ng, T + 1, nv, device=DEVICE, dtype=DTYPE)
    q = torch.zeros(T, ng, device=DEVICE, dtype=DTYPE)

    u = zeta[:T, NX:ntau]
    slack = zeta[:T, ntau:]
    b[0] = zeta[0, :NX] - x0.flatten()
    b[1:] = zeta[1:, :NX] - ora["dyn_f_next"]
    q[:, :NB] = ora["bar_f"][1:] - (1 - gamma) * ora["bar_f"][:T] + slack
    q[:, NB:nd] = torch.cat([u - umin, umax - u], dim=-1)
    q[:, nd:] = slack

    p[:, :ntau] = ora["cost_g"]
    p[T, NX:ntau] = 0
    p[:T, ntau:] = RHO1 + RHO2 * slack
    for t in range(T):
        A[t + 1, :, t, :ntau] = ora["dyn_j"][t]
        G[t, :NB, t, :ntau] = (1 - gamma) * (ora["bar_j"][t] @ cx.T)
        G[t, :NB, t + 1, :ntau] = -(ora["bar_j"][t + 1] @ cx.T)
        G[t, NB:nd, t, :ntau] = torch.cat([-cu.T, cu.T], dim=0)
    for t in range(T + 1):
        A[t, :, t, :ntau] = -cx.T
    eye_s = torch.eye(T * NB, device=DEVICE, dtype=DTYPE).view(T, NB, T, NB)
    G[:, :NB, :T, ntau:] = -eye_s
    G[:, nd:, :T, ntau:] = -eye_s

    for t in range(T + 1):
        Q[t, :ntau, t, :ntau] = ora["cost_h"][t]
    if mode == "full":
        for t in range(T):
            Q[t, :ntau, t, :ntau] = Q[t, :ntau, t, :ntau] + (dveq[t + 1].reshape(-1, 1, 1) * ora["dyn_h"][t]).sum(0)
        coeff = _current_barrier_coeff(dvneq, gamma)
        mapped = _map_barrier_hess(ora["bar_h"], cx)
        extra = (coeff.unsqueeze(-1).unsqueeze(-1) * mapped).sum(1)
        for t in range(T + 1):
            Q[t, :ntau, t, :ntau] = Q[t, :ntau, t, :ntau] + extra[t]
    Q[T, NX:ntau, T, :] = 0
    Q[T, :, T, NX:ntau] = 0
    Q[T, NX:ntau, T, NX:ntau] = 1e-3 * torch.eye(NU, device=DEVICE, dtype=DTYPE)
    Q[:T, ntau:, :T, ntau:] = RHO2 * eye_s
    return {"Q": Q, "p": p, "A": A, "b": b, "G": G, "q": q}


def _flat(mats: dict[str, Tensor]) -> dict[str, Tensor]:
    ntau, nv = NX + NU, NX + NU + NB
    vardim = (T + 1) * ntau + T * NB
    vdim = (T + 1) * nv
    xdim = (T + 1) * NX
    gdim = T * (2 * (NB + NU))
    Qv = mats["Q"].reshape(vdim, vdim)[:vardim, :vardim]
    pv = mats["p"].reshape(vdim)[:vardim]
    Av = mats["A"].reshape(xdim, vdim)[:, :vardim]
    bv = mats["b"].reshape(xdim)
    Gv = mats["G"].reshape(gdim, vdim)[:, :vardim]
    qv = mats["q"].reshape(gdim)
    return {"Qv": Qv, "pv": pv, "Av": Av, "bv": bv, "Gv": Gv, "qv": qv}


def _blank(mpc: DiffMPCBF) -> None:
    for name in ("_Q", "_p", "_A", "_b", "_G", "_q"):
        getattr(mpc, name).zero_()
    mpc._first_one = True
    mpc._first_two = True


def _run_current(mpc: DiffMPCBF, zeta, x0, dveq, dvneq, params, mode: str) -> dict[str, Tensor]:
    _blank(mpc)
    mpc._two_order_mode = mode
    mpc._zeta.copy_(zeta)
    mpc._dveq.copy_(dveq)
    mpc._dvneq.copy_(dvneq)
    with torch.no_grad():
        mpc._update_zero_order_term(mpc._zeta, x0, params)
        mpc._update_one_order_term(mpc._zeta, params)
        mpc._update_two_order_term(mpc._zeta, mpc._dveq, mpc._dvneq, params)
    return {k: getattr(mpc, name).detach().clone() for k, name in {
        "Q": "_Q", "p": "_p", "A": "_A", "b": "_b", "G": "_G", "q": "_q",
    }.items()}


def _per_time(diff: Tensor) -> list[float]:
    reduce_dims = tuple(range(1, diff.ndim))
    reduced = diff if not reduce_dims else diff.amax(dim=reduce_dims)
    return [v.item() for v in reduced]


def _print_time_series(title: str, diff: Tensor) -> None:
    vals = _per_time(diff)
    hot = [(t, v) for t, v in enumerate(vals) if v > ATOL]
    print(f"  {title}: times above {ATOL:.0e}: {hot if hot else 'none'}")


def _q_blocks(q: Tensor) -> dict[str, Tensor]:
    nd = NB + 2 * NU
    return {"q_cbf": q[:, :NB], "q_bound": q[:, NB:nd], "q_slack": q[:, nd:]}


def _p_blocks(p: Tensor) -> dict[str, Tensor]:
    ntau = NX + NU
    return {"p_tau": p[:, :ntau], "p_x": p[:, :NX], "p_u": p[:, NX:ntau], "p_slack": p[:, ntau:]}


def _compare_pair(cur: dict[str, Tensor], ref: dict[str, Tensor], tag: str) -> None:
    print(f"\n== {tag} ==")
    for key in ("b", "q", "p", "A", "G", "Q"):
        print(_fmt(key, cur[key], ref[key]))
    flats_c, flats_r = _flat(cur), _flat(ref)
    print("  flat QP data")
    for key in ("bv", "qv", "pv", "Av", "Gv", "Qv"):
        print(_fmt(key, flats_c[key], flats_r[key]))
    print("  blocks")
    for key, block in _q_blocks(cur["q"]).items():
        print(_fmt(key, block, _q_blocks(ref["q"])[key]))
    for key, block in _p_blocks(cur["p"]).items():
        print(_fmt(key, block, _p_blocks(ref["p"])[key]))
    ntau = NX + NU
    print(_fmt("Q_terminal_xx", cur["Q"][T, :NX, T, :NX], ref["Q"][T, :NX, T, :NX]))
    print(_fmt("Q_terminal_xu", cur["Q"][T, :NX, T, NX:ntau], ref["Q"][T, :NX, T, NX:ntau]))
    print(_fmt("Q_terminal_uu", cur["Q"][T, NX:ntau, T, NX:ntau], ref["Q"][T, NX:ntau, T, NX:ntau]))
    print(_fmt("p_terminal_u", cur["p"][T, NX:ntau], ref["p"][T, NX:ntau]))
    _print_time_series("b per time", (cur["b"] - ref["b"]).abs())
    _print_time_series("q_cbf per time", (cur["q"][:, :NB] - ref["q"][:, :NB]).abs())
    q_diag = torch.stack([(cur["Q"][t, :ntau, t, :ntau] - ref["Q"][t, :ntau, t, :ntau]).abs().max() for t in range(T + 1)])
    _print_time_series("Q tau-block per time", q_diag)


def _make_modules(seed: int):
    torch.manual_seed(seed)
    cost = TvToCost(NX, NU, NB, T, DEVICE, DTYPE).to(DEVICE, DTYPE)
    dynamic = TvToDynamic(NX, NU, T, DEVICE, DTYPE, delta_t=0.05).to(DEVICE, DTYPE)
    barrier = TvToBarrier(NX, NU, NB, T, DEVICE, DTYPE).to(DEVICE, DTYPE)
    umin = torch.tensor([-1.0, -1.0], device=DEVICE, dtype=DTYPE)
    umax = torch.tensor([1.0, 1.0], device=DEVICE, dtype=DTYPE)
    mpc = DiffMPCBF(
        nx=NX, nu=NU, nb=NB, T=T, batch_size=1,
        cost=cost, dynamic=dynamic, barrier=barrier,
        umin=umin, umax=umax,
        slack_weight=[RHO1, RHO2],
        two_order_mode="nodv",
        device=DEVICE, dtype=DTYPE,
    )
    return cost, dynamic, barrier, umin, umax, mpc


def _problem(seed: int, kind: str, mpc: DiffMPCBF):
    torch.manual_seed(seed)
    ntau, nv = NX + NU, NX + NU + NB
    ng = 2 * (NB + NU)
    if kind == "rollout":
        env = Env(torch.zeros(2).numpy(), torch.tensor([1.0, 1.0]).numpy(), delta_t=0.05)
        po_np, ro_np = env.get_obs_info(T + 1)
        po = torch.tensor(po_np, dtype=DTYPE)
        ro = torch.tensor(ro_np, dtype=DTYPE)
        pref = torch.tensor([1.0, 1.0], dtype=DTYPE)
        x0 = torch.tensor(env.state, dtype=DTYPE)
    else:
        po = torch.randn(NB, T + 1, 2, dtype=DTYPE)
        ro = torch.rand(NB, T + 1, dtype=DTYPE) * 0.05 + 0.04
        pref = torch.tensor([1.0, 1.0], dtype=DTYPE)
        x0 = torch.randn(NX, 1, dtype=DTYPE)
    ra = torch.tensor(0.02, dtype=DTYPE)
    gamma = torch.tensor(GAMMA, dtype=DTYPE)
    params = {
        "cost": {"pref": pref, "obs_info": (po, ro)},
        "dynamic": {},
        "barrier": {"obs_info": (po, ro), "radius_agent": ra, "gamma": gamma},
    }
    dveq = torch.randn(T + 1, NX, dtype=DTYPE)
    dvneq = torch.randn(T, ng, dtype=DTYPE)
    if kind == "rollout":
        mpc._zeta.zero_()
        mpc._zeta[:, NX:ntau] = (torch.rand(T + 1, NU, dtype=DTYPE) * 2 - 1)
        mpc._first_forward = True
        with torch.no_grad():
            mpc._reconstruct(x0, params)
        zeta = mpc._zeta.detach().clone()
    else:
        zeta = torch.randn(T + 1, nv, dtype=DTYPE)
        zeta[:, ntau:] = zeta[:, ntau:].abs() + 1e-3
    return zeta, x0, dveq, dvneq, params


def _diagnose(cur, ref, ora, zeta, x0, params) -> None:
    print("\n== diagnosis ==")
    b_shift = torch.zeros_like(cur["b"])
    b_shift[0] = zeta[0, :NX] - x0.flatten()
    b_shift[1:] = zeta[1:, :NX] - ora["dyn_f_next"]
    b_v2 = torch.zeros_like(cur["b"])
    b_v2[0] = zeta[0, :NX] - x0.flatten()
    b_v2[1:] = zeta[1:, :NX] - ora["dyn_f"]
    print(_fmt("b vs f(x_{t+1}) shift", cur["b"], b_shift))
    print(_fmt("b vs f(x_t) v2", cur["b"], b_v2))
    print(f"  rollout-style |b_v2[1:]| max {b_v2[1:].abs().max().item():.3e}  |b_cur[1:]| max {cur['b'][1:].abs().max().item():.3e}")

    gamma = params["barrier"]["gamma"]
    slack = zeta[:T, NX + NU :]
    q_v2 = ora["bar_f"][1:] - (1 - gamma) * ora["bar_f"][:T] + slack
    print(_fmt("q_cbf vs h(x_{t+1},t+1)-(1-g)h(x_t,t)", cur["q"][:, :NB], q_v2))
    print(_fmt("ref q_cbf is v2 expression", ref["q"][:, :NB], q_v2))


def run_case(kind: str, seed: int) -> None:
    print(f"\n######## case={kind} seed={seed} T={T} dtype={DTYPE} ########")
    cost, dynamic, barrier, umin, umax, mpc = _make_modules(seed)
    zeta, x0, dveq, dvneq, params = _problem(seed + 100, kind, mpc)
    print("evaluating derivatives...")
    ora = _oracles(cost, dynamic, barrier, zeta, params)

    cur_nodv = _run_current(mpc, zeta, x0, dveq, dvneq, params, "nodv")
    cur_full = _run_current(mpc, zeta, x0, dveq, dvneq, params, "full")
    ref_nodv = assemble_v2(zeta, x0, dveq, dvneq, params, umin, umax, ora, "nodv")
    ref_full = assemble_v2(zeta, x0, dveq, dvneq, params, umin, umax, ora, "full")
    replay_nodv = assemble_current_reading(zeta, x0, dveq, dvneq, params, umin, umax, ora, "nodv")
    replay_full = assemble_current_reading(zeta, x0, dveq, dvneq, params, umin, umax, ora, "full")

    _compare_pair(cur_nodv, replay_nodv, "current code vs transcription of current indexing (nodv)")
    _compare_pair(cur_full, replay_full, "current code vs transcription of current indexing (full)")
    _compare_pair(cur_nodv, ref_nodv, "current vs v2 loop, Gauss-Newton / nodv")
    _compare_pair(cur_full, ref_full, "current vs v2 loop, full second-order")
    _diagnose(cur_nodv, ref_nodv, ora, zeta, x0, params)

    coeff_v2 = _v2_barrier_coeff(dvneq, params["barrier"]["gamma"])
    coeff_cur = _current_barrier_coeff(dvneq, params["barrier"]["gamma"])
    print("\n== barrier-Hessian coefficients, max abs over barriers ==")
    for t in range(T + 1):
        print(
            f"  t={t:02d}  v2 {coeff_v2[t].abs().max().item():.3e}  "
            f"current {coeff_cur[t].abs().max().item():.3e}  "
            f"|diff| {(coeff_v2[t] - coeff_cur[t]).abs().max().item():.3e}"
        )

    # Second-order increment the running code actually adds.
    delta = cur_full["Q"] - cur_nodv["Q"]
    delta_replay = replay_full["Q"] - replay_nodv["Q"]
    print(_fmt("full-minus-nodv vs current formula", delta, delta_replay))
    delta_v2 = ref_full["Q"] - ref_nodv["Q"]
    print(_fmt("full-minus-nodv vs v2 formula", delta, delta_v2))


def main() -> None:
    run_case("rollout", seed=0)
    run_case("random", seed=1)


if __name__ == "__main__":
    main()
