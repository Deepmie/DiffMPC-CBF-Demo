"""Verify spnn_ex against the recommended neural-cost checklist."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "spnn_ex"))

from system import SpnnCost, SpnnDynamic
from diffmpc_cbf import DiffMPC as SQPMPC
from diffmpc_cbf.lqr import DiffMPC as ILQRMPC


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_cost_api() -> None:
    section("1) Cost formulation / API")
    torch.manual_seed(0)
    cost = SpnnCost(2, 1, 5)
    dynamic = SpnnDynamic(2, 1, 5, 0.1)
    xref = torch.tensor([[1.0], [0.0]], requires_grad=True)
    tau = torch.tensor([0.2, -0.1, 0.3], requires_grad=True)
    params = {"xref": xref, "t": 0}

    value = cost._define_stage_cost_function(tau, params)
    print("stage value", value, "shape", tuple(value.shape), "ndim", value.ndimension())
    print("network out > 0?", (cost.context_network(torch.tensor([1.0, 0.0, 0.0])) > 0).all().item())

    # weights depend on context, not on tau quadratic features
    ctx = torch.stack([xref[0], xref[1], torch.tensor([0.0])]).flatten()
    w = cost.context_network(ctx)
    print("weights", w.detach())

    # autograd derivatives
    try:
        g = cost.get_stage_cost(tau, params, order=1)
        H = cost.get_stage_cost(tau, params, order=2)
        print("grad shape", tuple(g.shape), "hess shape", tuple(H.shape))
        print("hess eigs", torch.linalg.eigvalsh(H.detach()).tolist())
        print("hess PSD?", bool(torch.linalg.eigvalsh(H.detach()).min() >= -1e-5))
    except Exception as e:
        print("derivative FAILED:", type(e).__name__, e)

    # dynamic: wrapper expects [nx,1], current stack returns [nx]
    try:
        xnext = dynamic.forward(tau, {"t": 0})
        print("dynamic out", xnext.detach(), "shape", tuple(xnext.shape), "grad_fn", xnext.grad_fn)
    except Exception as e:
        print("dynamic.forward(tau[3]) FAILS:", type(e).__name__, e)
        raw = dynamic._define_state_transition_equation(tau, {"t": 0})
        print("  raw dynamic shape", tuple(raw.shape), "(expected [nx,1] by DerivativesFunctionWrapper)")

    # document xref shape fragility
    try:
        cost._define_stage_cost_function(tau, {"xref": torch.tensor([1.0, 0.0]), "t": 0})
        print("xref shape (2,) OK")
    except Exception as e:
        print("xref shape (2,) FAILS:", type(e).__name__, e)


def run_rollout(solver_name: str, make_mpc, n_steps: int = 50, bound: float = 10.0):
    section(f"2) {solver_name} rollout ({n_steps} steps)")
    torch.manual_seed(1)
    nx, nu, T = 2, 1, 5
    cost = SpnnCost(nx, nu, T)
    dynamic = SpnnDynamic(nx, nu, T, 0.1)
    umin = torch.tensor([-bound])
    umax = torch.tensor([bound])
    mpc = make_mpc(nx, nu, T, cost, dynamic, umin, umax)
    xref = torch.tensor([[1.0], [0.0]])
    xs = torch.zeros(n_steps, nx)
    us = []
    eigmins = []
    for t in range(1, n_steps):
        if solver_name == "SQP":
            tau = mpc.step(xs[t - 1], {"cost": {"xref": xref}})
            u0 = tau.u[0].detach()
            # inspect local stage Hessian at current state/control
            z = torch.cat([xs[t - 1], u0.view(-1)])
            H = cost.get_stage_cost(z, {"xref": xref, "t": 0}, order=2).detach()
            eigmins.append(float(torch.linalg.eigvalsh(H).min()))
            xs[t] = dynamic.forward((xs[t - 1], u0), {"t": 0}).flatten().detach()
            us.append(float(u0))
        else:
            _, u = mpc.step(xs[t - 1], {"cost": {"xref": xref}})
            u0 = u[0].detach()
            z = torch.cat([xs[t - 1], u0.view(-1)])
            H = cost.get_stage_cost(z, {"xref": xref, "t": 0}, order=2).detach()
            eigmins.append(float(torch.linalg.eigvalsh(H).min()))
            xs[t] = dynamic.forward((xs[t - 1], u0), {"t": 0}).flatten().detach()
            us.append(float(u0))
    err = (xs[-1] - xref.flatten()).norm().item()
    print("final x", xs[-1].tolist())
    print("final |x-xref|", err)
    print("u0 first/last", us[0], us[-1])
    print("stage hess eig_min over traj: min/mean", min(eigmins), sum(eigmins) / len(eigmins))
    return xs, us


def check_grad_and_fd(solver_name: str, make_mpc, bound: float = 100.0):
    section(f"3) {solver_name} network-parameter gradients + FD")
    torch.manual_seed(1)
    nx, nu, T = 2, 1, 5
    cost = SpnnCost(nx, nu, T)
    dynamic = SpnnDynamic(nx, nu, T, 0.1)
    mpc = make_mpc(nx, nu, T, cost, dynamic, torch.tensor([-bound]), torch.tensor([bound]))
    xref = torch.tensor([[1.0], [0.0]])
    x0 = torch.zeros(nx)

    if solver_name == "SQP":
        tau = mpc.step(x0, {"cost": {"xref": xref}})
        u = tau.u.reshape(-1)
    else:
        _, u = mpc.step(x0, {"cost": {"xref": xref}})
        u = u.reshape(-1)

    params = list(cost.context_network.parameters())
    loss = 0.5 * (u**2).sum()
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    grad_norms = []
    for p, g in zip(params, grads):
        if g is None:
            grad_norms.append(0.0)
            print(f"  {tuple(p.shape)}: grad is None")
        else:
            grad_norms.append(float(g.norm()))
            print(f"  {tuple(p.shape)}: grad_norm={g.norm().item():.6e}")

    # directional finite difference on flattened params
    flat = torch.cat([p.detach().reshape(-1) for p in params])
    direction = torch.randn_like(flat)
    direction = direction / direction.norm()
    analytic = 0.0
    offset = 0
    for p, g in zip(params, grads):
        n = p.numel()
        if g is not None:
            analytic += float((g.detach().reshape(-1) * direction[offset : offset + n]).sum())
        offset += n

    def loss_at(eps: float) -> float:
        torch.manual_seed(1)
        cost2 = SpnnCost(nx, nu, T)
        # copy base weights then perturb
        with torch.no_grad():
            base = list(cost.context_network.parameters())
            cur = list(cost2.context_network.parameters())
            off = 0
            for b, c in zip(base, cur):
                n = b.numel()
                c.copy_(b + eps * direction[off : off + n].view_as(b))
                off += n
        dyn2 = SpnnDynamic(nx, nu, T, 0.1)
        mpc2 = make_mpc(nx, nu, T, cost2, dyn2, torch.tensor([-bound]), torch.tensor([bound]))
        if solver_name == "SQP":
            u2 = mpc2.step(x0, {"cost": {"xref": xref}}).u.reshape(-1)
        else:
            u2 = mpc2.step(x0, {"cost": {"xref": xref}})[1].reshape(-1)
        return float(0.5 * (u2**2).sum())

    eps = 1e-3
    fd = (loss_at(eps) - loss_at(-eps)) / (2 * eps)
    rel = abs(analytic - fd) / max(1.0, abs(analytic), abs(fd))
    print(f"analytic directional={analytic:.6e}")
    print(f"FD directional={fd:.6e}")
    print(f"relative error={rel:.6e}")
    return rel, grad_norms


def compare_sqp_ilqr(n_steps: int = 10, bound: float = 100.0):
    section("4) SQP vs iLQR control agreement")
    torch.manual_seed(1)
    nx, nu, T = 2, 1, 5
    cost = SpnnCost(nx, nu, T)
    dynamic = SpnnDynamic(nx, nu, T, 0.1)
    xref = torch.tensor([[1.0], [0.0]])
    sqp = SQPMPC(nx, nu, T, 1, cost, dynamic, torch.tensor([-bound]), torch.tensor([bound]))
    # separate cost/dynamic copies with same weights for iLQR
    cost_l = SpnnCost(nx, nu, T)
    with torch.no_grad():
        for a, b in zip(cost.context_network.parameters(), cost_l.context_network.parameters()):
            b.copy_(a)
    dynamic_l = SpnnDynamic(nx, nu, T, 0.1)
    ilqr = ILQRMPC(nx, nu, T, 1, cost_l, dynamic_l, torch.tensor([-bound]), torch.tensor([bound]))

    xs = torch.zeros(n_steps, nx)
    diffs = []
    for t in range(1, n_steps):
        tau = sqp.step(xs[t - 1], {"cost": {"xref": xref}})
        _, u_l = ilqr.step(xs[t - 1], {"cost": {"xref": xref}})
        du = (tau.u.detach() - u_l.detach()).norm().item()
        diffs.append(du)
        print(f"  t={t}: ||u_sqp-u_ilqr||={du:.3e}, u_sqp0={float(tau.u[0]):.4f}, u_ilqr0={float(u_l[0]):.4f}")
        xs[t] = dynamic.forward((xs[t - 1], tau.u[0].detach()), {"t": 0}).flatten()
    print("max control diff", max(diffs))


def make_sqp(nx, nu, T, cost, dynamic, umin, umax):
    return SQPMPC(nx, nu, T, 1, cost, dynamic, umin, umax)


def make_ilqr(nx, nu, T, cost, dynamic, umin, umax):
    return ILQRMPC(nx, nu, T, 1, cost, dynamic, umin, umax)


if __name__ == "__main__":
    check_cost_api()
    run_rollout("SQP", make_sqp, n_steps=50, bound=10.0)
    run_rollout("iLQR", make_ilqr, n_steps=50, bound=10.0)
    rel_sqp, _ = check_grad_and_fd("SQP", make_sqp, bound=100.0)
    rel_ilqr, _ = check_grad_and_fd("iLQR", make_ilqr, bound=100.0)
    compare_sqp_ilqr(n_steps=8, bound=100.0)
    section("Summary thresholds")
    print("FD relative error SQP ", rel_sqp, "OK?" , rel_sqp < 5e-2)
    print("FD relative error iLQR", rel_ilqr, "OK?", rel_ilqr < 5e-2)
