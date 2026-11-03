"""Reproduce tvto training while tracing Q conditioning and QP residuals."""
from __future__ import annotations

import importlib
import math
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "examples", "tvto"))

from diffmpc_cbf import DiffMPCBF
from diffmpc_cbf.utils import convert_any_to_tensor_batch, convert_tensor_iteration, tcon
from env import Env
from system import TvToBarrier, TvToCost, TvToDynamic


@dataclass
class Location:
    epoch: int = -1
    step: int = -1
    qp: int = 0


loc = Location()
solver_module = importlib.import_module("diffmpc_cbf.diffmpc_cbf")
original_qp_function = solver_module.QPFunction


def tensor_stats(x: torch.Tensor) -> str:
    x = x.detach()
    finite = bool(torch.isfinite(x).all())
    if not finite:
        return f"finite=False nan={torch.isnan(x).sum().item()} inf={torch.isinf(x).sum().item()}"
    return (
        f"finite=True min={x.min().item():.3e} max={x.max().item():.3e} "
        f"absmax={x.abs().max().item():.3e}"
    )


def q_stats(Q: torch.Tensor) -> dict[str, float | int | bool]:
    q = Q.detach()
    q64 = q.double()
    finite = bool(torch.isfinite(q).all())
    if not finite:
        return {
            "finite": False,
            "chol32": -1,
            "chol64": -1,
            "eig_min": math.nan,
            "eig_max": math.nan,
            "cond": math.inf,
            "sym": math.nan,
        }
    eig = torch.linalg.eigvalsh(0.5 * (q64 + q64.T))
    eig_min = eig.min().item()
    eig_max = eig.max().item()
    return {
        "finite": True,
        "chol32": int(torch.linalg.cholesky_ex(q).info.max().item()),
        "chol64": int(torch.linalg.cholesky_ex(q64).info.max().item()),
        "eig_min": eig_min,
        "eig_max": eig_max,
        "cond": abs(eig_max / eig_min) if eig_min != 0 else math.inf,
        "sym": (q64 - q64.T).abs().max().item(),
    }


def traced_qp_function(*args, **kwargs):
    apply_qp = original_qp_function(*args, **kwargs)

    def apply(Q, p, G, q, A, b):
        loc.qp += 1
        stats = q_stats(Q)
        prefix = f"[epoch={loc.epoch} step={loc.step} qp={loc.qp}]"
        suspicious = (
            not stats["finite"]
            or stats["chol32"] != 0
            or stats["eig_min"] < 5e-5
            or stats["cond"] > 1e7
        )
        if suspicious:
            print(prefix, "Q_PRE", stats, "p", tensor_stats(p))
        try:
            primal, eq_dual, ineq_dual = apply_qp(Q, p, G, q, A, b)
        except Exception:
            print(prefix, "QP_EXCEPTION", stats)
            print(prefix, "Q", tensor_stats(Q))
            print(prefix, "p", tensor_stats(p))
            print(prefix, "G", tensor_stats(G), "q", tensor_stats(q))
            print(prefix, "A", tensor_stats(A), "b", tensor_stats(b))
            raise

        with torch.no_grad():
            z = primal.flatten()
            y = eq_dual.flatten()
            lam = ineq_dual.flatten()
            eq = A @ z - b
            gap = q - G @ z
            stationarity = Q @ z + p + A.T @ y + G.T @ lam
            residuals = {
                "eq_inf": eq.abs().max().item(),
                "ineq_inf": torch.relu(-gap).max().item(),
                "dual_inf": torch.relu(-lam).max().item(),
                "stationarity_inf": stationarity.abs().max().item(),
                "complementarity_inf": (lam * gap).abs().max().item(),
                "gap_min": gap.min().item(),
                "lambda_max": lam.abs().max().item(),
            }
            if suspicious or max(residuals.values()) > 1e-2:
                print(prefix, "QP_POST", residuals)
        return primal, eq_dual, ineq_dual

    return apply


def grad_stats(module: torch.nn.Module) -> tuple[float, float, bool]:
    total_sq = 0.0
    max_abs = 0.0
    finite = True
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(grad).all())
        total_sq += grad.double().square().sum().item()
        max_abs = max(max_abs, grad.abs().max().item())
    return math.sqrt(total_sq), max_abs, finite


def parameter_stats(module: torch.nn.Module) -> tuple[float, bool]:
    max_abs = 0.0
    finite = True
    for parameter in module.parameters():
        value = parameter.detach()
        finite = finite and bool(torch.isfinite(value).all())
        max_abs = max(max_abs, value.abs().max().item())
    return max_abs, finite


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    device = torch.device("cpu")
    dtype = torch.float32
    nx, nu, nb, horizon = 4, 2, 2, 15
    delta_t, episode_steps = 0.05, 80

    cost = TvToCost(nx, nu, horizon)
    dynamic = TvToDynamic(nx, nu, horizon, delta_t)
    barrier = TvToBarrier(nx, nu, horizon, nb)
    mpc = DiffMPCBF(
        nx=nx,
        nu=nu,
        nb=nb,
        T=horizon,
        batch_size=1,
        cost=cost,
        dynamic=dynamic,
        barrier=barrier,
        umin=torch.tensor([-1.0, -1.0]),
        umax=torch.tensor([1.0, 1.0]),
        dtype=dtype,
        start_dv_eps_detection=True,
    )
    env = Env(np.zeros(2), np.ones(2), delta_t)
    pref = torch.ones(2, 1, requires_grad=True)
    optimizer = torch.optim.Adam(mpc.parameters(), lr=5e-4)
    solver_module.QPFunction = traced_qp_function

    try:
        for epoch in range(5):
            loc.epoch = epoch
            loc.qp = 0
            env.reset()
            mpc.reset()
            xs, xs_real, us, barriers = [], [], [], []
            statuses = {}

            for step in range(episode_steps - 1):
                loc.step = step
                obs, radius, gamma = convert_any_to_tensor_batch(
                    [env.get_obs_info(horizon + 1), env.radius_agent, 0.2],
                    device,
                    dtype,
                )
                params = {
                    "cost": {"pref": pref, "obs_info": obs},
                    "barrier": {
                        "obs_info": obs,
                        "radius_agent": radius,
                        "gamma": gamma,
                    },
                }
                state = torch.from_numpy(env.state.copy()).requires_grad_(True).float()
                tau, status = mpc(state, params)
                statuses[status.name] = statuses.get(status.name, 0) + 1
                state_real, done = env.step(tau.u[0].detach().numpy().reshape(-1, 1))
                xs.append(tau.x[1])
                us.append(tau.u[0])
                barriers.append(
                    barrier(
                        tau.x[1].reshape(-1, 1),
                        tcon(params["barrier"], 1, device, dtype),
                    ).flatten()
                )
                xs_real.append(convert_tensor_iteration(state_real.copy(), device, dtype).flatten())
                if done:
                    break

            xs_t = torch.stack(xs)
            us_t = torch.stack(us)
            barriers_t = torch.stack(barriers)
            xs_real_t = torch.stack(xs_real)
            loss_obj = (xs_t[:, :2] - pref.reshape(1, -1)).square().mean()
            loss_control = us_t.square().mean()
            loss_safe = torch.relu(-barriers_t).square().mean()
            loss_dynamic = (xs_t - xs_real_t).square().mean()
            loss = loss_obj + loss_control + 1e4 * loss_safe + 1e3 * loss_dynamic

            optimizer.zero_grad()
            loss.backward()
            grad_norm, grad_max, grad_finite = grad_stats(mpc)
            param_before, param_finite_before = parameter_stats(mpc)
            optimizer.step()
            param_after, param_finite_after = parameter_stats(mpc)
            print(
                f"EPOCH {epoch} loss={loss.item():.6e} "
                f"parts=({loss_obj.item():.3e},{loss_control.item():.3e},"
                f"{loss_safe.item():.3e},{loss_dynamic.item():.3e}) "
                f"grad_norm={grad_norm:.3e} grad_max={grad_max:.3e} "
                f"grad_finite={grad_finite} param_max={param_before:.3e}->{param_after:.3e} "
                f"param_finite={param_finite_before}/{param_finite_after} "
                f"statuses={statuses} qps={loc.qp}",
                flush=True,
            )
    finally:
        solver_module.QPFunction = original_qp_function


if __name__ == "__main__":
    main()
