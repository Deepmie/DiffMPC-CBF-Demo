"""Trace NLP KKT stationarity over successive MPC/environment steps."""
from __future__ import annotations

import os
import random
import sys
import types

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "tvto"))

from diffmpc_cbf import DiffMPCBF
from diffmpc_cbf.base import SQPStatus
from diffmpc_cbf.utils import convert_any_to_tensor_batch
from env import Env
from system import TvToBarrier, TvToCost, TvToDynamic


def norm_inf(x: torch.Tensor) -> float:
    return float(x.detach().norm(torch.inf).cpu())


def main() -> None:
    kkt_policy = os.getenv("KKT_POLICY", "stored")
    if kkt_policy not in {"stored", "current_qp"}:
        raise ValueError("KKT_POLICY must be 'stored' or 'current_qp'")
    print(f"kkt_policy={kkt_policy}")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    dtype = torch.float64
    nx, nu, nb, horizon = 4, 2, 2, 15
    delta_t = 0.05

    cost = TvToCost(nx, nu, horizon).to(device, dtype)
    dynamic = TvToDynamic(nx, nu, horizon, delta_t).to(device, dtype)
    barrier = TvToBarrier(nx, nu, horizon, nb).to(device, dtype)
    mpc = DiffMPCBF(
        nx=nx,
        nu=nu,
        nb=nb,
        T=horizon,
        batch_size=1,
        cost=cost,
        dynamic=dynamic,
        barrier=barrier,
        umin=torch.tensor([-1.0, -1.0], device=device, dtype=dtype),
        umax=torch.tensor([1.0, 1.0], device=device, dtype=dtype),
        dtype=dtype,
    ).to(device)

    env = Env(np.zeros(2), np.ones(2), delta_t=delta_t)
    pref = torch.tensor([[1.0], [1.0]], device=device, dtype=dtype)
    mpc.reset()

    current_k = -1
    checks: list[dict[str, float | int]] = []
    alphas: list[float] = []
    last_step_inf = float("nan")
    last_qp_dual_stat = float("nan")
    last_qp_dual_comp = float("nan")
    last_metric_dir = float("nan")
    last_dveq_qp: torch.Tensor | None = None
    last_dvneq_qp: torch.Tensor | None = None

    original_check = mpc._check_nlp_kkt
    original_solve = mpc._solve_qp_subproblem
    original_line_search = mpc._line_search

    def traced_solve(self, *args, **kwargs):
        nonlocal last_step_inf, last_qp_dual_stat, last_qp_dual_comp
        nonlocal last_metric_dir, last_dveq_qp, last_dvneq_qp
        result = original_solve(*args, **kwargs)
        step, dveq_qp, dvneq_qp = result
        last_step_inf = norm_inf(step)
        last_qp_dual_stat = norm_inf(
            self._pv
            + self._Av.T @ dveq_qp.flatten()
            + self._Gv.T @ dvneq_qp.flatten()
        )
        last_qp_dual_comp = norm_inf(dvneq_qp.flatten() * self._qv)
        last_metric_dir = float(
            self._metricfunc_direct(step, dveq_qp, dvneq_qp).detach().cpu()
        )
        last_dveq_qp = dveq_qp
        last_dvneq_qp = dvneq_qp
        return result

    def traced_check(self):
        p = self._pv
        eq_term = self._Av.T @ self._dveqv
        ineq_term = self._Gv.T @ self._dvneqv
        residual = p + eq_term + ineq_term
        stat = norm_inf(residual)
        max_term = max(1.0, norm_inf(p), norm_inf(eq_term), norm_inf(ineq_term))
        sum_scale = 1.0 + norm_inf(p) + norm_inf(eq_term) + norm_inf(ineq_term)
        max_index = int(residual.detach().abs().argmax())
        checks.append(
            {
                "k": current_k,
                "iteration": len(checks),
                "stat": stat,
                "relative_max": stat / max_term,
                "relative_sum": stat / sum_scale,
                "p": norm_inf(p),
                "eq": norm_inf(eq_term),
                "ineq": norm_inf(ineq_term),
                "dual": norm_inf((-self._dvneqv).clamp_min(0)),
                "comp": norm_inf(self._dvneqv * self._qv),
                "step": last_step_inf,
                "qp_dual_stat": last_qp_dual_stat,
                "qp_dual_comp": last_qp_dual_comp,
                "metric_dir": last_metric_dir,
                "max_index": max_index,
            }
        )
        if kkt_policy == "stored":
            return original_check()
        assert last_dveq_qp is not None and last_dvneq_qp is not None
        current_qp_dual = norm_inf((-last_dvneq_qp).clamp_min(0))
        return torch.tensor(
            last_qp_dual_stat < 1e-3
            and current_qp_dual < 1e-4
            and last_qp_dual_comp < 1e-4,
            device=self._device,
        )

    def traced_line_search(self, *args, **kwargs):
        result = original_line_search(*args, **kwargs)
        alphas.append(float(result[1]))
        return result

    mpc._solve_qp_subproblem = types.MethodType(traced_solve, mpc)
    mpc._check_nlp_kkt = types.MethodType(traced_check, mpc)
    mpc._line_search = types.MethodType(traced_line_search, mpc)

    for k in range(79):
        current_k = k
        checks.clear()
        alphas.clear()

        obs, radius, gamma = convert_any_to_tensor_batch(
            [env.get_obs_info(horizon + 1), env.radius_agent, 0.2], device, dtype
        )
        params = {
            "cost": {"pref": pref, "obs_info": obs},
            "barrier": {
                "obs_info": obs,
                "radius_agent": radius,
                "gamma": gamma,
            },
        }
        state = torch.from_numpy(env.state.copy()).requires_grad_(True).to(device, dtype)
        solution, status = mpc(state, params)

        stats = [entry for entry in checks if entry["k"] == k]
        best = min(stats, key=lambda entry: float(entry["stat"]))
        final = stats[-1]
        print(
            f"k={k:02d} status={status.name:<18} checks={len(stats):02d} "
            f"best_stat={float(best['stat']):.3e} final_stat={float(final['stat']):.3e} "
            f"final_rel={float(final['relative_max']):.3e} "
            f"final_step={float(final['step']):.3e} "
            f"min_alpha={min(alphas) if alphas else float('nan'):.3e}"
        )

        if status != SQPStatus.CONVERGED:
            print("\nDetailed failed-step trace:")
            for i, entry in enumerate(stats):
                variable_index = int(entry["max_index"])
                stage, within_stage = divmod(variable_index, mpc._nv)
                print(
                    f"  i={i:02d} stat={float(entry['stat']):.6e} "
                    f"rel_max={float(entry['relative_max']):.3e} "
                    f"rel_sum={float(entry['relative_sum']):.3e} "
                    f"terms=({float(entry['p']):.3e},{float(entry['eq']):.3e},"
                    f"{float(entry['ineq']):.3e}) "
                    f"dual={float(entry['dual']):.3e} comp={float(entry['comp']):.3e} "
                    f"step={float(entry['step']):.3e} "
                    f"qp_dual_stat={float(entry['qp_dual_stat']):.3e} "
                    f"qp_dual_comp={float(entry['qp_dual_comp']):.3e} "
                    f"metric_dir={float(entry['metric_dir']):+.3e} "
                    f"max_at=(stage={stage},component={within_stage})"
                )
            break

        control = solution[nx : nx + nu].detach().cpu().numpy().reshape(-1, 1)
        _, done = env.step(control)
        if done:
            print(f"environment finished at k={k}")
            break


if __name__ == "__main__":
    main()
