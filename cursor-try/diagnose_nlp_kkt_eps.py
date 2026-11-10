"""Diagnose NLP KKT residual scales without modifying solver source."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "examples", "tvto"))

from diffmpc_cbf import DiffMPCBF  # noqa: E402
from diffmpc_cbf.utils import convert_any_to_tensor_batch  # noqa: E402
from env import Env  # noqa: E402
from system import TvToBarrier, TvToCost, TvToDynamic  # noqa: E402


def build_case(max_iter: int = 10, dtype: torch.dtype = torch.float32):
    torch.manual_seed(1)
    np.random.seed(1)
    nx, nu, nb, horizon = 4, 2, 2, 5
    cost = TvToCost(nx, nu, horizon).to(dtype=dtype)
    dynamic = TvToDynamic(nx, nu, horizon, 0.05).to(dtype=dtype)
    barrier = TvToBarrier(nx, nu, horizon, nb).to(dtype=dtype)
    mpc = DiffMPCBF(
        nx=nx,
        nu=nu,
        nb=nb,
        T=horizon,
        batch_size=1,
        cost=cost,
        dynamic=dynamic,
        barrier=barrier,
        umin=torch.tensor([-1.0, -1.0], dtype=dtype),
        umax=torch.tensor([1.0, 1.0], dtype=dtype),
        max_iter_nums=max_iter,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    env = Env(np.zeros(2), np.ones(2), delta_t=0.05)
    obs, radius, gamma = convert_any_to_tensor_batch(
        [env.get_obs_info(horizon + 1), env.radius_agent, 0.5],
        torch.device("cpu"),
        dtype,
    )
    params = {
        "cost": {"pref": torch.ones(2, 1, dtype=dtype), "obs_info": obs},
        "dynamic": {},
        "barrier": {"obs_info": obs, "radius_agent": radius, "gamma": gamma},
    }
    x = torch.from_numpy(env.state.copy()).to(dtype=dtype).unsqueeze(-1)
    return mpc, x, params


def residuals(mpc, delta_w, dveq, dvneq, x, params):
    candidate = mpc._zeta + delta_w
    mpc._update_zero_order_term(candidate, x, params, "temp")
    mpc._update_one_order_term(candidate, params, "temp")

    eq = (-mpc._btmpv).norm(torch.inf)
    ineq = (-mpc._qtmpv).clamp_min(0.0).norm(torch.inf)
    stationarity = (
        mpc._ptmpv + mpc._Atmpv.T @ dveq + mpc._Gtmpv.T @ dvneq
    ).norm(torch.inf)
    dual = (-dvneq).clamp_min(0.0).norm(torch.inf)
    comp = (dvneq * mpc._qtmpv).norm(torch.inf)

    grad_scale = (
        1.0
        + mpc._ptmpv.norm(torch.inf)
        + (mpc._Atmpv.T @ dveq).norm(torch.inf)
        + (mpc._Gtmpv.T @ dvneq).norm(torch.inf)
    )
    comp_scale = 1.0 + dvneq.norm(torch.inf) * mpc._qtmpv.norm(torch.inf)
    return {
        "eq": eq.item(),
        "ineq": ineq.item(),
        "stat": stationarity.item(),
        "dual": dual.item(),
        "comp": comp.item(),
        "stat_scaled": (stationarity / grad_scale).item(),
        "comp_scaled": (comp / comp_scale).item(),
        "q_min": mpc._qtmpv.min().item(),
    }


def run_case(correct_initial_slack: bool, dtype: torch.dtype):
    mpc, x, params = build_case(dtype=dtype)
    label = "corrected-zeta-slack" if correct_initial_slack else "current-source"
    label += f"-{str(dtype).removeprefix('torch.')}"
    print(f"\n=== {label} ===")

    with torch.no_grad():
        mpc._reconstruct(x, params)
        zeta_slack = mpc._zeta[: mpc._T, mpc._ntau :]
        print(
            "after reconstruct:",
            f"self._slack max={mpc._slack.abs().max().item():.3e}",
            f"zeta slack max={zeta_slack.abs().max().item():.3e}",
            f"shares_storage={mpc._slack.data_ptr() == zeta_slack.data_ptr()}",
        )

        if correct_initial_slack:
            # q_cbf = barrier margin + slack; with current zeta slack=0,
            # the minimum nonnegative slack is max(0, -margin) plus a margin.
            mpc._update_zero_order_term(mpc._zeta, x, params)
            slack = (-mpc._q[:, : mpc._nb]).clamp_min(0.0) + 1e-3
            mpc._zeta[: mpc._T, mpc._ntau :].copy_(slack)

        for iteration in range(mpc._max_iter_nums):
            delta, dveq, dvneq = mpc._solve_qp_subproblem(mpc._zeta, x, params)
            delta_w = torch.nn.functional.pad(
                delta, (0, mpc._nb), mode="constant", value=0
            ).view(mpc._T + 1, mpc._nv)
            model = 0.5 * (delta @ mpc._Qv @ delta) + mpc._pv @ delta
            values = residuals(mpc, delta_w, dveq, dvneq, x, params)
            source_kkt = bool(
                mpc._check_nlp_kkt(delta_w, dveq, dvneq, x, params).item()
            )
            print(
                f"iter={iteration:02d}",
                f"model={model.item(): .3e}",
                f"step={delta.norm(torch.inf).item():.3e}",
                f"source_kkt={source_kkt}",
                " ".join(f"{key}={value:.3e}" for key, value in values.items()),
            )

            accepted, alpha = mpc._line_search(
                delta_w, dveq, dvneq, x, params
            )
            print(f"         line_search accepted={accepted} alpha={alpha:.3e}")
            if not accepted:
                break
            mpc._zetav += alpha * delta
            mpc._dveq.copy_(dveq.view_as(mpc._dveq))
            mpc._dvneq.copy_(dvneq.view_as(mpc._dvneq))


def run_unmodified_forward(dtype: torch.dtype):
    mpc, x, params = build_case(dtype=dtype)
    _, status = mpc.forward(x, params)
    print(f"unmodified forward {dtype}: {status}")


if __name__ == "__main__":
    for test_dtype in (torch.float32, torch.float64):
        run_unmodified_forward(test_dtype)
        run_case(correct_initial_slack=False, dtype=test_dtype)
        run_case(correct_initial_slack=True, dtype=test_dtype)
