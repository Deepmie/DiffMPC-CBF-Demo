"""Trace QP direction quality and slack feasibility on the first few tvto steps."""
from __future__ import annotations

import importlib
import os
import random
import sys

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "examples", "tvto"))

from diffmpc_cbf import DiffMPCBF
from diffmpc_cbf.utils import convert_any_to_tensor_batch
from env import Env
from system import TvToBarrier, TvToCost, TvToDynamic


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)

    nx, nu, nb, horizon = 4, 2, 2, 15
    dtype = torch.float64
    device = torch.device("cpu")
    mpc = DiffMPCBF(
        nx=nx,
        nu=nu,
        nb=nb,
        T=horizon,
        batch_size=1,
        cost=TvToCost(nx, nu, horizon),
        dynamic=TvToDynamic(nx, nu, horizon, 0.05),
        barrier=TvToBarrier(nx, nu, horizon, nb),
        umin=torch.tensor([-1.0, -1.0], dtype=dtype),
        umax=torch.tensor([1.0, 1.0], dtype=dtype),
        start_dv_eps_detection=True,
        device=device,
        dtype=dtype,
    )
    mpc.to(device=device, dtype=dtype)
    env = Env(np.zeros(2), np.ones(2), 0.05)
    pref = torch.ones(2, 1, dtype=dtype)

    solver_module = importlib.import_module("diffmpc_cbf.diffmpc_cbf")
    original_qp = solver_module.QPFunction
    original_metric = mpc._metricfunc
    original_zero = mpc._get_zero_order_term
    original_init_eq = mpc._set_init_dveq
    original_init_ineq = mpc._set_init_dvneq

    context = {"step": -1, "qp": 0}

    def traced_qp(*args, **kwargs):
        solve = original_qp(*args, **kwargs)

        def apply(Q, p, G, q, A, b):
            context["qp"] += 1
            primal, eq_dual, ineq_dual = solve(Q, p, G, q, A, b)
            d = primal.flatten()
            model = float(p @ d + 0.5 * (d @ Q @ d))
            p_dot = float(p @ d)
            eq_res = float((A @ d - b).abs().max())
            ineq_res = float((G @ d - q).clamp_min(0).max())
            stationarity = float(
                (Q @ d + p + A.T @ eq_dual.flatten() + G.T @ ineq_dual.flatten()).abs().max()
            )
            eig_min = float(torch.linalg.eigvalsh(0.5 * (Q + Q.T)).min())
            print(
                f"step={context['step']} qp={context['qp']} "
                f"eigmin={eig_min:.3e} pTd={p_dot:.3e} model={model:.3e} "
                f"eq={eq_res:.3e} ineq={ineq_res:.3e} stat={stationarity:.3e} "
                f"dmax={d.abs().max().item():.3e} slack_dmax={d[mpc._var_dims:].abs().max().item():.3e}",
                flush=True,
            )
            return primal, eq_dual, ineq_dual

        return apply

    def traced_metric(*args, **kwargs):
        requested = kwargs.get("include_res", False)
        kwargs["include_res"] = True
        value, residual = original_metric(*args, **kwargs)
        slack = args[1]
        dveq = args[2]
        if requested:
            print(
                f"  base merit={float(value):.9e} res={residual.flatten().tolist()} "
                f"slack[min,max]=[{float(slack.min()):.3e},{float(slack.max()):.3e}] "
                f"eq_dual_inf={float(dveq.abs().max()):.3e} eq_dual_l1={float(dveq.abs().sum()):.3e}",
                flush=True,
            )
            return value, residual
        print(
            f"  trial merit={float(value):.9e} res={residual.flatten().tolist()} "
            f"slack[min,max]=[{float(slack.min()):.3e},{float(slack.max()):.3e}]",
            flush=True,
        )
        return value

    def traced_zero(tau, x0, params):
        h, g_cbf, g_boundary = original_zero(tau, x0, params)
        if context.get("print_zero"):
            print(
                f"  init h_inf={float(h.abs().max()):.3e} "
                f"g_cbf[min,max]=[{float(g_cbf.min()):.3e},{float(g_cbf.max()):.3e}] "
                f"g_bound_max={float(g_boundary.max()):.3e}",
                flush=True,
            )
            context["print_zero"] = False
        return h, g_cbf, g_boundary

    def traced_init_eq():
        before = mpc._dveq.abs().max().item() if hasattr(mpc, "_dveq") else float("nan")
        original_init_eq()
        print(
            f"  dveq after shift inf={float(mpc._dveq.abs().max()):.3e} before={before:.3e}",
            flush=True,
        )
        return mpc._dveq

    def traced_init_ineq():
        original_init_ineq()
        print(
            f"  dvneq after shift inf={float(mpc._dvneq.abs().max()):.3e}",
            flush=True,
        )
        return mpc._dvneq

    solver_module.QPFunction = traced_qp
    mpc._metricfunc = traced_metric
    mpc._get_zero_order_term = traced_zero
    mpc._set_init_dveq = traced_init_eq
    mpc._set_init_dvneq = traced_init_ineq

    try:
        for step in range(4):
            context["step"] = step
            context["qp"] = 0
            context["print_zero"] = True
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
            print(f"\n=== STEP {step} is_first={mpc._is_first} ===", flush=True)
            state = torch.from_numpy(env.state.copy()).to(dtype=dtype)
            tau, status = mpc(state, params)
            print(
                f"STATUS {status.name} u0={tau.u[0].detach().tolist()} "
                f"final_slack[min,max]=[{float(mpc._slack.min()):.3e},{float(mpc._slack.max()):.3e}]",
                flush=True,
            )
            context["print_zero"] = False
            _, done = env.step(tau.u[0].detach().numpy().reshape(-1, 1))
            if status.name == "LINE_SEARCH_FAILED" or done:
                break
    finally:
        solver_module.QPFunction = original_qp


if __name__ == "__main__":
    main()
