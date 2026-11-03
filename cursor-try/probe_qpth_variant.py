"""Compare one tvto rollout under QP-conditioning variants."""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
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


parser = argparse.ArgumentParser()
parser.add_argument("--dtype", choices=["float32", "float64"], required=True)
parser.add_argument("--rho1", type=float, required=True)
parser.add_argument("--rho2", type=float, required=True)
parser.add_argument("--jitter", type=float, default=0.0)
parser.add_argument("--max-iter", type=int)
parser.add_argument("--not-improved", type=int)
parser.add_argument("--hessian", choices=["exact", "cost-only"], default="exact")
args = parser.parse_args()

dtype = torch.float32 if args.dtype == "float32" else torch.float64
stats = {
    "qps": 0,
    "warnings": 0,
    "min_eig": float("inf"),
    "max_cond": 0.0,
    "max_eq": 0.0,
    "max_ineq": 0.0,
    "max_stationarity": 0.0,
    "max_complementarity": 0.0,
}

module = importlib.import_module("diffmpc_cbf.diffmpc_cbf")
original = module.QPFunction


def traced_qp(*qp_args, **qp_kwargs):
    if args.max_iter is not None:
        qp_kwargs["maxIter"] = args.max_iter
    if args.not_improved is not None:
        qp_kwargs["notImprovedLim"] = args.not_improved
    solve = original(*qp_args, **qp_kwargs)

    def apply(Q, p, G, q, A, b):
        if args.jitter:
            Q = Q + args.jitter * torch.eye(Q.shape[-1], device=Q.device, dtype=Q.dtype)
        stats["qps"] += 1
        q64 = Q.detach().double()
        eig = torch.linalg.eigvalsh(0.5 * (q64 + q64.T))
        eig_min, eig_max = eig.min().item(), eig.max().item()
        stats["min_eig"] = min(stats["min_eig"], eig_min)
        if eig_min != 0:
            stats["max_cond"] = max(stats["max_cond"], abs(eig_max / eig_min))

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            primal, eq_dual, ineq_dual = solve(Q, p, G, q, A, b)
        stats["warnings"] += output.getvalue().count("qpth warning")

        with torch.no_grad():
            z = primal.flatten()
            y = eq_dual.flatten()
            lam = ineq_dual.flatten()
            gap = q - G @ z
            stats["max_eq"] = max(stats["max_eq"], (A @ z - b).abs().max().item())
            stats["max_ineq"] = max(stats["max_ineq"], torch.relu(-gap).max().item())
            stationarity = Q @ z + p + A.T @ y + G.T @ lam
            stats["max_stationarity"] = max(
                stats["max_stationarity"], stationarity.abs().max().item()
            )
            stats["max_complementarity"] = max(
                stats["max_complementarity"], (lam * gap).abs().max().item()
            )
        return primal, eq_dual, ineq_dual

    return apply


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)

    nx, nu, nb, horizon = 4, 2, 2, 15
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
        slack_weight=[args.rho1, args.rho2],
        start_dv_eps_detection=True,
        device=device,
        dtype=dtype,
    )
    mpc.to(device=device, dtype=dtype)
    env = Env(np.zeros(2), np.ones(2), 0.05)
    pref = torch.ones(2, 1, dtype=dtype)
    if args.hessian == "cost-only":
        original_two_order = mpc._get_two_order_term

        def cost_only_hessian(tau, dveq, dvneq, params):
            return original_two_order(
                tau,
                torch.zeros_like(dveq),
                torch.zeros_like(dvneq),
                params,
            )

        mpc._get_two_order_term = cost_only_hessian
    statuses = {}
    module.QPFunction = traced_qp
    error = None
    steps = 0
    try:
        for step in range(79):
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
            state = torch.from_numpy(env.state.copy()).to(dtype=dtype)
            tau, status = mpc(state, params)
            statuses[status.name] = statuses.get(status.name, 0) + 1
            _, done = env.step(tau.u[0].detach().numpy().reshape(-1, 1))
            steps += 1
            if done:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        module.QPFunction = original

    print(
        {
            "dtype": args.dtype,
            "slack_weight": [args.rho1, args.rho2],
            "jitter": args.jitter,
            "max_iter": args.max_iter,
            "not_improved": args.not_improved,
            "hessian": args.hessian,
            "steps": steps,
            "statuses": statuses,
            "error": error,
            **stats,
        }
    )


if __name__ == "__main__":
    main()
