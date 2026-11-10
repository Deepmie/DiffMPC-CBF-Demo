"""Reproduce autograd version errors caused by reusing SQP matrix buffers."""
from __future__ import annotations

import importlib
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


def install_cloned_qp_inputs() -> None:
    solver_module = importlib.import_module("diffmpc_cbf.diffmpc_cbf")
    original_factory = solver_module.QPFunction

    def cloned_factory(*factory_args, **factory_kwargs):
        original_apply = original_factory(*factory_args, **factory_kwargs)

        def apply(Q, p, G, q, A, b):
            # clone() keeps the autograd connection but gives qpth immutable
            # storage for tensors saved until the episode-level backward.
            return original_apply(
                Q.clone(),
                p.clone(),
                G.clone(),
                q.clone(),
                A.clone(),
                b.clone(),
            )

        return apply

    solver_module.QPFunction = cloned_factory


def main() -> None:
    mode = os.getenv("QP_INPUT_MODE", "shared")
    steps = int(os.getenv("MPC_STEPS", "3"))
    epochs = int(os.getenv("TRAIN_EPOCHS", "1"))
    reset_timing = os.getenv("MATRIX_GRAPH_RESET", "none")
    if os.getenv("RESET_MATRIX_GRAPH", "0") == "1":
        reset_timing = "before_solve"
    if reset_timing not in {
        "none",
        "before_solve",
        "after_forward",
        "after_forward_detach_only",
        "after_forward_detach_inplace",
    }:
        raise ValueError(
            "MATRIX_GRAPH_RESET must be none, before_solve, after_forward, "
            "after_forward_detach_only, or after_forward_detach_inplace"
        )
    if mode not in {"shared", "cloned"}:
        raise ValueError("QP_INPUT_MODE must be shared or cloned")
    if mode == "cloned":
        install_cloned_qp_inputs()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.autograd.set_detect_anomaly(True)

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
    pref = torch.tensor(
        [[1.0], [1.0]], device=device, dtype=dtype, requires_grad=True
    )
    optimizer = torch.optim.Adam(mpc.parameters(), lr=5e-4)

    print(
        f"mode={mode} Q_shape={tuple(mpc._Qv.shape)} "
        f"Q_storage={mpc._Qv.untyped_storage().data_ptr()} "
        f"matrix_graph_reset={reset_timing}"
    )

    def refresh_matrix_graph(self, clone: bool = True) -> None:
        for name in ("_Q", "_p", "_G", "_q", "_A", "_b"):
            value = getattr(self, name).detach()
            setattr(self, name, value.clone() if clone else value)
        self._get2Dview()

    if reset_timing == "before_solve":
        original_solve = mpc._solve_qp_subproblem

        def solve_with_fresh_matrix_graph(self, *args, **kwargs):
            if torch.is_grad_enabled():
                refresh_matrix_graph(self)
            return original_solve(*args, **kwargs)

        mpc._solve_qp_subproblem = types.MethodType(
            solve_with_fresh_matrix_graph, mpc
        )

    for epoch in range(epochs):
        env.reset()
        mpc.reset()
        losses = []
        for k in range(steps):
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
            state = (
                torch.from_numpy(env.state.copy())
                .requires_grad_(True)
                .to(device, dtype)
            )
            solution, status = mpc(state, params)
            if status != SQPStatus.CONVERGED:
                raise RuntimeError(f"SQP failed at epoch={epoch}, k={k}: {status.name}")

            losses.append(solution.square().mean())
            if reset_timing in {"after_forward", "after_forward_detach_only"}:
                refresh_matrix_graph(
                    mpc, clone=reset_timing != "after_forward_detach_only"
                )
            elif reset_timing == "after_forward_detach_inplace":
                for name in ("_Q", "_p", "_G", "_q", "_A", "_b"):
                    getattr(mpc, name).detach_()
            control = solution[nx : nx + nu].detach().cpu().numpy().reshape(-1, 1)
            env.step(control)

        loss = torch.stack(losses).sum()
        optimizer.zero_grad()
        print(
            f"epoch={epoch} calling backward after {steps} forwards, "
            f"loss={float(loss.detach()):.6g} Q_grad_fn={type(mpc._Q.grad_fn).__name__}"
        )
        loss.backward()
        finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in mpc.parameters()
        )
        optimizer.step()
        print(
            f"epoch={epoch} backward succeeded, "
            f"finite_parameter_gradients={finite}"
        )


if __name__ == "__main__":
    main()
