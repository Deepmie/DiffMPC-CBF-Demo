"""Diagnose tvto training and Q SPD failures without modifying user code."""
import os
import sys
import warnings
from types import MethodType

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "examples", "tvto"))

from diffmpc_cbf import DiffMPCBF
from env import Env
from system import TvToBarrier, TvToCost, TvToDynamic

warnings.filterwarnings("ignore")

NX, NU, NB, T = 4, 2, 2, 5
DT, N = 0.05, 80
GAMMA = 0.5


def build():
    torch.manual_seed(1)
    np.random.seed(1)
    cost = TvToCost(NX, NU, T)
    dynamic = TvToDynamic(NX, NU, T, DT)
    barrier = TvToBarrier(NX, NU, T, NB)
    mpc = DiffMPCBF(
        nx=NX, nu=NU, nb=NB, T=T, batch_size=1,
        cost=cost, dynamic=dynamic, barrier=barrier,
        umin=torch.tensor([-1.0, -1.0]),
        umax=torch.tensor([1.0, 1.0]),
    )
    pref = torch.tensor([[1.0], [1.0]], requires_grad=True)
    env = Env(np.array([0.0, 0.0]), np.array([1.0, 1.0]), delta_t=DT)
    return mpc, cost, dynamic, barrier, pref, env


def finite_summary(mpc):
    ps = list(mpc.named_parameters())
    vals = torch.cat([p.detach().flatten() for _, p in ps])
    return {
        "param_absmax": vals.abs().max().item(),
        "param_norm": vals.norm().item(),
        "param_finite": torch.isfinite(vals).all().item(),
    }


def install_q_logger(mpc, log):
    original = mpc._build_sqp_params

    def wrapped(self, *args, **kwargs):
        out = original(*args, **kwargs)
        Q = out[0].detach()
        ev = torch.linalg.eigvalsh(Q)
        info = torch.linalg.cholesky_ex(Q)[1]
        dveq = args[2]
        dvneq = args[3]
        log.append({
            "q_finite": torch.isfinite(Q).all().item(),
            "eig_min": ev.min().item(),
            "eig_max": ev.max().item(),
            "q_absmax": Q.abs().max().item(),
            "chol_info": int(info.item()),
            "eq_mul_max": dveq.abs().max().item(),
            "cbf_mul_max": dvneq[:T * NB].abs().max().item(),
            "bound_mul_max": dvneq[T * NB:].abs().max().item(),
        })
        return out

    mpc._build_sqp_params = MethodType(wrapped, mpc)


def check_barrier_grad(barrier, env, pref):
    obs = env.get_obs_info(T + 1)
    params = {"obs_info": obs, "radius_agent": env.radius_agent, "gamma": GAMMA, "t": 0}
    x = torch.zeros(4, 1, requires_grad=True)
    b = barrier.forward(x, params).sum()
    grads = torch.autograd.grad(
        b, list(barrier.parameters()), allow_unused=True, retain_graph=True
    )
    print("Barrier network gradients from h:", [
        None if g is None else g.abs().max().item() for g in grads
    ])


def run():
    mpc, cost, dynamic, barrier, pref, env = build()
    optimizer = torch.optim.Adam(mpc.parameters(), lr=1e-3)
    qlog = []
    install_q_logger(mpc, qlog)
    check_barrier_grad(barrier, env, pref)

    for epoch in range(10):
        env.reset()
        xs_list, xs_real_list, us_list, bs_list = [], [], [], []
        for k in range(N - 1):
            obs = env.get_obs_info(T + 1)
            params = {
                "cost": {"pref": pref, "obs_info": obs},
                "barrier": {
                    "obs_info": obs,
                    "radius_agent": env.radius_agent,
                    "gamma": GAMMA,
                },
            }
            x = torch.from_numpy(env.state.copy()).requires_grad_(True)
            start = len(qlog)
            try:
                tau = mpc(x, params)
            except Exception as exc:
                print(f"FAIL epoch={epoch}, env_step={k}, error={exc}")
                print("parameters:", finite_summary(mpc))
                print("last Q builds:")
                for i, row in enumerate(qlog[max(start, len(qlog)-12):], start=max(start, len(qlog)-12)):
                    print(i, row)
                return

            x_real, _ = env.step(tau.u[0].detach().numpy().reshape(-1, 1))
            xs_list.append(tau.x[1])
            us_list.append(tau.u[0])
            bs_list.append(
                barrier.forward(
                    tau.x[1].reshape(-1, 1),
                    {**params["barrier"], "t": 1},
                ).flatten()
            )
            xs_real_list.append(torch.from_numpy(x_real.copy()).flatten())

        xs = torch.stack(xs_list)
        us = torch.stack(us_list)
        bs = torch.stack(bs_list)
        xs_real = torch.stack(xs_real_list)
        loss_obj = (xs[:, :2] - pref.reshape(1, -1)).square().sum()
        loss_control = us.square().sum()
        loss_safe = torch.nn.functional.relu(-bs).square().sum()
        loss_dynamic = (xs - xs_real).square().sum()
        loss = loss_obj + loss_control + 0.5 * loss_safe + 0.5 * loss_dynamic

        optimizer.zero_grad()
        loss.backward()
        grad_rows = {}
        for name, module in (
            ("cost", cost), ("dynamic", dynamic), ("barrier", barrier)
        ):
            gs = [
                p.grad.detach().flatten()
                for p in module.parameters()
                if p.grad is not None
            ]
            grad_rows[name] = (
                None if not gs else {
                    "norm": torch.cat(gs).norm().item(),
                    "absmax": torch.cat(gs).abs().max().item(),
                    "finite": torch.isfinite(torch.cat(gs)).all().item(),
                }
            )
        print(
            f"epoch={epoch} loss={loss.item():.5g} "
            f"parts={[z.item() for z in (loss_obj, loss_control, loss_safe, loss_dynamic)]}"
        )
        print("grads:", grad_rows)
        optimizer.step()
        print("after step:", finite_summary(mpc))


if __name__ == "__main__":
    run()
