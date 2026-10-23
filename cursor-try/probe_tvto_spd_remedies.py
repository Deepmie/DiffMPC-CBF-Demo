"""Probe causes/remedies for tvto Q-is-not-SPD without changing project files."""
import contextlib
import io
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

NX, NU, NB, T, DT = 4, 2, 2, 5, 0.05
PREF = torch.tensor([[1.0], [1.0]])
UMIN = torch.tensor([-1.0, -1.0])
UMAX = torch.tensor([1.0, 1.0])
GAMMA = 0.5


class SmallDifferentiableBarrier(TvToBarrier):
    """Diagnostic barrier: bounded small delta and no Tensor.item()."""
    def _define_barrier_function(self, xt, params):
        pt = xt[:2]
        t = params["t"]
        tt = torch.tensor([[t]])
        obs = params["obs_info"]
        radius_agent = torch.tensor(params["radius_agent"])
        values = []
        for o in obs:
            ot = o[t]
            os = torch.from_numpy(ot["state"])
            orad = torch.tensor(ot["radius"])
            dp = pt - os
            ctx = torch.concat([
                dp, torch.linalg.norm(dp).reshape(1, 1),
                orad.reshape(1, 1), tt
            ]).float().flatten()
            # Diagnostic scale: 0 < delta < 0.02.
            delta = 0.02 * torch.sigmoid(self.delta_network._net(ctx)).reshape(())
            values.append((dp.square().sum() - (orad + radius_agent + delta).square()).reshape(1))
        return torch.stack(values)


def make(barrier_cls=TvToBarrier):
    torch.manual_seed(1)
    np.random.seed(1)
    cost = TvToCost(NX, NU, T)
    dyn = TvToDynamic(NX, NU, T, DT)
    barrier = barrier_cls(NX, NU, T, NB)
    mpc = DiffMPCBF(
        nx=NX, nu=NU, nb=NB, T=T, batch_size=1,
        cost=cost, dynamic=dyn, barrier=barrier,
        umin=UMIN, umax=UMAX,
    )
    env = Env(np.zeros(2), np.ones(2), delta_t=DT)
    return mpc, cost, dyn, barrier, env


def params(env):
    obs = env.get_obs_info(T + 1)
    return {
        "cost": {"pref": PREF, "obs_info": obs},
        "barrier": {
            "obs_info": obs,
            "radius_agent": env.radius_agent,
            "gamma": GAMMA,
        },
    }


def delta_values(barrier, env, x):
    p = torch.as_tensor(x[:2], dtype=torch.float32).reshape(2, 1)
    obs = env.get_obs_info(T + 1)
    vals = []
    for o in obs:
        ot = o[0]
        os = torch.from_numpy(ot["state"])
        r = torch.tensor(ot["radius"]).reshape(1, 1)
        dp = p - os
        ctx = torch.concat([
            dp, torch.linalg.norm(dp).reshape(1, 1), r, torch.tensor([[0.0]])
        ]).float().flatten()
        vals.append(barrier.delta_network(ctx).item())
    return vals


def install_relative_spd(mpc, rel=1e-4, abs_floor=1e-4):
    original = mpc._build_sqp_params

    def wrapped(self, *args, **kwargs):
        out = list(original(*args, **kwargs))
        Q = 0.5 * (out[0] + out[0].T)
        eig = torch.linalg.eigvalsh(Q)
        target = max(abs_floor, rel * max(1.0, eig.abs().max().item()))
        if eig.min().item() < target:
            Q = Q + (target - eig.min().item()) * torch.eye(Q.shape[0])
        # Roundoff-safe Cholesky retry.
        jitter = target
        for _ in range(5):
            if torch.linalg.cholesky_ex(Q)[1].item() == 0:
                break
            Q = Q + jitter * torch.eye(Q.shape[0])
            jitter *= 10
        out[0] = Q
        return tuple(out)

    mpc._build_sqp_params = MethodType(wrapped, mpc)


def rollout(mpc, env, n=79):
    warnings_count = 0
    h_min = float("inf")
    max_state = 0.0
    for k in range(n):
        p = params(env)
        x = torch.from_numpy(env.state.copy()).requires_grad_(True)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as capture:
                tau = mpc(x, p)
            warnings_count += capture.getvalue().count("qpth warning")
        except Exception as exc:
            return {
                "ok": False, "step": k, "error": str(exc),
                "warnings": warnings_count, "h_min": h_min,
                "max_state": max_state,
            }
        b = mpc._barrier.forward(
            tau.x[1].reshape(-1, 1), {**p["barrier"], "t": 1}
        )
        h_min = min(h_min, b.min().item())
        state, _ = env.step(tau.u[0].detach().numpy().reshape(-1, 1))
        max_state = max(max_state, float(np.abs(state).max()))
    return {
        "ok": True, "step": n, "warnings": warnings_count,
        "h_min": h_min, "max_state": max_state,
        "final": env.state.flatten().tolist(),
    }


if __name__ == "__main__":
    mpc, cost, dyn, barrier, env = make()
    print("Initial learned deltas:", delta_values(barrier, env, env.state))
    print("Physical clearance base radius = 0.08 + agent 0.02 = 0.10")
    print("Effective barrier radii:",
          [0.10 + d for d in delta_values(barrier, env, env.state)])
    b0 = barrier.forward(
        torch.from_numpy(env.state.copy()),
        {**params(env)["barrier"], "t": 0},
    )
    print("Initial h:", b0.flatten().tolist())
    print("Barrier output requires_grad:", b0.requires_grad)
    print("Barrier param grads reachable:",
          [g is not None for g in torch.autograd.grad(
              b0.sum(), list(barrier.parameters()),
              allow_unused=True, retain_graph=True
          )])

    mpc1, _, _, _, env1 = make()
    print("Baseline:", rollout(mpc1, env1))

    mpc2, _, _, _, env2 = make()
    install_relative_spd(mpc2, rel=1e-4)
    print("Relative SPD floor:", rollout(mpc2, env2))

    mpc3, _, _, barrier3, env3 = make(SmallDifferentiableBarrier)
    p3 = params(env3)
    b3 = barrier3.forward(torch.from_numpy(env3.state.copy()), {**p3["barrier"], "t": 0})
    print("Small differentiable barrier initial h:", b3.flatten().tolist())
    print("Small barrier param grads reachable:",
          [g is not None for g in torch.autograd.grad(
              b3.sum(), list(barrier3.parameters()),
              allow_unused=True, retain_graph=True
          )])
    print("Small differentiable delta:", rollout(mpc3, env3))
