from diffmpc_cbf import DiffMPCBF, Tau
from diffmpc_cbf.utils import *
from diffmpc_cbf.base import SQPStatus
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from system import TvToCost, TvToDynamic, TvToBarrier
from env import Env
from typing import List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import time
import random

def get_timenow() -> str:
    return datetime.now().strftime('%H:%M:%S')

def train(
        env: Env,
        mpc: DiffMPCBF,
        pref: Tensor,
        gamma: float,
        N: int,
        num_epoch: int,
        lr: float,
        loss_weight: List[float],
        device: torch.device,
        dtype: torch.dtype,
    ):
    # initial set
    nx = mpc._nx
    nu = mpc._nu
    nb = mpc._nb
    T  = mpc._T
    ntau = mpc._ntau
    nv = mpc._nv
    vardim = mpc._vardim
    _zeta = mpc._zeros((T+1), nv)
    _zetav = _zeta.view((T+1) * nv)
    _cw = mpc._ones(T+1); _cw[-1] = 10.0; _cw.requires_grad_(False)
    cost = mpc._cost
    dynamic = mpc._dynamic
    barrier = mpc._barrier

    # set log
    with open('./log/train.txt', mode='w', encoding='utf-8') as writer:
        writer.write(f'[{get_timenow()}] Start to record\n')

    w1, w2, w3 = loss_weight
    # set optim
    opt_policy = torch.optim.Adam(list(cost.parameters()) + list(barrier.parameters()), lr=lr)
    opt_dynamic = torch.optim.Adam(dynamic.parameters(), lr=lr)
    loss_list: List[float] = []
    evaluate_num: int = 0
    start = time.time()

    # start to train
    for epoch in range(num_epoch):
        env.reset()
        mpc.reset()

        xs: List[Tensor] = []
        us: List[Tensor] = []
        bs: List[Tensor] = []
        xcurrs: List[Tensor] = []
        xnexts: List[Tensor] = []
        # one episode
        for k in range(N):
            _obs_info = env.get_obs_info(T+1)
            _obs_info_ts, _radius_agent_ts, _gamma_ts = convert_any_to_tensor_batch([_obs_info, env.radius_agent, gamma], device, dtype)
            params = {
                'cost': {'pref': pref, 'obs_info': _obs_info_ts},
                'barrier': {'obs_info': _obs_info_ts, 'radius_agent': _radius_agent_ts, 'gamma': _gamma_ts},
            }
            x_curr = torch.as_tensor(env.state.reshape(-1).copy(), dtype=dtype, device=device) # (nx, )
            zeta, sqp_status = mpc.forward(x_curr, params)
            if sqp_status != SQPStatus.CONVERGED: raise RuntimeError(f'SQP solve Failed!')
            
            # interact with env
            u_env = zeta[nx: ntau].detach()
            x_env, done = env.step(u_env.cpu().numpy().reshape(-1, 1))
            x_next = torch.as_tensor(x_env.reshape(-1), dtype=dtype, device=device) # (nx, )
            
            # collect training data
            _zetav[:vardim] = zeta
            xs.append(_zeta[:, :nx].clone()) # (T+1, nx)
            us.append(u_env) # (nu, )
            bs.append(barrier.forward(zeta[nv: nv+nx].unsqueeze(0), mpc.tcon(params['barrier'], 1)).flatten()) # (nb, )
            xcurrs.append(x_curr) # (nx, )
            xnexts.append(x_next)
            if done: break

        _xs = torch.stack(xs) # (N, T+1, nx)
        _us = torch.stack(us) # (N, nu)
        _bs = torch.stack(bs) # (N, nb)
        _xcurrs = torch.stack(xcurrs) # (N, nx)
        _xnexts = torch.stack(xnexts) # (N, nx)
        _taus = torch.concat([_xcurrs, _us], dim=-1) # (N, ntau)

        # step1: policy training
        loss_obj = (_cw.view(1, T+1, 1) * (_xs[..., 0:2].reshape(N*(T+1), 2) - pref.reshape(-1)).square().reshape(N, T+1, 2)).mean()
        # loss_control = us.square().mean()
        loss_safe = (torch.nn.functional.relu(-_bs)).square().mean()
        loss_policy: Tensor = w1*loss_obj + w2*loss_safe

        opt_policy.zero_grad()
        loss_policy.backward()
        opt_policy.step()

        # step2: dynamic training
        xpreds = dynamic.forward(_taus, dict()) # (N, nx)
        loss_dynamic: Tensor = (xpreds - _xnexts).square().mean()

        opt_dynamic.zero_grad()
        loss_dynamic.backward()
        opt_dynamic.step()


        # record loss
        with open('./log/train.txt', mode='a', encoding='utf-8') as writer:
            writer.write(f'[{get_timenow()}] Epoch: {epoch}, policy: {loss_policy:4f} (obj: {loss_obj:4f}, safe: {loss_safe:4f}), dynamic: {loss_dynamic:4f}\n')

        # evaluate
        if (epoch+1) % 10 == 0:
            loss_policy_eval, loss_dynamic_eval = \
            eval(
                env,
                mpc,
                pref,
                gamma,
                N,
                loss_weight,
                device=device,
                dtype=dtype,
            )
            # record loss
            with open('./log/eval.txt', mode='a', encoding='utf-8') as writer:
                writer.write(f'[{get_timenow()}] Num: {evaluate_num}, policy: {loss_policy_eval:4f}, dynamic: {loss_dynamic_eval:4f}\n')

            env.show(f'./imgs/tvto_{evaluate_num}.gif')
            evaluate_num += 1

    end = time.time()
    with open('./log/train.txt', mode='a', encoding='utf-8') as writer:
        writer.write(f'[{get_timenow()}] total time: {end-start}\n')
    return loss_list


def eval(
        env: Env,
        mpc: DiffMPCBF,
        pref: Tensor,
        gamma: float,
        N: int,
        loss_weight: List[float],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    # initial set
    nx = mpc._nx
    nu = mpc._nu
    nb = mpc._nb
    T  = mpc._T
    ntau = mpc._ntau
    nv = mpc._nv
    vardim = mpc._vardim
    _zeta = mpc._zeros((T+1), nv)
    _zetav = _zeta.view((T+1) * nv)
    _cw = mpc._ones(T+1); _cw[-1] = 10.0
    cost = mpc._cost
    dynamic = mpc._dynamic
    barrier = mpc._barrier
    
    w1, w2, w3 = loss_weight
    # start to evaluate
    env.reset()
    mpc.reset()

    xs: List[Tensor] = []
    us: List[Tensor] = []
    bs: List[Tensor] = []
    xcurrs: List[Tensor] = []
    xnexts: List[Tensor] = []
    # one episode
    for k in range(N):
        _obs_info = env.get_obs_info(T+1)
        _obs_info_ts, _radius_agent_ts, _gamma_ts = convert_any_to_tensor_batch([_obs_info, env.radius_agent, gamma], device, dtype)
        params = {
            'cost': {'pref': pref, 'obs_info': _obs_info_ts},
            'barrier': {'obs_info': _obs_info_ts, 'radius_agent': _radius_agent_ts, 'gamma': _gamma_ts},
        }
        x_curr = torch.as_tensor(env.state.copy(), dtype=dtype, device=device) # (nx, )
        zeta, sqp_status = mpc.forward(x_curr, params)
        if sqp_status != SQPStatus.CONVERGED: raise RuntimeError(f'SQP solve Failed!')
        
        # interact with env
        u_env = zeta[nx: ntau].cpu().detach().numpy().reshape(-1, 1)
        x_env, done = env.step(u_env)
        x_next = torch.as_tensor(x_env, dtype=dtype, device=device) # (nx, )
        
        # collect training data
        _zetav[:vardim] = zeta
        xs.append(_zeta[:, :nx].clone()) # (T+1, nx)
        us.append(zeta[nx: ntau]) # (nu, )
        bs.append(barrier.forward(zeta[nv: nv+nx].unsqueeze(0), mpc.tcon(params['barrier'], 1)).flatten()) # (nb, )
        xcurrs.append(x_curr) # (nx, )
        xnexts.append(x_next)
        if done: break

    _xs: Tensor = torch.stack(xs) # (N, T+1, nx)
    _us: Tensor = torch.stack(us) # (N, nu)
    _bs: Tensor = torch.stack(bs) # (N, nb)
    _xcurrs: Tensor = torch.stack(xcurrs) # (N, nx)
    _xnexts: Tensor = torch.stack(xnexts) # (N, nx)
    _taus: Tensor = torch.concat([xcurrs, us], dim=-1) # (N, ntau)

    # step1: policy training
    loss_obj = (_cw.view(1, T+1, 1) * (_xs[..., 0:2].reshape(N*(T+1), 2) - pref.reshape(-1)).square()).mean()
    # loss_control = us.square().mean()
    loss_safe = (torch.nn.functional.relu(-_bs)).square().mean()
    loss_policy: Tensor = w1*loss_obj + w2*loss_safe

    # step2: dynamic training
    xpreds = dynamic.forward(_taus, dict()) # (N, nx)
    loss_dynamic: Tensor = (xpreds - _xnexts).square().mean()
    return loss_policy, loss_dynamic

def main():
    # basic environment set
    seed: int = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # device: torch.device = torch.device('cuda', index=0)
    device: torch.device = torch.device('cuda', index=0)
    dtype: torch.dtype = torch.float64
    nx: int = 4
    nu: int = 2
    nb: int = 2
    T: int  = 15
    delta_t: float = 0.05
    N: int  = 80
    ntau = nx + nu
    nl = T * nu
    # build mpc-cbf solver
    cost      = TvToCost(nx, nu, nb, T, device, dtype)
    dynamic   = TvToDynamic(nx, nu, T, device, dtype, delta_t)
    barrier   = TvToBarrier(nx, nu, nb, T, device, dtype)

    umin      = torch.tensor([-1.0, -1.0], device=device, dtype=dtype)
    umax      = torch.tensor([1.0, 1.0], device=device, dtype=dtype)
    p0        = torch.tensor([[0.], [0.]], device=device, dtype=dtype, requires_grad=True)
    pref      = torch.tensor([[1.], [1.]], device=device, dtype=dtype, requires_grad=True) # (2, 1)
    gamma: float = 0.2

    mpc = DiffMPCBF(
        nx         = nx,
        nu         = nu,
        nb         = nb,
        T          = T,
        batch_size = 1,
        cost       = cost,
        dynamic    = dynamic,
        barrier    = barrier,
        umin       = umin,
        umax       = umax,
        dtype      = dtype,
    )
    mpc.to(device)
    
    env = Env(
        p0.flatten().cpu().detach().numpy(),
        pref.flatten().cpu().detach().numpy(),
        delta_t=delta_t,
    )

    ls_train= train(
        env,
        mpc,
        pref,
        gamma,
        N,
        num_epoch=100,
        lr=5e-4,
        loss_weight=[1., 1e4, 1e3],
        device=device,
        dtype=dtype,
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(ls_train, marker='x', alpha=0.5)
    ax.set_xlabel('step'); ax.set_ylabel('loss')
    ax.set_title('Loss Curve')
    fig.savefig('./imgs/loss.png')
    

if __name__ == '__main__':
    main()