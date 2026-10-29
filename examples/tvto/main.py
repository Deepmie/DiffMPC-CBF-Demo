from diffmpc_cbf import DiffMPCBF, Tau
from diffmpc_cbf.utils import *
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from system import TvToCost, TvToDynamic, TvToBarrier
from env import Env
from typing import List
from datetime import datetime
import matplotlib.pyplot as plt
import time

def get_timenow() -> str:
    return datetime.now().strftime('%H:%M:%S')

def train(
        env: Env,
        mpc: DiffMPCBF,
        barrier: TvToBarrier,
        pref: Tensor,
        nx: int,
        nu: int,
        nb: int,
        T: int,
        gamma: float,
        N: int,
        num_epoch: int,
        lr: float,
        loss_weight: List[float],
        device: torch.device,
        dtype: torch.dtype,
    ):
    with open('./log/train.txt', mode='w', encoding='utf-8') as writer:
        writer.write(f'[{get_timenow()}] Start to record\n')

    w1, w2, w3 = loss_weight
    loss_list: List[float] = []
    # set optim
    optimizer = torch.optim.Adam(mpc.parameters(), lr=lr)

    # set loss
    start = time.time()
    for epoch in range(num_epoch):
        env.reset()
        mpc.reset()

        xs_list: List[Tensor] = []
        xs_real_list: List[Tensor] = []
        us_list: List[Tensor] = []
        bs_list: List[Tensor] = []
        # one episode
        for k in range(N-1):
            _obs_info = env.get_obs_info(T+1)
            _obs_info_ts, _radius_agent_ts, _gamma_ts = convert_any_to_tensor_batch([_obs_info, env.radius_agent, gamma], device, dtype)
            params = {
                'cost': {'pref': pref, 'obs_info': _obs_info_ts},
                'barrier': {'obs_info': _obs_info_ts, 'radius_agent': _radius_agent_ts, 'gamma': _gamma_ts},
            }
            x = torch.from_numpy(env.state.copy()).requires_grad_(True).to(device=device, dtype=dtype)
            tau, _ = mpc.forward(x, params)

            # interact with env
            x, done = env.step(tau.u[0].cpu().detach().numpy().reshape(-1, 1))

            
            # append data
            xs_list.append(tau.x[1])
            us_list.append(tau.u[0])
            bs_list.append(barrier.forward(tau.x[1].reshape(-1, 1), tcon(params['barrier'], 1, device, dtype)).flatten())
            xs_real_list.append(convert_tensor_iteration(x.copy(), device, dtype).flatten())
            print(f'Epoch{epoch}: {k} / {N-1} finished\n')
            if done: break

        xs = torch.stack(xs_list)
        us = torch.stack(us_list)
        bs = torch.stack(bs_list)
        xs_real = torch.stack(xs_real_list)

        loss_obj = (xs[:, 0:2] - pref.reshape(1, -1)).square().mean()
        loss_control = us.square().mean()
        loss_safe = (torch.nn.functional.relu(-bs)).square().mean()
        loss_dynamic = (xs-xs_real).square().mean()

        loss = w1*(loss_obj + loss_control) + w2*loss_safe + w3*loss_dynamic
        loss_list.append(loss.item())

        # update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss
        with open('./log/train.txt', mode='a', encoding='utf-8') as writer:
            writer.write(f'[{get_timenow()}] Epoch: {epoch}, loss: {loss:4f}, {loss_obj:4f} | {loss_control:4f} | {loss_safe:4f} | {loss_dynamic:4f}\n')

    end = time.time()
    with open('./log/train.txt', mode='a', encoding='utf-8') as writer:
        writer.write(f'[{get_timenow()}] total time: {end-start}\n')

    return loss_list


def main():
    # basic environment set
    torch.manual_seed(1)
    device: torch.device = torch.device('cuda', index=0)
    dtype: torch.dtype = torch.float32
    nx: int = 4
    nu: int = 2
    nb: int = 2
    T: int  = 5
    delta_t: float = 0.05
    N: int  = 10
    ntau = nx + nu
    nl = T * nu
    # build mpc-cbf solver
    cost      = TvToCost(nx, nu, T)
    dynamic   = TvToDynamic(nx, nu, T, delta_t)
    barrier   = TvToBarrier(nx, nu, T, nb)

    umin      = torch.tensor([-1.0, -1.0], device=device, dtype=dtype)
    umax      = torch.tensor([1.0, 1.0], device=device, dtype=dtype)
    p0        = torch.tensor([[0.], [0.]], device=device, dtype=dtype, requires_grad=True)
    pref      = torch.tensor([[1.], [1.]], device=device, dtype=dtype, requires_grad=True)
    gamma: float = 0.5

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
        barrier,
        pref,
        nx,
        nu,
        nb,
        T,
        gamma,
        N,
        num_epoch=10,
        lr=1e-3,
        loss_weight=[1., 0.5, 0.5],
        device=device,
        dtype=dtype,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(ls_train, marker='x', alpha=0.5)
    ax.set_xlabel('step'); ax.set_ylabel('loss')
    ax.set_title('Loss Curve')
    plt.show()
    

if __name__ == '__main__':
    main()