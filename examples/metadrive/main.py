from diffmpc_cbf import DiffMPCBF, Tau
from diffmpc_cbf.utils import *
from diffmpc_cbf.base import SQPStatus
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from numpy import ndarray
from env import EnvGenerator
from metadrive.env import MetaDriveEnv
from system import *
from typing import List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import time
import random
import os
from copy import deepcopy
from math import pi
from PIL import Image
from PIL.Image import Image as PILImage
import shutil

class Trainer:
    def __init__(
            self,
            env: MetaDriveEnv,
            env_eval: MetaDriveEnv,
            mpc: DiffMPCBF,
            delta_t: float,
            N: int,
            num_epoch: int,
            lr: float,
            loss_weight: List[float],
            vref: float,
            yref: float,
            gamma: float,
            dtype: torch.dtype,
            device: torch.device,
            seed: int,
            train_log_path: str = './log/train.txt',
            tensorboard_log_path: str = './log/exp',
            eval_gif_path: str = './log/eval',
        ):
        self._env = env
        self._env_eval = env_eval
        self._mpc = mpc
        self._delta_t = delta_t

        # train params
        self._N = N
        self._num_epoch = num_epoch
        self._lr = lr
        self._loss_weight = loss_weight
        self._vref = vref
        self._yref = yref
        self._gamma = gamma
        self._dtype = dtype
        self._device = device
        self._seed = seed
        self._train_log_path = train_log_path
        self._eval_gif_path = eval_gif_path
        self._nx = mpc._nx
        self._nu = mpc._nu
        self._nb = mpc._nb
        self._nobs = self._nb-2
        self._T  = mpc._T
        self._ntau = mpc._ntau
        self._nv = mpc._nv
        self._vardim = mpc._vardim
        self._eval_nums: int = 0
        
        if os.path.exists(tensorboard_log_path):
            shutil.rmtree(tensorboard_log_path)
        self._writer = SummaryWriter(log_dir=tensorboard_log_path)
            

    def train(self):
        _cw = self._mpc._ones(self._T+1); _cw[-1] = 10.0
        w1, w2 = self._loss_weight

        self._write_log('Start to record')
        # set optim
        optimizer_ply = torch.optim.Adam(
            list(self._mpc._cost.parameters()) + list(self._mpc._barrier.parameters()),
            lr = self._lr,
        )
        optimizer_dym = torch.optim.Adam(
            list(self._mpc._dynamic.parameters()),
            lr = self._lr,
        )

        for epoch in range(self._num_epoch):
            obs, info = self._env.reset(); self._seed += 1
            self._mpc.reset()

            xs: List[Tensor] = []
            us: List[Tensor] = []
            bs: List[Tensor] = []
            xcurrs: List[Tensor] = []
            xnexts: List[Tensor] = []
            prefs: List[Tensor] = []
            psirefs: List[Tensor] = []

            # one episode
            for k in range(self._N):
                # build params
                vehicle_position = self._predict_vehicle_position()
                pref, psiref = self._path_planning()
                params = {
                    'cost': {'pref': pref, 'psiref': psiref, 'vref': self._vref},
                    'barrier': {'vehicle_position': vehicle_position, 'gamma': self._gamma, 'pref': pref, 'psiref': psiref},
                }
                x_curr = torch.as_tensor(self._base.vehicle.state, dtype=self._dtype, device=self._device)
                zeta, sqp_status = self._mpc.forward(x_curr, params)
                if sqp_status != SQPStatus.CONVERGED: raise RuntimeError(f'SQP solve failed. Reason: {sqp_status}.')
                
                # build full
                pad = zeta.new_zeros((self._T+1) * self._nv - self._vardim)
                zeta_full = torch.concat([zeta, pad]).view(self._T+1, self._nv)

                u_env = zeta[self._nx: self._ntau].detach()
                action: ndarray = (u_env.cpu().numpy() / [5.0, pi/4]).reshape(-1)
                obs, reward, terminated, truncated, info = self._env.step(action)

                x_next = torch.as_tensor(self._base.vehicle.state, dtype=self._dtype, device=self._device)

                # collect traning data
                xs.append(zeta_full[:, :self._nx]) # (T+1, nx)
                us.append(u_env) # (nu, )
                bs.append(
                    self._mpc._barrier.forward(zeta_full[:, :self._nx], self._mpc.tcon(params['barrier'], 'full')).flatten()
                ) # (T+1, nb)
                xcurrs.append(x_curr) # (nx, )
                xnexts.append(x_next)
                prefs.append(pref)
                psirefs.append(psiref)

                if terminated or truncated:
                    obs, info = self._env.reset(); self._seed += 1
                    self._mpc.reset()

            _xs = torch.stack(xs) # (N, T+1, nx)
            _us = torch.stack(us) # (N, nu)
            _bs = torch.stack(bs) # (N, T+1, nb)
            _xcurrs = torch.stack(xcurrs) # (N, nx)
            _xnexts = torch.stack(xnexts) # (N, nx)
            _taus = torch.concat([_xcurrs, _us], dim=-1) # (N, ntau)
            _prefs = torch.stack(prefs) # (N, T+1, 2)
            _psirefs = torch.stack(psirefs) # (N, T+1, 1)

            # step1: policy training
            es, en, epsi = self._get_lateral_error(_xs, _prefs, _psirefs)
            loss_obj = (_cw.view(1, self._T+1, 1) * (
                es.square() + en.square() + epsi.square() + (_xs[..., 3: 4] - self._vref).square()
            )).mean()
            # loss_control = us.square().mean()
            loss_safe = (torch.nn.functional.relu(-_bs)).square().mean()
            loss_policy: Tensor = w1*loss_obj + w2*loss_safe
    
            optimizer_ply.zero_grad()
            loss_policy.backward()
            # grad clip
            torch.nn.utils.clip_grad_norm_(
                list(self._mpc._cost.parameters()) + list(self._mpc._barrier.parameters()),
                1.0,
            )
            
            optimizer_ply.step()
    
            # step2: dynamic training
            xpreds = self._mpc._dynamic.forward(_taus, dict()) # (N, nx)
            loss_dynamic: Tensor = (xpreds - _xnexts).square().mean()

            optimizer_dym.zero_grad()
            loss_dynamic.backward()
            optimizer_dym.step()

            # record loss
            with open('./log/train.txt', mode='a', encoding='utf-8') as writer:
                writer.write(f'[{self._timenow}] Epoch: {epoch}, policy: {loss_policy:4f} (obj: {loss_obj:4f}, safe: {loss_safe:4f}), dynamic: {loss_dynamic:4f}\n')

            if (epoch+1) % 10 == 0:
                self.eval()
                self._eval_nums += 1

            lsply = loss_policy.item()
            lsdym = loss_dynamic.item()
            lstot = lsply + lsdym
            self._writer.add_scalars(
                'loss',
                {
                    'total': lstot,
                    'policy': lsply,
                    'dynamic': lsdym,
                },
                epoch,
            )

        self._writer.close()

    def eval(self):
        _zeta = self._mpc._zeros((self._T+1), self._nv)
        _zetav = _zeta.view((self._T+1) * self._nv)
        _cw = self._mpc._ones(self._T+1); _cw[-1] = 10.0
        w1, w2 = self._loss_weight

        obs, info = self._env_eval.reset()
        self._mpc.reset()

        xs: List[Tensor] = []
        us: List[Tensor] = []
        bs: List[Tensor] = []
        xcurrs: List[Tensor] = []
        xnexts: List[Tensor] = []
        prefs: List[Tensor] = []
        psirefs: List[Tensor] = []
        frames: List[PILImage] = []

        # one episode
        for k in range(self._N):
            # build params
            vehicle_position = self._predict_vehicle_position(self._base_eval)
            pref, psiref = self._path_planning(self._base_eval)
            params = {
                'cost': {'pref': pref, 'psiref': psiref, 'vref': self._vref},
                'barrier': {'vehicle_position': vehicle_position, 'gamma': self._gamma, 'pref': pref, 'psiref': psiref},
            }
            x_curr = torch.as_tensor(self._base_eval.vehicle.state, dtype=self._dtype, device=self._device)
            zeta, sqp_status = self._mpc.forward(x_curr, params)
            if sqp_status != SQPStatus.CONVERGED: raise RuntimeError(f'SQP solve failed. Reason: {sqp_status}.')

            # build full
            pad = zeta.new_zeros((self._T+1) * self._nv - self._vardim)
            zeta_full = torch.concat([zeta, pad]).view(self._T+1, self._nv)

            u_env = zeta[self._nx: self._ntau].detach()
            action: ndarray = (u_env.cpu().numpy() / [5.0, pi/4]).reshape(-1)
            obs, reward, terminated, truncated, info = self._env_eval.step(action)

            x_next = torch.as_tensor(self._base_eval.vehicle.state, dtype=self._dtype, device=self._device)

            # collect traning data
            xs.append(zeta_full[:, :self._nx].clone()) # (T+1, nx)
            us.append(u_env) # (nu, )
            bs.append(
                self._mpc._barrier.forward(zeta_full[:, :self._nx], self._mpc.tcon(params['barrier'], 'full')).flatten()
            ) # (T+1, nb)
            xcurrs.append(x_curr) # (nx, )
            xnexts.append(x_next)
            prefs.append(pref)
            psirefs.append(psiref)

            # render
            frame = self._env_eval.render()
            frames.append(Image.fromarray(frame))

            if terminated or truncated:
                break

        _xs = torch.stack(xs) # (N, T+1, nx)
        _us = torch.stack(us) # (N, nu)
        _bs = torch.stack(bs) # (N, nb)
        _xcurrs = torch.stack(xcurrs) # (N, nx)
        _xnexts = torch.stack(xnexts) # (N, nx)
        _taus = torch.concat([_xcurrs, _us], dim=-1) # (N, ntau)
        _prefs = torch.stack(prefs) # (N, T+1, 2)
        _psirefs = torch.stack(psirefs) # (N, T+1, 1)


        # step1: policy training
        es, en, epsi = self._get_lateral_error(_xs, _prefs, _psirefs)
        loss_obj = (_cw.view(1, self._T+1, 1) * (
            es.square() + en.square() + epsi.square() + (_xs[..., 3: 4] - self._vref).square()
        )).mean()
        # loss_control = us.square().mean()
        loss_safe = (torch.nn.functional.relu(-_bs)).square().mean()
        loss_policy: Tensor = w1*loss_obj + w2*loss_safe

        # step2: dynamic training
        xpreds = self._mpc._dynamic.forward(_taus, dict()) # (N, nx)
        loss_dynamic: Tensor = (xpreds - _xnexts).square().mean()

        if frames:
            frames[0].save(
                os.path.join(self._eval_gif_path, f'eval{self._eval_nums}.gif'),
                save_all      = True,
                append_images = frames[1:],
                duration      = int(self._delta_t * 1000),
                loop          = 0,
            )

        lsply = loss_policy.item()
        lsdym = loss_dynamic.item()
        lstot = lsply + lsdym
        return lstot, lsply, lsdym

    def _predict_vehicle_position(self, base: MetaDriveEnv) -> Tensor:
        src = base.road
        ego_idx: int = src.vehicles.index(base.vehicle)
        road = deepcopy(src)
        ego = road.vehicles[ego_idx]
        ego.act({'acceleration': 0.0, 'steering': 0.0})
        others = [i for i in range(len(road.vehicles)) if i != ego_idx]
        others.sort(key=lambda i: np.linalg.norm(road.vehicles[i].position - ego.position, ord=2))
        chosen = others[:self._nobs]
        pos = 1e-3 * np.ones([self._T+1, self._nobs, 2])
        for k in range(self._T+1):
            for j, i in enumerate(chosen):
                pos[k, j] = road.vehicles[i].position

            if k == self._T:
                break

            road.act()
            road.step(self._delta_t)
        return torch.as_tensor(pos, dtype=self._dtype, device=self._device)

    def _path_planning(self, base: MetaDriveEnv) -> Tuple[Tensor, Tensor]:
        vehicle = base.vehicle
        destination = base.config.get('destination')

        # get lane
        road = vehicle.road
        lane_index = vehicle.lane_index

        path = road.network.shortest_path(lane_index[1], destination)
        route = [lane_index] + [
            (path[i], path[i + 1], 0) for i in range(len(path) - 1)
        ]
        lanes = [road.network.get_lane(idx) for idx in route]
        s, _ = vehicle.lane.local_coordinates(vehicle.position)
        lane_i = 0
        pref = np.zeros([self._T+1, 2])
        psiref = np.zeros([self._T+1, 1])
        for i in range(self._T+1):
            lane = lanes[lane_i]
            while s > lane.length and lane_i + 1 < len(lanes):
                s -= lane.length
                lane_i += 1
                lane = lanes[lane_i]
            s_clip = min(max(s, 0.0), lane.length)
            pref[i] = lane.position(s_clip, 0.0)
            psiref[i] = float(lane.heading_at(s_clip))
            s += self._vref * self._delta_t
        return self._mpc._tensor(pref), self._mpc._tensor(psiref)

    def _get_lateral_error(self, xs: Tensor, prefs: Tensor, psirefs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        '''
        Argument:
            - xs (N, T+1, nx)
            - pref (N, T+1, 2)
            - psiref (N, T+1, 1)
        Returns:
            - es (N, T+1, 1)
            - en (N, T+1, 1)
            - epsi (N, T+1, 1)
        '''
        psi = xs[..., 2: 3] # (N, T+1, 1)
        c, s = torch.cos(psirefs), torch.sin(psirefs)
        p_delta = xs[..., 0: 2] - prefs # (N, T+1, 2)
        es = c * p_delta[..., 0: 1] + s * p_delta[..., 1: 2] # (N, T+1, 1)
        en = -s * p_delta[..., 0: 1] + c * p_delta[..., 1: 2] # (N, T+1, 1)
        epsi = torch.atan2(torch.sin(psi - psirefs), torch.cos(psi - psirefs)) # (N, T+1, 1)
        return es, en, epsi

    @property
    def _timenow(self) -> str: return datetime.now().strftime('%H:%M:%S')

    def _write_log(self, s: str, mode: str='w') -> str:
        '''
        Arguments:
            - mode: options, ['w', 'a']
        '''
        log_string: str = f'[{self._timenow}] {s}\n'
        with open(self._train_log_path, mode=mode, encoding='utf-8') as file:
            file.write(log_string)
        return log_string



def main():
    # basic environment set
    seed: int = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device: torch.device = torch.device('cuda', index=0)
    dtype: torch.dtype = torch.float64
    nx: int = 4
    nu: int = 2
    nb: int = 3 + 2
    T: int  = 15
    vref: float = 10.0
    yref: float = 0.0
    L: float = 5.0
    W: float = 2.0
    ymin: float = -2.0
    ymax: float = 14.0
    delta_t: float = 1 / 15


    env_generator = EnvGenerator()
    env_generator_eval = EnvGenerator()
    env = env_generator.env
    env_eval = env_generator_eval.env
    cost    = MetaDriveCost(nx, nu, nb, T, device, dtype, vref, yref)
    dynamic = MetaDriveDynamic(nx, nu, T,device, dtype, L, delta_t)
    barrier = MetaDriveBarrier(nx, nu, nb, T, device, dtype, L, W, ymin, ymax)
    umin    = torch.tensor([-5.0, -pi/4], device=device, dtype=dtype)
    umax    = torch.tensor([ 5.0,  pi/4], device=device, dtype=dtype)
    mpc     = DiffMPCBF(
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
    trainer = Trainer(
        env = env,
        env_eval = env_eval,
        mpc = mpc,
        delta_t = delta_t,
        N = 200,
        num_epoch = 100,
        lr = 5e-4,
        loss_weight = [1.0, 1.0],
        vref = vref,
        yref = yref,
        gamma = 0.2,
        dtype = dtype,
        device = device,
        seed = 53,
    )
    trainer.train()
    # lstot_el, lsply_el, lsdym_el = trainer.eval()
    # print(f'total: {lstot_el}, policy: {lsply_el}, dynamic: {lsdym_el}')


if __name__ == '__main__':
    main()