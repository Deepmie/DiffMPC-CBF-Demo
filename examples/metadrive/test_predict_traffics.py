from env import *
from transitor import Transitor
from metadrive.component.vehicle.vehicle_type import BaseVehicle, DefaultVehicle
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.component.lane.pg_lane import PGLane
from metadrive.utils.doc_utils import generate_gif
from typing import List
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from copy import deepcopy
from pygame import Surface
import os

ROOT = os.path.dirname(__file__)

# 按距离排序后的槽位颜色：最近、次近、第三近
_SLOT_COLORS = (
    (220, 30, 30),
    (30, 140, 255),
    (255, 150, 0),
    (150, 40, 220),
)


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def mark_chosen(
        env: MetaDriveEnv,
        pos: ndarray, # (T+1, nh, 2)
        heading: ndarray, # (T+1, nh) 或 (T+1, nh, 1)
        length: ndarray, # (nh,)
        width: ndarray, # (nh,)
        mask: ndarray, # (nh,)
    ) -> ndarray:
    '''
    绘制每个备选车辆的预测轨迹.
    线框是当前车身，折线是未来 T 步的中心轨迹，大点是当前时刻。
    '''
    frame: ndarray = env.render(**render_kwargs)
    canvas: Surface = env.top_down_renderer._frame_canvas
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    
    def pix(point) -> tuple:
        return canvas.pos2pix(float(point[0]), float(point[1]))

    for j in range(pos.shape[1]):
        if _scalar(mask[j]) < 0.5:
            continue
        color = _SLOT_COLORS[j % len(_SLOT_COLORS)]
        traj = [pix(pos[k, j]) for k in range(pos.shape[0])]
        if len(traj) >= 2:
            draw.line(traj, fill=color, width=2)
        for k, (x, y) in enumerate(traj):
            r = 4 if k == 0 else 2
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

        center = np.asarray(pos[0, j], dtype=float)
        psi = _scalar(heading[0, j])
        half_l = _scalar(length[j]) / 2.0
        half_w = _scalar(width[j]) / 2.0
        forward = np.array([np.cos(psi), np.sin(psi)])
        left = np.array([-np.sin(psi), np.cos(psi)])
        corners = [
            center + forward * half_l + left * half_w,
            center + forward * half_l - left * half_w,
            center - forward * half_l - left * half_w,
            center - forward * half_l + left * half_w,
        ]
        draw.polygon([pix(corner) for corner in corners], outline=color)
    return np.asarray(image)


def main(seed: int = 0):
    env = create_env(seed, use_policy=True)
    T: int = 15
    delta_t: float = 0.02 * 5
    nobs: int = 3
    env.reset(seed=seed)
    vehicle: DefaultVehicle = env.agent
    transitor = Transitor(env, T, delta_t)

    frames = []
    for i in range(200):
        obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
        # 触发式车流会在行驶中生成，预测函数读的是这两份引用
        transitor._traffic = list(env.engine.traffic_manager.traffic_vehicles)
        transitor.ego = env.agent
        pos, heading, length, width, mask = transitor._predict_traffic_info(nobs)
        frames.append(mark_chosen(env, pos, heading, length, width, mask))

    generate_gif(frames, os.path.join(ROOT, 'predict_traffics.gif'))


if __name__ == '__main__':
    main()