from env import *
from transitor import Transitor
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
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

def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def get_frame_withpt(
        env: MetaDriveEnv,
        pref: ndarray,
        psiref: ndarray,
        nleft: ndarray,
        nright: ndarray,
    ) -> ndarray:
    frame = env.render(**render_kwargs)
    canvas: Surface = env.top_down_renderer._frame_canvas
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)

    def pix(point) -> tuple:
        return canvas.pos2pix(float(point[0]), float(point[1]))

    # 左右法向与 Transitor._clearance 一致：左法向 (-sin, cos)
    bounds = []
    for i, p in enumerate(pref):
        psi = _scalar(psiref[i])
        left_n = np.array([-np.sin(psi), np.cos(psi)])
        left_pt = np.asarray(p, dtype=float) + _scalar(nleft[i]) * left_n
        right_pt = np.asarray(p, dtype=float) - _scalar(nright[i]) * left_n
        bounds.append((pix(left_pt), pix(right_pt)))

    for left_pix, right_pix in bounds:
        draw.line([left_pix, right_pix], fill=(255, 220, 0), width=1)
    for left_pix, _ in bounds:
        draw.ellipse((left_pix[0] - 2, left_pix[1] - 2, left_pix[0] + 2, left_pix[1] + 2), fill=(0, 255, 255))
    for _, right_pix in bounds:
        draw.ellipse((right_pix[0] - 2, right_pix[1] - 2, right_pix[0] + 2, right_pix[1] + 2), fill=(255, 140, 0))

    pixels = [pix(p) for p in pref]
    draw.line(pixels, fill=(255, 0, 0), width=1)
    for i, (x, y) in enumerate(pixels):
        r = 4 if i == 0 else 2
        color = (255, 0, 0) if i == 0 else (0, 80, 255)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    return np.asarray(image)


def main(seed=0):
    env = create_env(seed, use_policy=True)
    T = 15
    delta_t = 0.02 * 5
    vref = 10.0
    env.reset(seed=seed)
    vehicle: DefaultVehicle = env.agent
    transitor = Transitor(env, T, delta_t)
    
    frames = []
    for i in range(200):
        obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
        pref, psiref, nleft, nright = transitor._path_planning(vref)
        frames.append(get_frame_withpt(env, pref, psiref, nleft, nright))
        if terminated or truncated:
            break

    generate_gif(frames, os.path.join(ROOT, 'path_planning.gif'))
    env.close()
    

if __name__ == '__main__':
    main()



    