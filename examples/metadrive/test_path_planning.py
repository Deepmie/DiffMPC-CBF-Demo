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

def get_frame_withpt(env: MetaDriveEnv, pref: ndarray) -> ndarray:
    frame = env.render(**render_kwargs)
    canvas: Surface = env.top_down_renderer._frame_canvas
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    pixels = [canvas.pos2pix(float(p[0]), float(p[1])) for p in pref]
    draw.line(pixels, fill=(255, 0, 0), width=1)
    for i, (x, y) in enumerate(pixels):
        r = 4 if i == 0 else 2
        color = (255, 0, 0) if i == 0 else (0, 80, 255)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
    return np.asarray(image)


def main(seed=0):
    env = create_env(seed)
    T = 15
    delta_t = 0.02 * 5
    vref = 10.0
    env.reset(seed=seed)
    vehicle: DefaultVehicle = env.agent
    transitor = Transitor(env, T, delta_t)
    
    frames = []
    for i in range(200):
        if i <= 80:
            action = [0.0, 0.5]
        elif 80 < i <= 95:
            action = [-0.5, 0.0]
        else:
            action = [0.0, 0.0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        pref, psiref, nleft, nright = transitor._path_planning(vref)
        frames.append(get_frame_withpt(env, pref))

    generate_gif(frames, 'demo.gif')
    env.close()
    

if __name__ == '__main__':
    main()



    