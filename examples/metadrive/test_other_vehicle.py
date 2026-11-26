from env import *
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

def route_lanes(vehicle):
    navi = vehicle.navigation
    network = navi.map.road_network
    nodes = navi.checkpoints
    lane = vehicle.lane
    start_i = navi._target_checkpoints_index[0]
    for i in range(len(nodes) - 1):
        if nodes[i] == lane.index[0] and nodes[i + 1] == lane.index[1]:
            start_i = i
            break
    lanes = []
    prev = None
    for i in range(start_i, len(nodes) - 1):
        parallel = network.graph[nodes[i]][nodes[i + 1]]
        if prev is None:
            chosen = lane if lane in parallel else parallel[min(lane.index[-1], len(parallel) - 1)]
        else:
            linked = [c for c in parallel if prev.is_previous_lane_of(c)]
            same = [c for c in linked if c.index[-1] == prev.index[-1]]
            chosen = same[0] if same else (linked[0] if linked else parallel[min(prev.index[-1], len(parallel) - 1)])
        lanes.append(chosen)
        prev = chosen
    return lanes

def predict_vehicle_position(env: MetaDriveEnv, nobs: int, T: int, delta_t: float) -> ndarray:
    ego: DefaultVehicle = env.agent
    others: List[BaseVehicle] = list(env.engine.traffic_manager.traffic_vehicles)
    others.sort(key=lambda v: np.linalg.norm(np.asarray(v.position) - np.asarray(ego.position)))
    chosen = others[:nobs]
    pos = np.empty([T + 1, nobs, 2])
    pos[:] = np.asarray(ego.position) + np.array([1.0e4, 0.0])
    for j, v in enumerate(chosen):
        if v.lane is None or v.navigation is None:
            p = np.asarray(v.position, dtype=float)
            vel = np.asarray(v.velocity, dtype=float)
            for k in range(T + 1):
                pos[k, j] = p + vel * (k * delta_t)
            continue
        lanes = route_lanes(v)
        long, lat = lanes[0].local_coordinates(v.position)
        for k in range(T + 1):
            s = long + float(v.speed) * k * delta_t
            lane_i = 0
            lane = lanes[lane_i]
            while s > lane.length and lane_i + 1 < len(lanes):
                s -= lane.length
                lane_i += 1
                lane = lanes[lane_i]
            s = min(max(s, 0.0), lane.length)
            pos[k, j] = lane.position(s, lat)
    return chosen, pos


def main(seed: int = 0):
    env = create_env(seed)
    T = 15
    delta_t = 0.02 * 5
    env.reset(seed=seed)
    vehicle: DefaultVehicle = env.agent

    frames = []
    for i in range(200):
        if i <= 80:
            action = [0.0, 0.5]
        elif 80 < i <= 95:
            action = [-0.5, 0.0]
        else:
            action = [0.0, 0.0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        chosen, pos = predict_vehicle_position(env, 3, T, delta_t)
        # mark
        chosen_id = {v.id for v in chosen}
        for v in env.engine.traffic_manager.traffic_vehicles:
            if not hasattr(v, '_cbf_color'):
                v._cbf_color = v._panda_color
            v._panda_color = (1.0, 0.0, 0.0) if v.id in chosen_id else v._cbf_color
        
        frames.append(env.render(**render_kwargs))

    generate_gif(frames, 'demo.gif')


if __name__ == '__main__':
    main()