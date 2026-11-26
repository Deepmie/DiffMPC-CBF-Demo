import gymnasium as gym
from gymnasium import Env
import highway_env
from highway_env.envs import HighwayEnv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pygame
import os
gym.register_envs(highway_env)


def route_reference(base: HighwayEnv, delta_t, T, vref):
    vehicle = base.vehicle
    # frames = []
    # frames.append(Image.fromarray(env.render()))
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
    pref, psiref = [], []
    for _ in range(T + 1):
        lane = lanes[lane_i]
        while s > lane.length and lane_i + 1 < len(lanes):
            s -= lane.length
            lane_i += 1
            lane = lanes[lane_i]
        s_clip = min(max(s, 0.0), lane.length)
        pref.append(lane.position(s_clip, 0.0))
        psiref.append(float(lane.heading_at(s_clip)))
        s += vref * delta_t
    return pref, psiref


def main(seed=0):
    env = gym.make(
        'intersection-v2',
        config = {
            'action': {'type': 'ContinuousAction'},
            'simulation_frequency': 15,
            'policy_frequency': 15,
            'destination': 'o1',
        },
        render_mode = 'rgb_array',
    )
    T = 15
    delta_t = 1 / 15
    vref = 10.0
    env.reset(seed=seed)
    base: HighwayEnv = env.unwrapped
    vehicle = base.vehicle

    lane = vehicle.lane
    s0 = 20.0
    vehicle.position = lane.position(s0, 0.0)
    vehicle.heading = lane.heading_at(s0)
    
    frames = []
    for i in range(100):
        pref, _ = route_reference(base, delta_t, T, vref)
        env.render()
        surface = base.viewer.sim_surface
        for p in pref:
            pygame.draw.circle(surface, (255, 0, 0), surface.vec2pix(p), 3)
            frame = Image.fromarray(base.viewer.get_image())

        if i <= 60:
            action = [1.0, 0.0]
        elif 60 <= i <= 70:
            action = [0.0, -0.5]
        elif 70 <= i:
            action = [0.0, 0.0]
        env.step(action)
        frames.append(frame)
    
    
    if frames:
        frames[0].save(
            'route_demo.gif',
            save_all      = True,
            append_images = frames[1:],
            duration      = int(delta_t * 1000),
            loop          = 0,
        )
    env.close()
    

if __name__ == '__main__':
    main()



    