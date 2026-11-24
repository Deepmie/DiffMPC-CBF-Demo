import gymnasium as gym
import highway_env
from highway_env.envs import HighwayEnvFast
from copy import deepcopy
import numpy as np
from numpy import ndarray
from PIL import Image
from typing import List

gym.register_envs(highway_env)

N: int = 2
T: int = 200
delta_t: float = 1 / 15
# Initialise the environment
env = gym.make(
"highway-fast-v0",
config={
    "action": {"type": "ContinuousAction"},
    "simulation_frequency": 15,
    "policy_frequency": 15,
}, render_mode="rgb_array")
env.reset()
env_row: HighwayEnvFast = env.unwrapped
src = env_row.road
ego_idx = src.vehicles.index(env_row.vehicle)
road = deepcopy(src)
ego = road.vehicles[ego_idx]

print(type(ego))
input()
ego.act({"acceleration": 0.0, "steering": 0.0})

others = [i for i in range(len(road.vehicles)) if i != ego_idx]

pos = 1e3 * np.ones([T+1, N, 2])
env_row_road = env_row.road
env_row_controlled = list(env_row.controlled_vehicles)
frames: List[ndarray] = list()
try:
    env_row.road = road
    env_row.controlled_vehicles = [ego]
    for k in range(T+1):
        
        # chose the closed vehicle
        others.sort(key=lambda i: np.linalg.norm(road.vehicles[i].position - ego.position, ord=2))
        chosen = others[:N]
        for i in others:
            if i in chosen:
                road.vehicles[i].color = (255, 80, 80)
            else:
                road.vehicles[i].color = (100, 200, 255)
        
        for j, i in enumerate(chosen):
            pos[k, j] = road.vehicles[i].position
        
        frame = env.render()
        frames.append(Image.fromarray(frame))
        if k == T:
            break
        road.act()
        road.step(delta_t)
finally:
    env_row.road = env_row_road
    env_row.controlled_vehicles = env_row_controlled

if frames:
    frames[0].save(
        f"./predict.gif",
        save_all=True,
        append_images=frames[1:],
        duration=int(delta_t * 1000),
        loop=0,
    )
