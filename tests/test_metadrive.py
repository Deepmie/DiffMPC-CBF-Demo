from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils.doc_utils import generate_gif
from PIL import Image

def main():
    seed = 0
    config = dict(
        num_scenarios=1,
        start_seed=seed,
        map="O",
        random_lane_width=False,       # 车道宽固定 3.5 m
        random_lane_num=False,         # 车道数固定为默认 3
        random_spawn_lane_index=False, # 自车固定在出生道路的 0 号车道
        random_traffic=False,          # 车流由场景种子决定，每次 reset 相同
        traffic_density=0.1,
        traffic_mode="trigger",
        accident_prob=0.0,
        discrete_action=False, # 动作 [转向, 油门]，范围 [-1, 1]
        out_of_road_done=True,
        crash_vehicle_done=True,
        crash_object_done=True,
        horizon=1000,
    )
    env = MetaDriveEnv(config)
    obs, info = env.reset(seed)
    frames = []
    render_kwargs = dict(
        mode="topdown",
        window=False,
        film_size=(800, 800),
        screen_size=(800, 800),
        scaling=None,
        center_on_map=True,
        target_agent_heading_up=False,
    )

    for i in range(200):
        if i <= 80:
            action = [0.0, 0.5]
        elif 80 < i <= 95:
            action = [0.5, 0.0]
        else:
            action = [0.0, 0.0]            

        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render(**render_kwargs))
        if terminated or truncated:
            env.reset(seed)
    env.close()
    generate_gif(frames, 'demo.gif')

if __name__ == '__main__':
    main()