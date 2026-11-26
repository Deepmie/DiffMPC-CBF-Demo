from metadrive.envs import MetaDriveEnv, SafeMetaDriveEnv

class EnvGenerator:
    def __init__(self):
        self._config = dict(
            num_scenarios=1, # 包含的场景数量
            start_seed=0,
            random_traffic=True, # 随机化背景车流生成位置与车型
            random_lane_width=True, # 随机化车道宽度 (3.0m ~ 4.5m)，考验横向安全边界鲁棒性
            random_lane_num=True, # 随机化车道数 (2 ~ 4 车道)
            map=7, # 包含7个随机组合模块，易出现连续弯道、环岛、十字路口组合
            traffic_density=0.25, # 高密度车流 (默认0.1，0.25~0.35 博弈极强)
            traffic_mode="trigger", # 触发式生成，保证自车沿途始终处于密集车流中
            accident_prob=0.5, # 50%概率在路网中生成事故车/施工锥桶，迫使借道绕行
            discrete_action=False, # 连续动作空间: [转向角, 油门/刹车]
            use_lateral_reward=True, # 鼓励车道保持，抑制画龙
            vehicle_config=dict(
                lidar=dict(
                    num_lasers=240, # 高分辨率激光雷达，方便构建安全距离约束或障碍物集合
                    distance=50,
                    num_others=4, # 同时返回最近的4辆车精确运动学状态 (x, y, vx, vy, heading)
                ),
                show_lidar=False,
                show_navi_mark=True,
            ),
            out_of_road_done=True,
            crash_vehicle_done=True,
            crash_object_done=True,
            horizon=1500, # 复合长地图需调大最大步数，避免正常通行超时
        )
        self._env = MetaDriveEnv(self._config)

    @property
    def env(self) -> MetaDriveEnv:
        return self._env