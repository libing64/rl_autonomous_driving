"""highway-v0 换道决策环境。

动作是离散的高层决策，不是方向盘转角：
0 左换道，1 保持，2 右换道，3 加速，4 减速。
观测是自车和最近 4 辆车的运动学，每辆 5 维，压成 25 维向量给 PPO。
"""

from __future__ import annotations

import gymnasium as gym
import highway_env  # noqa: F401  注册 highway-v0
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor


ACTIONS = ("LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER")
FEATURES = ("presence", "x", "y", "vx", "vy")
VEHICLES = 5

# 策略 1 Hz，duration 单位是秒，40 秒就是 40 个决策步。
# 默认仿真 15 Hz、50 辆背景车，一步大约要 100ms 以上。
# 决策频率不变，把仿真降到 5 Hz、背景车 20 辆，换道任务还在，采样才能跟上 PPO。
ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": VEHICLES,
        "features": list(FEATURES),
        "absolute": False,
        "normalize": True,
        "order": "sorted",
    },
    "action": {"type": "DiscreteMetaAction"},
    "simulation_frequency": 5,
    "lanes_count": 4,
    "vehicles_count": 20,
    "duration": 40,
    "collision_reward": -1,
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.4,
    "lane_change_reward": 0,
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
}


class FlattenKinematics(gym.ObservationWrapper):
    """把 (车辆数, 特征) 的运动学观测拉成一条向量。"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        dim = VEHICLES * len(FEATURES)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32).reshape(-1)


def make_highway_env(seed: int | None = None, render_mode: str | None = None):
    """返回 VecEnv 用的环境工厂。seed 只作用于这一份环境。"""

    def _init():
        env = gym.make("highway-v0", render_mode=render_mode, config=dict(ENV_CONFIG))
        env = FlattenKinematics(env)
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        return env

    return _init
