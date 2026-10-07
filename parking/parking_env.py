"""parking-v0 环境封装。

highway-env 的 parking-v0 是目标条件任务，观测是 dict：
observation / achieved_goal / desired_goal，各为 6 维运动学
[x, y, vx, vy, cos_h, sin_h]。PPO 的 MlpPolicy 需要向量观测，
这里拼成「自车状态 + 目标位姿 + 位姿误差」。
"""

from __future__ import annotations

import gymnasium as gym
import highway_env  # noqa: F401  注册 parking-v0
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor


FEATURES = ("x", "y", "vx", "vy", "cos_h", "sin_h")

# duration 的单位是秒。策略频率 5 Hz，环境默认 100 秒，也就是 500 个控制步。
# 这个长度下，没泊进去的累积惩罚远大于自带的碰撞 -5，PPO 会学着早点撞墙。
# 收成 100 个控制步（20 秒），配合下面的撞车重罚，成功才是回报最高的结局。
POLICY_HZ = 5
MAX_EPISODE_STEPS = 100
ENV_CONFIG = {"duration": MAX_EPISODE_STEPS / POLICY_HZ}
CRASH_PENALTY = -100.0


class CrashPenaltyWrapper(gym.Wrapper):
    """撞车时额外扣一次分。

    parking-v0 自带的 collision_reward 只有 -5。默认回合有 500 个控制步，
    没泊进去会累积大约 -250，早点撞墙反而更赚。这里在碰撞终止的那一步
    再扣一笔，配合收短后的 100 步回合，让撞车差于耗满时限。
    """

    def __init__(self, env: gym.Env, penalty: float = CRASH_PENALTY):
        super().__init__(env)
        self.penalty = float(penalty)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated and info.get("crashed", False) and self.penalty != 0.0:
            reward = float(reward) + self.penalty
            info = dict(info)
            info["crash_penalty"] = self.penalty
        return obs, reward, terminated, truncated, info


class ParkingGoalWrapper(gym.ObservationWrapper):
    """把 KinematicsGoal 的 dict 观测压成 float32 向量。"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        dim = 3 * len(FEATURES)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
        )

    def observation(self, obs: dict) -> np.ndarray:
        ego = np.asarray(obs["observation"], dtype=np.float32)
        goal = np.asarray(obs["desired_goal"], dtype=np.float32)
        # achieved_goal 与 observation 相同，误差直接用目标减自车。
        error = goal - ego
        return np.concatenate([ego, goal, error])


def make_parking_env(
    seed: int | None = None,
    render_mode: str | None = None,
    crash_penalty: float = CRASH_PENALTY,
):
    """返回 VecEnv 用的环境工厂。seed 只作用于该进程里的这一份环境。"""

    def _init():
        env = gym.make("parking-v0", render_mode=render_mode, config=dict(ENV_CONFIG))
        env = ParkingGoalWrapper(env)
        env = CrashPenaltyWrapper(env, penalty=crash_penalty)
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        return env

    return _init
