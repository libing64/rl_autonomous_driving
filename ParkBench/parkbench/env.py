"""ParkBench Gym 环境。观测是自车坐标系下的目标和障碍点。"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from parkbench.curriculum import STAGES, sample_start
from parkbench.data import Scene, load_scenes
from parkbench.vehicle import (
    ACTIONS,
    MAX_STEER,
    bicycle_step,
    collides,
    geometric_center,
    normalize_angle,
    world_to_body,
)

# Hybrid A* 丢掉 25m 以外的点。感知范围沿用这个数，论文没有另外给。
HORIZON = 25.0
MAX_POINTS = 256
GOAL_POS = 0.2
GOAL_YAW = np.deg2rad(3.0)
OUT_OF_BOUNDS = 30.0

R_GOAL = 3.0
R_COLLISION = -3.0
R_OUT = -3.0
R_GEAR = -0.01
R_IDLE = -0.2
R_TIME = -0.01


class ParkEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenes: list[Scene] | None = None, stage: int = 0):
        super().__init__()
        self.scenes = scenes if scenes is not None else load_scenes()
        if not self.scenes:
            raise FileNotFoundError("no scenarios under ParkBench/scenarios/rear_in")
        self.stage = int(stage)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
                "obstacles": spaces.Box(-np.inf, np.inf, (MAX_POINTS, 2), dtype=np.float32),
                "mask": spaces.Box(0.0, 1.0, (MAX_POINTS,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.scene: Scene | None = None
        self.x = self.y = self.theta = self.delta = 0.0
        self.steps = 0
        self.last_sign = 0
        self.path_xy: list[tuple[float, float]] = []
        self.travel = 0.0
        self.pivots = 0

    def set_stage(self, stage: int) -> None:
        self.stage = int(np.clip(stage, 0, len(STAGES) - 1))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "scenario_index" in options:
            index = int(options["scenario_index"])
        else:
            index = int(self.np_random.integers(0, len(self.scenes)))
        stage = int(options.get("stage", self.stage))
        self.scene = self.scenes[index]
        rng = np.random.default_rng(int(self.np_random.integers(0, 2**31 - 1)))
        self.x, self.y, self.theta, self.delta = sample_start(self.scene, stage, rng)
        self.steps = 0
        self.last_sign = 0
        self.travel = 0.0
        self.pivots = 0
        self.path_xy = [(self.x, self.y)]
        self._stage_limit = STAGES[stage]["max_steps"]
        self._phi_prev = self._phi()
        return self._observe(), self._info(False)

    def step(self, action: int):
        ddelta, v = ACTIONS[int(action)]
        self.x, self.y, self.theta, self.delta, ds = bicycle_step(
            self.x, self.y, self.theta, self.delta, ddelta, v
        )
        self.steps += 1
        self.travel += abs(ds)
        self.path_xy.append((self.x, self.y))

        sign = 1 if v > 0 else -1 if v < 0 else 0
        reward = R_TIME
        if sign == 0:
            reward += R_IDLE
        elif self.last_sign != 0 and sign != self.last_sign:
            reward += R_GEAR
            self.pivots += 1
        if sign != 0:
            self.last_sign = sign
        # γ=1 下的势函数差分：靠近目标、摆正航向有中间奖励，最优策略不变。
        phi = self._phi()
        reward += phi - self._phi_prev
        self._phi_prev = phi

        terminated = False
        success = False
        crashed = collides(self.scene.points, self.x, self.y, self.theta)
        if crashed:
            reward += R_COLLISION
            terminated = True
        else:
            center = geometric_center(self.x, self.y, self.theta)
            goal_center = geometric_center(*self.scene.target)
            pos_err = float(np.linalg.norm(center - goal_center))
            yaw_err = float(abs(normalize_angle(self.theta - self.scene.target[2])))
            if pos_err <= GOAL_POS and yaw_err <= GOAL_YAW:
                reward += R_GOAL
                terminated = True
                success = True
            elif float(np.hypot(self.x - self.scene.target[0], self.y - self.scene.target[1])) > OUT_OF_BOUNDS:
                reward += R_OUT
                terminated = True

        truncated = (not terminated) and self.steps >= self._stage_limit
        return self._observe(), float(reward), terminated, truncated, self._info(success)

    def _phi(self) -> float:
        """先减小横向偏差和航向差，再沿车位中线靠近目标。"""
        target = self.scene.target
        dx = self.x - target[0]
        dy = self.y - target[1]
        c, s = np.cos(target[2]), np.sin(target[2])
        fwd = c * dx + s * dy
        lat = -s * dx + c * dy
        yaw = float(abs(normalize_angle(self.theta - target[2])))
        return -0.7 * abs(float(lat)) - 2.0 * yaw - 0.2 * abs(float(fwd))

    def _observe(self):
        target = self.scene.target
        dx = target[0] - self.x
        dy = target[1] - self.y
        c, s = np.cos(self.theta), np.sin(self.theta)
        gx = c * dx + s * dy
        gy = -s * dx + c * dy
        gth = float(normalize_angle(target[2] - self.theta))
        dist = float(np.hypot(gx, gy))
        state = np.array(
            [
                gx / HORIZON,
                gy / HORIZON,
                np.cos(gth),
                np.sin(gth),
                self.delta / MAX_STEER,
                dist / HORIZON,
            ],
            dtype=np.float32,
        )

        obstacles = np.zeros((MAX_POINTS, 2), dtype=np.float32)
        mask = np.zeros(MAX_POINTS, dtype=np.float32)
        pts = self.scene.points
        if len(pts):
            body = world_to_body(pts, self.x, self.y, self.theta)
            dist_b = np.hypot(body[:, 0], body[:, 1])
            keep = dist_b <= HORIZON
            body = body[keep]
            dist_b = dist_b[keep]
            if len(body) > MAX_POINTS:
                chosen = np.argpartition(dist_b, MAX_POINTS)[:MAX_POINTS]
                body = body[chosen]
            n = len(body)
            if n:
                obstacles[:n] = (body / HORIZON).astype(np.float32)
                mask[:n] = 1.0
        return {"state": state, "obstacles": obstacles, "mask": mask}

    def _info(self, success: bool) -> dict:
        return {
            "is_success": success,
            "travel": self.travel,
            "pivots": self.pivots,
            "scene": None if self.scene is None else self.scene.name,
        }


class ActionChunkWrapper(gym.Wrapper):
    """把 h 个底层动作当成一个宏动作执行。h=4 对应论文的 action chunking。"""

    def __init__(self, env: gym.Env, chunk: int = 4):
        super().__init__(env)
        self.chunk = int(chunk)
        if self.chunk <= 1:
            self.action_space = spaces.Discrete(env.action_space.n)
        else:
            self.action_space = spaces.MultiDiscrete([env.action_space.n] * self.chunk)

    def step(self, action):
        if self.chunk <= 1:
            primitives = [int(action)]
        else:
            primitives = [int(a) for a in np.asarray(action).reshape(-1)]
        total = 0.0
        obs = info = None
        terminated = truncated = False
        for primitive in primitives:
            obs, reward, terminated, truncated, info = self.env.step(primitive)
            total += float(reward)
            if terminated or truncated:
                break
        return obs, total, terminated, truncated, info
