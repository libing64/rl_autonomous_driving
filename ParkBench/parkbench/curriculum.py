"""从目标位姿沿自行车模型往外滚，生成由近到远的课程初始位姿。

图 5 没有给出每阶段的米数和航向范围，这里按正文复原：
前两阶段沿用滚出的航向，之后在该航向附近采一个不撞的航向；
横向偏差只来自滚出轨迹。最后一阶段用日志里的起点。
"""

from __future__ import annotations

import numpy as np

from parkbench.vehicle import ACTIONS, bicycle_step, collides, normalize_angle

# distance: 从目标沿可行方向滚出的路程。heading_range: 相对滚出航向的采样半宽（度）。
STAGES = (
    {"distance": 0.6, "heading": "rollout", "heading_range": 0.0, "max_steps": 100},
    {"distance": 1.5, "heading": "rollout", "heading_range": 0.0, "max_steps": 200},
    {"distance": 3.0, "heading": "sample", "heading_range": 15.0, "max_steps": 400},
    {"distance": 5.0, "heading": "sample", "heading_range": 30.0, "max_steps": 400},
    {"distance": 8.0, "heading": "sample", "heading_range": 45.0, "max_steps": 800},
    {"distance": 11.0, "heading": "sample", "heading_range": 60.0, "max_steps": 800},
    {"distance": 15.0, "heading": "sample", "heading_range": 90.0, "max_steps": 800},
    {"distance": None, "heading": "logged", "heading_range": 0.0, "max_steps": 1000},
)


def rollout_from_target(target, points, distance: float, rng: np.random.Generator):
    """从车位往外开，得到一个不撞的起点。开不动就停在最远的可行位姿。"""
    x, y, theta = (float(target[0]), float(target[1]), float(target[2]))
    delta = 0.0
    traveled = 0.0
    # 只使用向前的三个动作，对应倒车入库的反向。
    forward = (0, 1, 2)
    while traveled < distance - 1e-6:
        order = rng.permutation(forward)
        moved = False
        for index in order:
            ddelta, v = ACTIONS[int(index)]
            nx, ny, nth, nd, ds = bicycle_step(x, y, theta, delta, ddelta, v)
            if collides(points, nx, ny, nth):
                continue
            x, y, theta, delta = nx, ny, nth, nd
            traveled += abs(ds)
            moved = True
            break
        if not moved:
            break
    return x, y, theta, delta


def sample_start(scene, stage: int, rng: np.random.Generator):
    spec = STAGES[stage]
    if spec["heading"] == "logged":
        x, y, theta = scene.ego
        return float(x), float(y), float(theta), 0.0

    x, y, theta, delta = rollout_from_target(scene.target, scene.points, spec["distance"], rng)
    if spec["heading"] == "sample":
        half = np.deg2rad(spec["heading_range"])
        for _ in range(24):
            candidate = float(normalize_angle(theta + rng.uniform(-half, half)))
            if not collides(scene.points, x, y, candidate):
                theta = candidate
                break
    return x, y, theta, delta
