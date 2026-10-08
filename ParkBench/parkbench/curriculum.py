"""从车位沿自行车模型向外滚出可行起点。前进和后退都允许。

先直线驶出车位，再一边打方向一边换挡把车头摆进过道，最后沿新航向开出横向距离。
课程按离目标的远近截取这段轨迹。最后一阶段用日志里的真实起点。
"""

from __future__ import annotations

import numpy as np

from parkbench.demos import chain_for
from parkbench.vehicle import ACTIONS, bicycle_step, collides, normalize_angle

# 沿可倒回车位的轨迹取起点。index 是离开车位的步数，最后一阶段是日志起点。
STAGES = (
    {"index": 24, "max_steps": 160},
    {"index": 48, "max_steps": 220},
    {"index": 70, "max_steps": 320},
    {"index": 90, "max_steps": 400},
    {"index": 110, "max_steps": 500},
    {"index": 130, "max_steps": 600},
    {"index": 150, "max_steps": 700},
    {"index": 170, "max_steps": 800},
    {"index": 190, "max_steps": 900, "logged_mix": 0.4},
    {"logged": True, "max_steps": 1000},
)

_PATHS: dict[str, list[tuple[float, float, float]]] = {}


def _try_step(points, x, y, theta, delta, action: int):
    ddelta, v = ACTIONS[action]
    nx, ny, nth, nd, ds = bicycle_step(x, y, theta, delta, ddelta, v)
    if collides(points, nx, ny, nth):
        return None
    return nx, ny, nth, nd, abs(ds)


def _drive(points, x, y, theta, delta, action: int, limit: int, path: list):
    moved = 0
    for _ in range(limit):
        nxt = _try_step(points, x, y, theta, delta, action)
        if nxt is None:
            break
        x, y, theta, delta, _ds = nxt
        path.append((x, y, theta))
        moved += 1
    return x, y, theta, delta, moved


def _lateral(target, x: float, y: float) -> float:
    dx, dy = x - target[0], y - target[1]
    c, s = np.cos(target[2]), np.sin(target[2])
    return float(-s * dx + c * dy)


def _maneuver(scene, side: int) -> list[tuple[float, float, float]]:
    """side > 0 向左摆，side < 0 向右摆。返回从车位向外的位姿序列。"""
    x, y, theta = (float(scene.target[0]), float(scene.target[1]), float(scene.target[2]))
    delta = 0.0
    path = [(x, y, theta)]
    x, y, theta, delta, _ = _drive(scene.points, x, y, theta, delta, 1, 80, path)

    forward = 2 if side > 0 else 0
    reverse = 3 if side > 0 else 5
    # 先沿打方向前进。只有这一侧被挡住时才倒车，倒出来再继续。
    for _ in range(8):
        _x, _y, _t, _d, n_fwd = _drive(scene.points, x, y, theta, delta, forward, 20, path)
        x, y, theta, delta = _x, _y, _t, _d
        if n_fwd > 0:
            continue
        x, y, theta, delta, n_rev = _drive(scene.points, x, y, theta, delta, reverse, 20, path)
        if n_rev == 0:
            break

    for _ in range(200):
        lat = abs(_lateral(scene.target, x, y))
        best = None
        best_lat = lat
        for action in range(6):
            nxt = _try_step(scene.points, x, y, theta, delta, action)
            if nxt is None:
                continue
            nlat = abs(_lateral(scene.target, nxt[0], nxt[1]))
            if nlat > best_lat + 1e-4:
                best_lat = nlat
                best = nxt
        if best is None:
            break
        x, y, theta, delta, _ds = best
        path.append((x, y, theta))
    return path


def build_path(scene) -> list[tuple[float, float, float]]:
    cached = _PATHS.get(scene.name)
    if cached is not None:
        return cached
    poses = _maneuver(scene, side=1) + _maneuver(scene, side=-1)
    _PATHS[scene.name] = poses
    return poses


def _pose_features(scene, poses: list[tuple[float, float, float]]):
    target = scene.target
    xs = np.array([p[0] for p in poses])
    ys = np.array([p[1] for p in poses])
    th = np.array([p[2] for p in poses])
    dx, dy = xs - target[0], ys - target[1]
    c, s = np.cos(target[2]), np.sin(target[2])
    dist = np.hypot(dx, dy)
    lat = np.abs(-s * dx + c * dy)
    yaw = np.abs(np.degrees(np.arctan2(np.sin(th - target[2]), np.cos(th - target[2]))))
    return dist, lat, yaw


def sample_start(scene, stage: int, rng: np.random.Generator):
    stage = int(stage)
    spec = STAGES[stage]
    if spec.get("logged") or float(rng.random()) < float(spec.get("logged_mix", 0.0)):
        return float(scene.ego[0]), float(scene.ego[1]), float(scene.ego[2]), 0.0

    side = 1 if float(rng.random()) < 0.5 else -1
    states = chain_for(scene, side)
    index = int(spec["index"]) + int(rng.integers(-6, 7))
    index = int(np.clip(index, 8, min(200, len(states) - 1)))
    x, y, theta, delta = states[index]
    return float(x), float(y), float(normalize_angle(theta)), float(delta)
