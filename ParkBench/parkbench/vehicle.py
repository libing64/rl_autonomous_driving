"""自行车模型和八边形车体。参数来自论文附录 B。"""

from __future__ import annotations

import numpy as np

WHEELBASE = 3.0
WIDTH = 2.0
LENGTH = 4.95
REAR_OVERHANG = 1.025
FRONT_OVERHANG = 3.925
MAX_STEER_DEG = 32.0
MAX_STEER = np.deg2rad(MAX_STEER_DEG)
CROP_L = 0.3
CROP_W = 0.2
# 几何中心在后轴前方。后悬到前悬的中点。
CENTER_OFFSET = 0.5 * (FRONT_OVERHANG - REAR_OVERHANG)

DT = 0.1
# 附录 D：Δδ ∈ {-8, 0, +8} 度，v ∈ {-0.8, 0, +0.8} m/s。不含 (0, 0)。
ACTIONS = (
    (-8.0, 0.8),
    (0.0, 0.8),
    (8.0, 0.8),
    (-8.0, -0.8),
    (0.0, -0.8),
    (8.0, -0.8),
    (-8.0, 0.0),
    (8.0, 0.0),
)

# 后轴为原点，+x 向前，+y 向左。顶点按逆时针。
POLYGON = np.array(
    [
        [-REAR_OVERHANG + CROP_L, -WIDTH / 2],
        [FRONT_OVERHANG - CROP_L, -WIDTH / 2],
        [FRONT_OVERHANG, -WIDTH / 2 + CROP_W],
        [FRONT_OVERHANG, WIDTH / 2 - CROP_W],
        [FRONT_OVERHANG - CROP_L, WIDTH / 2],
        [-REAR_OVERHANG + CROP_L, WIDTH / 2],
        [-REAR_OVERHANG, WIDTH / 2 - CROP_W],
        [-REAR_OVERHANG, -WIDTH / 2 + CROP_W],
    ],
    dtype=np.float64,
)


def normalize_angle(angle: float | np.ndarray) -> np.ndarray:
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def geometric_center(x: float, y: float, theta: float) -> np.ndarray:
    return np.array(
        [x + CENTER_OFFSET * np.cos(theta), y + CENTER_OFFSET * np.sin(theta)],
        dtype=np.float64,
    )


def bicycle_step(x, y, theta, delta, ddelta_deg: float, v: float):
    """先更新前轮转角，再用新转角积分一步。Δs = v · dt。"""
    delta = float(np.clip(delta + np.deg2rad(ddelta_deg), -MAX_STEER, MAX_STEER))
    ds = float(v) * DT
    x = x + ds * np.cos(theta)
    y = y + ds * np.sin(theta)
    theta = float(normalize_angle(theta + ds / WHEELBASE * np.tan(delta)))
    return float(x), float(y), theta, delta, ds


def world_to_body(points: np.ndarray, x: float, y: float, theta: float) -> np.ndarray:
    dx = points[:, 0] - x
    dy = points[:, 1] - y
    c, s = np.cos(theta), np.sin(theta)
    out = np.empty_like(points, dtype=np.float64)
    out[:, 0] = c * dx + s * dy
    out[:, 1] = -s * dx + c * dy
    return out


def points_in_convex(points: np.ndarray, polygon: np.ndarray = POLYGON) -> np.ndarray:
    """逆时针凸多边形。边上的点算在内部。"""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    a = polygon
    b = np.roll(polygon, -1, axis=0)
    edge = b - a
    rel = points[:, None, :] - a[None, :, :]
    cross = edge[:, 0] * rel[:, :, 1] - edge[:, 1] * rel[:, :, 0]
    return np.all(cross >= -1e-8, axis=1)


def collides(points: np.ndarray, x: float, y: float, theta: float, radius: float = 8.0) -> bool:
    if len(points) == 0:
        return False
    d2 = (points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2
    near = points[d2 <= radius * radius]
    if len(near) == 0:
        return False
    body = world_to_body(near, x, y, theta)
    return bool(np.any(points_in_convex(body)))
