"""读取官方 ParkBench JSON。障碍点插值与车位内点过滤沿用官方 utils。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from parkbench.vehicle import FRONT_OVERHANG, REAR_OVERHANG, WIDTH, normalize_angle

# 目标位姿车体包围盒。官方 utils 引用了这四个量但没有赋值。
VEHICLE_X_MIN = -REAR_OVERHANG
VEHICLE_X_MAX = FRONT_OVERHANG
VEHICLE_Y_MIN = -WIDTH / 2
VEHICLE_Y_MAX = WIDTH / 2

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "rear_in"


@dataclass
class Scene:
    name: str
    ego: np.ndarray  # (x, y, theta) 后轴
    target: np.ndarray
    points: np.ndarray  # (N, 2) 世界坐标障碍点


def _interpolate(nodes: list[dict], interp_dist: float) -> list[list[float]]:
    tmp: list[list[float]] = []
    last = None
    for node in nodes:
        current = [float(node["m_x"]), float(node["m_y"])]
        if last is None:
            tmp.append(current)
        else:
            dist = float(np.hypot(current[0] - last[0], current[1] - last[1]))
            if dist > interp_dist:
                num = int(dist // interp_dist)
                xs = np.linspace(last[0], current[0], num + 2)[1:-1]
                ys = np.linspace(last[1], current[1], num + 2)[1:-1]
                for x, y in zip(xs, ys):
                    if not tmp or np.hypot(x - tmp[-1][0], y - tmp[-1][1]) > 1e-6:
                        tmp.append([float(x), float(y)])
            if not tmp or np.hypot(current[0] - tmp[-1][0], current[1] - tmp[-1][1]) > 1e-6:
                tmp.append(current)
        last = current
    return tmp


def load_scene(path: Path, interp_dist: float = 0.1) -> Scene:
    import json

    data = json.loads(path.read_text())
    frame = data["Frames"]["0"]
    try:
        nfm_origin = frame["m_nfmOrigin"]
    except KeyError:
        nfm_origin = [0, 0]
    try:
        path_origin = frame["PlanningRequest"]["m_origin"]
    except KeyError:
        path_origin = [0, 0]

    def to_world(pose):
        return [
            pose[0] + path_origin[0] - nfm_origin[0],
            pose[1] + path_origin[1] - nfm_origin[1],
            float(normalize_angle(pose[2])),
        ]

    ego = to_world(frame["PlanningRequest"]["m_startPosture"]["m_pose"])
    request = frame["PlanningRequest"]
    try:
        target_pose = request["m_targetArea"]["m_targetPosture"]["m_pose"]
    except KeyError:
        target_pose = request["m_targetAreas"]["m_targetPosture"][0]["m_pose"]
    target = to_world(target_pose)

    th = target[2]
    c, s = np.cos(th), np.sin(th)
    points = []
    for obj in frame.get("NfmAggregatedPolygonObjects", []):
        nodes = obj.get("nfmPolygonObjectNodes", [])
        if len(nodes) < 2:
            continue
        for x, y in _interpolate(nodes, interp_dist):
            x_t = c * (x - target[0]) + s * (y - target[1])
            y_t = -s * (x - target[0]) + c * (y - target[1])
            inside = VEHICLE_X_MIN <= x_t <= VEHICLE_X_MAX and VEHICLE_Y_MIN <= y_t <= VEHICLE_Y_MAX
            if not inside:
                points.append((x, y))
    pts = np.asarray(points, dtype=np.float64) if points else np.zeros((0, 2))
    return Scene(path.stem, np.asarray(ego, dtype=np.float64), np.asarray(target, dtype=np.float64), pts)


def load_scenes(directory: Path | None = None) -> list[Scene]:
    directory = directory or SCENARIO_DIR
    return [load_scene(path) for path in sorted(directory.glob("*.json"))]
