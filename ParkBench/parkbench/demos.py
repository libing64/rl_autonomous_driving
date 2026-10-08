"""从车位开出一条前进加倒车的轨迹，再逐步倒回去。

沿这条轨迹采样的状态都有一条能泊入的动作序列。
"""

from __future__ import annotations

import numpy as np

from parkbench.vehicle import ACTIONS, bicycle_step, collides, normalize_angle

_CHAINS: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}


def chain_for(scene, side: int) -> list[tuple[float, float, float, float]]:
    key = (scene.name, 1 if side > 0 else -1)
    cached = _CHAINS.get(key)
    if cached is not None:
        return cached
    x, y, theta = (float(scene.target[0]), float(scene.target[1]), float(scene.target[2]))
    delta = 0.0
    states = [(x, y, theta, delta)]

    def go(action: int, limit: int) -> int:
        nonlocal x, y, theta, delta
        moved = 0
        for _ in range(limit):
            ddelta, v = ACTIONS[action]
            nx, ny, nth, nd, _ds = bicycle_step(x, y, theta, delta, ddelta, v)
            if collides(scene.points, nx, ny, nth):
                break
            x, y, theta, delta = nx, ny, nth, float(nd)
            states.append((x, y, theta, delta))
            moved += 1
        return moved

    go(1, 80)
    forward = 2 if side > 0 else 0
    reverse = 3 if side > 0 else 5
    for _ in range(8):
        if go(forward, 20) > 0:
            continue
        if go(reverse, 20) == 0:
            break

    def lateral(px: float, py: float) -> float:
        dx, dy = px - scene.target[0], py - scene.target[1]
        c, s = np.cos(scene.target[2]), np.sin(scene.target[2])
        return abs(float(-s * dx + c * dy))

    for _ in range(160):
        current = lateral(x, y)
        best = None
        for action in range(6):
            ddelta, v = ACTIONS[action]
            nx, ny, nth, nd, _ds = bicycle_step(x, y, theta, delta, ddelta, v)
            if collides(scene.points, nx, ny, nth):
                continue
            gained = lateral(nx, ny)
            if best is None or gained > best[0]:
                best = (gained, nx, ny, nth, float(nd))
        if best is None or best[0] <= current + 1e-4:
            break
        _, x, y, theta, delta = best
        states.append((x, y, theta, delta))
    _CHAINS[key] = states
    return states


def expert_chunk_from(points, states, cursor: int, pose, n: int = 4) -> list[int]:
    """从给定位姿朝轨迹上更靠近车位的点走 n 步。"""
    x, y, theta, delta = pose
    cursor = int(cursor)
    actions: list[int] = []
    for _ in range(n):
        if cursor <= 0:
            actions.append(4)
            continue
        target = states[cursor - 1]
        best_action = 4
        best_score = 1e9
        best_state = (x, y, theta, delta)
        for action, (ddelta, v) in enumerate(ACTIONS):
            nx, ny, nth, nd, _ds = bicycle_step(x, y, theta, delta, ddelta, v)
            if collides(points, nx, ny, nth):
                continue
            score = np.hypot(nx - target[0], ny - target[1]) + 0.35 * abs(
                float(normalize_angle(nth - target[2]))
            )
            if score < best_score:
                best_score = score
                best_action = action
                best_state = (nx, ny, nth, float(nd))
        actions.append(best_action)
        x, y, theta, delta = best_state
        cursor -= 1
    return actions


def expert_chunk(points, states, index: int, n: int = 4) -> list[int]:
    """从轨迹上的一个状态朝车位走 n 步，返回宏动作。"""
    index = int(np.clip(index, 0, len(states) - 1))
    return expert_chunk_from(points, states, index, states[index], n)
