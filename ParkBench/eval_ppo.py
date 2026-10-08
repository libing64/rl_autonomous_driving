#!/usr/bin/env python3
"""在 51 个日志起点上评估策略，统计成功率和路径，并画出轨迹。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from parkbench.curriculum import STAGES
from parkbench.data import load_scenes
from parkbench.env import HORIZON, ActionChunkWrapper, ParkEnv
from parkbench.vehicle import geometric_center


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a ParkBench PPO policy")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="")
    return parser.parse_args()


def rollout(model: PPO, env: ParkEnv, index: int):
    obs, _ = env.reset(seed=0, options={"scenario_index": index, "stage": len(STAGES) - 1})
    done = False
    reward_sum = 0.0
    t0 = time.perf_counter()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_sum += float(reward)
        done = terminated or truncated
    elapsed = time.perf_counter() - t0
    base = env.unwrapped
    return {
        "scene": base.scene.name,
        "success": bool(info["is_success"]),
        "time_s": elapsed,
        "distance_m": float(info["travel"]),
        "pivots": int(info["pivots"]),
        "return": reward_sum,
        "path": list(base.path_xy),
    }


def save_grid(scenes, results, path: Path) -> None:
    cols = 3
    rows = int(np.ceil(len(scenes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0))
    for ax, scene, result in zip(axes.ravel(), scenes, results):
        pts = scene.points
        if len(pts):
            center = geometric_center(*scene.target)
            keep = np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1]) < HORIZON
            ax.scatter(pts[keep, 0], pts[keep, 1], s=1, c="tab:red", linewidths=0)
        xy = np.asarray(result["path"])
        color = "tab:blue" if result["success"] else "0.6"
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.0)
        ax.scatter(scene.ego[0], scene.ego[1], c="magenta", s=18, zorder=3)
        ax.scatter(scene.target[0], scene.target[1], c="cyan", s=18, zorder=3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("ok" if result["success"] else "fail", fontsize=8, color=color)
    for ax in axes.ravel()[len(scenes) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    model_path = Path(args.model) if args.model else root / "runs" / "ppo_parkbench" / "ppo_parkbench.zip"
    out_dir = Path(args.out) if args.out else model_path.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = load_scenes()
    env = ActionChunkWrapper(ParkEnv(scenes), chunk=args.chunk)
    model = PPO.load(model_path, device="cpu")
    results = [rollout(model, env, i) for i in range(len(scenes))]
    env.close()

    success = np.array([r["success"] for r in results])
    ok = [r for r in results if r["success"]]
    summary = {
        "episodes": len(results),
        "success_rate": float(success.mean()),
        "successes": int(success.sum()),
        "time_s_success_mean": float(np.mean([r["time_s"] for r in ok])) if ok else None,
        "distance_m_success_mean": float(np.mean([r["distance_m"] for r in ok])) if ok else None,
        "pivots_success_mean": float(np.mean([r["pivots"] for r in ok])) if ok else None,
    }
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "episodes": [
        {k: v for k, v in r.items() if k != "path"} for r in results
    ]}, indent=2))
    save_grid(scenes, results, out_dir / "paths.png")
    print(
        f"success={summary['successes']}/{summary['episodes']} ({summary['success_rate']:.1%})  "
        f"time={summary['time_s_success_mean']}  dist={summary['distance_m_success_mean']}  "
        f"pivots={summary['pivots_success_mean']}"
    )
    print(f"wrote {out_dir / 'summary.json'} and {out_dir / 'paths.png'}")


if __name__ == "__main__":
    main()
