#!/usr/bin/env python3
"""加载 PPO 换道策略，统计存活和动作，并把推理过程写成 MP4。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# 无显示器时 pygame 走 dummy 驱动，必须在 import highway_env 之前设置。
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from lane_env import ACTIONS, make_highway_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PPO highway lane-change policy")
    parser.add_argument("--run", type=str, default="", help="训练目录，默认 highway/runs/latest")
    parser.add_argument("--model", type=str, default="final", choices=["best", "final"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--video", type=str, default="", help="把这些回合写成该 MP4；空字符串则不录像")
    parser.add_argument("--fps", type=int, default=8)
    return parser.parse_args()


def resolve_run(path: str) -> Path:
    if path:
        return Path(path).resolve()
    return (Path(__file__).resolve().parent / "runs" / "latest").resolve()


def load_model(run_dir: Path, which: str, seed: int, render_mode: str | None):
    if which == "best":
        model_path = run_dir / "best" / "best_model.zip"
        stats_path = run_dir / "best" / "vecnormalize.pkl"
    else:
        model_path = run_dir / "ppo_highway.zip"
        stats_path = run_dir / "vecnormalize.pkl"
    if not model_path.exists() or not stats_path.exists():
        raise FileNotFoundError(f"missing {model_path} or {stats_path}")

    vec_env = DummyVecEnv([make_highway_env(seed, render_mode=render_mode)])
    vec_env = VecNormalize.load(stats_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(model_path, env=vec_env, device="cpu")
    return model, vec_env


def rollout(model: PPO, vec_env: VecNormalize, n_episodes: int, collect_frames: bool):
    obs = vec_env.reset()
    ep_reward = 0.0
    ep_len = 0
    survived: list[float] = []
    rewards: list[float] = []
    lengths: list[float] = []
    action_counts = np.zeros(len(ACTIONS), dtype=np.int64)
    frames: list[np.ndarray] = []

    if collect_frames:
        frame = vec_env.env_method("render")[0]
        if frame is not None:
            frames.append(np.asarray(frame))

    while len(survived) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        action_index = int(action[0])
        action_counts[action_index] += 1
        obs, reward, dones, infos = vec_env.step(action)
        ep_reward += float(reward[0])
        ep_len += 1
        if collect_frames:
            frame = vec_env.env_method("render")[0]
            if frame is not None:
                frames.append(np.asarray(frame))
        if dones[0]:
            crashed = bool(infos[0].get("crashed", False))
            survived.append(float(not crashed))
            rewards.append(ep_reward)
            lengths.append(float(ep_len))
            ep_reward = 0.0
            ep_len = 0
    return survived, rewards, lengths, action_counts, frames


def main() -> None:
    args = parse_args()
    run_dir = resolve_run(args.run)
    record = bool(args.video)
    model, vec_env = load_model(
        run_dir, args.model, args.seed, render_mode="rgb_array" if record else None
    )
    survived, rewards, lengths, action_counts, frames = rollout(
        model, vec_env, args.episodes, collect_frames=record
    )
    vec_env.close()

    print(f"run={run_dir}  model={args.model}  episodes={args.episodes}")
    print(f"survive_rate={np.mean(survived):.2%}  crash_rate={1 - np.mean(survived):.2%}")
    print(f"mean_reward={np.mean(rewards):.2f}  std={np.std(rewards):.2f}")
    print(f"mean_length={np.mean(lengths):.1f}")
    total = max(int(action_counts.sum()), 1)
    counts = "  ".join(f"{name}={action_counts[i] / total:.1%}" for i, name in enumerate(ACTIONS))
    print(f"actions  {counts}")

    if not record:
        return
    if not frames:
        raise RuntimeError("render returned no frames")

    out = Path(args.video)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps, codec="libx264", quality=8, macro_block_size=1)
    print(f"wrote {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()
