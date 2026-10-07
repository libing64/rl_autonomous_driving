#!/usr/bin/env python3
"""加载训练好的 PPO，在 parking-v0 上统计成功率，并可录一段泊车过程。"""

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

from parking_env import CRASH_PENALTY, make_parking_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PPO parking policy")
    parser.add_argument("--run", type=str, default="", help="训练目录，默认 parking/runs/latest")
    parser.add_argument("--model", type=str, default="best", choices=["best", "final"])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--record", action="store_true", help="额外录一个回合的 gif")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--crash-penalty", type=float, default=CRASH_PENALTY)
    return parser.parse_args()


def resolve_run(path: str) -> Path:
    if path:
        return Path(path).resolve()
    return (Path(__file__).resolve().parent / "runs" / "latest").resolve()


def load_model(run_dir: Path, which: str, seed: int, render_mode: str | None, crash_penalty: float):
    if which == "best":
        model_path = run_dir / "best" / "best_model.zip"
        stats_path = run_dir / "best" / "vecnormalize.pkl"
    else:
        model_path = run_dir / "ppo_parking.zip"
        stats_path = run_dir / "vecnormalize.pkl"
    if not model_path.exists() or not stats_path.exists():
        raise FileNotFoundError(f"missing {model_path} or {stats_path}")

    vec_env = DummyVecEnv(
        [make_parking_env(seed, render_mode=render_mode, crash_penalty=crash_penalty)]
    )
    vec_env = VecNormalize.load(stats_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(model_path, env=vec_env, device="cpu")
    return model, vec_env


def rollout(model: PPO, vec_env: VecNormalize, n_episodes: int, collect_frames: bool = False):
    obs = vec_env.reset()
    ep_reward = 0.0
    ep_len = 0
    successes: list[float] = []
    rewards: list[float] = []
    lengths: list[float] = []
    frames: list[np.ndarray] = []

    if collect_frames:
        frame = vec_env.env_method("render")[0]
        if frame is not None:
            frames.append(frame)

    while len(successes) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        ep_reward += float(reward[0])
        ep_len += 1
        if collect_frames and len(successes) == 0:
            frame = vec_env.env_method("render")[0]
            if frame is not None:
                frames.append(frame)
        if dones[0]:
            successes.append(float(bool(infos[0].get("is_success", False))))
            rewards.append(ep_reward)
            lengths.append(float(ep_len))
            ep_reward = 0.0
            ep_len = 0
            if collect_frames:
                break
    return successes, rewards, lengths, frames


def main() -> None:
    args = parse_args()
    run_dir = resolve_run(args.run)
    model, vec_env = load_model(
        run_dir, args.model, args.seed, render_mode=None, crash_penalty=args.crash_penalty
    )
    successes, rewards, lengths, _ = rollout(model, vec_env, args.episodes)
    vec_env.close()

    print(f"run={run_dir}  model={args.model}  episodes={args.episodes}")
    print(f"success_rate={np.mean(successes):.2%}")
    print(f"mean_reward={np.mean(rewards):.2f}  std={np.std(rewards):.2f}")
    print(f"mean_length={np.mean(lengths):.1f}")

    if not args.record:
        return

    model, vec_env = load_model(
        run_dir, args.model, args.seed, render_mode="rgb_array", crash_penalty=args.crash_penalty
    )
    successes, rewards, lengths, frames = rollout(model, vec_env, n_episodes=1, collect_frames=True)
    vec_env.close()
    if not frames:
        raise RuntimeError("render returned no frames")

    out = Path(args.out) if args.out else run_dir / "parking.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=8, loop=0)
    tag = "success" if successes and successes[0] else "fail"
    print(f"recorded {tag} episode  reward={rewards[0]:.2f}  len={lengths[0]:.0f}  gif={out}")


if __name__ == "__main__":
    main()
