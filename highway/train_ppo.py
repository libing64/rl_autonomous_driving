#!/usr/bin/env python3
"""用 PPO 训练 highway-v0 的换道决策。

每步从左换道、保持、右换道、加速、减速里选一个。
奖励鼓励开得快、靠右行驶、别撞车；撞车回合立刻结束。
训练结束后用最终策略录一段 MP4。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from lane_env import make_highway_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO lane-change training for highway-v0")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=10_000, help="每隔多少环境步评估一次")
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--dummy", action="store_true", help="单进程串行采样，默认用子进程并行")
    # MlpPolicy 的 PPO 在 CPU 上更快，SB3 也不建议这种网络走 GPU。
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--video-episodes", type=int, default=5, help="最终推理录进 MP4 的回合数")
    return parser.parse_args()


def make_vec_env(n_envs: int, seed: int, subproc: bool, render_mode: str | None = None):
    env_fns = [make_highway_env(seed + i, render_mode=render_mode) for i in range(n_envs)]
    vec_cls = SubprocVecEnv if subproc else DummyVecEnv
    return vec_cls(env_fns)


def evaluate(model: PPO, vec_env: VecNormalize, n_episodes: int):
    """返回存活率（走满时限且没撞车）、回报、长度。"""
    obs = vec_env.reset()
    ep_reward = np.zeros(vec_env.num_envs, dtype=np.float64)
    ep_len = np.zeros(vec_env.num_envs, dtype=np.int32)
    survived: list[float] = []
    rewards: list[float] = []
    lengths: list[float] = []

    while len(survived) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        ep_reward += reward
        ep_len += 1
        for i, done in enumerate(dones):
            if not done:
                continue
            crashed = bool(infos[i].get("crashed", False))
            survived.append(float(not crashed))
            rewards.append(float(ep_reward[i]))
            lengths.append(float(ep_len[i]))
            ep_reward[i] = 0.0
            ep_len[i] = 0
            if len(survived) >= n_episodes:
                break
    return survived, rewards, lengths


class HighwayEvalCallback(BaseCallback):
    """按存活率（其次平均回报）保存最好的模型和归一化统计。"""

    def __init__(
        self,
        eval_env: VecNormalize,
        train_env: VecNormalize,
        save_dir: Path,
        eval_freq: int,
        n_eval_episodes: int,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.train_env = train_env
        self.save_dir = save_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_eval_step = 0
        self.best_survived = -1.0
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True
        self.last_eval_step = self.num_timesteps

        survived, rewards, lengths = evaluate(self.model, self.eval_env, self.n_eval_episodes)
        survive_rate = float(np.mean(survived))
        mean_reward = float(np.mean(rewards))
        mean_len = float(np.mean(lengths))

        self.logger.record("eval/survive_rate", survive_rate)
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", mean_len)
        print(
            f"[eval] step={self.num_timesteps}  "
            f"survive={survive_rate:.2%}  reward={mean_reward:.2f}  len={mean_len:.1f}"
        )

        better = survive_rate > self.best_survived + 1e-6 or (
            abs(survive_rate - self.best_survived) <= 1e-6 and mean_reward > self.best_reward
        )
        if better:
            self.best_survived = survive_rate
            self.best_reward = mean_reward
            best_dir = self.save_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(best_dir / "best_model")
            self.train_env.save(best_dir / "vecnormalize.pkl")
            print(f"[eval] new best  survive={survive_rate:.2%}  reward={mean_reward:.2f}")
        return True


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else Path(__file__).resolve().parent / "runs" / "ppo_highway"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = run_dir.parent / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.resolve())

    subproc = args.n_envs > 1 and not args.dummy
    train_env = make_vec_env(args.n_envs, args.seed, subproc)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    eval_env = make_vec_env(1, args.seed + 10_000, subproc=False)
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )
    eval_env.obs_rms = train_env.obs_rms

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=str(run_dir / "tb"),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    callback = HighwayEvalCallback(
        eval_env=eval_env,
        train_env=train_env,
        save_dir=run_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
    )
    print(
        f"train highway-v0 lane changes with PPO  timesteps={args.timesteps}  "
        f"n_envs={args.n_envs}  device={model.device}  run={run_dir}"
    )
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)

    model.save(run_dir / "ppo_highway")
    train_env.save(run_dir / "vecnormalize.pkl")
    train_env.close()
    eval_env.close()
    print(f"saved final model to {run_dir}", flush=True)

    video = run_dir / "lane_change.mp4"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("eval_ppo.py")),
            "--run",
            str(run_dir),
            "--model",
            "final",
            "--episodes",
            str(args.video_episodes),
            "--video",
            str(video),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
