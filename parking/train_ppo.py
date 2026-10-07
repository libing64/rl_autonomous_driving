#!/usr/bin/env python3
"""用 PPO 训练 highway-env parking-v0。

动作是连续的纵向加速度和前轮转角，车辆不能横移。
回合在泊入成功、撞到障碍，或走满 100 个控制步时结束。
奖励是位姿误差的加权 p-范数（越接近 0 越好）。
环境自带碰撞 -5；碰撞终止时再额外 -100，避免策略靠早点撞墙刷分。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from parking_env import CRASH_PENALTY, make_parking_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO training for parking-v0")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=10_000, help="每隔多少环境步评估一次")
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--subproc", action="store_true", help="用 SubprocVecEnv 并行采样")
    # MlpPolicy 的 PPO 在 CPU 上更快，SB3 也不建议这种网络走 GPU。
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--crash-penalty",
        type=float,
        default=CRASH_PENALTY,
        help="碰撞终止时额外奖励，用来抵消「早点撞墙更划算」",
    )
    parser.add_argument("--run-dir", type=str, default="")
    return parser.parse_args()


def make_vec_env(
    n_envs: int,
    seed: int,
    subproc: bool,
    render_mode: str | None = None,
    crash_penalty: float = CRASH_PENALTY,
):
    env_fns = [
        make_parking_env(seed + i, render_mode=render_mode, crash_penalty=crash_penalty)
        for i in range(n_envs)
    ]
    vec_cls = SubprocVecEnv if subproc else DummyVecEnv
    return vec_cls(env_fns)


def evaluate(model: PPO, vec_env: VecNormalize, n_episodes: int):
    """在已归一化的评估环境上跑若干回合，返回成功率、回报、长度。"""
    obs = vec_env.reset()
    ep_reward = np.zeros(vec_env.num_envs, dtype=np.float64)
    ep_len = np.zeros(vec_env.num_envs, dtype=np.int32)
    successes: list[float] = []
    rewards: list[float] = []
    lengths: list[float] = []

    while len(successes) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        ep_reward += reward
        ep_len += 1
        for i, done in enumerate(dones):
            if not done:
                continue
            successes.append(float(bool(infos[i].get("is_success", False))))
            rewards.append(float(ep_reward[i]))
            lengths.append(float(ep_len[i]))
            ep_reward[i] = 0.0
            ep_len[i] = 0
            if len(successes) >= n_episodes:
                break
    return successes, rewards, lengths


class ParkingEvalCallback(BaseCallback):
    """按环境步数评估，用成功率（其次平均回报）保存最好的模型和归一化统计。"""

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
        self.best_success = -1.0
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True
        self.last_eval_step = self.num_timesteps

        successes, rewards, lengths = evaluate(self.model, self.eval_env, self.n_eval_episodes)
        success_rate = float(np.mean(successes))
        mean_reward = float(np.mean(rewards))
        mean_len = float(np.mean(lengths))

        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", mean_len)
        print(
            f"[eval] step={self.num_timesteps}  "
            f"success={success_rate:.2%}  reward={mean_reward:.2f}  len={mean_len:.1f}"
        )

        better = success_rate > self.best_success + 1e-6 or (
            abs(success_rate - self.best_success) <= 1e-6 and mean_reward > self.best_reward
        )
        if better:
            self.best_success = success_rate
            self.best_reward = mean_reward
            best_dir = self.save_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(best_dir / "best_model")
            self.train_env.save(best_dir / "vecnormalize.pkl")
            print(f"[eval] new best  success={success_rate:.2%}  reward={mean_reward:.2f}")
        return True


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else Path(__file__).resolve().parent / "runs" / "ppo_parking"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = run_dir.parent / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.resolve())

    train_env = make_vec_env(args.n_envs, args.seed, args.subproc, crash_penalty=args.crash_penalty)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    eval_env = make_vec_env(1, args.seed + 10_000, subproc=False, crash_penalty=args.crash_penalty)
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )
    # 评估使用训练过程中更新的观测统计，且不把评估样本写回去。
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

    callback = ParkingEvalCallback(
        eval_env=eval_env,
        train_env=train_env,
        save_dir=run_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
    )
    print(
        f"train parking-v0 with PPO  timesteps={args.timesteps}  "
        f"n_envs={args.n_envs}  device={model.device}  run={run_dir}"
    )
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)

    model.save(run_dir / "ppo_parking")
    train_env.save(run_dir / "vecnormalize.pkl")
    train_env.close()
    eval_env.close()
    print(f"saved final model to {run_dir}")


if __name__ == "__main__":
    main()
