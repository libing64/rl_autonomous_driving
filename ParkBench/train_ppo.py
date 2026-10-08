#!/usr/bin/env python3
"""用课程学习 + 动作分块的 PPO 训练 ParkBench。超参对应论文表 3。"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from parkbench.curriculum import STAGES
from parkbench.env import ActionChunkWrapper, ParkEnv
from parkbench.model import AttentionExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO on ParkBench")
    parser.add_argument("--steps-per-stage", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--resume", action="store_true", help="从 run 目录里的 ppo_parkbench.zip 接着训")
    return parser.parse_args()


def make_env(chunk: int, seed: int):
    def _init():
        env = ParkEnv()
        env = ActionChunkWrapper(env, chunk=chunk)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


class CurriculumCallback(BaseCallback):
    def __init__(self, steps_per_stage: int, run_dir: Path, stage: int = 0, save_every: int = 50_000):
        super().__init__()
        self.steps_per_stage = steps_per_stage
        self.run_dir = run_dir
        self.stage = stage
        self.save_every = save_every
        self.last_save = 0

    def _save(self) -> None:
        self.model.save(self.run_dir / "ppo_parkbench")
        self.last_save = self.num_timesteps
        print(f"[checkpoint] step={self.num_timesteps}  stage={self.stage + 1}", flush=True)

    def _on_step(self) -> bool:
        stage = min(self.num_timesteps // self.steps_per_stage, len(STAGES) - 1)
        if stage != self.stage:
            self.stage = stage
            self.training_env.env_method("set_stage", stage)
            limit = STAGES[stage]["max_steps"]
            print(f"[curriculum] stage={stage + 1}/{len(STAGES)}  max_steps={limit}", flush=True)
            self._save()
        elif self.num_timesteps - self.last_save >= self.save_every:
            self._save()
        return True


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else Path(__file__).resolve().parent / "runs" / "ppo_parkbench"
    run_dir.mkdir(parents=True, exist_ok=True)
    total = args.steps_per_stage * len(STAGES)
    # 表 3 的 “batch size per GPU 1024” 当作每次更新收集的样本量。
    n_steps = max(1024 // args.n_envs, 1)
    env = DummyVecEnv([make_env(args.chunk, args.seed + i) for i in range(args.n_envs)])
    ckpt = run_dir / "ppo_parkbench.zip"
    resume_from = args.resume and ckpt.exists()
    if resume_from:
        model = PPO.load(ckpt, env=env, device=args.device)
        start_stage = min(model.num_timesteps // args.steps_per_stage, len(STAGES) - 1)
        env.env_method("set_stage", start_stage)
        print(f"resume step={model.num_timesteps}  stage={start_stage + 1}", flush=True)
    else:
        start_stage = 0
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=256,
            n_epochs=10,
            gamma=1.0,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                features_extractor_class=AttentionExtractor,
                features_extractor_kwargs=dict(features_dim=128, d_model=64),
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
            ),
            tensorboard_log=str(run_dir / "tb"),
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
    remaining = total - model.num_timesteps
    print(
        f"train ParkBench PPO  remaining={remaining}  total={total}  stages={len(STAGES)}  "
        f"chunk={args.chunk}  n_steps={n_steps}  device={model.device}  run={run_dir}",
        flush=True,
    )
    if remaining > 0:
        model.learn(
            total_timesteps=remaining,
            callback=CurriculumCallback(args.steps_per_stage, run_dir, stage=start_stage),
            progress_bar=True,
            reset_num_timesteps=not resume_from,
        )
    model.save(run_dir / "ppo_parkbench")
    env.close()
    print(f"saved {run_dir / 'ppo_parkbench.zip'}", flush=True)


if __name__ == "__main__":
    main()
