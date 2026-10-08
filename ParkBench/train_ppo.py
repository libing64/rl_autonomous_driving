#!/usr/bin/env python3
"""用课程学习 + 动作分块的 PPO 训练 ParkBench。超参对应论文表 3。"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from parkbench.curriculum import STAGES
from parkbench.data import load_scenes
from parkbench.demos import chain_for, expert_chunk_from
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
    parser.add_argument(
        "--restart-curriculum",
        action="store_true",
        help="加载已有权重，但从第 1 阶段重新上课，并允许提前结束",
    )
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-steps-per-stage", type=int, default=60_000)
    parser.add_argument("--final-stage-steps", type=int, default=200_000)
    parser.add_argument("--success-to-advance", type=float, default=0.85)
    parser.add_argument("--stop-success", type=float, default=0.90, help="51 个日志起点达到该成功率就停")
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--behavior-clone", action="store_true", help="先模仿可泊入的轨迹，再 PPO")
    parser.add_argument(
        "--dagger",
        action="store_true",
        help="用当前策略走出的状态，再由沿轨迹的专家打标签，然后 PPO",
    )
    return parser.parse_args()


def make_env(chunk: int, seed: int):
    def _init():
        env = ParkEnv()
        env = ActionChunkWrapper(env, chunk=chunk)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def benchmark_logged(model: PPO, chunk: int) -> float:
    """在全部日志起点上跑一遍，返回成功率。"""
    scenes = load_scenes()
    env = ActionChunkWrapper(ParkEnv(scenes), chunk=chunk)
    successes = 0
    for index in range(len(scenes)):
        obs, _ = env.reset(seed=0, options={"scenario_index": index, "stage": len(STAGES) - 1})
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        successes += int(bool(info.get("is_success", False)))
    env.close()
    return successes / max(len(scenes), 1)


class CurriculumCallback(BaseCallback):
    """成功率够了就进入下一阶段。最后一阶段用日志起点，达标即停。"""

    def __init__(
        self,
        run_dir: Path,
        chunk: int,
        max_steps_per_stage: int,
        final_stage_steps: int,
        success_to_advance: float,
        stop_success: float,
        stage: int = 0,
    ):
        super().__init__()
        self.run_dir = run_dir
        self.chunk = chunk
        self.max_steps_per_stage = max_steps_per_stage
        self.final_stage_steps = final_stage_steps
        self.success_to_advance = success_to_advance
        self.stop_success = stop_success
        self.stage = stage
        self.stage_start = 0
        self.recent: deque[float] = deque(maxlen=200)
        self.last_benchmark = -10**9

    def _save(self) -> None:
        self.model.save(self.run_dir / "ppo_parkbench")
        self.model.save(self.run_dir / f"ppo_parkbench_stage{self.stage + 1}")
        print(f"[checkpoint] step={self.num_timesteps}  stage={self.stage + 1}", flush=True)

    def _benchmark(self) -> bool:
        rate = benchmark_logged(self.model, self.chunk)
        self.last_benchmark = self.num_timesteps
        print(f"[benchmark] step={self.num_timesteps}  logged_success={rate:.1%}", flush=True)
        self._save()
        return rate + 1e-9 >= self.stop_success

    def _on_training_start(self) -> None:
        self.stage_start = self.num_timesteps
        self.training_env.env_method("set_stage", self.stage)
        print(
            f"[curriculum] stage={self.stage + 1}/{len(STAGES)}  "
            f"max_steps={STAGES[self.stage]['max_steps']}",
            flush=True,
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done:
                self.recent.append(float(bool(info.get("is_success", False))))

        spent = self.num_timesteps - self.stage_start
        last = self.stage >= len(STAGES) - 1
        limit = self.final_stage_steps if last else self.max_steps_per_stage
        rate = float(np.mean(self.recent)) if self.recent else 0.0
        ready = len(self.recent) >= 80 and rate >= self.success_to_advance
        # 只有这一阶段的成功率稳住才进入下一阶段，避免没学会就被推进去。
        advance = (not last) and spent >= 8_000 and ready
        if last and spent >= 20_000 and self.num_timesteps - self.last_benchmark >= 20_000:
            if self._benchmark():
                return False
        if advance:
            self.stage += 1
            self.stage_start = self.num_timesteps
            self.recent.clear()
            self.training_env.env_method("set_stage", self.stage)
            print(
                f"[curriculum] stage={self.stage + 1}/{len(STAGES)}  "
                f"max_steps={STAGES[self.stage]['max_steps']}",
                flush=True,
            )
            self._save()
            if self.stage >= len(STAGES) - 1 and self._benchmark():
                return False
        elif (not last) and spent >= limit:
            pass
        if last and spent >= self.final_stage_steps:
            self._benchmark()
            return False
        return True


def _rollout_chain(model: PPO, chunk: int, cursor: int) -> float:
    """在示范轨迹的固定步数上，用当前策略倒车，统计泊入比例。"""
    scenes = load_scenes()
    env = ActionChunkWrapper(ParkEnv(scenes), chunk=chunk)
    successes = 0
    for index, scene in enumerate(scenes):
        states = chain_for(scene, side=1)
        if len(chain_for(scene, side=-1)) > len(states):
            states = chain_for(scene, side=-1)
        at = min(cursor, len(states) - 1)
        obs, _ = env.reset(seed=0, options={"scenario_index": index, "stage": 0})
        base = env.unwrapped
        base.x, base.y, base.theta, base.delta = states[at]
        base.steps = 0
        base._stage_limit = 1000
        base._phi_prev = base._phi()
        obs = base._observe()
        done = False
        info: dict = {}
        guard = 0
        while not done and guard < 400:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            guard += 1
        successes += int(bool(info.get("is_success", False)))
    env.close()
    return successes / max(len(scenes), 1)


def _imitate(model: PPO, observations: list[dict], targets: np.ndarray, epochs: int, batch_size: int, tag: str) -> None:
    policy = model.policy
    policy.set_training_mode(True)
    order = np.arange(len(observations))
    for epoch in range(epochs):
        rng = np.random.default_rng(epoch)
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            obs = {
                "state": np.stack([observations[i]["state"] for i in batch]),
                "obstacles": np.stack([observations[i]["obstacles"] for i in batch]),
                "mask": np.stack([observations[i]["mask"] for i in batch]),
            }
            obs_t, _ = policy.obs_to_tensor(obs)
            dist = policy.get_distribution(obs_t)
            act_t = torch.as_tensor(targets[batch], device=model.device)
            loss = -dist.log_prob(act_t).mean()
            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"{tag} epoch {epoch + 1}/{epochs}  loss={float(np.mean(losses)):.3f}", flush=True)
    policy.set_training_mode(False)


def _report_chain(model: PPO, chunk: int) -> None:
    for cursor in (80, 140, 180):
        rate = _rollout_chain(model, chunk, cursor)
        print(f"[benchmark] demo index={cursor}  success={rate:.1%}", flush=True)


def behavior_clone(model: PPO, chunk: int, epochs: int = 24, batch_size: int = 256) -> None:
    """用倒回车位的动作监督当前策略。"""
    scenes = load_scenes()
    env = ParkEnv(scenes)
    observations: list[dict] = []
    actions: list[list[int]] = []
    for index, scene in enumerate(scenes):
        for side in (1, -1):
            states = chain_for(scene, side)
            last = min(len(states) - 1, 200)
            cursors = list(range(12, last, 3))
            cursors += list(range(90, last, 2))
            for cursor in cursors:
                pose = states[cursor]
                env.reset(seed=0, options={"scenario_index": index, "stage": 0})
                env.x, env.y, env.theta, env.delta = pose
                observations.append(env._observe())
                actions.append(expert_chunk_from(scene.points, states, cursor, pose, chunk))
    env.close()
    print(f"behavior clone samples={len(observations)}", flush=True)
    _imitate(model, observations, np.asarray(actions, dtype=np.int64), epochs, batch_size, "bc")
    _report_chain(model, chunk)


def dagger(model: PPO, chunk: int, rounds: int = 3, epochs: int = 6, batch_size: int = 256) -> None:
    """策略自己开，专家按当前位姿标注下一段动作。只有靠近轨迹上一点时，标注才往车位推进。"""
    scenes = load_scenes()
    env = ActionChunkWrapper(ParkEnv(scenes), chunk=chunk)
    observations: list[dict] = []
    actions: list[list[int]] = []
    betas = (0.7, 0.35, 0.1)
    for round_i in range(rounds):
        beta = betas[min(round_i, len(betas) - 1)]
        rng = np.random.default_rng(10_000 + round_i)
        for index, scene in enumerate(scenes):
            for side in (1, -1):
                states = chain_for(scene, side)
                if len(states) < 16:
                    continue
                last = min(len(states) - 1, 180)
                for cursor0 in (60, 100, 140, last):
                    cursor = min(int(cursor0), len(states) - 1)
                    obs, _ = env.reset(seed=index, options={"scenario_index": index, "stage": 0})
                    base = env.unwrapped
                    base.x, base.y, base.theta, base.delta = states[cursor]
                    base.steps = 0
                    base._stage_limit = 1000
                    base._phi_prev = base._phi()
                    obs = base._observe()
                    done = False
                    guard = 0
                    while not done and guard < cursor // chunk + 2 and cursor > 0:
                        pose = (float(base.x), float(base.y), float(base.theta), float(base.delta))
                        label = expert_chunk_from(scene.points, states, cursor, pose, chunk)
                        observations.append(obs)
                        actions.append(label)
                        if rng.random() < beta:
                            action = np.asarray(label, dtype=np.int64)
                        else:
                            action, _ = model.predict(obs, deterministic=False)
                        obs, _reward, terminated, truncated, _info = env.step(action)
                        done = bool(terminated or truncated)
                        target = states[cursor - 1]
                        dist_now = float(np.hypot(base.x - target[0], base.y - target[1]))
                        dist_then = float(np.hypot(pose[0] - target[0], pose[1] - target[1]))
                        if not done and dist_now + 1e-3 < dist_then:
                            cursor = max(0, cursor - chunk)
                        guard += 1
        print(f"dagger round {round_i + 1}/{rounds}  samples={len(observations)}  beta={beta:.2f}", flush=True)
        _imitate(model, observations, np.asarray(actions, dtype=np.int64), epochs, batch_size, "dagger")
    env.close()
    _report_chain(model, chunk)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else Path(__file__).resolve().parent / "runs" / "ppo_parkbench"
    run_dir.mkdir(parents=True, exist_ok=True)
    # 表 3 的 “batch size per GPU 1024” 当作每次更新收集的样本量。
    n_steps = max(1024 // args.n_envs, 1)
    env = DummyVecEnv([make_env(args.chunk, args.seed + i) for i in range(args.n_envs)])
    ckpt = run_dir / "ppo_parkbench.zip"
    near = run_dir / "ppo_parkbench_neargoal.zip"
    restart = args.restart_curriculum and (near.exists() or ckpt.exists())
    resume_from = args.resume and ckpt.exists() and not restart
    # 重新上课时用最新权重。neargoal 只在最新文件不存在时兜底。
    if restart and not ckpt.exists() and near.exists():
        ckpt = near
    if restart or resume_from:
        model = PPO.load(ckpt, env=env, device=args.device)
        model.ent_coef = args.ent_coef
        print(f"loaded {ckpt}  timesteps={model.num_timesteps}  ent_coef={model.ent_coef}", flush=True)
    else:
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
        model.ent_coef = args.ent_coef
    if restart:
        total = args.max_steps_per_stage * (len(STAGES) - 1) + args.final_stage_steps
        reset_steps = True
        start_stage = args.start_stage
    elif resume_from:
        total = args.steps_per_stage * len(STAGES)
        total = max(total - model.num_timesteps, 0)
        reset_steps = False
        start_stage = min(model.num_timesteps // max(args.steps_per_stage, 1), len(STAGES) - 1)
    else:
        total = args.steps_per_stage * len(STAGES)
        reset_steps = True
        start_stage = args.start_stage
    if args.behavior_clone:
        behavior_clone(model, args.chunk)
    if args.dagger:
        dagger(model, args.chunk)
    if args.behavior_clone or args.dagger:
        rate = benchmark_logged(model, args.chunk)
        print(f"[benchmark] after imitation  logged_success={rate:.1%}", flush=True)
        model.save(run_dir / "ppo_parkbench_bc")
        model.save(run_dir / "ppo_parkbench")
    env.env_method("set_stage", start_stage)
    print(
        f"train ParkBench PPO  steps={total}  start_stage={start_stage + 1}  "
        f"chunk={args.chunk}  ent_coef={model.ent_coef}  device={model.device}  run={run_dir}",
        flush=True,
    )
    if total > 0:
        model.learn(
            total_timesteps=total,
            callback=CurriculumCallback(
                run_dir,
                args.chunk,
                args.max_steps_per_stage,
                args.final_stage_steps,
                args.success_to_advance,
                args.stop_success,
                stage=start_stage,
            ),
            progress_bar=True,
            reset_num_timesteps=reset_steps,
        )
    model.save(run_dir / "ppo_parkbench")
    env.close()
    print(f"saved {run_dir / 'ppo_parkbench.zip'}", flush=True)


if __name__ == "__main__":
    main()
