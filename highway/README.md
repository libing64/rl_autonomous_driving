# highway-v0 换道决策 + PPO

用 highway-env 的 `highway-v0` 训练换道决策。环境是 conda 的 `deep-rl-class`。

四车道高速公路上，自车每步选一个高层动作，底层控制器负责跟踪目标车道和目标速度。

| 项目 | 内容 |
| --- | --- |
| 动作 | `LANE_LEFT`、`IDLE`、`LANE_RIGHT`、`FASTER`、`SLOWER` |
| 观测 | 自车加最近 4 辆车，每辆 `presence, x, y, vx, vy`（相对自车、已归一化），拉成 25 维 |
| 奖励 | 高速约 0.4、靠右车道约 0.1，归一化到 `[0, 1]`；撞车该步为 0 并结束回合 |
| 场景 | 4 车道、约 20 辆背景车。仿真 5 Hz（环境默认 15 Hz），决策仍是每秒一次 |
| 回合 | 最长 40 秒（40 步），撞车提前结束 |
| 存活 | 走满 40 步且没有撞车 |

速度奖励在 20–30 m/s 之间线性升高。前车慢时，换道超车才能保住高速奖励。

## 训练

```bash
conda activate deep-rl-class
cd highway
python train_ppo.py --timesteps 100000
```

默认 8 个子进程并行采样，每 1 万步评估 20 个回合。权重写到 `runs/ppo_highway/`。训练结束会用最终策略推理若干回合，写成 `runs/ppo_highway/lane_change.mp4`。

## 单独评估

```bash
python eval_ppo.py --episodes 20
python eval_ppo.py --model final --video runs/ppo_highway/lane_change.mp4
```
