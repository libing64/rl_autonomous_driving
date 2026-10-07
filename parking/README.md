# parking-v0 + PPO

用 highway-env 的 `parking-v0` 训练一个连续控制的 PPO 策略。环境是 conda 的 `deep-rl-class`（已安装 gymnasium、highway-env、stable-baselines3）。

## 任务

停车场里把车开进目标车位。速度低、空间窄，动力学是自行车模型：动作只有纵向加速度和前轮转角，车不能横移，只能靠前进、倒车和转向凑出横向位移。

| 项目 | 内容 |
| --- | --- |
| 动作 | `Box(2,)`，`[-1, 1]` 的加速度和转角（最大转角 45°） |
| 观测 | 自车与目标各 6 维：`x, y, vx, vy, cos_h, sin_h`（位置除以 100，速度除以 5） |
| 策略输入 | 自车状态、目标位姿、二者之差，共 18 维 |
| 奖励 | 位姿误差的加权 p-范数取负，权重 `[1, 0.3, 0, 0, 0.02, 0.02]`；环境自带碰撞 -5，训练时碰撞终止再额外 -100 |
| 成功 | 位姿奖励 > -0.12，也就是位置和航向都对齐 |
| 结束 | 成功、碰撞，或 100 个控制步截断 |

`parking-v0` 的 `duration` 单位是秒，策略 5 Hz，默认 100 秒等于 500 个控制步。开满这么久的位姿惩罚大约是 -250，远大于碰撞的 -5，所以不改奖励的话 PPO 会学「尽快撞墙」。这里把回合收成 20 秒（100 步），并在碰撞那一步再扣 100，让撞车差于耗满时限，成功泊入最好。成功率的判定没有改。

奖励每步都有，但只有贴进车位才算成功，远处的梯度很平，所以成功率上升会比较慢。

## 训练

```bash
conda activate deep-rl-class
cd parking
python train_ppo.py --timesteps 200000
```

默认 8 个环境并行、每 1 万步评估 20 个回合。模型和观测归一化统计写到 `runs/ppo_parking/`，`runs/latest` 指向这次运行。TensorBoard：

```bash
tensorboard --logdir runs/ppo_parking/tb
```

## 评估

```bash
python eval_ppo.py --episodes 30
python eval_ppo.py --record
```

`--model best` 用评估成功率最高的检查点，`--model final` 用训练结束时的权重。`--record` 把一个回合存成 `runs/ppo_parking/parking.gif`。
