# ParkBench

复现 [Adapting Reinforcement Learning for Path Planning in Constrained Parking Scenarios](https://arxiv.org/abs/2601.22545)（arXiv:2601.22545）。

官方仓库只发布了 51 个倒车入库场景和 JSON 解析脚本，没有训练代码。这里按论文补了自行车模型仿真、稀疏奖励、八阶段课程、动作分块和带交叉注意力的 PPO。

场景 JSON 来自 [Constrained_Parking_Scenarios](https://github.com/dqm5rtfg9b-collab/Constrained_Parking_Scenarios)，许可是 CC-BY-NC-4.0，只用于学术和非商业用途。

conda 环境用 `deep-rl-class`。论文的软件版本是 Python 3.9、PyTorch 2.6、SB3 2.2.1；当前环境更新，算法设置仍按附录 C、D。

## 和论文对齐的部分

- 后轴自行车模型，轴距 3 m，车宽 2 m，车长 4.95 m，最大转角 32°。碰撞用裁角八边形。
- 8 个离散动作：转角增量 ±8° 或 0，速度 ±0.8 或 0 m/s，步长 0.1 s。没有 (0, 0)。
- 奖励：到位 +3，碰撞 -3，出界 -3，换挡 -0.01，原地打方向 -0.2，每步 -0.01。到位容差是几何中心 0.2 m、航向 ±3°。
- 观测转到自车坐标系。障碍点只保留 25 m 内最近的 256 个，这个上限论文没写，是为了让注意力层能实时跑。
- PPO：batch 256，每次更新 1024 步，10 个 epoch，γ = 1，学习率 3e-4，熵系数 0.001。动作分块长度 h = 4。
- 课程最大步数 `[100, 200, 400, 400, 800, 800, 800, 1000]`。

图 5 没有标每阶段滚出多远、航向扰动多大。实现里前 7 阶段的滚出距离是 0.6、1.5、3、5、8、11、15 米；前两阶段沿用滚出航向，之后的航向半宽是 15°、30°、45°、60°、90°。第 8 阶段用日志起点。论文也没写每个阶段训多少步，默认每阶段 10 万步。

## 训练和评估

```bash
conda activate deep-rl-class
cd ParkBench
python train_ppo.py --steps-per-stage 100000
python eval_ppo.py
```

评估在 51 个日志起点上各跑一次，把成功轨迹画到 `runs/ppo_parkbench/eval/paths.png`。成功样本上的平均耗时、路程和换挡次数对应论文表 1。论文里带课程和分块的 PPO 是 92.2% 成功、规划 0.20 s、路程 19.2 m、换挡 4.3 次。
