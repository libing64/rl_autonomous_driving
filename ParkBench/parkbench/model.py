"""自车状态查询障碍点的一层交叉注意力。对应论文图 3 的轻量特征提取器。"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128, d_model: int = 64):
        super().__init__(observation_space, features_dim)
        state_dim = int(observation_space["state"].shape[0])
        self.state_net = nn.Sequential(nn.Linear(state_dim, d_model), nn.ReLU())
        self.point_net = nn.Sequential(nn.Linear(2, d_model), nn.ReLU())
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, features_dim),
            nn.ReLU(),
        )
        self._features_dim = features_dim

    def forward(self, obs: dict) -> torch.Tensor:
        state = self.state_net(obs["state"])
        points = self.point_net(obs["obstacles"])
        mask = obs["mask"]
        # 全被 mask 掉时 MultiheadAttention 会出 NaN，留一个虚拟点再把结果清零。
        empty = mask.sum(dim=-1) < 0.5
        key_padding = mask < 0.5
        if torch.any(empty):
            key_padding = key_padding.clone()
            key_padding[empty, 0] = False
        query = state.unsqueeze(1)
        context, _ = self.attn(query, points, points, key_padding_mask=key_padding, need_weights=False)
        context = context.squeeze(1)
        context = torch.where(empty.unsqueeze(-1), torch.zeros_like(context), context)
        return self.out(torch.cat([state, context], dim=-1))
