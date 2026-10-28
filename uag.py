"""
Uncertainty-Adaptive Gate (方向A)
=================================
让"域盲"模块变成"域感知"：先用一个轻量方差头估计每个空间位置的
特征可靠性，再用该可靠性门控一个 refine 模块。

核心公式:
    w     = sigmoid(-logvar)        # 置信度 ∈ (0,1)，方差小 → w 大（可靠）
    out   = x + gamma * (w * refine(x))

性质:
- gamma 初始化为 0 → 训练起点 out = x（等价 G0），fail-safe，不会比基线更差
- 噪声区域（超声 speckle）方差大 → w≈0 → 不 refine（不放大噪声、不丢信号）
- 清晰区域（息肉边界）方差小 → w≈1 → 充分 refine

与 VIB (ib_skip.py) 的对比:
- VIB 用 KL 强迫 z→噪声，是"信息压缩"，在噪声数据上丢真信号
- UAG 用方差门控 refine，是"按可靠性调度"，在噪声数据上安全跳过
"""

import torch
import torch.nn as nn


class UncertaintyAdaptiveGate(nn.Module):
    """Variance-gated residual refinement at a bottleneck position.

    Args:
        channels: feature channels (refine preserves channels)
    """

    def __init__(self, channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        # 1-channel per-position log-variance -> spatial uncertainty map
        self.logvar_head = nn.Conv2d(channels, 1, 1)
        # 残差缩放，init=0 -> fail-safe 起点
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        refined = self.refine(x)
        logvar = self.logvar_head(x)
        w = torch.sigmoid(-logvar)          # (B,1,H,W) 空间置信度
        out = x + self.gamma * (w * refined)
        return out

    def extra_repr(self):
        return f'ch={self.logvar_head.in_channels}'
