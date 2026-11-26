# -*- coding: utf-8 -*-
"""
losses.py
=========
分割损失，对应稿件 Section 3.4 的 Eq. (25)-(27)：

    L_CE   = -sum_k y_k * log(y_hat_k)                                     (25)
    L_Dice = 1 - (1/K) * sum_k (2*sum(y_hat_k*y_k) + 0.8)
                               / (sum(y_hat_k) + sum(y_k) + 0.8)           (26)
    L_Bd   = sum(W * l_ce) / sum(W)                                        (27)

按 Table 2 的权重组合（1.0 / 0.8 / 0.4）：

    L_seg = 1.0*L_CE + 0.8*L_Dice + 0.4*L_Bd

其中求和 sum 覆盖 batch 与空间维度，l_ce 是未加权的逐像素交叉熵，W 是边界权重图，
K 是类别数。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 超参：Table 2
# ---------------------------------------------------------------------------
W_CE = 1.0          # Loss weight of L_CE
W_DICE = 0.8        # Loss weight of L_Dice
W_BD = 0.4          # Loss weight of L_Bd

DICE_SMOOTH = 0.8   # Eq. (26) 的平滑项

# ---------------------------------------------------------------------------
# Eq. (27) 的边界权重图 W
#
# 稿件只写了 "W is the boundary weight"，没有给出 W 的构造方式。这里采用按距离
# 衰减的形式
#
#       W = 1 + BD_ALPHA * exp(-d / BD_SIGMA)
#
# d 为像素到最近 GT 类别边界的距离（像素数）。BD_ALPHA、BD_SIGMA 是仅有的两个
# 自由参数：运行实验前请确认取值，或替换为实际使用过的权重图构造。
# ---------------------------------------------------------------------------
BD_ALPHA = 1.0
BD_SIGMA = 5.0
BD_MAX_DIST = 32    # 距离截断（迭代膨胀的最大步数）


def boundary_weight_map(label: torch.Tensor,
                        num_classes: int,
                        alpha: float = BD_ALPHA,
                        sigma: float = BD_SIGMA,
                        max_dist: int = BD_MAX_DIST) -> torch.Tensor:
    """由 GT 标签构造 Eq. (27) 的边界权重图 W，返回 [B, H, W]。"""
    label = label.long().clamp(0, num_classes - 1)
    with torch.no_grad():
        onehot = F.one_hot(label, num_classes).permute(0, 3, 1, 2).float()
        dilated = F.max_pool2d(onehot, kernel_size=3, stride=1, padding=1)
        # 邻域里出现别的类别 -> 该像素位于类别边界上（边界两侧都被标记）
        boundary = ((dilated - onehot).sum(dim=1) > 0).float()          # [B, H, W]

        dist = torch.where(boundary > 0,
                           torch.zeros_like(boundary),
                           torch.full_like(boundary, float(max_dist)))
        filled = boundary.clone()
        frontier = boundary.unsqueeze(1)
        for step in range(1, max_dist + 1):
            frontier = F.max_pool2d(frontier, kernel_size=3, stride=1, padding=1)
            new = (frontier.squeeze(1) > 0) & (filled == 0)
            if not bool(new.any()):
                break
            dist = torch.where(new, torch.full_like(dist, float(step)), dist)
            filled = filled + new.float()

    return 1.0 + alpha * torch.exp(-dist / sigma)


def dice_loss(logits: torch.Tensor,
              label: torch.Tensor,
              num_classes: int,
              smooth: float = DICE_SMOOTH) -> torch.Tensor:
    """Eq. (26)。"""
    prob = torch.softmax(logits, dim=1)
    onehot = F.one_hot(label.long().clamp(0, num_classes - 1),
                       num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (prob * onehot).sum(dims)
    card = prob.sum(dims) + onehot.sum(dims)
    return 1.0 - ((2.0 * inter + smooth) / (card + smooth)).mean()


def boundary_loss(logits: torch.Tensor,
                  label: torch.Tensor,
                  num_classes: int) -> torch.Tensor:
    """Eq. (27)：以边界权重 W 加权的逐像素交叉熵。"""
    l_ce = F.cross_entropy(logits,
                           label.long().clamp(0, num_classes - 1),
                           reduction='none')
    weight = boundary_weight_map(label, num_classes)
    return (weight * l_ce).sum() / weight.sum().clamp_min(1e-6)


class SegmentationLoss(nn.Module):
    """L_seg = W_CE*L_CE + W_DICE*L_Dice + W_BD*L_Bd（Eq. 25-27，Table 2）。"""

    def __init__(self,
                 num_classes: int = 6,
                 w_ce: float = W_CE,
                 w_dice: float = W_DICE,
                 w_bd: float = W_BD):
        super().__init__()
        self.num_classes = num_classes
        self.w_ce = w_ce
        self.w_dice = w_dice
        self.w_bd = w_bd

    def forward(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        label = label.long().clamp(0, self.num_classes - 1)
        l_ce = F.cross_entropy(logits, label)
        l_dice = dice_loss(logits, label, self.num_classes)
        l_bd = boundary_loss(logits, label, self.num_classes)
        return self.w_ce * l_ce + self.w_dice * l_dice + self.w_bd * l_bd

    def components(self, logits: torch.Tensor, label: torch.Tensor) -> dict:
        """分别返回三项损失，便于写日志核对权重是否生效。"""
        label = label.long().clamp(0, self.num_classes - 1)
        return {
            'ce': F.cross_entropy(logits, label).detach(),
            'dice': dice_loss(logits, label, self.num_classes).detach(),
            'bd': boundary_loss(logits, label, self.num_classes).detach(),
        }
