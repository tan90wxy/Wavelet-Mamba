"""
GAN.py
======
语义分割 GAN 架构（遥感/UAV 场景）

Generator  : DualEncoderFusionNet
             双编码器（CNN + Mamba-DWConv）+ CAB 精化 + 对应阶段 Fusion
             输入  : [B, 3, H, W]  RGB 图像
             输出  : [B, num_classes, H, W]  分割 logits

Discriminator : 5 层全卷积 CNN（来自 GATrans，适配任意 num_classes）
             输入  : [B, num_classes, H, W]  softmax 概率图 or one-hot GT
             输出  : [B, num_classes, H', W']  patch-level 判别 logits（原始，无 Sigmoid）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_wavelets import DWTForward, DWTInverse
from timm.models.layers import DropPath
from Vim import SS2D

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn, selective_scan_ref,
    )
except ImportError:
    pass

try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except ImportError:
    pass


# ============================================================
# 一、基础卷积模块
# ============================================================

class ConvBNReLU(nn.Sequential):
    """Conv → BN → ReLU6"""
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3,
        dilation: int = 1, stride: int = 1,
        norm_layer=nn.BatchNorm2d, groups: int = 1, bias: bool = False,
    ):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                groups=groups, bias=bias,
            ),
            norm_layer(out_channels),
            nn.ReLU6(inplace=True),
        )


class ConvBN(nn.Sequential):
    """Conv → BN（无激活）"""
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3,
        dilation: int = 1, stride: int = 1,
        norm_layer=nn.BatchNorm2d, bias: bool = False,
    ):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding, bias=bias,
            ),
            norm_layer(out_channels),
        )


class SeparableConvBNReLU(nn.Sequential):
    """深度可分离卷积 → BN → ReLU6"""
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3,
        stride: int = 1, dilation: int = 1, norm_layer=nn.BatchNorm2d,
    ):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(
                in_channels, in_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                groups=in_channels, bias=False,
            ),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6(inplace=True),
        )


class DepthWiseConv(nn.Module):
    """深度卷积 + 逐点卷积"""
    def __init__(self, in_channel: int, out_channel: int, kernel: int):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channel, in_channel, kernel_size=kernel,
            padding=(kernel - 1) // 2, groups=in_channel,
        )
        self.point_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.depth_conv(x))


# ============================================================
# 二、注意力模块
# ============================================================

class SpatialAttention(nn.Module):
    """CBAM 空间注意力"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.sigmoid(self.relu1(self.conv1(torch.cat([avg_out, max_out], dim=1))))
        return x * out.view(out.size(0), 1, out.size(-2), out.size(-1))


class channelattention(nn.Module):  # noqa: N801 — 保持原命名以兼容 CAB
    """SE 风格通道注意力"""
    def __init__(self, dim: int, fc_ratio: int = 4):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // fc_ratio, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(dim // fc_ratio, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.avgpool(x)) * x


# ============================================================
# 三、频域感知深度可分卷积（FreqAwareDWConv）
# ============================================================

class FreqAwareDWConv(nn.Module):
    """
    频域感知深度可分卷积。
    DWT 分解 → LL/HF 分别精化 → 子带门控加权 → IDWT 重建 → 残差连接。

    接口：输入输出均为 [B, C, H, W]，可无缝替换 WaveletConv。
    """

    def __init__(self, channels: int, wave: str = 'haar'):
        super().__init__()
        self.dwt  = DWTForward(J=1, wave=wave, mode='reflect')
        self.idwt = DWTInverse(wave=wave, mode='reflect')

        # LL 子带精化：大感受野（k=5）捕捉语义级结构
        self.ll_dw = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2,
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # HF 子带精化：小感受野（k=3）保留边缘锐度
        self.hf_dw = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1,
                      groups=channels * 3, bias=False),
            nn.Conv2d(channels * 3, channels * 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels * 3),
            nn.ReLU(inplace=True),
        )

        # 子带门控：由 LL 全局信息决定 LL vs HF 的贡献权重
        self.band_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels // 4, 2, 1, bias=False),
            nn.Softmax(dim=1),
        )

        # 残差对齐
        self.res_align = ConvBN(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        yl, yh = self.dwt(x)
        # yl:  [B, C, Hh, Wh]
        # yh[0]: [B, C, 3, Hh, Wh]

        # LL 子带精化
        yl_out = self.ll_dw(yl)

        # HF 子带精化
        hf0 = yh[0][:, :, 0, :, :]
        hf1 = yh[0][:, :, 1, :, :]
        hf2 = yh[0][:, :, 2, :, :]
        hf_stack = torch.cat([hf0, hf1, hf2], dim=1)        # [B, 3C, Hh, Wh]
        hf_proc  = self.hf_dw(hf_stack)                     # [B, 3C, Hh, Wh]
        hf0_out, hf1_out, hf2_out = torch.split(hf_proc, C, dim=1)
        hf_out = torch.stack([hf0_out, hf1_out, hf2_out], dim=2)  # [B, C, 3, Hh, Wh]

        # 子带门控
        weights = self.band_gate(yl_out)                     # [B, 2, 1, 1]
        w_ll    = weights[:, 0:1, :, :]
        w_hf    = weights[:, 1:2, :, :]

        yl_weighted = yl_out * w_ll
        hf_weighted = hf_out * w_hf.unsqueeze(2)

        # IDWT 重建
        x_rec = self.idwt((yl_weighted, [hf_weighted]))      # [B, C, H', W']

        if x_rec.shape[-2] != H or x_rec.shape[-1] != W:
            x_rec = F.interpolate(x_rec, size=(H, W), mode='bilinear', align_corners=False)

        return x_rec + self.res_align(x)


# ============================================================
# 四、Mamba 路径模块（CrossGuidedMambaDWConv）
# ============================================================

class CrossGuidedMambaDWConv(nn.Module):
    """
    Mamba ‖ FreqAwareDWConv 并行模块。

    两路从同一 x_proj 出发，独立处理后相加：
        Mamba → 全局序列上下文
        FreqAwareDWConv → 局部纹理 + 频域感知
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        ksize: int = 5,
        act_layer=nn.ReLU6,
        drop: float = 0.0,
        dim: int = 512,
        atten_drop: float = 0.0,
    ):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features

        self.fc1      = ConvBNReLU(in_features, hidden_features, kernel_size=1)
        self.mamba    = SS2D(d_model=hidden_features, dropout=atten_drop, d_state=16, scan=8)
        self.mamba_bn = nn.BatchNorm2d(hidden_features)
        self.freq_dw  = FreqAwareDWConv(channels=hidden_features, wave='haar')
        self.fc2      = ConvBN(hidden_features, out_features, kernel_size=1)
        self.act      = act_layer()
        self.drop     = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj  = self.fc1(x)
        x_m     = self.mamba_bn(self.mamba(x_proj))
        x_d     = self.freq_dw(x_proj)
        x_fused = x_m + x_d
        return self.act(self.drop(self.fc2(x_fused)))


# ============================================================
# 五、CAB 精化模块（BiFPN 风格多尺度加权融合）
# ============================================================

class FixedCAB(nn.Module):
    """
    BiFPN 风格多尺度上下文聚合模块（Cross-scale Attention Block）。

    输入  : 四个不同分辨率的特征图（从大到小：res1, res2, res3, res4）
    输出  : 同等尺度的精化特征（out1 最小, out4 最大）
    通道数均为 channels（默认 144），由 proj_cnn/proj_mamba 统一对齐。
    """

    def __init__(self, channels: int = 144, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

        self.weights0    = nn.Parameter(torch.ones(3), requires_grad=True)
        self.attention0  = channelattention(channels)
        self.conv4       = ConvBNReLU(channels, channels, 1)

        self.weights1    = nn.Parameter(torch.ones(3), requires_grad=True)
        self.attention1  = channelattention(channels)
        self.conv3       = ConvBNReLU(channels, channels, 1)

        self.weights2    = nn.Parameter(torch.ones(3), requires_grad=True)
        self.attention2  = channelattention(channels)
        self.conv2       = ConvBNReLU(channels, channels, 1)

        self.weights3    = nn.Parameter(torch.ones(3), requires_grad=True)
        self.attention3  = channelattention(channels)
        self.conv1       = ConvBNReLU(channels, channels, 1)

    def forward(
        self,
        res1: torch.Tensor,
        res2: torch.Tensor,
        res3: torch.Tensor,
        res4: torch.Tensor,
    ):
        # 第 4 层（最小尺度）
        h, w    = res4.shape[-2:]
        feat1_4 = F.interpolate(res1, (h, w), mode='bilinear', align_corners=False)
        feat2_4 = F.interpolate(res2, (h, w), mode='bilinear', align_corners=False)
        feat3_4 = F.interpolate(res3, (h, w), mode='bilinear', align_corners=False)
        w0      = F.relu6(self.weights0) / (F.relu6(self.weights0).sum() + self.eps)
        out1    = self.conv4(self.attention0(
            w0[0] * feat1_4 + w0[1] * feat2_4 + w0[2] * feat3_4 + res4
        ))

        # 第 3 层
        h, w    = res3.shape[-2:]
        feat1_3 = F.interpolate(res1, (h, w), mode='bilinear', align_corners=False)
        feat2_3 = F.interpolate(res2, (h, w), mode='bilinear', align_corners=False)
        feat4_3 = F.interpolate(res4, (h, w), mode='bilinear', align_corners=False)
        w1      = F.relu6(self.weights1) / (F.relu6(self.weights1).sum() + self.eps)
        out2    = self.conv3(self.attention1(
            w1[0] * feat1_3 + w1[1] * feat2_3 + w1[2] * feat4_3 + res3
        ))

        # 第 2 层
        h, w    = res2.shape[-2:]
        feat1_2 = F.interpolate(res1, (h, w), mode='bilinear', align_corners=False)
        feat3_2 = F.interpolate(res3, (h, w), mode='bilinear', align_corners=False)
        feat4_2 = F.interpolate(res4, (h, w), mode='bilinear', align_corners=False)
        w2      = F.relu6(self.weights2) / (F.relu6(self.weights2).sum() + self.eps)
        out3    = self.conv2(self.attention2(
            w2[0] * feat1_2 + w2[1] * res2 + w2[2] * feat3_2 + feat4_2
        ))

        # 第 1 层（最大尺度）
        h, w    = res1.shape[-2:]
        feat2_1 = F.interpolate(res2, (h, w), mode='bilinear', align_corners=False)
        feat3_1 = F.interpolate(res3, (h, w), mode='bilinear', align_corners=False)
        feat4_1 = F.interpolate(res4, (h, w), mode='bilinear', align_corners=False)
        w3      = F.relu6(self.weights3) / (F.relu6(self.weights3).sum() + self.eps)
        out4    = self.conv1(self.attention3(
            w3[0] * res1 + w3[1] * feat2_1 + w3[2] * feat3_1 + feat4_1
        ))

        return out1, out2, out3, out4


# ============================================================
# 六、编码器模块
# ============================================================

class CNNEncoder(nn.Module):
    """四阶段 CNN 编码器：每阶段 stride=2 下采样，通道数翻倍"""

    def __init__(self, in_ch: int = 3, base_ch: int = 32, num_stages: int = 4):
        super().__init__()
        self.stages = nn.ModuleList()
        ch = base_ch
        for _ in range(num_stages):
            self.stages.append(nn.Sequential(
                ConvBNReLU(in_ch, ch, kernel_size=3, stride=2),
                ConvBNReLU(ch, ch, kernel_size=3, stride=1),
            ))
            in_ch = ch
            ch   *= 2

    def forward(self, x: torch.Tensor):
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return outs  # 四个尺度特征列表


class MambaDWEncoder(nn.Module):
    """四阶段 Mamba+DWConv 并行编码器：stride=2 下采样 + CrossGuidedMambaDWConv"""

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 32,
        num_stages: int = 4,
    ):
        super().__init__()
        self.stages = nn.ModuleList()
        ch = base_ch
        for _ in range(num_stages):
            downsample  = ConvBNReLU(in_ch, ch, kernel_size=3, stride=2)
            mamba_block = CrossGuidedMambaDWConv(
                in_features=ch, hidden_features=ch, out_features=ch,
                dim=ch, atten_drop=0.0,
            )
            self.stages.append(nn.ModuleList([downsample, mamba_block]))
            in_ch = ch
            ch   *= 2

    def forward(self, x: torch.Tensor):
        outs = []
        for downsample, mamba_block in self.stages:
            x = downsample(x)
            x = mamba_block(x)
            outs.append(x)
        return outs


# ============================================================
# 七、Fusion 模块（可学习加权融合）
# ============================================================

class Fusion(nn.Module):
    """双输入可学习加权融合，自动对齐分辨率（双线性插值到较大分辨率）"""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.weights   = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps       = eps
        self.post_conv = SeparableConvBNReLU(dim, dim, 5)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        th = max(a.shape[-2], b.shape[-2])
        tw = max(a.shape[-1], b.shape[-1])
        if a.shape[-2] != th or a.shape[-1] != tw:
            a = F.interpolate(a, (th, tw), mode='bilinear', align_corners=False)
        if b.shape[-2] != th or b.shape[-1] != tw:
            b = F.interpolate(b, (th, tw), mode='bilinear', align_corners=False)
        w      = F.relu6(self.weights) / (F.relu6(self.weights).sum() + self.eps)
        return self.post_conv(w[0] * b + w[1] * a)


# ============================================================
# 八、Generator = DualEncoderFusionNet
# ============================================================

class Generator(nn.Module):
    """
    双编码器语义分割网络（原 DualEncoderFusionNet）。

    输入  : [B, in_ch, H, W]
    输出  : [B, num_classes, H, W]  分割 logits（不含 softmax）
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 32,
        num_stages: int = 4,
        num_classes: int = 6,
        proj_ch: int = 144,
    ):
        super().__init__()
        self.proj_ch     = proj_ch
        self.cnn_encoder = CNNEncoder(in_ch, base_ch, num_stages)
        self.mamba_encoder = MambaDWEncoder(in_ch, base_ch, num_stages)

        # 将各阶段输出统一投影到 proj_ch 维
        self.proj_cnn   = nn.ModuleList()
        self.proj_mamba = nn.ModuleList()
        ch = base_ch
        for _ in range(num_stages):
            self.proj_cnn.append(ConvBNReLU(ch, proj_ch, 1))
            self.proj_mamba.append(ConvBNReLU(ch, proj_ch, 1))
            ch *= 2

        # CAB 精化（BiFPN 多尺度）
        self.cab = FixedCAB(channels=proj_ch)

        # 对应阶段特征融合
        self.fusions = nn.ModuleList([Fusion(proj_ch) for _ in range(num_stages)])

        # 独立分割头（每个头处理一个尺度，结果累加）
        self.heads = nn.ModuleList([
            nn.Sequential(
                ConvBNReLU(proj_ch, proj_ch, kernel_size=3),
                nn.Dropout2d(0.1),
                nn.Conv2d(proj_ch, num_classes, kernel_size=1),
            )
            for _ in range(num_stages)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        # 双编码器前向
        cnn_feats   = self.cnn_encoder(x)
        mamba_feats = self.mamba_encoder(x)

        # 投影到统一通道数
        cnn_feats   = [p(f) for p, f in zip(self.proj_cnn,   cnn_feats)]
        mamba_feats = [p(f) for p, f in zip(self.proj_mamba, mamba_feats)]

        # CAB 精化 CNN 特征（从大到小：c1 最大）
        c1, c2, c3, c4 = self.cab(*cnn_feats)

        # 对应阶段融合
        fused = [
            self.fusions[i](m, c)
            for i, (c, m) in enumerate(zip([c1, c2, c3, c4], mamba_feats))
        ]

        # 各尺度分割头输出累加
        out = None
        for i, f in enumerate(fused):
            if f.shape[-2] != H or f.shape[-1] != W:
                f = F.interpolate(f, (H, W), mode='bilinear', align_corners=False)
            pred = self.heads[i](f)
            out  = pred if out is None else out + pred

        return out


# ============================================================
# 九、Discriminator（来自 GATrans，纯 CNN）
# ============================================================

class Discriminator(nn.Module):
    """
    全卷积 CNN 判别器（改自 GATrans.py；卷积核按稿件 Section 3.4 取 3×3）。

    结构  : 5 × (Conv2d + LeakyReLU(0.2))，3×3 卷积、步长 2 逐步下采样，无归一化层
    输入  : [B, num_classes, H, W]  softmax 概率图 or one-hot GT
    输出  : [B, num_classes, H', W']  patch-level 原始 logits（无 Sigmoid）
              H' = H // 32, W' = W // 32
    损失  : 配合 BCEWithLogitsLoss 使用（外部计算）
    """

    def __init__(self, num_classes: int = 6, ndf: int = 64):
        super().__init__()
        self.conv1      = nn.Conv2d(num_classes, ndf,      kernel_size=3, stride=2, padding=1)
        self.conv2      = nn.Conv2d(ndf,         ndf * 2,  kernel_size=3, stride=2, padding=1)
        self.conv3      = nn.Conv2d(ndf * 2,     ndf * 4,  kernel_size=3, stride=2, padding=1)
        self.conv4      = nn.Conv2d(ndf * 4,     ndf * 8,  kernel_size=3, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 8,     num_classes, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        return self.classifier(x)   # 原始 logits，配合 BCEWithLogitsLoss


# ============================================================
# 十、快速自测
# ============================================================

if __name__ == '__main__':
    import time
    from thop import profile, clever_format

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 3, 512, 512).to(device)

    print('=' * 50)
    print('Generator (DualEncoderFusionNet) test')
    G = Generator(in_ch=3, base_ch=32, num_stages=4, num_classes=6).to(device)
    with torch.no_grad():
        t0 = time.time()
        out_g = G(x)
        print(f'  output shape : {out_g.shape}')   # [1, 6, 512, 512]
        print(f'  forward time : {(time.time() - t0) * 1000:.1f} ms')
    flops, params = profile(G, (x,), verbose=False)
    print('  FLOPs/Params :', *clever_format([flops, params], '%.3f'))

    print('=' * 50)
    print('Discriminator test')
    prob_map = torch.softmax(out_g, dim=1)           # [1, 6, 512, 512]
    D = Discriminator(num_classes=6, ndf=64).to(device)
    with torch.no_grad():
        out_d = D(prob_map)
        print(f'  output shape : {out_d.shape}')    # [1, 6, 16, 16]
    flops_d, params_d = profile(D, (prob_map,), verbose=False)
    print('  FLOPs/Params :', *clever_format([flops_d, params_d], '%.3f'))
