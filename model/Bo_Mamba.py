"""
Bo_Mamba_CrossGuided.py
=======================
在保留 Mamba ‖ DWConv 并行主题的基础上，针对原版四大问题的创新改进。

原版四大问题
─────────────────────────────────────────────────────────
① WaveletConv 近恒等：DWT→IDWT 之间无任何处理，mode='zero' 边界效应
② GLSS 等权求和：Mamba 路径与 DWConv 路径直接相加，无门控，语义空间不同
③ CAB weights 参数从未使用：forward() 里实际执行等权加法，权重梯度为零
④ CAB conv4/conv3/conv2 未使用：outfeature1~3 缺少卷积精化步骤

改进方案
─────────────────────────────────────────────────────────
改进① FreqAwareDWConv（替换 WaveletConv）
      LL 子带（语义低频）→ DWConv-k5（大感受野）
      HF 子带（边缘高频）→ DWConv-k3（精确局部）
      可学习子带门控：由 LL 全局均值决定 LL/HF 各自贡献权重
      mode='reflect' 减少边界伪影

改进② CrossGuidedMambaDWConv（替换 GLSS）
      两路从同一 x_proj 出发，真正并行：
        Mamba 路径    → x_m（全局序列上下文）
        DWConv+频域路径 → x_d（局部纹理+频域感知）
      互引导机制：
        x_m → spatial_gate = σ(conv1×1(x_m)) → x_d_guided（Mamba 告诉 DWConv 哪里看）
        x_d → channel_gate = σ(SE(x_d))      → x_m_guided（DWConv 告诉 Mamba 看什么）
      融合：x_m_guided + x_d_guided → fc2 → act

改进③④ WeightedCAB（替换 CAB，同时修复两个 Bug）
      修复 Bug③：实际使用 weights 参数做 BiFPN 风格 relu-归一化加权融合
      修复 Bug④：启用 conv4/conv3/conv2 对 outfeature1~3 进行卷积精化

维度速查（decode_channels=144, mlp_ratio=4）
──────────────────────────────────────────
  Block 输入/输出  [B, 144, H, W]
  fc1 输出 hidden  [B,  36, H, W]   (144 // 4 = 36)
  x_m / x_d        [B,  36, H, W]
  gate_d (空间)     [B,   1, H, W]   ← 由 x_m 生成
  gate_m (通道)     [B,  36, 1, 1]   ← 由 x_d 生成
  fc2 输出          [B, 144, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from Vim import SS2D
from backbone.HRNet.config.default import _C as config
from backbone.HRNet.models.seg_hrnet import get_seg_model

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass


# ══════════════════════════════════════════════════════════════════
#  基础卷积模块（完全保留原版）
# ══════════════════════════════════════════════════════════════════

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, norm_layer=nn.BatchNorm2d, groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )

class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                                    kernel_size=kernel, stride=1,
                                    padding=(kernel - 1) // 2, groups=in_channel)
        self.point_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                    kernel_size=1, stride=1, padding=0, groups=1)
    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


# ══════════════════════════════════════════════════════════════════
#  注意力与 ASPP 模块（完全保留原版）
# ══════════════════════════════════════════════════════════════════

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1   = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.relu1   = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.relu1(self.conv1(out)))
        return x * out.view(out.size(0), 1, out.size(-2), out.size(-1))

class channelattention(nn.Module):
    def __init__(self, dim, fc_ratio=4):
        super(channelattention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(dim // fc_ratio, dim, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        u = x.clone()
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        return c_attn * u

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, dilation):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, dilation=dilation,
                                   padding=((dilation * 2) // 2), groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        kw = {} if norm_kwargs is None else norm_kwargs
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1))
        self.add_module('bn1',   norm_layer(inter_channels, **kw))
        self.add_module('relu1', nn.ReLU(True))
        self.add_module('conv2', depthwise_separable_conv(
            nin=inter_channels, nout=out_channels, dilation=atrous_rate))
        self.add_module('bn2',   norm_layer(out_channels, **kw))
        self.add_module('relu2', nn.ReLU(True))
        self.drop_rate = drop_rate
    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features

class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels
        rate1, rate2, rate3 = atrous_rates
        self.b0 = _DenseASPPConv(in_channels, out_channels // 2,
                                  out_channels, rate1, 0.1)
        self.b1 = _DenseASPPConv(in_channels + out_channels,
                                  (in_channels + out_channels) // 2,
                                  out_channels, rate2, 0.1)
        self.b2 = _DenseASPPConv(in_channels + out_channels * 2,
                                  (in_channels + out_channels * 2) // 2,
                                  out_channels, rate3, 0.1)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels + 3 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x):
        u = x.clone()
        aspp1 = self.b0(x);         x = torch.cat([aspp1, x], dim=1)
        aspp2 = self.b1(x);         x = torch.cat([aspp2, x], dim=1)
        aspp3 = self.b2(x);         x = torch.cat([aspp3, x], dim=1)
        return self.project(x) * u

class CSDF(nn.Module):
    """原版 CSDF，完全保留"""
    def __init__(self, dim, num_heads, atten_drop=0., proj_drop=0.,
                 dilation=None, fc_ratio=4):
        super(CSDF, self).__init__()
        if dilation is None:
            dilation = [3, 5, 7]
        assert dim % num_heads == 0
        self.dim        = dim
        self.num_heads  = num_heads
        self.scale      = (dim // num_heads) ** -0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop  = nn.Dropout(proj_drop)
        self.DMSC       = ASPPModule(dim, dilation)
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(dim // fc_ratio, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.spat_atten = SpatialAttention()
        self.kv         = Conv(dim, 2 * dim, 1)
        self.fc2        = ConvBN(in_channels=dim, out_channels=dim, kernel_size=1)
        self.act        = nn.ReLU6()
    def forward(self, x):
        u      = x.clone()
        attn   = self.DMSC(x)
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn) * u
        s_attn = self.spat_atten(x)
        x      = self.fc2(attn + c_attn + s_attn)
        return self.act(x)


# ══════════════════════════════════════════════════════════════════
#  改进① FreqAwareDWConv — 真正利用子带差异化处理（替换原 WaveletConv）
# ══════════════════════════════════════════════════════════════════

class FreqAwareDWConv(nn.Module):
    """
    频域感知深度可分卷积。

    原版 WaveletConv 的根本问题：
        conv1 → BN → ReLU → DWT → IDWT（无任何子带处理）→ conv2
        DWT 之后立刻 IDWT 相当于 x + 边界噪声，mode='zero' 让噪声更严重。

    改进逻辑：
        DWT 分解后，LL 和 HF 子带携带不同的信息：
        · LL（低频）= 语义轮廓、区域整体结构 → 适合大感受野（k=5）精化
        · HF（LH/HL/HH 三个高频方向）= 边缘、纹理细节 → 适合小感受野（k=3）保留锐度

        子带门控权重：
        · 由 LL 子带的全局均值预测一个 2 维向量 [w_ll, w_hf]，经 Softmax 归一化
        · w_ll 高 → LL 信息主导（平坦语义区域）
        · w_hf 高 → HF 信息主导（边界、纹理区域）
        · 这样边界区域的高频响应被自动增强，内部区域的低频结构被自动保留

        残差连接 + 1×1 对齐卷积：
        · 防止 DWT/IDWT 边界效应积累（reflect 模式已减少，但仍有 ±1 误差）
        · 保证梯度畅通，稳定深层训练

    接口：输入输出均为 [B, channels, H, W]，可无缝替换 WaveletConv。
    """

    def __init__(self, channels: int, wave: str = 'haar'):
        super().__init__()
        # reflect 模式比 zero 更适合遥感图像（边界连续性更好）
        self.dwt  = DWTForward(J=1, wave=wave, mode='reflect')
        self.idwt = DWTInverse(wave=wave, mode='reflect')

        # LL 子带精化：大感受野（k=5），捕捉语义级结构
        # 深度卷积 → 点卷积 的分离结构保持轻量
        self.ll_dw = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2,
                      groups=channels, bias=False),      # 逐通道：保持各通道独立
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),  # 点卷积：通道混合
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # HF 子带精化：小感受野（k=3），保留边缘锐度
        # 三个方向子带合并为 C*3，groups=C*3 保持各方向子带独立处理
        self.hf_dw = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1,
                      groups=channels * 3, bias=False),  # 各子带独立逐通道卷积
            nn.Conv2d(channels * 3, channels * 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels * 3),
            nn.ReLU(inplace=True)
        )

        # 子带门控：由 LL 全局信息决定 LL vs HF 的贡献权重
        # 输出 2 维 Softmax 权重：w[0] 给 LL，w[1] 给 HF
        self.band_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                           # [B, C, 1, 1]
            nn.Conv2d(channels, channels // 4, 1, bias=False), # 压缩
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels // 4, 2, 1, bias=False),        # 输出 2 个权重
            nn.Softmax(dim=1)                                   # 归一化，二者之和为 1
        )

        # 残差对齐：1×1 BN，对齐原始输入的数值范围
        self.res_align = ConvBN(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── 小波分解 ──────────────────────────────────────────────
        yl, yh = self.dwt(x)
        # yl:    [B, C, Hh, Wh]       Hh ≈ H/2
        # yh[0]: [B, C, 3, Hh, Wh]   三个高频子带
        _, _, Hh, Wh = yl.shape

        # ── LL 子带精化（大核） ────────────────────────────────────
        yl_out = self.ll_dw(yl)                          # [B, C, Hh, Wh]

        # ── HF 子带精化（小核） ────────────────────────────────────
        hf      = yh[0].reshape(B, C * 3, Hh, Wh)       # 合并三方向 → [B, 3C, Hh, Wh]
        hf_out  = self.hf_dw(hf)                         # [B, 3C, Hh, Wh]
        hf_out  = hf_out.reshape(B, C, 3, Hh, Wh)        # 还原 → [B, C, 3, Hh, Wh]

        # ── 子带门控：学习 LL vs HF 的空间权重分配 ─────────────────
        # 以 yl_out 为输入：基于增强后的低频计算门控，更稳定
        weights  = self.band_gate(yl_out)                 # [B, 2, 1, 1]
        w_ll     = weights[:, 0:1, :, :]                  # [B, 1, 1, 1]
        w_hf     = weights[:, 1:2, :, :]                  # [B, 1, 1, 1]

        # 加权 LL 子带
        yl_weighted  = yl_out * w_ll                      # [B, C, Hh, Wh]
        # 加权 HF 子带（需 unsqueeze 匹配 5D）
        hf_weighted  = hf_out * w_hf.unsqueeze(2)         # [B, C, 3, Hh, Wh]

        # ── IDWT 重建 ──────────────────────────────────────────────
        x_rec = self.idwt((yl_weighted, [hf_weighted]))   # [B, C, H', W']

        # 处理 reflect 模式可能的 ±1 尺寸误差
        if x_rec.shape[-2] != H or x_rec.shape[-1] != W:
            x_rec = F.interpolate(x_rec, size=(H, W),
                                  mode='bilinear', align_corners=False)

        # ── 残差连接 ──────────────────────────────────────────────
        return x_rec + self.res_align(x)


# ══════════════════════════════════════════════════════════════════
#  改进② CrossGuidedMambaDWConv — 交叉引导并行（替换原 GLSS）
# ══════════════════════════════════════════════════════════════════

class CrossGuidedMambaDWConv(nn.Module):
    """
    交叉引导的 Mamba ‖ DWConv 并行模块，替换原 GLSS。

    核心设计思想——互引导而非独立并行：

    原版 GLSS 的本质问题：
        x1 = DWConv(wavelet(x_proj))
        x3 = Mamba(x_proj)
        output = fc2(x1 + x2 + x3)   ← 等权求和，两路互不感知

    改进：两路相互提供引导信号
        ┌─────────────────── x_proj ───────────────────┐
        │                                               │
      Mamba → x_m (全局上下文)        FreqAwareDWConv → x_d (局部+频域)
        │                                               │
        ▼                                               ▼
    Mamba 对 DWConv 的引导：           DWConv 对 Mamba 的引导：
    gate_d = σ(conv1×1(x_m))          gate_m = σ(SE(gap(x_d)))
    → 空间注意力图 [B,1,H,W]           → 通道注意力图 [B,C,1,1]
    "全局感知告诉局部：哪里重要"        "局部纹理告诉全局：哪些通道活跃"
        │                                               │
        └──── x_d_guided = x_d * gate_d ────────────────┘
        └──── x_m_guided = x_m * gate_m ────────────────┘
                                │
                    x_m_guided + x_d_guided
                                │
                           fc2 → act

    为什么这样的引导方向是合理的：
    · Mamba 经过全局序列扫描后，x_m 中每个位置包含全局上下文信息，
      它"知道"整体语义结构中哪些区域更关键（如建筑轮廓/植被边界）。
      将 x_m 压缩成空间权重图，可以让 DWConv 在重要区域投入更多"注意力"。

    · DWConv 经过频域感知后，x_d 中不同通道对应不同频率的局部响应
      （某些通道对边缘敏感，某些对纹理敏感）。
      将 x_d 全局池化后做 SE，可以让 Mamba 优先处理当前图像中最活跃的纹理类型。
    """

    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, ksize: int = 5,
                 act_layer=nn.ReLU6, drop: float = 0.,
                 dim: int = 512, atten_drop: float = 0.):
        super(CrossGuidedMambaDWConv, self).__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features

        # ── 输入投影：in → hidden ──────────────────────────────────
        self.fc1 = ConvBNReLU(in_channels=in_features,
                               out_channels=hidden_features, kernel_size=1)

        # ── 分支 A：Mamba 全局路径 ────────────────────────────────
        self.mamba    = SS2D(d_model=hidden_features, dropout=atten_drop,
                             d_state=16, scan=8)
        self.mamba_bn = nn.BatchNorm2d(hidden_features)

        # ── 分支 B：频域感知 DWConv 局部路径 ─────────────────────
        self.freq_dw  = FreqAwareDWConv(channels=hidden_features, wave='haar')

        # ── 交叉引导模块 ──────────────────────────────────────────
        #
        # (A) Mamba → 空间门控 → DWConv
        #     x_m [B,C,H,W] → 1×1 conv → sigmoid → [B,1,H,W]
        #     空间权重告诉 DWConv "哪些位置值得关注"
        self.mamba_to_spatial = nn.Sequential(
            nn.Conv2d(hidden_features, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        #
        # (B) DWConv → 通道门控 → Mamba
        #     x_d [B,C,H,W] → GAP → SE-style → sigmoid → [B,C,1,1]
        #     通道权重告诉 Mamba "哪些通道的纹理模式当前最活跃"
        squeeze_ch = max(hidden_features // 4, 4)  # 防止 hidden 很小时除以4太小
        self.dw_to_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_features, squeeze_ch, kernel_size=1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(squeeze_ch, hidden_features, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # ── 输出投影：hidden → out ─────────────────────────────────
        self.fc2  = ConvBN(in_channels=hidden_features,
                            out_channels=out_features, kernel_size=1)
        self.act  = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── 输入投影 ──────────────────────────────────────────────
        x_proj = self.fc1(x)                                 # [B, hidden, H, W]

        # ── 两路并行处理（均以同一 x_proj 为输入）────────────────
        x_m = self.mamba_bn(self.mamba(x_proj))             # 全局路径 [B, hidden, H, W]
        x_d = self.freq_dw(x_proj)                          # 局部路径 [B, hidden, H, W]

        # ── 交叉引导 ─────────────────────────────────────────────
        #
        # Mamba 为 DWConv 提供空间引导
        # 直觉：Mamba 扫描全序列后"知道"哪些位置对当前语义最重要
        gate_d       = self.mamba_to_spatial(x_m)           # [B, 1, H, W]
        x_d_guided   = x_d * gate_d                         # 空间精化的局部特征

        # DWConv 为 Mamba 提供通道引导
        # 直觉：DWConv 对局部纹理的响应告知 Mamba 哪些通道"信噪比最高"
        gate_m       = self.dw_to_channel(x_d)              # [B, hidden, 1, 1]
        x_m_guided   = x_m * gate_m                         # 通道精化的全局特征

        # ── 融合与输出 ────────────────────────────────────────────
        x_fused = x_m_guided + x_d_guided                   # [B, hidden, H, W]
        return self.act(self.drop(self.fc2(x_fused)))


# ══════════════════════════════════════════════════════════════════
#  Block（将 mlp 替换为 CrossGuidedMambaDWConv，attn 保持 CSDF）
# ══════════════════════════════════════════════════════════════════

class Block(nn.Module):
    """
    与原版 Block 结构完全相同，仅将 mlp=GLSS 替换为 mlp=CrossGuidedMambaDWConv。
    attn=CSDF 保留（原版已是有效的注意力模块，无需修改）。
    双重残差（内层 attn/mlp 各一次 + 外层 u）保持不变。
    """
    def __init__(self, dim: int = 512, num_heads: int = 16, mlp_ratio: int = 4,
                 pool_ratio: int = 16, drop: float = 0., dilation=None,
                 drop_path: float = 0., act_layer=nn.ReLU6,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        if dilation is None:
            dilation = [3, 5, 7]
        self.norm1      = norm_layer(dim)
        self.norm2      = norm_layer(dim)
        # 保留 CSDF 作为 attn 分支
        self.attn       = CSDF(dim, num_heads=num_heads, atten_drop=drop,
                                proj_drop=drop, dilation=dilation, fc_ratio=mlp_ratio)
        self.drop_path  = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim  = int(dim // mlp_ratio)
        # 用 CrossGuidedMambaDWConv 替换 GLSS
        self.mlp = CrossGuidedMambaDWConv(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.clone()                                        # 外层残差锚点
        x = x + self.drop_path(self.attn(self.norm1(x)))    # attn 分支
        x = x + self.drop_path(self.mlp(self.norm2(x)))     # mlp 分支（交叉引导）
        return x + u                                         # 外层残差叠加


# ══════════════════════════════════════════════════════════════════
#  Fusion（原版保留）
# ══════════════════════════════════════════════════════════════════

class Fusion(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(Fusion, self).__init__()
        self.dim       = dim
        self.weights   = nn.Parameter(torch.ones(2, dtype=torch.float32),
                                       requires_grad=True)
        self.eps       = eps
        self.post_conv = SeparableConvBNReLU(dim, dim, 5)
        self.upsample  = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)

    def forward(self, x, res):
        if x.shape[2] != res.shape[2]:
            x = self.upsample(x)
        weights      = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x            = fuse_weights[0] * res + fuse_weights[1] * x
        return self.post_conv(x)


# ══════════════════════════════════════════════════════════════════
#  改进③④ WeightedCAB — 修复 weights 未用 + conv4/3/2 未用两个 Bug
# ══════════════════════════════════════════════════════════════════

class WeightedCAB(nn.Module):
    """
    修复原版 CAB 的两个 Bug，实现真正的加权跨尺度融合。

    Bug ③：weights 定义但从未使用
    ──────────────────────────────
    原版 forward() 中直接用 f1+f2+f3+f4，四路等权相加。
    self.weights0~3 的参数梯度永远为零（未参与计算图）。
    修复：用 _bfpn_fuse() 真正调用 weights，实现 BiFPN 风格加权融合。

    Bug ④：conv4/conv3/conv2 定义但从未使用
    ──────────────────────────────────────────
    原版只对 outfeature4 应用了 conv1(144→18)，
    outfeature1~3 只过了 channelattention，没有通过卷积做精化。
    修复：对 outfeature1~3 分别应用对应的 conv4/conv3/conv2(144→144)。

    加权策略：BiFPN relu-归一化
    ──────────────────────────
    w = relu(weights) / (sum(relu(weights)) + eps)
    · relu 保证非负（不用 softmax，避免极端情况下的 winner-takes-all）
    · 归一化保证各路贡献总和为 1，防止量级爆炸
    · 初始化为全 1，等价于初始等权，通过训练自适应

    维度一致性：
    · outfeature1 (144ch, res4 分辨率) → conv4(144→144)
    · outfeature2 (144ch, res3 分辨率) → conv3(144→144)
    · outfeature3 (144ch, res2 分辨率) → conv2(144→144)
    · outfeature4 (144ch, res1 分辨率) → conv1(144→18)  ← 与原版一致
    """

    def __init__(self, eps: float = 1e-8):
        super(WeightedCAB, self).__init__()
        channels  = 144
        self.eps  = eps

        # Bug③ 修复：权重参数扩展为 4 路（原版错误地定义为 3 路）
        # w[i] 代表第 i 路（res1/res2/res3/res4）的贡献权重
        self.weights0 = nn.Parameter(torch.ones(4, dtype=torch.float32),
                                      requires_grad=True)
        self.weights1 = nn.Parameter(torch.ones(4, dtype=torch.float32),
                                      requires_grad=True)
        self.weights2 = nn.Parameter(torch.ones(4, dtype=torch.float32),
                                      requires_grad=True)
        self.weights3 = nn.Parameter(torch.ones(4, dtype=torch.float32),
                                      requires_grad=True)

        # 各分辨率通道注意力精化（保留原版）
        self.attention0 = channelattention(channels)
        self.attention1 = channelattention(channels)
        self.attention2 = channelattention(channels)
        self.attention3 = channelattention(channels)

        # Bug④ 修复：conv4/conv3/conv2 正式接入 forward
        self.conv4 = ConvBNReLU(channels, channels, 1)  # outfeature1（最低分辨率）
        self.conv3 = ConvBNReLU(channels, channels, 1)  # outfeature2
        self.conv2 = ConvBNReLU(channels, channels, 1)  # outfeature3
        self.conv1 = ConvBNReLU(channels, 18, 1)        # outfeature4（18ch，与原版一致）

    # ── BiFPN 风格归一化加权融合 ─────────────────────────────────
    def _bfpn_fuse(self, feats: list, weights: torch.Tensor) -> torch.Tensor:
        """
        feats:   list of 4 tensors, 已对齐到同一分辨率
        weights: Parameter [4]
        """
        w      = F.relu(weights)              # 非负化（relu 比 softmax 更平滑）
        w_norm = w / (w.sum() + self.eps)     # 归一化：Σw ≈ 1
        return sum(w_norm[i] * feats[i] for i in range(4))

    def _align(self, feats: list, h: int, w: int) -> list:
        """将所有特征双线性插值到目标 (h, w)"""
        return [F.interpolate(f, size=(h, w), mode='bilinear', align_corners=False)
                for f in feats]

    def forward(self, res1, res2, res3, res4):
        """
        输入：res1~res4 均为 [B, 144, H_i, W_i]，分辨率从高到低
        输出顺序与原版 CAB 完全一致：
            outfeature1 (144ch, 最低分辨率 res4)
            outfeature2 (144ch, 次低分辨率 res3)
            outfeature3 (144ch, 次高分辨率 res2)
            outfeature4 ( 18ch, 最高分辨率 res1)
        """
        feats = [res1, res2, res3, res4]

        # outfeature1：对齐到 res4 分辨率（最低）
        h0, w0 = res4.shape[-2:]
        out1   = self.conv4(                          # Bug④ 修复：现在真正用 conv4
                     self.attention0(
                         self._bfpn_fuse(             # Bug③ 修复：用 weights0 加权
                             self._align(feats, h0, w0),
                             self.weights0
                         )
                     )
                 )

        # outfeature2：对齐到 res3 分辨率
        h1, w1 = res3.shape[-2:]
        out2   = self.conv3(                          # Bug④ 修复：现在真正用 conv3
                     self.attention1(
                         self._bfpn_fuse(
                             self._align(feats, h1, w1),
                             self.weights1
                         )
                     )
                 )

        # outfeature3：对齐到 res2 分辨率
        h2, w2 = res2.shape[-2:]
        out3   = self.conv2(                          # Bug④ 修复：现在真正用 conv2
                     self.attention2(
                         self._bfpn_fuse(
                             self._align(feats, h2, w2),
                             self.weights2
                         )
                     )
                 )

        # outfeature4：对齐到 res1 分辨率（最高），输出 18ch
        h3, w3 = res1.shape[-2:]
        out4   = self.conv1(                          # 原版 conv1(144→18) 保留
                     self.attention3(
                         self._bfpn_fuse(
                             self._align(feats, h3, w3),
                             self.weights3
                         )
                     )
                 )

        return out1, out2, out3, out4


# ══════════════════════════════════════════════════════════════════
#  Segmentation_Head（原版保留）
# ══════════════════════════════════════════════════════════════════

class Segmentation_Head(nn.Module):
    def __init__(self, dim, atten_drop=0., proj_drop=0., dilation=None,
                 fc_ratio=4, pool_ratio=16, dropout=0., num_classes=6):
        super(Segmentation_Head, self).__init__()
        if dilation is None:
            dilation = [3, 5, 7]
        self.dim        = dim
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop  = nn.Dropout(proj_drop)
        self.MSC        = ASPPModule(dim, dilation)
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(dim // fc_ratio, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.spat_atten = SpatialAttention()
        self.kv   = Conv(dim, 2 * dim, 1)
        self.head = nn.Sequential(
            SeparableConvBNReLU(dim, dim, kernel_size=3),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(18, num_classes, kernel_size=1)
        )

    def forward(self, x):
        u      = x.clone()
        B, C, H, W = x.shape
        attn   = self.MSC(x)
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn) * u
        s_attn = self.spat_atten(x)
        return self.head(attn + c_attn + s_attn)


# ══════════════════════════════════════════════════════════════════
#  Decoder（接口与原版完全一致，使用 WeightedCAB 替换 CAB）
# ══════════════════════════════════════════════════════════════════

class Decoder(nn.Module):
    def __init__(self,
                 encode_channels=None,
                 decode_channels: int = 128,
                 dilation=None,
                 fc_ratio: int = 4,
                 dropout: float = 0.1,
                 num_classes: int = 6):
        super(Decoder, self).__init__()
        if encode_channels is None:
            encode_channels = [32, 64, 128, 256]
        if dilation is None:
            dilation = [[6, 12, 18]] * 4

        self.Conv1 = ConvBNReLU(encode_channels[-2], decode_channels, 1)
        self.Conv2 = ConvBNReLU(encode_channels[-3], decode_channels, 1)
        self.Conv4 = ConvBNReLU(encode_channels[-4], decode_channels, 1)

        self.b4 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4,
                         pool_ratio=16, dilation=dilation[0])
        self.p3 = Fusion(decode_channels)
        self.b3 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4,
                         pool_ratio=16, dilation=dilation[1])
        self.p2 = Fusion(decode_channels)
        self.b2 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4,
                         pool_ratio=16, dilation=dilation[2])

        self.Conv3    = ConvBN(decode_channels, encode_channels[-4], 1)
        self.p1       = Fusion(encode_channels[-4])
        self.seg_head = Segmentation_Head(
            encode_channels[-4], fc_ratio=fc_ratio,
            dilation=dilation[3], dropout=dropout, num_classes=num_classes
        )

        # 使用修复了两个 Bug 的 WeightedCAB
        self.CAB = WeightedCAB()
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h: int, w: int):
        res3 = self.Conv1(res3)
        res2 = self.Conv2(res2)
        res1 = self.Conv4(res1)

        outfeature1, outfeature2, outfeature3, outfeature4 = \
            self.CAB(res1, res2, res3, res4)

        decoder_feature = []

        x = self.b4(outfeature1);             decoder_feature.append(x)
        x = self.p3(x, outfeature2)
        x = self.b3(x);                       decoder_feature.append(x)
        x = self.p2(x, outfeature3)
        x = self.b2(x);                       decoder_feature.append(x)
        x = self.Conv3(x)
        x = self.p1(x, outfeature4);          decoder_feature.append(x)
        x = self.seg_head(x)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ══════════════════════════════════════════════════════════════════
#  Bo_Mamba（顶层模型，接口与原版完全一致，直接替换即可）
# ══════════════════════════════════════════════════════════════════

class Bo_Mamba(nn.Module):
    def __init__(self,
                 encode_channels=None,
                 decode_channels: int = 144,
                 dropout: float = 0.1,
                 num_classes: int = 6):
        super().__init__()
        if encode_channels is None:
            encode_channels = [18, 36, 72, 144]
        self.backbone = get_seg_model(config)
        self.decoder  = Decoder(encode_channels, decode_channels,
                                  dropout=dropout, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        return self.decoder(res1, res2, res3, res4, h, w)


# ══════════════════════════════════════════════════════════════════
#  快速验证
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from thop import profile, clever_format

    DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {DEVICE}')

    x   = torch.randn(1, 3, 512, 512).to(DEVICE)
    net = Bo_Mamba().to(DEVICE)

    with torch.no_grad():
        out = net(x)
    print(f'Output shape: {out.shape}')

    flops, params = profile(net, (x,), verbose=False)
    flops_s, params_s = clever_format([flops, params], '%.3f')
    print(f'运算量：{flops_s}，参数量：{params_s}')
