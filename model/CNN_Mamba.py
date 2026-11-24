import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from timm.models.layers import DropPath
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

# 假设原文件中所有基础模块已定义，我们直接复用
# 这里为了代码完整性，将需要的基础模块再次导入（实际运行时只需确保它们已定义）
# 原代码中的类：ConvBNReLU, ConvBN, Conv, SeparableConvBNReLU, SeparableConvBN,
# SeparableConv, DepthWiseConv, SpatialAttention, channelattention,
# depthwise_separable_conv, _DenseASPPConv, ASPPModule, CSDF,
# FreqAwareDWConv, CrossGuidedMambaDWConv, Fusion, CAB (需修复), Segmentation_Head

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

#  注意力与 ASPP 模块（已改版）
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
        # 三个方向子带沿通道维堆叠为 C*3，groups=C*3 保持各通道独立处理
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
        # yh[0]: [B, C, 3, Hh, Wh]   三个高频子带（LH / HL / HH）
        _, _, Hh, Wh = yl.shape
 
        # ── LL 子带精化（大核） ────────────────────────────────────
        yl_out = self.ll_dw(yl)                               # [B, C, Hh, Wh]
 
        # ── HF 子带精化（小核） ────────────────────────────────────
        # 第一步：显式提取三个高频子带，各自形状 [B, C, Hh, Wh]
        hf0 = yh[0][:, :, 0, :, :]                           # LH 子带 [B, C, Hh, Wh]
        hf1 = yh[0][:, :, 1, :, :]                           # HL 子带 [B, C, Hh, Wh]
        hf2 = yh[0][:, :, 2, :, :]                           # HH 子带 [B, C, Hh, Wh]
 
        # 第二步：沿通道维堆叠为 [B, 3C, Hh, Wh]
        # cat 保证子带在通道维连续对齐：前 C 通道=LH，中 C=HL，后 C=HH
        hf_stack = torch.cat([hf0, hf1, hf2], dim=1)         # [B, 3C, Hh, Wh]
 
        # 第三步：整体送入 3×3 深度可分卷积处理
        hf_proc = self.hf_dw(hf_stack)                       # [B, 3C, Hh, Wh]
 
        # 第四步：split 还原为三个对齐的高频子带，各自 [B, C, Hh, Wh]
        hf0_out, hf1_out, hf2_out = torch.split(hf_proc, C, dim=1)
 
        # 第五步：stack 重组为 IDWT 所需的 5D 格式 [B, C, 3, Hh, Wh]
        hf_out = torch.stack([hf0_out, hf1_out, hf2_out], dim=2)  # [B, C, 3, Hh, Wh]
 
        # ── 子带门控：学习 LL vs HF 的空间权重分配 ─────────────────
        # 以 yl_out 为输入：基于增强后的低频计算门控，更稳定
        weights  = self.band_gate(yl_out)                     # [B, 2, 1, 1]
        w_ll     = weights[:, 0:1, :, :]                      # [B, 1, 1, 1]
        w_hf     = weights[:, 1:2, :, :]                      # [B, 1, 1, 1]
 
        # 加权 LL 子带
        yl_weighted = yl_out * w_ll                           # [B, C, Hh, Wh]
 
        # 加权 HF 子带（unsqueeze 使 w_hf 从 [B,1,1,1] 广播到 [B,C,3,Hh,Wh]）
        hf_weighted = hf_out * w_hf.unsqueeze(2)              # [B, C, 3, Hh, Wh]
 
        # ── IDWT 重建 ──────────────────────────────────────────────
        x_rec = self.idwt((yl_weighted, [hf_weighted]))       # [B, C, H', W']
 
        # 处理 reflect 模式可能的 ±1 尺寸误差
        if x_rec.shape[-2] != H or x_rec.shape[-1] != W:
            x_rec = F.interpolate(x_rec, size=(H, W),
                                  mode='bilinear', align_corners=False)
 
        # ── 残差连接 ──────────────────────────────────────────────
        return x_rec + self.res_align(x)

class CrossGuidedMambaDWConv(nn.Module):
    """
    Mamba ‖ DWConv 并行模块。

    两路从同一 x_proj 出发，独立处理后直接相加融合：
        Mamba → x_m（全局序列上下文）
        FreqAwareDWConv → x_d（局部纹理+频域感知）
        x_fused = x_m + x_d → fc2 → act
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

        # ── 直接相加融合 ──────────────────────────────────────────
        x_fused = x_m + x_d                                  # [B, hidden, H, W]

        # ── 输出投影 ──────────────────────────────────────────────
        return self.act(self.drop(self.fc2(x_fused)))

# 修复 CAB 类，使其真正使用 weights 和 conv2/3/4
class FixedCAB(nn.Module):
    """修复后的 CAB，使用可学习权重进行加权融合，并对每个输出使用通道注意力和卷积精化"""
    def __init__(self, eps=1e-8):
        super().__init__()
        channels = 144  # 假设统一通道数，实际应可配置
        self.eps = eps

        # 每个尺度的融合权重 (BiFPN 风格，3 个输入源)
        self.weights0 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.attention0 = channelattention(channels)
        self.conv4 = ConvBNReLU(channels, channels, 1)

        self.weights1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.attention1 = channelattention(channels)
        self.conv3 = ConvBNReLU(channels, channels, 1)

        self.weights2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.attention2 = channelattention(channels)
        self.conv2 = ConvBNReLU(channels, channels, 1)

        self.weights3 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.attention3 = channelattention(channels)
        self.conv1 = ConvBNReLU(channels, channels, 1)

    def forward(self, res1, res2, res3, res4):
        h, w = res4.size()[-2:]

        # 第4层 (最小尺度)
        feat1_4 = F.interpolate(res1, size=(h, w), mode='bilinear')
        feat2_4 = F.interpolate(res2, size=(h, w), mode='bilinear')
        feat3_4 = F.interpolate(res3, size=(h, w), mode='bilinear')
        # 加权融合 (3 个源: res1, res2, res3)
        w0 = F.relu6(self.weights0)
        w0 = w0 / (torch.sum(w0) + self.eps)
        out1 = w0[0] * feat1_4 + w0[1] * feat2_4 + w0[2] * feat3_4 + res4
        out1 = self.attention0(out1)
        out1 = self.conv4(out1)  # 精化

        # 第3层
        h, w = res3.size()[-2:]
        feat1_3 = F.interpolate(res1, size=(h, w), mode='bilinear')
        feat2_3 = F.interpolate(res2, size=(h, w), mode='bilinear')
        feat4_3 = F.interpolate(res4, size=(h, w), mode='bilinear')
        w1 = F.relu6(self.weights1)
        w1 = w1 / (torch.sum(w1) + self.eps)
        out2 = w1[0] * feat1_3 + w1[1] * feat2_3 + w1[2] * feat4_3 + res3
        out2 = self.attention1(out2)
        out2 = self.conv3(out2)

        # 第2层
        h, w = res2.size()[-2:]
        feat1_2 = F.interpolate(res1, size=(h, w), mode='bilinear')
        feat3_2 = F.interpolate(res3, size=(h, w), mode='bilinear')
        feat4_2 = F.interpolate(res4, size=(h, w), mode='bilinear')
        w2 = F.relu6(self.weights2)
        w2 = w2 / (torch.sum(w2) + self.eps)
        out3 = w2[0] * feat1_2 + w2[1] * res2 + w2[2] * feat3_2 + feat4_2
        out3 = self.attention2(out3)
        out3 = self.conv2(out3)

        # 第1层 (最大尺度)
        h, w = res1.size()[-2:]
        feat2_1 = F.interpolate(res2, size=(h, w), mode='bilinear')
        feat3_1 = F.interpolate(res3, size=(h, w), mode='bilinear')
        feat4_1 = F.interpolate(res4, size=(h, w), mode='bilinear')
        w3 = F.relu6(self.weights3)
        w3 = w3 / (torch.sum(w3) + self.eps)
        out4 = w3[0] * res1 + w3[1] * feat2_1 + w3[2] * feat3_1 + feat4_1
        out4 = self.attention3(out4)
        out4 = self.conv1(out4)

        return out1, out2, out3, out4


class CNNEncoder(nn.Module):
    """简单的 CNN 编码器，四个阶段，通道数依次翻倍，分辨率减半"""
    def __init__(self, in_ch=3, base_ch=32, num_stages=4):
        super().__init__()
        self.stages = nn.ModuleList()
        ch = base_ch
        for i in range(num_stages):
            # 每个阶段：一个步长2的卷积下采样 + 一个步长1的卷积（可选）
            layers = [
                ConvBNReLU(in_ch, ch, kernel_size=3, stride=2),
                ConvBNReLU(ch, ch, kernel_size=3, stride=1)
            ]
            self.stages.append(nn.Sequential(*layers))
            in_ch = ch
            ch *= 2

    def forward(self, x):
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return outs  # 四个尺度的特征


class MambaDWEncoder(nn.Module):
    """Mamba + DWConv 并行编码器，四个阶段，通道数翻倍，分辨率减半"""
    def __init__(self, in_ch=3, base_ch=32, num_stages=4, mamba_kwargs=None):
        super().__init__()
        self.stages = nn.ModuleList()
        ch = base_ch
        for i in range(num_stages):
            # 每个阶段：下采样卷积 + CrossGuidedMambaDWConv
            downsample = ConvBNReLU(in_ch, ch, kernel_size=3, stride=2)
            mamba_block = CrossGuidedMambaDWConv(
                in_features=ch,
                hidden_features=ch,   # 保持通道数不变
                out_features=ch,
                dim=ch,
                atten_drop=0.0
            )
            self.stages.append(nn.ModuleList([downsample, mamba_block]))
            in_ch = ch
            ch *= 2

    def forward(self, x):
        outs = []
        for downsample, mamba_block in self.stages:
            x = downsample(x)
            x = mamba_block(x)
            outs.append(x)
        return outs

class Fusion(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(Fusion, self).__init__()
        self.dim = dim
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = SeparableConvBNReLU(dim, dim, 5)
        self.upsample = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)

    def forward(self, a, b):
        """
        自动对齐 a 和 b 的分辨率，将较小分辨率上采样至较大分辨率，然后加权融合。
        """
        # 获取目标尺寸（取较大）
        target_h = max(a.shape[-2], b.shape[-2])
        target_w = max(a.shape[-1], b.shape[-1])

        # 如果尺寸不一致，进行上采样
        if a.shape[-2] != target_h or a.shape[-1] != target_w:
            # 使用双线性插值上采样，更稳定
            a = F.interpolate(a, size=(target_h, target_w), mode='bilinear', align_corners=False)
        if b.shape[-2] != target_h or b.shape[-1] != target_w:
            b = F.interpolate(b, size=(target_h, target_w), mode='bilinear', align_corners=False)

        # 可学习加权融合
        weights = F.relu6(self.weights)
        fuse_weights = weights / (torch.sum(weights) + self.eps)
        out = fuse_weights[0] * b + fuse_weights[1] * a
        return self.post_conv(out)


class DualEncoderFusionNet(nn.Module):
    """
    双编码器架构：
    - CNN 编码器 → CAB 精化
    - Mamba-DWconv 编码器
    - 对应阶段通过 Fusion 融合
    - 每个融合特征分别上采样后通过独立头，再相加得到最终分割图
    """
    def __init__(self, in_ch=3, base_ch=32, num_stages=4, num_classes=6):
        super().__init__()
        self.cnn_encoder = CNNEncoder(in_ch, base_ch, num_stages)
        self.mamba_encoder = MambaDWEncoder(in_ch, base_ch, num_stages)

        # 投影到统一通道数（例如 144）
        self.proj_cnn = nn.ModuleList()
        self.proj_mamba = nn.ModuleList()
        ch = base_ch
        for i in range(num_stages):
            self.proj_cnn.append(ConvBNReLU(ch, 144, 1))
            self.proj_mamba.append(ConvBNReLU(ch, 144, 1))
            ch *= 2

        # CAB (使用修复版)
        self.cab = FixedCAB()

        # 对应阶段融合模块
        self.fusions = nn.ModuleList([Fusion(144) for _ in range(num_stages)])

        # 四个独立的轻量分割头，每个输入144通道，输出 num_classes
        self.heads = nn.ModuleList([
            nn.Sequential(
                ConvBNReLU(144, 144, kernel_size=3),
                nn.Dropout2d(0.1),
                nn.Conv2d(144, num_classes, kernel_size=1)
            ) for _ in range(num_stages)
        ])

    def forward(self, x):
        B, _, H, W = x.shape

        # CNN 编码器输出
        cnn_feats = self.cnn_encoder(x)
        cnn_feats = [proj(f) for proj, f in zip(self.proj_cnn, cnn_feats)]

        # Mamba-DWconv 编码器输出
        mamba_feats = self.mamba_encoder(x)
        mamba_feats = [proj(f) for proj, f in zip(self.proj_mamba, mamba_feats)]

        # CAB 精化 CNN 特征（四个尺度）
        c1, c2, c3, c4 = self.cab(*cnn_feats)   # 顺序：从小到大（c4 最小）

        # 对应阶段融合
        fused_feats = []
        for i, (c, m) in enumerate(zip([c1, c2, c3, c4], mamba_feats)):
            f = self.fusions[i](m, c)            # m 是较小分辨率，c 是较大分辨率
            fused_feats.append(f)

        # 对每个融合特征分别上采样并通过独立 head，然后累加
        out = None
        for i, f in enumerate(fused_feats):
            if f.shape[-2] != H or f.shape[-1] != W:
                f = F.interpolate(f, size=(H, W), mode='bilinear', align_corners=False)
            pred = self.heads[i](f)
            if out is None:
                out = pred
            else:
                out = out + pred

        return out


# 测试
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 3, 512, 512).to(device)
    model = DualEncoderFusionNet(in_ch=3, base_ch=32, num_stages=4, num_classes=6).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")  # 预期 [1, 6, 512, 512]
    from thop import profile, clever_format
    flops, params = profile(model, (x,), verbose=False)
    flops_s, params_s = clever_format([flops, params], '%.3f')
    print(f'FLOPs: {flops_s}, Params: {params_s}')
