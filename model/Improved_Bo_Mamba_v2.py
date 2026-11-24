"""
Improved_Bo_Mamba_v2.py
============================================================
主题架构：Mamba ‖ Multi-scale DWConv 并行

六项核心改动（相对原始 Bo_Mamba.py）：
 ① WaveletSubBandModule    — LL/HF 分子带差异化处理，替换无效直通
 ② MultiScaleDWConvBranch  — DWConv 分支三路（k=3/k=5/wav）可学习权重融合
 ③ ParallelMambaConvBlock  — 替换 GLSS；真正并行 + 自适应空间门控
 ④ SEBlock                 — 替换 nn.Identity()，激活 norm1 分支
 ⑤ ImprovedBlock           — 替换 Block；norm1→SE + norm2→ParallelMamba
 ⑥ BiFPNCAB               — 替换 CAB；BiFPN 可学习归一化跨尺度权重
============================================================
"""

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
except Exception:
    pass
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except Exception:
    pass


# =========================================================
#  基础卷积模块（保持原始不变）
# =========================================================

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, norm_layer=nn.BatchNorm2d, groups=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=groups),
            norm_layer(out_channels), nn.ReLU6())

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels))

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2))

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, norm_layer=nn.BatchNorm2d):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride-1)+dilation*(kernel_size-1))//2, groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels), nn.ReLU6())

class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, norm_layer=nn.BatchNorm2d):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride-1)+dilation*(kernel_size-1))//2, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels))

class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride-1)+dilation*(kernel_size-1))//2, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))

class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_channel, in_channel, kernel_size=kernel,
                                    stride=1, padding=(kernel-1)//2, groups=in_channel)
        self.point_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


# =========================================================
#  注意力辅助（保持原始，供 CSDF/Seg_Head 使用）
# =========================================================

class channelattention(nn.Module):
    def __init__(self, dim, fc_ratio=4):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim//fc_ratio, 1), nn.ReLU6(),
            nn.Conv2d(dim//fc_ratio, dim, 1), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(self.avgpool(x))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1   = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.relu1   = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out    = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.sigmoid(self.relu1(self.conv1(torch.cat([avg_out, max_out], dim=1))))
        return x * out


# =========================================================
#  Dense ASPP（保持原始，供 CSDF/Seg_Head 使用）
# =========================================================

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, dilation):
        super().__init__()
        self.depthwise = nn.Conv2d(nin, nin, 3, dilation=dilation,
                                   padding=(dilation*2)//2, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, 1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super().__init__()
        kw = {} if norm_kwargs is None else norm_kwargs
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1))
        self.add_module('bn1',   norm_layer(inter_channels, **kw))
        self.add_module('relu1', nn.ReLU(True))
        self.add_module('conv2', depthwise_separable_conv(inter_channels, out_channels, atrous_rate))
        self.add_module('bn2',   norm_layer(out_channels, **kw))
        self.add_module('relu2', nn.ReLU(True))
        self.drop_rate = drop_rate
    def forward(self, x):
        feat = super().forward(x)
        if self.drop_rate > 0:
            feat = F.dropout(feat, p=self.drop_rate, training=self.training)
        return feat

class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super().__init__()
        oc = in_channels
        r1, r2, r3 = atrous_rates
        self.b0 = _DenseASPPConv(in_channels,           oc//2,                   oc, r1, 0.1)
        self.b1 = _DenseASPPConv(in_channels+oc,        (in_channels+oc)//2,     oc, r2, 0.1)
        self.b2 = _DenseASPPConv(in_channels+oc*2,      (in_channels+oc*2)//2,   oc, r3, 0.1)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels+3*oc, oc, 1, bias=False),
            nn.BatchNorm2d(oc), nn.ReLU(True))
    def forward(self, x):
        u = x.clone()
        a1 = self.b0(x); x = torch.cat([a1, x], 1)
        a2 = self.b1(x); x = torch.cat([a2, x], 1)
        a3 = self.b2(x); x = torch.cat([a3, x], 1)
        return self.project(x) * u


# =========================================================
#  CSDF（原始保留，未接入主流程；可选 Decoder 精化见注释）
# =========================================================

class CSDF(nn.Module):
    def __init__(self, dim, num_heads, atten_drop=0., proj_drop=0.,
                 dilation=None, fc_ratio=4):
        super().__init__()
        if dilation is None: dilation = [3, 5, 7]
        assert dim % num_heads == 0
        self.DMSC       = ASPPModule(dim, dilation)
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.fc         = nn.Sequential(
            nn.Conv2d(dim, dim//fc_ratio, 1), nn.ReLU6(),
            nn.Conv2d(dim//fc_ratio, dim, 1), nn.Sigmoid())
        self.spat_atten = SpatialAttention()
        self.fc2        = ConvBN(dim, dim, 1)
        self.act        = nn.ReLU6()
    def forward(self, x):
        u=x.clone(); a=self.DMSC(x)
        c_a=self.fc(self.avgpool(x))*u; s_a=self.spat_atten(x)
        return self.act(self.fc2(a+c_a+s_a))


# =========================================================
#  改进① WaveletSubBandModule
#  原问题：DWT→IDWT 直通，近于恒等变换，频域优势完全浪费
#  改进：LL（低频·语义）k=5 DWConv + HF（边缘）跨子带 DWConv
#        groups=channels 使同一原始通道的 LH/HL/HH 共享滤波
# =========================================================

class WaveletSubBandModule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dwt  = DWTForward(J=1, wave='haar', mode='zero')
        self.idwt = DWTInverse(wave='haar', mode='zero')

        # LL：语义低频 — k=5 DWConv + 逐点混合 + BN + ReLU
        self.ll_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

        # HF：边缘纹理 — 3 子带合并为 3×C 通道
        # groups=channels：每组 3 个通道 = 同一原始通道的 LH+HL+HH
        # 学习跨方向子带协同（水平↔垂直↔对角线边缘）
        self.hf_conv = nn.Sequential(
            nn.Conv2d(channels*3, channels*3, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels*3, channels*3, 1, bias=False),
            nn.BatchNorm2d(channels*3), nn.ReLU(inplace=True))

        # IDWT 重建后精化
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        yl, yh = self.dwt(x)              # yl:(B,C,H/2,W/2)  yh[0]:(B,C,3,H/2,W/2)
        _, _, Hh, Wh = yl.shape

        # LL 增强
        yl_out = self.ll_conv(yl)         # (B,C,H/2,W/2)

        # HF 增强
        hf     = yh[0].reshape(B, C*3, Hh, Wh)       # (B,3C,H/2,W/2)
        hf_out = self.hf_conv(hf).reshape(B, C, 3, Hh, Wh)  # (B,C,3,H/2,W/2)

        # 重建 + 残差
        x_rec  = self.idwt((yl_out, [hf_out]))        # (B,C,H,W)
        return self.out_proj(x_rec) + x


# =========================================================
#  改进② MultiScaleDWConvBranch
#  三路互补：k=3 精细 + k=5 中等 + WaveletSubBand 频域
#  BiFPN 风格可学习归一化权重，替代等权相加
# =========================================================

class MultiScaleDWConvBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_k3      = DepthWiseConv(channels, channels, kernel=3)
        self.conv_k5      = DepthWiseConv(channels, channels, kernel=5)
        self.wavelet      = WaveletSubBandModule(channels)
        self.path_weights = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.fuse_bn      = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k3  = self.conv_k3(x)
        k5  = self.conv_k5(x)
        wav = self.wavelet(x)
        w   = F.relu6(self.path_weights)
        w   = w / (w.sum() + 1e-8)
        return self.fuse_bn(w[0]*k3 + w[1]*k5 + w[2]*wav)


# =========================================================
#  改进③ ParallelMambaConvBlock — 替换 GLSS
#
#  原 GLSS 问题：
#  (A) 伪并行：y=wavelet(fc1(x))≈fc1(x)；Mamba(x) 与 y 从不同点出发
#  (B) 无门控：x1+x2+x3 等权，全局/局部贡献比无法自适应
#
#  改进：真正并行 + 自适应空间门控
#    fc1(x) ──┬── Mamba SS2D → x_m  （全局序列上下文）
#             └── MultiScaleDWConvBranch → x_d  （局部+频域）
#    gate = σ(Conv₁ₓ₁_BN(cat(x_m, x_d)))  → (B,hidden,H,W)
#    fused = gate·x_m + (1−gate)·x_d → fc2 → act
#
#  gate 空间自适应含义：
#    边界区域：DWConv 贡献大（局部细节精确定位）
#    同质区域：Mamba 贡献大（全局上下文避免误分）
# =========================================================

class ParallelMambaConvBlock(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, act_layer=nn.ReLU6,
                 drop: float = 0., atten_drop: float = 0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features    = out_features    or in_features

        self.fc1          = ConvBNReLU(in_features, hidden_features, 1)
        self.mamba        = SS2D(d_model=hidden_features, dropout=atten_drop, d_state=16, scan=8)
        self.mamba_bn     = nn.BatchNorm2d(hidden_features)
        self.dwconv_branch = MultiScaleDWConvBranch(hidden_features)
        self.gate         = nn.Sequential(
            nn.Conv2d(hidden_features*2, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features), nn.Sigmoid())
        self.fc2  = ConvBN(hidden_features, out_features, 1)
        self.act  = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x     = self.fc1(x)                                    # 共享投影
        x_m   = self.mamba_bn(self.mamba(x))                   # 全局分支
        x_d   = self.dwconv_branch(x)                          # 局部+频域分支
        gate  = self.gate(torch.cat([x_m, x_d], dim=1))        # 自适应门控
        fused = gate * x_m + (1.0 - gate) * x_d
        return self.act(self.drop(self.fc2(fused)))


# =========================================================
#  改进④ SEBlock — 替换 nn.Identity()
#  激活 norm1 分支，SE 通道注意力（avgpool→fc→sigmoid→缩放）
# =========================================================

class SEBlock(nn.Module):
    def __init__(self, dim: int, ratio: int = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite  = nn.Sequential(
            nn.Conv2d(dim, dim//ratio, 1, bias=False), nn.ReLU6(inplace=True),
            nn.Conv2d(dim//ratio, dim, 1, bias=False), nn.Sigmoid())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.excite(self.squeeze(x))


# =========================================================
#  改进⑤ ImprovedBlock — 替换 Block
#  原：attn=Identity（norm1 浪费）+ mlp=GLSS（有缺陷）
#  改：norm1→SEBlock + norm2→ParallelMambaConvBlock + 外层残差
# =========================================================

class ImprovedBlock(nn.Module):
    def __init__(self, dim: int = 144, num_heads: int = 16,
                 mlp_ratio: int = 4, pool_ratio: int = 16,
                 drop: float = 0., dilation=None, drop_path: float = 0.,
                 act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # num_heads/pool_ratio/dilation 仅保留签名，兼容原 Decoder 调用
        self.norm1     = norm_layer(dim)
        self.norm2     = norm_layer(dim)
        self.attn      = SEBlock(dim, ratio=4)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp       = ParallelMambaConvBlock(
            in_features=dim, hidden_features=int(dim//mlp_ratio),
            out_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.clone()
        x = x + self.drop_path(self.attn(self.norm1(x)))   # SE 通道注意力
        x = x + self.drop_path(self.mlp(self.norm2(x)))    # Mamba||DWConv
        return x + u                                        # 外层残差


# =========================================================
#  Fusion（保持原始不变）
# =========================================================

class Fusion(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.weights   = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.eps       = eps
        self.post_conv = SeparableConvBNReLU(dim, dim, 5)
        self.upsample  = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
    def forward(self, x, res):
        if x.shape[2] != res.shape[2]: x = self.upsample(x)
        w = F.relu6(self.weights); w = w / (w.sum() + self.eps)
        return self.post_conv(w[0]*res + w[1]*x)


# =========================================================
#  改进⑥ BiFPNCAB — 替换 CAB（等权 f1+f2+f3+f4）
#  每输出分辨率独立学习 4 个融合权重（BiFPN ReLU6/sum 归一化）
# =========================================================

class BiFPNCAB(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        C = 144; self.eps = eps
        self.w0 = nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.w1 = nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.w3 = nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.attn0 = channelattention(C); self.attn1 = channelattention(C)
        self.attn2 = channelattention(C); self.attn3 = channelattention(C)
        self.conv0 = ConvBNReLU(C, C, 1); self.conv1 = ConvBNReLU(C, C, 1)
        self.conv2 = ConvBNReLU(C, C, 1); self.conv3 = ConvBNReLU(C, 18, 1)

    def _wfuse(self, fs, w):
        w = F.relu6(w); w = w/(w.sum()+self.eps)
        return sum(w[i]*fs[i] for i in range(4))

    def forward(self, res1, res2, res3, res4):
        fs = [res1, res2, res3, res4]
        def _align(ref): return [F.interpolate(f, ref.shape[-2:], mode='bilinear', align_corners=False) for f in fs]
        out0 = self.conv0(self.attn0(self._wfuse(_align(res4), self.w0)))
        out1 = self.conv1(self.attn1(self._wfuse(_align(res3), self.w1)))
        out2 = self.conv2(self.attn2(self._wfuse(_align(res2), self.w2)))
        out3 = self.conv3(self.attn3(self._wfuse(_align(res1), self.w3)))
        return out0, out1, out2, out3


# =========================================================
#  Segmentation_Head（保持原始不变）
# =========================================================

class Segmentation_Head(nn.Module):
    def __init__(self, dim, atten_drop=0., proj_drop=0., dilation=None,
                 fc_ratio=4, pool_ratio=16, dropout=0., num_classes=6):
        super().__init__()
        if dilation is None: dilation = [3, 5, 7]
        self.MSC     = ASPPModule(dim, dilation)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Sequential(
            nn.Conv2d(dim, dim//fc_ratio, 1), nn.ReLU6(),
            nn.Conv2d(dim//fc_ratio, dim, 1), nn.Sigmoid())
        self.spat_atten = SpatialAttention()
        self.head       = nn.Sequential(
            SeparableConvBNReLU(dim, dim, kernel_size=3),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(18, num_classes, kernel_size=1))
    def forward(self, x):
        u=x.clone(); a=self.MSC(x)
        c_a=self.fc(self.avgpool(x))*u; s_a=self.spat_atten(x)
        return self.head(a+c_a+s_a)


# =========================================================
#  Decoder（ImprovedBlock + BiFPNCAB，结构与原始相同）
# =========================================================

class Decoder(nn.Module):
    def __init__(self, encode_channels=None, decode_channels=128,
                 dilation=None, fc_ratio=4, dropout=0.1, num_classes=6):
        super().__init__()
        if encode_channels is None: encode_channels = [32, 64, 128, 256]
        if dilation is None:        dilation = [[6,12,18]]*4

        self.Conv1 = ConvBNReLU(encode_channels[-2], decode_channels, 1)
        self.Conv2 = ConvBNReLU(encode_channels[-3], decode_channels, 1)
        self.Conv4 = ConvBNReLU(encode_channels[-4], decode_channels, 1)

        self.b4 = ImprovedBlock(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[0])
        self.p3 = Fusion(decode_channels)
        self.b3 = ImprovedBlock(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[1])
        self.p2 = Fusion(decode_channels)
        self.b2 = ImprovedBlock(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[2])

        self.Conv3    = ConvBN(decode_channels, encode_channels[-4], 1)
        self.p1       = Fusion(encode_channels[-4])
        self.seg_head = Segmentation_Head(encode_channels[-4], fc_ratio=fc_ratio,
                                           dilation=dilation[3], dropout=dropout, num_classes=num_classes)
        self.CAB      = BiFPNCAB()

        # -- 可选 CSDF 精化（大数据集解注释）--
        # self.csdf1 = CSDF(dim=decode_channels, num_heads=9)
        # self.csdf2 = CSDF(dim=decode_channels, num_heads=9)
        # self.csdf3 = CSDF(dim=decode_channels, num_heads=9)
        # self.csdf4 = CSDF(dim=decode_channels, num_heads=9)
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        res3 = self.Conv1(res3); res2 = self.Conv2(res2); res1 = self.Conv4(res1)

        # res1 = self.csdf1(res1)   # -- 可选 CSDF --
        # res2 = self.csdf2(res2)
        # res3 = self.csdf3(res3)
        # res4 = self.csdf4(res4)

        o1, o2, o3, o4 = self.CAB(res1, res2, res3, res4)
        x = self.b4(o1)
        x = self.p3(x, o2); x = self.b3(x)
        x = self.p2(x, o3); x = self.b2(x)
        x = self.Conv3(x);  x = self.p1(x, o4)
        x = self.seg_head(x)
        return F.interpolate(x, (h, w), mode='bilinear', align_corners=False)

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)


# =========================================================
#  顶层模型 Improved_Bo_Mamba_v2
# =========================================================

class Improved_Bo_Mamba_v2(nn.Module):
    """与原始 Bo_Mamba 接口完全兼容。"""
    def __init__(self, encode_channels=None, decode_channels=144,
                 dropout=0.1, num_classes=6):
        super().__init__()
        if encode_channels is None: encode_channels = [18, 36, 72, 144]
        self.backbone = get_seg_model(config)
        self.decoder  = Decoder(encode_channels, decode_channels,
                                 dropout=dropout, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        return self.decoder(res1, res2, res3, res4, h, w)


if __name__ == '__main__':
    from thop import profile, clever_format
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    x   = torch.randn(1, 3, 512, 512).to(DEVICE)
    net = Improved_Bo_Mamba_v2().to(DEVICE)
    with torch.no_grad():
        out = net(x)
    print(f'Output shape: {out.shape}')
    flops, params = profile(net, (x,), verbose=False)
    flops, params = clever_format([flops, params], '%.3f')
    print(f'FLOPs: {flops}  Params: {params}')
