"""
train_4.py
==========
基于 train_2.py 扩展的完整 GAN 训练脚本。

训练策略
--------
每个 iteration 执行两步更新：

Step-1  训练 Discriminator（D）
    · 真样本：one-hot(GT label)          → D 输出接近 1
    · 假样本：softmax(Generator 输出).detach() → D 输出接近 0
    · 损失：loss_D = BCE(d_real, 1) + BCE(d_fake, 0)

Step-2  训练 Generator（G）
    · 分割损失  : loss_seg = CrossEntropyLoss(pred, label)
    · 对抗损失  : loss_adv = BCE(D(softmax(pred)), 1)  [欺骗 D]
    · 总损失    : loss_G   = loss_seg + λ_adv * loss_adv
    · λ_adv 随训练进行线性衰减（warm-up → 稳定阶段）

模型保存
--------
根据验证集 mIoU 保存最优 Generator 权重（D 仅用于训练，不单独保存）。
日志同步写入 TensorBoard。
"""

from __future__ import annotations

import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tensorboardX import SummaryWriter

# ── 项目内部模块（与 train_2.py 保持一致）────────────────────
from model import BasicDataset
from model.Metrics import metric, ConfusionMatrix

# ── GAN 核心模块（来自 GAN.py）──────────────────────────────
from GAN import Generator, Discriminator


warnings.filterwarnings('ignore')

# ============================================================
# 超参数配置（集中管理，避免硬编码）
# ============================================================

NUM_CLASSES  : int   = 6
MAX_EPOCH    : int   = 200
LR_G         : float = 1e-3       # Generator 学习率
LR_D         : float = 1e-4       # Discriminator 学习率（通常比 G 低一量级）
BATCH_SIZE   : int   = 4
NDF          : int   = 64         # Discriminator 基础通道数
BASE_CH      : int   = 32         # Generator CNN 编码器基础通道数
NUM_STAGES   : int   = 4          # 编码器阶段数

# 对抗损失权重：前 WARMUP_EPOCH 个 epoch 线性从 0 升至 LAMBDA_ADV
LAMBDA_ADV   : float = 0.01
WARMUP_EPOCH : int   = 10

# 数据集路径
DATASET       : str  = 'test'          # 切换为 'Vaihingen' 等实际数据集
TRAIN_IMG     : str  = f'./datasets/{DATASET}/train/'
TRAIN_MASK    : str  = f'./datasets/{DATASET}/train_labels/'
TEST_IMG      : str  = f'./datasets/{DATASET}/test/'
TEST_MASK     : str  = f'./datasets/{DATASET}/test_labels/'
WEIGHT_SAVE_DIR : str = './weights/'
LOG_DIR         : str = './runs/GAN_train'

# 随机种子（可复现性）
SEED: int = 42


# ============================================================
# 工具函数
# ============================================================

def set_seed(seed: int) -> None:
    """固定随机种子以保证结果可复现"""
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    将整数标签转换为 one-hot 概率图。

    Parameters
    ----------
    labels      : [B, H, W]  long
    num_classes : int

    Returns
    -------
    one_hot     : [B, num_classes, H, W]  float32
    """
    B, H, W   = labels.shape
    one_hot   = torch.zeros(B, num_classes, H, W, device=labels.device)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0)


def get_adv_lambda(epoch: int, warmup: int, lam: float) -> float:
    """
    对抗损失权重调度：前 warmup 个 epoch 线性增大，之后保持 lam 不变。
    """
    if epoch < warmup:
        return lam * (epoch / warmup)
    return lam


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ============================================================
# 训练主程序
# ============================================================

def main() -> None:
    set_seed(SEED)
    ensure_dirs(WEIGHT_SAVE_DIR, LOG_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device: {device}')

    # ── 数据加载 ─────────────────────────────────────────────
    train_ds = BasicDataset(TRAIN_IMG, TRAIN_MASK)
    test_ds  = BasicDataset(TEST_IMG,  TEST_MASK)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    print(f'[INFO] Train samples: {len(train_ds)}, Test samples: {len(test_ds)}')

    # ── 模型初始化 ───────────────────────────────────────────
    G = Generator(
        in_ch=3, base_ch=BASE_CH, num_stages=NUM_STAGES,
        num_classes=NUM_CLASSES,
    ).to(device)

    D = Discriminator(num_classes=NUM_CLASSES, ndf=NDF).to(device)

    # ── 优化器 ───────────────────────────────────────────────
    optimizer_G = optim.Adam(G.parameters(), lr=LR_G, weight_decay=1e-4)
    optimizer_D = optim.Adam(D.parameters(), lr=LR_D, weight_decay=1e-4,
                             betas=(0.9, 0.99))

    # Generator 学习率调度（余弦退火，与 train_2.py 保持一致）
    scheduler_G = CosineAnnealingWarmRestarts(optimizer_G, T_0=10, T_mult=1)
    # Discriminator 学习率同步调度（步调一致，避免 G/D 能力失衡）
    scheduler_D = CosineAnnealingWarmRestarts(optimizer_D, T_0=10, T_mult=1)

    # ── 损失函数 ─────────────────────────────────────────────
    criterion_seg = nn.CrossEntropyLoss()
    # BCEWithLogitsLoss 接收原始 logits，内部含 Sigmoid，数值更稳定
    criterion_adv = nn.BCEWithLogitsLoss()

    # ── TensorBoard ──────────────────────────────────────────
    writer = SummaryWriter(LOG_DIR)

    # ── 训练状态追踪 ─────────────────────────────────────────
    best_miou       : float = 0.0
    best_ckpt_path  : str   = ''

    # ============================================================
    # 主训练循环
    # ============================================================
    for epoch in range(MAX_EPOCH):
        G.train()
        D.train()

        epoch_loss_G   = 0.0
        epoch_loss_D   = 0.0
        epoch_loss_seg = 0.0
        epoch_loss_adv = 0.0
        n_batches      = 0

        # 当前 epoch 的对抗权重
        lam_adv = get_adv_lambda(epoch, WARMUP_EPOCH, LAMBDA_ADV)

        for batch in tqdm(train_loader, desc=f'Epoch {epoch:>3d}/{MAX_EPOCH}'):
            imgs       = batch['image'].to(device, dtype=torch.float32)
            true_masks = batch['mask'].to(device,  dtype=torch.long)
            B = imgs.size(0)

            # ── real sample：one-hot GT（D 的正样本）─────────────
            real_prob = to_one_hot(true_masks, NUM_CLASSES)  # [B, C, H, W]

            # ────────────────────────────────────────────────────
            # Step-1  训练 Discriminator
            # ────────────────────────────────────────────────────
            with torch.no_grad():
                fake_logits = G(imgs)                         # [B, C, H, W]
            fake_prob = torch.softmax(fake_logits, dim=1)    # 连续概率图给 D

            d_real  = D(real_prob)                           # [B, C, H', W']
            d_fake  = D(fake_prob)                           # [B, C, H', W']

            loss_D_real = criterion_adv(d_real, torch.ones_like(d_real))
            loss_D_fake = criterion_adv(d_fake, torch.zeros_like(d_fake))
            loss_D      = 0.5 * (loss_D_real + loss_D_fake)

            optimizer_D.zero_grad()
            loss_D.backward()
            # 梯度裁剪，防止 D 训练不稳定
            nn.utils.clip_grad_norm_(D.parameters(), max_norm=5.0)
            optimizer_D.step()

            # ────────────────────────────────────────────────────
            # Step-2  训练 Generator
            # ────────────────────────────────────────────────────
            pred_logits = G(imgs)                            # 重新前向（含梯度）
            pred_prob   = torch.softmax(pred_logits, dim=1)

            # 分割损失
            loss_seg = criterion_seg(pred_logits, true_masks)

            # 对抗损失：Generator 希望 D 把 fake 判断为 real（目标全 1）
            d_gen    = D(pred_prob)
            loss_adv = criterion_adv(d_gen, torch.ones_like(d_gen))

            loss_G = loss_seg + lam_adv * loss_adv

            optimizer_G.zero_grad()
            loss_G.backward()
            nn.utils.clip_grad_norm_(G.parameters(), max_norm=5.0)
            optimizer_G.step()

            # 累积 loss 用于 epoch 统计
            epoch_loss_G   += loss_G.item()
            epoch_loss_D   += loss_D.item()
            epoch_loss_seg += loss_seg.item()
            epoch_loss_adv += loss_adv.item()
            n_batches      += 1

        # 学习率调度
        scheduler_G.step()
        scheduler_D.step()

        # ── epoch 级 TensorBoard 记录 ────────────────────────
        avg_loss_G   = epoch_loss_G   / n_batches
        avg_loss_D   = epoch_loss_D   / n_batches
        avg_loss_seg = epoch_loss_seg / n_batches
        avg_loss_adv = epoch_loss_adv / n_batches

        writer.add_scalar('Loss/Generator_total', avg_loss_G,   epoch)
        writer.add_scalar('Loss/Discriminator',   avg_loss_D,   epoch)
        writer.add_scalar('Loss/Segmentation',    avg_loss_seg, epoch)
        writer.add_scalar('Loss/Adversarial',     avg_loss_adv, epoch)
        writer.add_scalar('HyperParam/lr_G',      optimizer_G.param_groups[0]['lr'], epoch)
        writer.add_scalar('HyperParam/lr_D',      optimizer_D.param_groups[0]['lr'], epoch)
        writer.add_scalar('HyperParam/lambda_adv', lam_adv, epoch)

        print(
            f'[Epoch {epoch:>3d}] '
            f'loss_G={avg_loss_G:.4f}  '
            f'loss_seg={avg_loss_seg:.4f}  '
            f'loss_adv={avg_loss_adv:.4f}  '
            f'loss_D={avg_loss_D:.4f}  '
            f'λ_adv={lam_adv:.4f}'
        )

        # ============================================================
        # 验证阶段（与 train_2.py 保持完全一致）
        # ============================================================
        G.eval()
        matrix_ = [[0.0] * NUM_CLASSES for _ in range(NUM_CLASSES)]

        for batch in tqdm(test_loader, desc=f'  Eval epoch {epoch}'):
            imgs_val       = batch['image'].to(device, dtype=torch.float32)
            true_masks_val = batch['mask'].to(device,  dtype=torch.float32)

            with torch.no_grad():
                mask_pred   = G(imgs_val)
                predict_all = mask_pred.argmax(dim=1).flatten().cpu()
                label_all   = true_masks_val.flatten().cpu()
                matrix = ConfusionMatrix(
                    numClass=NUM_CLASSES,
                    imgPredict=predict_all,
                    Label=label_all,
                )
            matrix_ += matrix   # type: ignore[operator]  — 来自原项目实现

        precision, recall, OA, IoU, FWIOU, mIOU, f1score = metric(matrix_)

        print(
            f'  precision: {precision}\n'
            f'  recall:    {recall}\n'
            f'  OA:        {OA}\n'
            f'  IoU:       {IoU}\n'
            f'  FWIoU:     {FWIOU}\n'
            f'  mIoU:      {mIOU}\n'
            f'  F1 score:  {f1score}'
        )

        # TensorBoard 验证指标
        writer.add_scalar('Val/mIoU',  mIOU, epoch)
        writer.add_scalar('Val/OA',    OA,   epoch)
        writer.add_scalar('Val/FWIoU', FWIOU, epoch)

        # ── 最优模型保存（仅保留 mIoU 最高的 checkpoint）──────
        if mIOU >= best_miou:
            best_miou = mIOU

            # 删除上一个最优 checkpoint
            if best_ckpt_path and os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)

            best_ckpt_path = os.path.join(
                WEIGHT_SAVE_DIR,
                f'epoch{epoch}_GAN_Generator_mIOU_{round(mIOU, 4)}.pth',
            )
            torch.save(G.state_dict(), best_ckpt_path)
            print(f'  [✓] Best model saved → {best_ckpt_path}')

    writer.close()
    print(f'\n[INFO] Training done. Best mIoU = {best_miou:.4f}')
    print(f'[INFO] Best checkpoint : {best_ckpt_path}')


if __name__ == '__main__':
    main()
