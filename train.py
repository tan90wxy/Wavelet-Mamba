import os
import torch
import warnings
import torch.nn as nn
import torch.optim as optim

from model import *
from tqdm import tqdm
from model.Metrics import metric
from model.MF_Mamba import MF_Mamba  # 导入MF_mamba模型
from torch.utils.data import DataLoader
from model.Metrics import ConfusionMatrix
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def calc_alpha(curr_iter, all_iters, weight=1.0):
    r = (1.0 - float(curr_iter) / all_iters) ** 2.0
    return weight * r


num_classes = 6
max_epoch = 300
lr = 0.001
batchsize = 16
dataset = 'test'
# dataset = 'Vaihingen'
train_dir_img = f'./datasets/{dataset}/train/'
train_dir_mask = f'./datasets/{dataset}/train_labels/'
test_dir_img = f'./datasets/{dataset}/test/'
test_dir_mask = f'./datasets/{dataset}/test_labels/'
weight_save_dir = './weights/'

train = BasicDataset(train_dir_img, train_dir_mask)
test = BasicDataset(test_dir_img, test_dir_mask)

train_loader = DataLoader(train, batch_size=batchsize, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 替换Swin Transformer为MF_Mamba模型
net = MF_Mamba(
    encode_channels=[18, 36, 72, 144],
    decode_channels=144,
    dropout=0.1,
    num_classes=num_classes
)
net.to(device=device)

optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

criterion = nn.CrossEntropyLoss()
miou = 0.0
last = weight_save_dir + 'epoch' + str(0) + '_MF_Mamba_' + f'mIOU_0.0.pth'

print("开始训练MF_Mamba模型...")
for epoch in range(max_epoch):
    print(f"\nEpoch {epoch+1}/{max_epoch}")
    
    # 训练阶段
    net.train()
    epoch_loss = 0
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Train')
    
    for id1, batch in enumerate(train_pbar):
        imgs = batch['image'].to(device=device, dtype=torch.float32)
        true_masks = batch['mask'].to(device=device, dtype=torch.long)

        masks_pred = net(imgs)

        ce_loss = criterion(masks_pred, true_masks.long())
        loss = ce_loss
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    # 验证阶段
    net.eval()
    matrix_ = [[0., 0., 0., 0., 0., 0.], 
               [0., 0., 0., 0., 0., 0.], 
               [0., 0., 0., 0., 0., 0.], 
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.], 
               [0., 0., 0., 0., 0., 0.]]
    
    val_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1} Val')
    for id2, batch in enumerate(val_pbar):
        imgs = batch['image'].to(device=device, dtype=torch.float32)
        true_masks = batch['mask'].to(device=device, dtype=torch.float32)

        with torch.no_grad():
            mask_pred = net(imgs)
            predict_all = mask_pred.argmax(dim=1).flatten().cpu()
            label_all = true_masks.flatten().cpu()
            matrix = ConfusionMatrix(numClass=6, imgPredict=predict_all, Label=label_all)
        matrix_ += matrix
    
    precision, recall, OA, IoU, FWIOU, mIOU, f1score = metric(matrix_)

    print(f'Epoch {epoch+1}:')
    print('precision:{}\nrecall:{}\nOA:{}\n'
          'IoU:{}\nFWIoU:{}\nmIoU:{}\nf1score:{}'.format(precision, recall, OA, IoU, FWIOU, mIOU, f1score))
    
    # 更新学习率
    scheduler.step()
    
    # 保存最佳模型
    if mIOU >= miou:
        miou = mIOU
        if os.path.exists(last):
            os.remove(last)
        torch.save(net.state_dict(), weight_save_dir + 'epoch' + str(epoch) + '_MF_Mamba_' + f'mIOU_{round(mIOU, 4)}.pth')
        last = weight_save_dir + 'epoch' + str(epoch) + '_MF_Mamba_' + f'mIOU_{round(mIOU, 4)}.pth'
        print(f"New best model saved with mIOU: {round(mIOU, 4)}")
    
    print(f"Current best mIOU: {round(miou, 4)}")

print("Training completed!")
print(f"Best mIOU achieved: {round(miou, 4)}")
