from torch.utils.data import DataLoader
import torch
from model import *
from model.Metrics import metric, ConfusionMatrix
from tqdm import tqdm
from model.MF_Mamba import MF_Mamba   # 替换为 MF_Mamba

num_classes = 6
batchsize = 16
dataset = 'test'  # 请根据实际数据集名称修改（如 'test'）

# 使用与 train_1.py 一致的相对路径（当前目录下的 datasets）
test_dir_img = f'./datasets/{dataset}/test/'
test_dir_mask = f'./datasets/{dataset}/test_labels/'

test = BasicDataset(test_dir_img, test_dir_mask)

test_loader = DataLoader(test, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化 MF_Mamba 模型
net = MF_Mamba(num_classes=num_classes).to(device)

# 加载训练好的权重（请根据实际权重文件路径修改）
weight_path = './weights/epoch135_MF_Mamba_mIOU_0.714.pth'  # 替换为你训练出的权重文件
pretrained_dict = torch.load(weight_path, map_location=device)
model_dict = net.state_dict()
# 只加载匹配的键（忽略不匹配的，如分类头维度变化）
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)
print(f"Loaded weights from {weight_path}")

net.eval()
matrix_ = [[0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]]

for id2, batch in enumerate(tqdm(test_loader)):
    imgs = batch['image'].to(device=device, dtype=torch.float32)
    true_masks = batch['mask'].to(device=device, dtype=torch.float32)

    with torch.no_grad():
        mask_pred = net(imgs)
        predict_all = mask_pred.argmax(dim=1).flatten().cpu()
        label_all = true_masks.flatten().cpu()
        matrix = ConfusionMatrix(numClass=num_classes, imgPredict=predict_all, Label=label_all)
    matrix_ += matrix

precision, recall, OA, IoU, FWIOU, mIOU, f1score = metric(matrix_)

print('precision:{}\nrecall:{}\nOA:{}\n'
      'IoU:{}\nFWIoU:{}\nmIoU:{}\nf1score:{}'.format(precision, recall, OA, IoU, FWIOU, mIOU, f1score))
