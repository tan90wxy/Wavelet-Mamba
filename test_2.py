from torch.utils.data import DataLoader
import torch
from model import BasicDataset
from model.Metrics import metric, ConfusionMatrix
from tqdm import tqdm
from GAN_nonwave import Generator

# ── 配置区 (按需修改) ────────────────────────────────────────────────────────
num_classes = 6
batchsize   = 16
dataset     = 'test'

# Generator 结构参数（须与训练时 train_4.py 中完全一致）
base_ch    = 32
num_stages = 4

# 训练结果权重路径（替换为实际文件名）
weights_path = './weights/epoch83_VSSM_mIOU_0.7482——对比方法1最终版-监督值0.1.pth'
# ─────────────────────────────────────────────────────────────────────────────

test_dir_img  = f'./datasets/{dataset}/test/'
test_dir_mask = f'./datasets/{dataset}/test_labels/'

test = BasicDataset(test_dir_img, test_dir_mask)
test_loader = DataLoader(test, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 构建 Generator（与 train_4.py 中完全一致）
net = Generator(in_ch=3, base_ch=base_ch, num_stages=num_stages, num_classes=num_classes)
net.to(device=device)

# 加载 Generator 权重
# train_4.py 保存的是纯 G.state_dict()，可直接 load_state_dict；
# 同时保留 test.py 的部分加载逻辑，兼容 key 不完全匹配的情况
pretrained_dict = torch.load(weights_path, map_location=device)
model_dict = net.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)

net.eval()
matrix_ = [[0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]]

for id2, batch in enumerate(tqdm(test_loader)):
    imgs       = batch['image'].to(device=device, dtype=torch.float32)
    true_masks = batch['mask'].to(device=device,  dtype=torch.float32)

    with torch.no_grad():
        mask_pred   = net(imgs)                                 # [B, num_classes, H, W] logits
        predict_all = mask_pred.argmax(dim=1).flatten().cpu()
        label_all   = true_masks.flatten().cpu()
        matrix = ConfusionMatrix(numClass=num_classes, imgPredict=predict_all, Label=label_all)
    matrix_ += matrix

precision, recall, OA, IoU, FWIOU, mIOU, f1score = metric(matrix_)

print('precision:{}\nrecall:{}\nOA:{}\n'
      'IoU:{}\nFWIoU:{}\nmIoU:{}\nf1score:{}'.format(precision, recall, OA, IoU, FWIOU, mIOU, f1score))
