from torch.utils.data import DataLoader
import torch
import cv2
import numpy as np
from model import *
from tqdm import tqdm
from model.swinT import swin_base

num_classes = 6
dataset = 'test'

test_dir_img = f'./datasets/{dataset}/test/'
test_dir_mask = f'./datasets/{dataset}/test_labels/'

test = BasicDataset(test_dir_img, test_dir_mask)

test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = swin_base(nclass=num_classes, pretrained=False, aux=False, head='seghead', edge_aux=False)

net.to(device=device)
pretrained_dict = torch.load('../weights/xx.pth')
model_dict = net.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)


with torch.no_grad():
    net.eval()
    for id2, batch in enumerate(tqdm(test_loader)):
        imgs = batch['image'].to(device=device, dtype=torch.float32)
        mask_prd = net(imgs)
        mask = mask_prd.contiguous().argmax(dim=1)[0]
        label = mask.unsqueeze(0).permute(1, 2, 0)
        cmap = np.array([[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]])
        y = label.cpu().numpy()
        r = y.copy()
        g = y.copy()
        b = y.copy()
        for l in range(0, len(cmap)):
            r[y == l] = cmap[l, 0]
            g[y == l] = cmap[l, 1]
            b[y == l] = cmap[l, 2]
        label = np.concatenate((b, g, r), axis=-1)
        cv2.imwrite('../inference/{}.png'.format(batch['idx'][0]), label)
