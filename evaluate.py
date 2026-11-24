"""
evaluate.py
===========
Evaluate a trained checkpoint on the test split of a dataset.

The protocol is identical to the validation loop of ``train_4.py``: the
predicted class map is compared with the ground truth over all test pixels,
and precision / recall / OA / IoU / FWIoU / mIoU / F1-score are reported.

Usage
-----
    python evaluate.py --ckpt weights/<checkpoint>.pth
    python evaluate.py --ckpt weights/<checkpoint>.pth --dataset Vaihingen

The dataset must be arranged as

    datasets/<name>/test/           test images  (.png/.tif)
    datasets/<name>/test_labels/    test labels  (RGB colour maps, same names)

and the checkpoint must be a ``state_dict`` of ``GAN.Generator`` saved by
``train_4.py``.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import BasicDataset
from model.Metrics import ConfusionMatrix, metric
from GAN import Generator

NUM_CLASSES = 6


def main() -> None:
    ap = argparse.ArgumentParser(description='Evaluate GAN.Generator on a test split')
    ap.add_argument('--ckpt', required=True, help='path to the trained Generator state_dict')
    ap.add_argument('--dataset', default='test', help='folder name under ./datasets')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dir_img = f'./datasets/{args.dataset}/test/'
    test_dir_mask = f'./datasets/{args.dataset}/test_labels/'
    test_set = BasicDataset(test_dir_img, test_dir_mask)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Same configuration as in train_4.py
    G = Generator(in_ch=3, base_ch=32, num_stages=4, num_classes=NUM_CLASSES).to(device)
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    G.load_state_dict(state)
    G.eval()

    matrix_ = [[0.0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Eval'):
            imgs = batch['image'].to(device, dtype=torch.float32)
            true_masks = batch['mask'].to(device, dtype=torch.float32)

            mask_pred = G(imgs)
            predict_all = mask_pred.argmax(dim=1).flatten().cpu()
            label_all = true_masks.flatten().cpu()
            matrix_ += ConfusionMatrix(
                numClass=NUM_CLASSES,
                imgPredict=predict_all,
                Label=label_all,
            )

    precision, recall, OA, IoU, FWIOU, mIOU, f1score = metric(matrix_)
    print(
        f'precision: {precision}\n'
        f'recall:    {recall}\n'
        f'OA:        {OA}\n'
        f'IoU:       {IoU}\n'
        f'FWIoU:     {FWIOU}\n'
        f'mIoU:      {mIOU}\n'
        f'F1 score:  {f1score}'
    )


if __name__ == '__main__':
    main()
