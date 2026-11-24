# Wavelet-Mamba

Code for the manuscript

> **Dual-Encoder Lightweight Semantic Segmentation with Adaptive Filtering and Wavelet Enhancement via Generative Adversarial Learning**
> Xiyuan Wang, Jian Gao — *Applied Intelligence* (under review)

The model is a lightweight dual-encoder segmentation network for remote-sensing imagery:

- a **CNN branch** and a **Mamba branch** run in parallel and are fused stage by stage with
  adaptive weights (multi-level adaptive fusion enhancement, AFE);
- a **wavelet (frequency) branch** works on the high-frequency detail that the spatial path
  loses, and is coupled to the global branch through an eight-direction selective scan;
- a **conditional discriminator** (from GATrans, adapted to six classes) is used during
  training only and discarded at inference, so it does not add to the deployed model.

On the ISPRS Vaihingen and Potsdam benchmarks the model reaches 75.35 % and 72.15 % mIoU
with 4.96 M parameters.

## Relation to MF-Mamba

This repository builds on the official implementation of **MF-Mamba**
([Xiao et al., *IEEE TGRS*, 2025](https://ieeexplore.ieee.org/document/11098811)): the HRNet
backbone, the Mamba/SS2D blocks and the dataset/metric utilities come from that project.
The generator (dual encoder + AFE + wavelet branch), the discriminator coupling and the
adversarial training schedule are the contributions of this work. The original MF-Mamba
citation is reproduced at the end of this file; please cite it together with this paper.

## Repository layout

| Path | Content |
|---|---|
| `GAN.py` | Generator (`DualEncoderFusionNet`: CNN encoder + Mamba-DWConv encoder + CAB + stage-wise fusion) and the discriminator of the full model, **with** the wavelet branch |
| `GAN_nonwave.py` | Same generator **without** the wavelet branch (ablation variant) |
| `train_4.py` | Main training script: two-step G/D updates, linear warm-up of the adversarial weight, best-mIoU checkpointing |
| `evaluate.py` | Evaluation with a trained checkpoint (same protocol as the validation loop of `train_4.py`) |
| `model/` | Dataset loader (`Dataset.py`), metrics (`Metrics.py`), segmentation head, Swin/Mamba baselines |
| `backbone/HRNet/` | HRNet-W18 backbone (config + model) |
| `Vim.py`, `GT_Net.py`, `MF_Mamba.py` | SS2D / Mamba building blocks and the baseline networks used for comparison |
| `train*.py`, `test*.py`, `visual.py` | Earlier training/evaluation variants kept for reference (paths and checkpoint names must be edited before use) |
| `imgs/` | Figures |

## Environment

```bash
conda create -n wavelet-mamba python=3.8
conda activate wavelet-mamba
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
pip install timm tensorboardX tqdm opencv-python
```

If the CUDA extensions fail to build, install the matching precompiled wheels from
[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases) and
[mamba-ssm](https://github.com/state-spaces/mamba/releases).

## Data preparation

Download the ISPRS 2D semantic labelling benchmarks (Vaihingen and Potsdam) from the
ISPRS website and arrange the tiles as

```
datasets/
└── <name>/                 # e.g. Vaihingen, Potsdam, test
    ├── train/              # training images
    ├── train_labels/       # training labels (RGB colour maps, same file names)
    ├── test/
    └── test_labels/
```

Labels use the six-colour ISPRS palette (impervious surface, building, low vegetation,
tree, car, clutter); the mapping is in `model/Dataset.py`. The dataset itself is **not**
included in this repository.

The HRNet-W18 ImageNet weights (`hrnetv2_w18_imagenet_pretrained.pth`) are loaded from the
working directory by `backbone/HRNet/models/seg_hrnet.py`; download them from
[HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification). They are
not stored here because of their size.

## Training

Set `DATASET` in `train_4.py` to the folder name under `datasets/`, then run

```bash
python train_4.py
```

The script writes TensorBoard logs to `runs/GAN_train` and keeps the checkpoint with the
best validation mIoU under `weights/`.

## Evaluation

```bash
python evaluate.py --ckpt weights/<checkpoint>.pth --dataset <name>
```

It reports precision, recall, OA, IoU, FWIoU, mIoU and F1-score over all test pixels, i.e.
the same numbers that are logged during training.

## Ablation

Switch `from GAN import Generator` to `from GAN_nonwave import Generator` to train or
evaluate the variant without the wavelet branch.

## Citation

The entry below will be completed with volume, pages and DOI once the paper is accepted.

```bibtex
@article{wang2026waveletmamba,
  author  = {Wang, Xiyuan and Gao, Jian},
  title   = {Dual-Encoder Lightweight Semantic Segmentation with Adaptive Filtering
             and Wavelet Enhancement via Generative Adversarial Learning},
  journal = {Applied Intelligence},
  note    = {under review},
  year    = {2026}
}
```

The baseline this code is built on:

```bibtex
@ARTICLE{11098811,
  author  = {Xiao, Pu and Dong, Yuting and Zhao, Ji and Peng, Tieqi and Geiß, Christian
             and Zhong, Yanfei and Taubenb{\"o}ck, Hannes},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  title   = {MF-Mamba: Multiscale Convolution and Mamba Fusion Model for Semantic
             Segmentation of Remote Sensing Imagery},
  year    = {2025}
}
```

## Contact

Xiyuan Wang — 202321440020@stu.ppsuc.edu.cn
