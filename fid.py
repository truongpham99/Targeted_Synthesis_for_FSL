#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchmetrics.image.fid import FrechetInceptionDistance

SEED = 1 
DEVICE = "cuda"
IMG_SIZE = [299, 299]

def parse_args():
    p = argparse.ArgumentParser(description="Compute FID (TorchMetrics) between two image folders.")
    p.add_argument("real_dir", type=str, help="Directory with real/reference images.")
    p.add_argument("fake_dir", type=str, help="Directory with generated/synthetic images.")
    return p.parse_args()

def make_loader(img_dir, seed, device):
    # TorchMetrics FID expects either uint8 in [0,255] or float in [0,1] if normalize=True.
    t_list = [transforms.Resize(IMG_SIZE)]
    t_list.append(transforms.ToTensor())  # outputs float in [0,1]
    # Defer normalization to TorchMetrics via normalize flag; do NOT apply ImageNet mean/std here.
    tfm = transforms.Compose(t_list)

    ds = datasets.ImageFolder(
        root=img_dir,
        transform=tfm
    )
    g = torch.Generator()
    g.manual_seed(seed)
    # Collate to ensure dtype matches expectation

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    return loader, len(ds)

@torch.no_grad()
def accumulate(fid_metric, loader, real_flag, device):
    for img, _ in loader:
        img = img.to(device, non_blocking=True).to(torch.uint8)
        fid_metric.update(img, real=real_flag)

def main():
    args = parse_args()

    # Data loaders
    real_loader, n_real = make_loader(args.real_dir, SEED, DEVICE)
    fake_loader, n_fake = make_loader(args.fake_dir, SEED, DEVICE)

    if n_real == 0 or n_fake == 0:
        raise RuntimeError("One of the folders is empty or not readable by torchvision.datasets.ImageFolder.")

    # Metric
    fid = FrechetInceptionDistance(
        feature=2048, # inception-v3 output size
    ).to(DEVICE)

    # Accumulate features
    accumulate(fid, real_loader, real_flag=True, device=DEVICE)
    accumulate(fid, fake_loader, real_flag=False, device=DEVICE)

    # Compute FID
    score = float(fid.compute().cpu())
    print(f"FID: {score:.6f}")

if __name__ == "__main__":
    main()
