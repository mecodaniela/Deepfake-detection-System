"""
PyTorch Dataset + DataLoader për frame-t e nxjerra nga image_extraction.py.
data/frames/{split}/{label}/*.jpg  ->  ImageDataset  ->  transform/normalize
-> DataLoader -> batch -> CNN
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# KONFIGURIME

FRAMES_DIR = Path("data/frames")

IMAGE_SIZE = 224  # frame-t janë tashmë 224x224 nga image_extraction.py (referencë, s'përdoret në transform)

# Statistika standarde ImageNet — të dobishme nëse do përdorim backbone pretrained
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 16
NUM_WORKERS = 2

LABEL_TO_IDX = {"real": 0, "fake": 1}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


# DATASET

class DeepfakeFrameDataset(Dataset):
    def __init__(self, split: str, transform=None):
        self.split = split
        self.transform = transform
        self.samples = []  # lista e (path, label_idx)

        split_dir = FRAMES_DIR / split
        for label_name, label_idx in LABEL_TO_IDX.items():
            label_dir = split_dir / label_name
            if not label_dir.exists():
                continue
            for img_path in sorted(label_dir.glob("*.jpg")):
                self.samples.append((img_path, label_idx))

        if not self.samples:
            raise RuntimeError(f"S'u gjet asnjë imazh te {split_dir} — kontrollo path-in.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label_idx


# TRANSFORMS

def get_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])
    else:  # val / test — PA augmentim, vetëm normalizim
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])


# NDIHMËSE — KRIJIMI I DATALOADER-ËVE

def get_dataloader(split: str, batch_size: int = BATCH_SIZE, shuffle: bool = None, num_workers: int = NUM_WORKERS):
    if shuffle is None:
        shuffle = (split == "train")  # vetëm train duhet shuffle; val/test jo

    dataset = DeepfakeFrameDataset(split, transform=get_transforms(split))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # s'ka GPU, s'ka nevojë
        persistent_workers=(num_workers > 0),
    )

    return loader

# TEST I SHPEJTË (ekzekuto direkt këtë skedar për verifikim)

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        dataset = DeepfakeFrameDataset(split, transform=get_transforms(split))
        print(f"{split}: {len(dataset)} imazhe")

        loader = get_dataloader(split, batch_size=8)
        images, labels = next(iter(loader))
        print(f"  Shape batch: {images.shape}, labels: {labels.tolist()}")