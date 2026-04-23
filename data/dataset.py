"""
dataset.py — FaceForensics++ loader
-------------------------------------
Key change: TimeSformer receives FRAME DIFFERENCE clips instead of raw frames.
  clip[t] = frame[t] - frame[t-1], normalized to [-1, 1]
  This makes TimeSformer detect temporal flickering/blending artifacts
  rather than natural motion (which is identical in real and fake videos).

__getitem__ returns:
  frame : (C, H, W)     — single normalized frame      → ViT
  clip  : (T, C, H, W)  — frame differences normalized → TimeSformer
  label : long tensor
"""

import json
import random
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────

def get_transforms(split: str = "train", label: int = 0) -> A.Compose:
    normalize = A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
    if split != "train":
        return A.Compose([A.Resize(224, 224), normalize, ToTensorV2()])

    if label == 0:   # real — heavy augmentation
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.2, hue=0.1, p=0.6),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(p=0.5),
            A.RandomGamma(p=0.3),
            A.Affine(translate_percent=0.1,
                     scale=(0.85, 1.15), rotate=(-15, 15), p=0.5),
            A.ImageCompression(quality_range=(70, 100), p=0.4),
            A.CoarseDropout(num_holes_range=(1, 8),
                            hole_height_range=(10, 20),
                            hole_width_range=(10, 20), p=0.3),
            normalize, ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.15, contrast=0.15,
                          saturation=0.1, p=0.4),
            A.GaussNoise(p=0.3),
            A.CoarseDropout(num_holes_range=(1, 4),
                            hole_height_range=(8, 16),
                            hole_width_range=(8, 16), p=0.2),
            normalize, ToTensorV2(),
        ])


def get_diff_transforms(split: str = "train") -> A.Compose:
    """
    Transforms for frame difference images.
    No ColorJitter or normalization — differences are already zero-centered.
    Just resize and optional spatial augmentation.
    """
    if split != "train":
        return A.Compose([A.Resize(224, 224), ToTensorV2()])
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(p=0.3),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class FaceForensicsDataset(Dataset):
    """
    Loads face clips saved by preprocess.py as .npy files.
    record keys: "path", "label", "source", "video"
    .npy shape : (T, H, W, C) uint8
    """

    def __init__(self, records: List[Dict], split: str = "train",
                 n_frames: int = 16, img_size: int = 224):
        self.records  = records
        self.split    = split
        self.n_frames = n_frames
        self.img_size = img_size

    def __len__(self):
        return len(self.records)

    def _load_npy(self, path: str) -> np.ndarray:
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr[np.newaxis]
        return arr  # (T, H, W, C) uint8

    def _sample_frames(self, arr: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        T = arr.shape[0]
        # Need n_frames+1 raw frames to produce n_frames differences
        n_raw = self.n_frames + 1
        if T >= n_raw:
            indices = np.linspace(0, T - 1, n_raw, dtype=int)
        else:
            indices = np.array([i % T for i in range(n_raw)])
        raw_frames = [arr[i] for i in indices]
        single     = arr[random.randint(0, T - 1)]
        return single, raw_frames  # single: (H,W,C), raw: list of n_frames+1

    def _compute_diff_clip(self, raw_frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute frame differences: diff[t] = frame[t+1] - frame[t]
        Cast to int16 to allow negative values, then scale to uint8 [0,255]
        by shifting: diff_uint8 = (diff + 255) / 2
        This gives a neutral gray (128) for no change, bright for increase,
        dark for decrease — exactly what TimeSformer needs to see artifacts.
        """
        diffs = []
        for t in range(len(raw_frames) - 1):
            f1 = raw_frames[t].astype(np.int16)
            f2 = raw_frames[t + 1].astype(np.int16)
            diff = f2 - f1                          # [-255, 255]
            diff_uint8 = ((diff + 255) / 2).astype(np.uint8)  # [0, 255]
            diffs.append(diff_uint8)
        return diffs  # list of n_frames (H, W, C) uint8

    def __getitem__(self, idx: int):
        record = self.records[idx]
        label  = record["label"]
        arr    = self._load_npy(record["path"])

        single, raw_frames = self._sample_frames(arr)
        diff_frames        = self._compute_diff_clip(raw_frames)

        # ViT input: single normalized frame
        frame_transform = get_transforms(split=self.split, label=label)
        frame_t = frame_transform(image=single)["image"]  # (C, H, W)

        # TimeSformer input: frame differences
        diff_transform = get_diff_transforms(split=self.split)
        clip_t = torch.stack(
            [diff_transform(image=d)["image"] for d in diff_frames], dim=0
        )  # (T, C, H, W)

        # Normalize diff clip to [-1, 1] for TimeSformer
        # Input is [0,255] uint8 → float → [0,1] → [-0.5, 0.5]*2
        clip_t = clip_t.float() / 255.0      # [0, 1]
        clip_t = (clip_t - 0.5) * 2.0        # [-1, 1], mean≈0 for no-change regions

        return frame_t, clip_t, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_records(records_path: str, max_fake: int = None) -> List[Dict]:
    with open(records_path, "r") as f:
        records = json.load(f)

    if max_fake is not None:
        real_recs = [r for r in records if r["label"] == 0]
        fake_recs = [r for r in records if r["label"] == 1]

        if len(fake_recs) > max_fake:
            methods    = list(set(r["source"] for r in fake_recs))
            per_method = max_fake // len(methods)
            capped     = []
            for method in methods:
                method_recs = [r for r in fake_recs if r["source"] == method]
                random.seed(42)
                capped.extend(random.sample(
                    method_recs, min(per_method, len(method_recs))))
            remaining = max_fake - len(capped)
            if remaining > 0:
                leftover = [r for r in fake_recs if r not in capped]
                random.seed(42)
                capped.extend(random.sample(
                    leftover, min(remaining, len(leftover))))
            fake_recs = capped

        records = real_recs + fake_recs
        print(f"Dataset capped : {len(real_recs)} real, {len(fake_recs)} fake "
              f"(ratio 1:{len(fake_recs)/max(len(real_recs),1):.1f})")

    return records


def split_records(records: List[Dict], train_ratio: float = 0.8,
                  val_ratio: float = 0.1, seed: int = 42
                  ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    labels = [r["label"] for r in records]
    train_rec, temp_rec = train_test_split(
        records, test_size=1 - train_ratio,
        stratify=labels, random_state=seed)
    val_size    = val_ratio / (1 - train_ratio)
    temp_labels = [r["label"] for r in temp_rec]
    val_rec, test_rec = train_test_split(
        temp_rec, test_size=1 - val_size,
        stratify=temp_labels, random_state=seed)
    return train_rec, val_rec, test_rec


def make_weighted_sampler(records: List[Dict]) -> WeightedRandomSampler:
    labels        = [r["label"] for r in records]
    class_counts  = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True)


def get_dataloaders(records_path: str = "data/processed/records.json",
                    batch_size: int = 8, n_frames: int = 16,
                    num_workers: int = 4, train_ratio: float = 0.8,
                    val_ratio: float = 0.1, seed: int = 42,
                    max_fake: int = None,
                    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    records = load_records(records_path, max_fake=max_fake)
    train_rec, val_rec, test_rec = split_records(
        records, train_ratio, val_ratio, seed)
    train_ds = FaceForensicsDataset(
        train_rec, split="train", n_frames=n_frames)
    val_ds   = FaceForensicsDataset(
        val_rec,   split="val",   n_frames=n_frames)
    test_ds  = FaceForensicsDataset(
        test_rec,  split="test",  n_frames=n_frames)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=make_weighted_sampler(train_rec),
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader