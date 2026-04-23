"""
check_diff.py — verify frame difference clips look correct
Run: python check_diff.py
"""
import torch
import yaml
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from data.dataset import FaceForensicsDataset, load_records, split_records, make_weighted_sampler

cfg = yaml.safe_load(open('configs/config.yaml'))
recs = load_records(str(Path(cfg['data']['output_dir']) / 'records.json'), max_fake=2000)
train_recs, val_recs, _ = split_records(recs, 0.8, 0.1)

ds = FaceForensicsDataset(train_recs, split='train',
                          n_frames=cfg['data']['num_frames'])
loader = DataLoader(ds, batch_size=4, sampler=make_weighted_sampler(train_recs),
                    num_workers=0)

frames, clips, labels = next(iter(loader))

print(f"frames shape : {frames.shape}   dtype: {frames.dtype}")
print(f"clips  shape : {clips.shape}    dtype: {clips.dtype}")
print(f"labels       : {labels.tolist()}")
print(f"frames range : [{frames.min():.2f}, {frames.max():.2f}]")
print(f"clips  range : [{clips.min():.2f}, {clips.max():.2f}]")
print(f"clips  mean  : {clips.mean():.4f}  (should be ≈0.0 for no-change)")
print(f"frames NaN   : {torch.isnan(frames).any()}")
print(f"clips  NaN   : {torch.isnan(clips).any()}")

# Check that real and fake clips look different in variance
real_clips = clips[labels == 0]
fake_clips = clips[labels == 1]
if len(real_clips) > 0 and len(fake_clips) > 0:
    print(f"\nReal clip std : {real_clips.std():.4f}")
    print(f"Fake clip std : {fake_clips.std():.4f}")
    print("(Fake std should ideally be higher due to blending artifacts)")

print("\nSHAPES OK — ready to train" if clips.shape[1] == cfg['data']['num_frames']
      else f"\nWARNING: expected T={cfg['data']['num_frames']} got T={clips.shape[1]}")