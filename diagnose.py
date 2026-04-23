"""
diagnose.py — run this to find exactly what's broken
Usage: python diagnose.py
"""
import torch
import numpy as np
import yaml
from pathlib import Path

cfg    = yaml.safe_load(open('configs/config.yaml'))
device = torch.device('cuda')

print("="*60)
print("STEP 1 — Check one batch from DataLoader")
print("="*60)

from data.dataset import FaceForensicsDataset, split_records, load_records, make_weighted_sampler
from torch.utils.data import DataLoader

records_path = Path(cfg['data']['output_dir']) / 'records.json'
all_records  = load_records(str(records_path))
train_recs, val_recs, _ = split_records(all_records, 0.8, 0.1)

ds = FaceForensicsDataset(train_recs, split='train',
                          n_frames=cfg['data']['num_frames'],
                          img_size=cfg['data']['image_size'])
loader = DataLoader(ds, batch_size=8, sampler=make_weighted_sampler(train_recs),
                    num_workers=0)  # num_workers=0 for clean error messages

frames, clips, labels = next(iter(loader))
print(f"frames shape : {frames.shape}  dtype: {frames.dtype}")
print(f"clips  shape : {clips.shape}   dtype: {clips.dtype}")
print(f"labels       : {labels.tolist()}")
print(f"frames min/max: {frames.min():.3f} / {frames.max():.3f}")
print(f"frames has NaN: {torch.isnan(frames).any()}")
print(f"clips  has NaN: {torch.isnan(clips).any()}")

print()
print("="*60)
print("STEP 2 — Forward pass through ViT only")
print("="*60)

from models.cat_model import CATModelWithSupCon, FocalLoss, SupConLoss, compute_loss

model = CATModelWithSupCon(embed_dim=cfg['model']['embed_dim'],
                           proj_dim=128, num_classes=2).to(device)
model.eval()

frames_d = frames.to(device)
clips_d  = clips.to(device)
labels_d = labels.to(device)

with torch.no_grad():
    vit_feat = model.backbone.vit_stream(frames_d)
    print(f"ViT output   shape : {vit_feat.shape}")
    print(f"ViT output   has NaN: {torch.isnan(vit_feat).any()}")
    print(f"ViT output   min/max: {vit_feat.min():.3f} / {vit_feat.max():.3f}")

print()
print("="*60)
print("STEP 3 — Forward pass through TimeSformer only")
print("="*60)

with torch.no_grad():
    tsf_feat = model.backbone.tsf_stream(clips_d)
    print(f"TSF output   shape : {tsf_feat.shape}")
    print(f"TSF output   has NaN: {torch.isnan(tsf_feat).any()}")
    print(f"TSF output   min/max: {tsf_feat.min():.3f} / {tsf_feat.max():.3f}")

print()
print("="*60)
print("STEP 4 — Full forward + loss")
print("="*60)

with torch.no_grad():
    logits, proj = model.forward_supcon(frames_d, clips_d)
    print(f"logits shape : {logits.shape}")
    print(f"logits       : {logits[:4].cpu().tolist()}")
    print(f"logits has NaN: {torch.isnan(logits).any()}")
    print(f"proj   has NaN: {torch.isnan(proj).any()}")

    probs = torch.softmax(logits, dim=-1)
    print(f"probs (fake) : {probs[:4, 1].cpu().tolist()}")

    focal_fn  = FocalLoss(alpha=0.25, gamma=2.0)
    supcon_fn = SupConLoss(temperature=0.07)

    focal = focal_fn(logits, labels_d)
    print(f"\nFocal loss   : {focal.item():.6f}  NaN={torch.isnan(focal).item()}")

    supcon = supcon_fn(proj, labels_d)
    print(f"SupCon loss  : {supcon.item():.6f}  NaN={torch.isnan(supcon).item()}")

print()
print("="*60)
print("STEP 5 — Check label distribution in train set")
print("="*60)
train_labels = [r['label'] for r in train_recs]
n_real = sum(1 for l in train_labels if l == 0)
n_fake = sum(1 for l in train_labels if l == 1)
print(f"Real : {n_real}  Fake : {n_fake}  Ratio: 1:{n_fake/max(n_real,1):.1f}")

print()
print("="*60)
print("STEP 6 — Sampler label distribution (first 200 samples)")
print("="*60)
sampler = make_weighted_sampler(train_recs)
indices = list(sampler)[:200]
sampled_labels = [train_recs[i]['label'] for i in indices]
n_real_s = sampled_labels.count(0)
n_fake_s = sampled_labels.count(1)
print(f"Sampled Real : {n_real_s}  Fake : {n_fake_s}  (should be ~50/50)")

print()
print("DONE — paste full output above and share it")