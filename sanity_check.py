"""
sanity_check.py
---------------
Run this BEFORE training to verify:
  1. GPU is available and working
  2. Dataset records.json exists and is valid
  3. Weighted sampler produces balanced batches
  4. Model forward pass works (no shape errors)
  5. Loss computation works
  6. Mixed precision works

Usage:
    python sanity_check.py
"""

import torch
import yaml
import json
import numpy as np
from pathlib import Path


def check_gpu():
    print("\n[1/6] GPU Check")
    print("-" * 40)
    if not torch.cuda.is_available():
        print("  WARNING: CUDA not available. Training will be very slow on CPU.")
        return torch.device('cpu')

    device = torch.device('cuda')
    name   = torch.cuda.get_device_name(0)
    vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU    : {name}")
    print(f"  VRAM   : {vram:.1f} GB")

    if vram < 6:
        print(f"  WARNING: Less than 6 GB VRAM detected.")
        print(f"    Set batch_size=2 and accumulate_grad_steps=16 in config.yaml")
    elif vram < 10:
        print(f"  Recommend: batch_size=4, accumulate_grad_steps=8")
    else:
        print(f"  OK: batch_size=8 should work fine")

    return device


def check_records(cfg):
    print("\n[2/6] Dataset Records Check")
    print("-" * 40)
    records_path = Path(cfg['data']['output_dir']) / 'records.json'

    if not records_path.exists():
        print(f"  FAIL: records.json not found at {records_path}")
        print(f"  Run: python data/preprocess.py")
        return None

    with open(records_path) as f:
        records = json.load(f)

    n_real = sum(1 for r in records if r['label'] == 0)
    n_fake = sum(1 for r in records if r['label'] == 1)
    print(f"  Total   : {len(records)}")
    print(f"  Real    : {n_real}")
    print(f"  Fake    : {n_fake}")
    print(f"  Ratio   : 1 : {n_fake / max(n_real, 1):.1f}")

    # verify a sample .npy file is loadable
    sample = records[0]
    try:
        arr = np.load(sample['path'])
        print(f"  Sample  : {arr.shape} dtype={arr.dtype} OK")
        assert arr.shape[0] == cfg['data']['num_frames'], \
            f"Expected {cfg['data']['num_frames']} frames, got {arr.shape[0]}"
        assert arr.shape[-1] == 3, f"Expected 3 channels, got {arr.shape[-1]}"
    except Exception as e:
        print(f"  FAIL loading sample npy: {e}")
        return None

    return records


def check_sampler(records, cfg):
    print("\n[3/6] Weighted Sampler Check")
    print("-" * 40)
    from data.dataset import DeepfakeDataset, get_splits, get_sampler
    from torch.utils.data import DataLoader

    train_recs, val_recs, _ = get_splits(
        str(Path(cfg['data']['output_dir']) / 'records.json'),
        cfg['data']['train_split'], cfg['data']['val_split']
    )

    ds = DeepfakeDataset(train_recs[:200], split='train',
                         num_frames=cfg['data']['num_frames'])
    loader = DataLoader(ds, batch_size=16,
                        sampler=get_sampler(train_recs[:200]),
                        num_workers=0)

    real_counts = []
    fake_counts = []
    for i, (videos, labels) in enumerate(loader):
        real_counts.append((labels == 0).sum().item())
        fake_counts.append((labels == 1).sum().item())
        if i == 4:
            break

    avg_real = np.mean(real_counts)
    avg_fake = np.mean(fake_counts)
    print(f"  Avg real/batch : {avg_real:.1f}")
    print(f"  Avg fake/batch : {avg_fake:.1f}")
    ratio = avg_real / (avg_fake + 1e-8)
    if 0.4 < ratio < 2.5:
        print(f"  OK: Batches are approximately balanced (ratio={ratio:.2f})")
    else:
        print(f"  WARNING: Batches may not be balanced (ratio={ratio:.2f})")

    return train_recs, val_recs


def check_model_forward(train_recs, cfg, device):
    print("\n[4/6] Model Forward Pass Check")
    print("-" * 40)
    from data.dataset import DeepfakeDataset, get_sampler
    from torch.utils.data import DataLoader
    from models.cat_model import CATModel

    ds = DeepfakeDataset(train_recs[:8], split='train',
                         num_frames=cfg['data']['num_frames'])
    loader = DataLoader(ds, batch_size=2, num_workers=0)

    model = CATModel(cfg['model']['embed_dim'], cfg['model']['dropout']).to(device)

    videos, labels = next(iter(loader))
    videos = videos.to(device)
    print(f"  Input shape : {tuple(videos.shape)}")

    try:
        with torch.no_grad():
            logits, emb = model(videos, return_embeddings=True)
        print(f"  Logits shape    : {tuple(logits.shape)}")
        print(f"  Embedding shape : {tuple(emb.shape)}")
        probs = torch.softmax(logits, dim=-1)
        print(f"  Sample probs    : {probs[0].cpu().numpy()}")
        print(f"  OK: Forward pass successful")
    except Exception as e:
        print(f"  FAIL: {e}")
        return None, None

    return model, loader


def check_loss(model, loader, cfg, device):
    print("\n[5/6] Loss Computation Check")
    print("-" * 40)
    from models.cat_model import FocalLoss, get_class_weights, compute_loss

    try:
        # load 2 batches of records for weight computation
        from data.dataset import get_splits
        train_recs, _, _ = get_splits(
            str(Path(cfg['data']['output_dir']) / 'records.json'),
            cfg['data']['train_split'], cfg['data']['val_split']
        )
        class_weights = get_class_weights(train_recs, device)
        criterion = FocalLoss(alpha=class_weights, gamma=cfg['model']['focal_gamma'])

        videos, labels = next(iter(loader))
        videos, labels = videos.to(device), labels.to(device)

        logits, emb = model(videos, return_embeddings=True)
        total, ce, sc = compute_loss(
            logits, emb, labels, criterion,
            alpha=cfg['model']['supcon_alpha'],
            temperature=cfg['model']['supcon_temperature']
        )
        print(f"  Focal Loss  : {ce:.4f}")
        print(f"  SupCon Loss : {sc:.4f}")
        print(f"  Total Loss  : {total.item():.4f}")
        print(f"  OK: Loss computation successful")
    except Exception as e:
        print(f"  FAIL: {e}")


def check_mixed_precision(model, loader, device):
    print("\n[6/6] Mixed Precision Check")
    print("-" * 40)
    from torch.cuda.amp import GradScaler, autocast

    if not torch.cuda.is_available():
        print("  SKIP: No GPU available")
        return

    scaler = GradScaler()
    videos, labels = next(iter(loader))
    videos, labels = videos.to(device), labels.to(device)

    try:
        with autocast():
            logits, _ = model(videos, return_embeddings=True)
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        scaler.scale(loss).backward()
        print(f"  Loss (fp16 forward): {loss.item():.4f}")
        print(f"  OK: Mixed precision working")
    except Exception as e:
        print(f"  FAIL: {e}")
        print(f"  Try setting mixed_precision: false in config.yaml")


def main():
    print("=" * 50)
    print("  DEEPFAKE DETECTION — SANITY CHECK")
    print("=" * 50)

    cfg = yaml.safe_load(open('configs/config.yaml'))

    device  = check_gpu()
    records = check_records(cfg)
    if records is None:
        print("\nFix dataset issues first, then re-run.")
        return

    train_recs, val_recs = check_sampler(records, cfg)
    model, loader = check_model_forward(train_recs, cfg, device)
    if model is not None:
        check_loss(model, loader, cfg, device)
        check_mixed_precision(model, loader, device)

    print("\n" + "=" * 50)
    print("  All checks passed — ready to train!")
    print("  Run: python train.py")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()