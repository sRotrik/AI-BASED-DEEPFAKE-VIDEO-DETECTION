"""
fast_ablations.py — Quick ablation study for IEEE paper table
Trains ViT-only and TimeSformer-only for 10 epochs each (~1 hour total).
Evaluates all 3 models and prints the final comparison table.

Run: python fast_ablations.py
Results saved to: results/ablation_table.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import json
from pathlib import Path
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from tqdm import tqdm
from transformers import ViTModel, TimesformerModel

from data.dataset import (
    FaceForensicsDataset, load_records,
    split_records, make_weighted_sampler,
)
from models.cat_model import CATModelWithSupCon, FocalLoss

torch.backends.cudnn.benchmark = True

ABLATION_EPOCHS = 10
MAX_FAKE        = 2000
DEVICE          = torch.device('cuda')
THRESHOLD       = 0.5308   # optimal threshold found earlier


# ─────────────────────────────────────────────────────────────
# Ablation models
# ─────────────────────────────────────────────────────────────

class ViTOnly(nn.Module):
    """ViT spatial stream only — no TimeSformer."""
    def __init__(self, embed_dim: int = 512, num_classes: int = 2):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        for param in self.vit.parameters():
            param.requires_grad = False
        for i in range(4):
            for param in self.vit.encoder.layer[11 - i].parameters():
                param.requires_grad = True
        for param in self.vit.layernorm.parameters():
            param.requires_grad = True
        self.proj = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, embed_dim),
            nn.GELU(), nn.Dropout(0.1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes))

    def forward(self, frames, clips=None):
        out  = self.vit(pixel_values=frames)
        feat = self.proj(out.last_hidden_state[:, 0, :])
        return self.classifier(feat)


class TimeSformerOnly(nn.Module):
    """TimeSformer temporal stream only — receives frame differences."""
    def __init__(self, embed_dim: int = 512, num_classes: int = 2):
        super().__init__()
        self.tsf = TimesformerModel.from_pretrained(
            "facebook/timesformer-base-finetuned-k400")
        for param in self.tsf.parameters():
            param.requires_grad = False
        num_blocks = len(self.tsf.encoder.layer)
        for i in range(2):
            for param in self.tsf.encoder.layer[num_blocks - 1 - i].parameters():
                param.requires_grad = True
        for param in self.tsf.layernorm.parameters():
            param.requires_grad = True
        self.proj = nn.Sequential(
            nn.Linear(self.tsf.config.hidden_size, embed_dim),
            nn.GELU(), nn.Dropout(0.1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes))

    def forward(self, frames, clips):
        x   = clips.float()
        out = self.tsf(pixel_values=x)
        feat = self.proj(out.last_hidden_state[:, 0, :])
        return self.classifier(feat)


# ─────────────────────────────────────────────────────────────
# Train one ablation model
# ─────────────────────────────────────────────────────────────

def train_ablation(model, train_loader, val_loader, name, epochs=10):
    print(f"\n{'='*55}")
    print(f"Training: {name}  ({epochs} epochs)")
    print(f"{'='*55}")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5, weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler     = GradScaler('cuda', enabled=True)
    criterion  = FocalLoss(alpha=0.75, gamma=0.0)

    best_auc  = 0.0
    save_path = Path(f"checkpoints/{name.lower().replace(' ', '_')}.pth")
    save_path.parent.mkdir(exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader,
                    desc=f"  [{name}] Epoch {epoch+1}/{epochs}")

        for frames, clips, labels in pbar:
            frames = frames.to(DEVICE, non_blocking=True)
            clips  = clips.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast('cuda', enabled=True):
                logits = model(frames, clips)
                loss   = criterion(logits, labels)

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()

        auc, bal_acc, f1 = evaluate(model, val_loader)
        print(f"  Epoch {epoch+1:02d} | "
              f"Loss: {epoch_loss/len(train_loader):.4f} | "
              f"AUC: {auc:.4f} | BalAcc: {bal_acc:.4f} | F1: {f1:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save({
                'model_state': model.state_dict(),
                'auc': auc, 'bal_acc': bal_acc, 'f1': f1,
            }, str(save_path))
            print(f"  *** Best {name} saved (AUC: {auc:.4f}) ***")

    return save_path


# ─────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for frames, clips, labels in loader:
        frames = frames.to(DEVICE, non_blocking=True)
        clips  = clips.to(DEVICE, non_blocking=True)
        logits = model(frames, clips)
        probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = (all_probs >= THRESHOLD).astype(int)
    auc     = roc_auc_score(all_labels, all_probs)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1      = f1_score(all_labels, all_preds, zero_division=0)
    return auc, bal_acc, f1


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    cfg = yaml.safe_load(open('configs/config.yaml'))

    # ── Data ─────────────────────────────────────────────────
    records_path = Path(cfg['data']['output_dir']) / 'records.json'
    all_records  = load_records(str(records_path), max_fake=MAX_FAKE)
    train_recs, val_recs, _ = split_records(all_records, 0.8, 0.1)

    train_ds = FaceForensicsDataset(
        train_recs, split='train', n_frames=cfg['data']['num_frames'])
    val_ds = FaceForensicsDataset(
        val_recs, split='val', n_frames=cfg['data']['num_frames'])

    train_loader = DataLoader(
        train_ds, batch_size=8,
        sampler=make_weighted_sampler(train_recs),
        num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False,
        num_workers=4, pin_memory=True)

    results = {}

    # ── 1. Main CAT+SupCon model (no training needed) ────────
    print("\nEvaluating CAT + SupCon (best_model.pth)...")
    main_model = CATModelWithSupCon(
        embed_dim=cfg['model']['embed_dim'],
        proj_dim=cfg['model'].get('proj_dim', 128),
    ).to(DEVICE)

    ckpt_path = Path("checkpoints_backup/best_model_0813.pth")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/best_model.pth")
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    main_model.load_state_dict(ckpt['model_state'])
    auc, bal_acc, f1 = evaluate(main_model, val_loader)
    print(f"  CAT+SupCon | AUC: {auc:.4f} | BalAcc: {bal_acc:.4f} | F1: {f1:.4f}")
    results['CAT + SupCon (Proposed)'] = {
        'auc': round(auc, 4),
        'bal_acc': round(bal_acc, 4),
        'f1': round(f1, 4),
        'epochs': 30,
    }
    del main_model
    torch.cuda.empty_cache()

    # ── 2. ViT Only ──────────────────────────────────────────
    vit_model = ViTOnly(embed_dim=cfg['model']['embed_dim']).to(DEVICE)
    save_path = train_ablation(
        vit_model, train_loader, val_loader, "ViT Only", ABLATION_EPOCHS)
    ckpt = torch.load(str(save_path), map_location=DEVICE)
    vit_model.load_state_dict(ckpt['model_state'])
    auc, bal_acc, f1 = evaluate(vit_model, val_loader)
    results['ViT Only'] = {
        'auc': round(auc, 4),
        'bal_acc': round(bal_acc, 4),
        'f1': round(f1, 4),
        'epochs': ABLATION_EPOCHS,
    }
    del vit_model
    torch.cuda.empty_cache()

    # ── 3. TimeSformer Only ──────────────────────────────────
    tsf_model = TimeSformerOnly(embed_dim=cfg['model']['embed_dim']).to(DEVICE)
    save_path = train_ablation(
        tsf_model, train_loader, val_loader,
        "TimeSformer Only", ABLATION_EPOCHS)
    ckpt = torch.load(str(save_path), map_location=DEVICE)
    tsf_model.load_state_dict(ckpt['model_state'])
    auc, bal_acc, f1 = evaluate(tsf_model, val_loader)
    results['TimeSformer Only (ΔFrames)'] = {
        'auc': round(auc, 4),
        'bal_acc': round(bal_acc, 4),
        'f1': round(f1, 4),
        'epochs': ABLATION_EPOCHS,
    }
    del tsf_model
    torch.cuda.empty_cache()

    # ── Print final table ────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  ABLATION STUDY — FaceForensics++")
    print(f"{'='*65}")
    print(f"  {'Model':<32} {'AUC':>7} {'Bal Acc':>9} {'F1':>8}")
    print(f"  {'-'*60}")
    for name, m in results.items():
        marker = ' ◄ proposed' if 'Proposed' in name else ''
        print(f"  {name:<32} {m['auc']:>7.4f} "
              f"{m['bal_acc']:>9.4f} {m['f1']:>8.4f}{marker}")
    print(f"{'='*65}")

    # ── Save ─────────────────────────────────────────────────
    results_dir = Path(cfg['logging']['results_dir'])
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / 'ablation_table.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_path}")
    print("Use these numbers in your IEEE paper Table IV.")


if __name__ == '__main__':
    main()