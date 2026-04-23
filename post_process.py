import json
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, roc_curve
)
from tqdm import tqdm

from data.dataset import FaceForensicsDataset, load_records, split_records
from models.cat_model import CATModelWithSupCon


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

CHECKPOINT_PATH = 'checkpoints/best_model.pth'

# If you have more checkpoints, add their paths here:
ALL_CHECKPOINTS = [
    'checkpoints/best_model.pth',
    # 'checkpoints/epoch_25.pth',
    # 'checkpoints/epoch_20.pth',
]

TTA_AUGMENTATIONS = 5   # number of noisy forward passes
NOISE_STD         = 0.02


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def find_best_threshold(labels, scores):
    """Find threshold that maximizes balanced accuracy."""
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_score = 0.5, 0.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        bal = balanced_accuracy_score(labels, preds)
        if bal > best_score:
            best_score, best_t = bal, t
    return best_t


def print_metrics(label, probs, labels):
    best_t  = find_best_threshold(labels, probs)
    preds   = (probs >= best_t).astype(int)
    auc     = roc_auc_score(labels, probs)
    ap      = average_precision_score(labels, probs)
    eer     = compute_eer(labels, probs)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1      = f1_score(labels, preds, zero_division=0)
    prec    = precision_score(labels, preds, zero_division=0)
    rec     = recall_score(labels, preds, zero_division=0)

    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    print(f"  AUC               : {auc:.4f}  ({auc*100:.2f}%)")
    print(f"  Balanced Accuracy : {bal_acc:.4f}  ({bal_acc*100:.2f}%)")
    print(f"  F1 Score          : {f1:.4f}")
    print(f"  Precision         : {prec:.4f}")
    print(f"  Recall            : {rec:.4f}")
    print(f"  Average Precision : {ap:.4f}")
    print(f"  EER               : {eer:.4f}  ({eer*100:.2f}%)")
    print(f"  Optimal Threshold : {best_t:.3f}")
    print(f"{'='*52}")

    return {
        'auc': round(float(auc), 4),
        'auc_pct': round(float(auc)*100, 2),
        'balanced_accuracy': round(float(bal_acc), 4),
        'f1': round(float(f1), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'average_precision': round(float(ap), 4),
        'eer': round(float(eer), 4),
        'optimal_threshold': round(float(best_t), 3),
    }


# ─────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device, cfg):
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = CATModelWithSupCon(
        embed_dim=cfg['model']['embed_dim'],
        proj_dim=cfg['model'].get('proj_dim', 128),
        num_classes=2,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  Loaded: {checkpoint_path}  "
          f"| Epoch: {ckpt.get('epoch','?')}  "
          f"| Val AUC: {ckpt.get('val_auc', 0):.4f}")
    return model, ckpt


# ─────────────────────────────────────────────────────────────
# Standard inference
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    all_probs, all_labels = [], []
    for frames, clips, labels in tqdm(loader, desc='  Inference'):
        frames = frames.to(device, non_blocking=True)
        clips  = clips.to(device, non_blocking=True)
        emb    = model.backbone.get_embeddings(frames, clips)
        logits = model.backbone.classifier(emb)
        probs  = torch.softmax(logits, dim=-1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ─────────────────────────────────────────────────────────────
# TTA inference
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference_tta(model, loader, device, n_aug=TTA_AUGMENTATIONS):
    all_probs, all_labels = [], []
    for frames, clips, labels in tqdm(loader, desc=f'  TTA x{n_aug}'):
        aug_probs = []
        for _ in range(n_aug):
            # add tiny noise to frames
            f = (frames + torch.randn_like(frames) * NOISE_STD).to(device)
            c = clips.to(device)
            emb    = model.backbone.get_embeddings(f, c)
            logits = model.backbone.classifier(emb)
            probs  = torch.softmax(logits, dim=-1)[:, 1]
            aug_probs.append(probs.cpu().numpy())
        # average over augmentations
        all_probs.append(np.mean(aug_probs, axis=0))
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ─────────────────────────────────────────────────────────────
# Platt Scaling calibration
# ─────────────────────────────────────────────────────────────

def apply_platt_scaling(val_probs, val_labels, test_probs):
    print("\n  Fitting Platt Scaling on val probs...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(val_probs.reshape(-1, 1), val_labels)
    calibrated = lr.predict_proba(test_probs.reshape(-1, 1))[:, 1]
    print("  Platt Scaling applied.")
    return calibrated


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    cfg    = yaml.safe_load(open('configs/config.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ── Load checkpoint + records ─────────────────────────────
    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    model, ckpt = load_model(CHECKPOINT_PATH, device, cfg)

    test_recs = ckpt.get('test_records', None)
    val_recs  = ckpt.get('val_records',  None)

    if test_recs is None:
        print("  No test_records in checkpoint — rebuilding from records.json")
        records_path = Path(cfg['data']['output_dir']) / 'records.json'
        all_records  = load_records(str(records_path), max_fake=2000)
        train_recs, val_recs, test_recs = split_records(
            all_records,
            train_ratio=cfg['data']['train_split'],
            val_ratio=cfg['data']['val_split'],
        )

    # ── Build DataLoaders ─────────────────────────────────────
    def make_loader(records):
        ds = FaceForensicsDataset(
            records, split='test',
            n_frames=cfg['data']['num_frames'],
            img_size=cfg['data']['image_size'],
        )
        return DataLoader(
            ds, batch_size=8, shuffle=False,
            num_workers=cfg['hardware']['num_workers'],
            pin_memory=True,
        )

    test_loader = make_loader(test_recs)
    print(f"  Test set : {len(test_recs)} samples")

    # ── BASELINE (original) ───────────────────────────────────
    print("\n[1/4] Baseline inference...")
    base_probs, labels = run_inference(model, test_loader, device)
    baseline_metrics   = print_metrics("BASELINE", base_probs, labels)

    # ── TTA ───────────────────────────────────────────────────
    print(f"\n[2/4] Test-Time Augmentation (TTA x{TTA_AUGMENTATIONS})...")
    tta_probs, _   = run_inference_tta(model, test_loader, device)
    tta_metrics    = print_metrics("TTA", tta_probs, labels)

    # ── PLATT SCALING ─────────────────────────────────────────
    print("\n[3/4] Platt Scaling calibration...")
    if val_recs is not None:
        val_loader          = make_loader(val_recs)
        val_probs, val_lbls = run_inference(model, val_loader, device)
        cal_probs           = apply_platt_scaling(val_probs, val_lbls, base_probs)
        cal_metrics         = print_metrics("PLATT SCALED", cal_probs, labels)

        # Platt + TTA combined
        cal_tta_probs   = apply_platt_scaling(val_probs, val_lbls, tta_probs)
        cal_tta_metrics = print_metrics("PLATT + TTA", cal_tta_probs, labels)
    else:
        print("  Skipping Platt Scaling — no val_records in checkpoint.")
        cal_metrics     = {}
        cal_tta_metrics = {}

    # ── ENSEMBLE (if multiple checkpoints) ────────────────────
    print("\n[4/4] Checkpoint Ensembling...")
    existing_ckpts = [p for p in ALL_CHECKPOINTS if Path(p).exists()]
    print(f"  Found {len(existing_ckpts)} checkpoint(s): {existing_ckpts}")

    if len(existing_ckpts) > 1:
        all_probs_list = []
        for ckpt_path in existing_ckpts:
            m, _ = load_model(ckpt_path, device, cfg)
            p, _ = run_inference(m, test_loader, device)
            all_probs_list.append(p)
        ensemble_probs   = np.mean(all_probs_list, axis=0)
        ensemble_metrics = print_metrics("ENSEMBLE", ensemble_probs, labels)
    else:
        print("  Only 1 checkpoint found — skipping ensemble.")
        print("  Add more checkpoint paths to ALL_CHECKPOINTS in post_process.py")
        ensemble_metrics = {}

    # ── Save all results ──────────────────────────────────────
    results_dir = Path(cfg['logging']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'baseline'   : baseline_metrics,
        'tta'        : tta_metrics,
        'platt'      : cal_metrics,
        'platt_tta'  : cal_tta_metrics,
        'ensemble'   : ensemble_metrics,
    }

    out_path = results_dir / 'post_process_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {out_path}")


if __name__ == '__main__':
    main()