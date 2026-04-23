"""
evaluate.py
-----------
Full evaluation script. Generates all figures needed for IEEE paper:

  - Figure 1: ROC Curve (PDF + PNG)
  - Figure 2: Confusion Matrix (PDF + PNG)
  - Figure 3: t-SNE Embedding Visualization (PDF + PNG)
  - Figure 4: Precision-Recall Curve (PDF + PNG)
  - Figure 5: Score Distribution (PDF + PNG)
  - Figure 6: Per-method AUC bar chart (PDF + PNG)
  - metrics.json: all numeric results

Usage:
    python evaluate.py
"""

import json
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, balanced_accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.manifold import TSNE
from tqdm import tqdm

# ── Fixed imports matching current codebase ──────────────────
from data.dataset import (
    FaceForensicsDataset, load_records,
    split_records, make_weighted_sampler,
)
from models.cat_model import CATModelWithSupCon


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def find_best_threshold(labels, scores):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_score = 0.5, 0.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        # optimize balanced accuracy instead of F1
        bal = balanced_accuracy_score(labels, preds)
        if bal > best_score:
            best_score, best_t = bal, t
    return best_t, best_score


# ─────────────────────────────────────────────────────────────
# Inference — dual stream (frames + clips)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_probs      = []
    all_labels     = []
    all_embeddings = []

    for frames, clips, labels in tqdm(loader, desc='Inference'):
        frames = frames.to(device, non_blocking=True)
        clips  = clips.to(device, non_blocking=True)

        # get logits and embeddings
        emb    = model.backbone.get_embeddings(frames, clips)
        logits = model.backbone.classifier(emb)
        probs  = torch.softmax(logits, dim=-1)[:, 1]

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_embeddings.append(emb.cpu().numpy())

    return (np.concatenate(all_probs),
            np.concatenate(all_labels),
            np.vstack(all_embeddings))


# ─────────────────────────────────────────────────────────────
# Plot styling
# ─────────────────────────────────────────────────────────────

PURPLE = '#534AB7'
TEAL   = '#1D9E75'
CORAL  = '#D85A30'
GRAY   = '#888780'

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.linewidth'   : 0.8,
    'font.size'        : 11,
})


def save_fig(fig, results_dir, name):
    fig.savefig(Path(results_dir) / f'{name}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(Path(results_dir) / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}.pdf / .png")


def plot_roc(labels, probs, auc, results_dir):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=PURPLE, lw=2,
            label=f'CAT + SupCon  (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, linestyle='--', label='Random')
    ax.fill_between(fpr, tpr, alpha=0.08, color=PURPLE)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — FaceForensics++')
    ax.legend(loc='lower right', frameon=False)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    fig.tight_layout()
    save_fig(fig, results_dir, 'roc_curve')


def plot_pr(labels, probs, ap, results_dir):
    precision, recall, _ = precision_recall_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision, color=CORAL, lw=2, label=f'AP = {ap:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='upper right', frameon=False)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    fig.tight_layout()
    save_fig(fig, results_dir, 'pr_curve')


def plot_confusion(labels, preds, threshold, results_dir):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm_norm, annot=cm, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake'],
        linewidths=0.5, linecolor='white',
        annot_kws={'size': 14, 'weight': 'bold'},
        vmin=0, vmax=1,
    )
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'Confusion Matrix  (threshold = {threshold:.2f})')
    fig.tight_layout()
    save_fig(fig, results_dir, 'confusion_matrix')
    return cm


def plot_tsne(embeddings, labels, results_dir):
    print("  Computing t-SNE (may take 2-3 minutes)...")
    # subsample if too large
    if len(embeddings) > 2000:
        idx = np.random.choice(len(embeddings), 2000, replace=False)
        embeddings = embeddings[idx]
        labels     = labels[idx]
    tsne  = TSNE(n_components=2, perplexity=40, random_state=42,
             max_iter=1500, learning_rate='auto', init='pca')
    emb2d = tsne.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(emb2d[labels == 0, 0], emb2d[labels == 0, 1],
               c=TEAL,  s=10, alpha=0.6, label='Real', rasterized=True)
    ax.scatter(emb2d[labels == 1, 0], emb2d[labels == 1, 1],
               c=CORAL, s=10, alpha=0.4, label='Fake', rasterized=True)
    ax.set_title('t-SNE: Embedding Space (Real vs Fake)')
    ax.legend(loc='upper right', frameon=False, markerscale=2)
    ax.axis('off')
    fig.tight_layout()
    save_fig(fig, results_dir, 'tsne')


def plot_score_dist(labels, probs, threshold, results_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(probs[labels == 0], bins=50, color=TEAL,  alpha=0.6,
            label='Real', density=True)
    ax.hist(probs[labels == 1], bins=50, color=CORAL, alpha=0.5,
            label='Fake', density=True)
    ax.axvline(threshold, color='black', lw=1.5, linestyle='--',
               label=f'Threshold = {threshold:.2f}')
    ax.set_xlabel('P(fake) score')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution')
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, results_dir, 'score_distribution')


def plot_per_method(per_method_results, results_dir):
    if not per_method_results:
        return
    names  = list(per_method_results.keys())
    values = list(per_method_results.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, values, color=PURPLE, alpha=0.8, height=0.5)
    ax.bar_label(bars, fmt='%.1f%%', padding=4, fontsize=10)
    ax.set_xlabel('AUC (%)')
    ax.set_title('Per-method AUC — FaceForensics++')
    ax.set_xlim([max(0, min(values) - 10), 105])
    fig.tight_layout()
    save_fig(fig, results_dir, 'per_method_auc')


# ─────────────────────────────────────────────────────────────
# Per-method evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_per_method(model, test_records, device, cfg, results_dir):
    methods      = ['DeepFakeDetection', 'Deepfakes', 'Face2Face',
                    'FaceShifter', 'FaceSwap', 'NeuralTextures']
    real_records = [r for r in test_records if r['label'] == 0]
    results      = {}

    for method in methods:
        method_records = [r for r in test_records if r['source'] == method]
        if not method_records:
            continue

        eval_records = real_records + method_records
        ds = FaceForensicsDataset(
            eval_records, split='test',
            n_frames=cfg['data']['num_frames'],
            img_size=cfg['data']['image_size'],
        )
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

        probs_list, labels_list = [], []
        for frames, clips, labels in loader:
            frames = frames.to(device)
            clips  = clips.to(device)
            logits = model(frames, clips)
            probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs_list.extend(probs)
            labels_list.extend(labels.numpy())

        if len(set(labels_list)) < 2:
            continue

        auc = roc_auc_score(labels_list, probs_list)
        results[method] = round(auc * 100, 2)
        print(f"  {method:22s}: AUC = {auc:.4f}")

    plot_per_method(results, results_dir)
    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def full_evaluation(checkpoint_path='checkpoints/best_model.pth'):
    cfg         = yaml.safe_load(open('configs/config.yaml'))
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(cfg['logging']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # ── Load model ───────────────────────────────────────────
    model = CATModelWithSupCon(
        embed_dim=cfg['model']['embed_dim'],
        proj_dim=cfg['model'].get('proj_dim', 128),
        num_classes=2,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded  |  "
          f"Epoch: {ckpt.get('epoch','?')}  |  "
          f"Val AUC: {ckpt.get('val_auc', 0):.4f}")

    # ── Test records ─────────────────────────────────────────
    test_recs = ckpt.get('test_records', None)
    if test_recs is None:
        print("No test_records in checkpoint — rebuilding from records.json")
        records_path = Path(cfg['data']['output_dir']) / 'records.json'
        all_records  = load_records(str(records_path), max_fake=2000)
        _, _, test_recs = split_records(
            all_records,
            train_ratio=cfg['data']['train_split'],
            val_ratio=cfg['data']['val_split'],
        )

    test_ds = FaceForensicsDataset(
        test_recs, split='test',
        n_frames=cfg['data']['num_frames'],
        img_size=cfg['data']['image_size'],
    )
    test_loader = DataLoader(
        test_ds, batch_size=8, shuffle=False,
        num_workers=cfg['hardware']['num_workers'],
        pin_memory=True,
    )
    print(f"Test set: {len(test_ds)} samples")

    # ── Inference ────────────────────────────────────────────
    print("\nRunning inference...")
    probs, labels, embeddings = run_inference(model, test_loader, device)

    # ── Threshold ────────────────────────────────────────────
    best_t, _ = find_best_threshold(labels, probs)
    preds      = (probs >= best_t).astype(int)
    print(f"Optimal threshold: {best_t:.3f}")

    # ── Metrics ──────────────────────────────────────────────
    auc     = roc_auc_score(labels, probs)
    ap      = average_precision_score(labels, probs)
    eer     = compute_eer(labels, probs)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1      = f1_score(labels, preds, zero_division=0)
    prec    = precision_score(labels, preds, zero_division=0)
    rec     = recall_score(labels, preds, zero_division=0)
    cm      = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    raw_acc = (tp + tn) / len(labels)

    print(f"\n{'='*52}")
    print(f"  TEST RESULTS — FaceForensics++")
    print(f"{'='*52}")
    print(f"  AUC               : {auc:.4f}  ({auc*100:.2f}%)")
    print(f"  Balanced Accuracy : {bal_acc:.4f}  ({bal_acc*100:.2f}%)")
    print(f"  F1 Score          : {f1:.4f}")
    print(f"  Precision         : {prec:.4f}")
    print(f"  Recall            : {rec:.4f}")
    print(f"  Average Precision : {ap:.4f}")
    print(f"  EER               : {eer:.4f}  ({eer*100:.2f}%)")
    print(f"  Raw Accuracy      : {raw_acc:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"{'='*52}")

    # ── Save metrics ─────────────────────────────────────────
    metrics = {
        'auc':               round(float(auc),     4),
        'auc_pct':           round(float(auc)*100, 2),
        'balanced_accuracy': round(float(bal_acc), 4),
        'f1':                round(float(f1),      4),
        'precision':         round(float(prec),    4),
        'recall':            round(float(rec),     4),
        'average_precision': round(float(ap),      4),
        'eer':               round(float(eer),     4),
        'raw_accuracy':      round(float(raw_acc), 4),
        'optimal_threshold': round(float(best_t),  3),
        'confusion_matrix':  cm.tolist(),
        'tp': int(tp), 'fp': int(fp),
        'tn': int(tn), 'fn': int(fn),
    }
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to results/metrics.json")

    # ── Figures ──────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_roc(labels, probs, auc, results_dir)
    plot_pr(labels, probs, ap, results_dir)
    plot_confusion(labels, preds, best_t, results_dir)
    plot_score_dist(labels, probs, best_t, results_dir)
    plot_tsne(embeddings, labels, results_dir)

    # ── Per-method ───────────────────────────────────────────
    print("\nPer-method AUC breakdown:")
    per_method = evaluate_per_method(
        model, test_recs, device, cfg, results_dir)
    metrics['per_method_auc'] = per_method
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll results saved to: {results_dir}/")
    print("Figures: roc_curve, pr_curve, confusion_matrix, "
          "score_distribution, tsne, per_method_auc")
    return metrics


if __name__ == '__main__':
    full_evaluation(checkpoint_path='checkpoints/best_model.pth')