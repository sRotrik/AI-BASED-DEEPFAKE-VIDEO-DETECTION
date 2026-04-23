"""
train.py
--------
Key fix: replaced OneCycleLR (caused LR spike → classifier explosion) with
linear warmup + cosine decay. Also uses separate param groups:
  - backbone (ViT/TSF unfrozen layers): lr = 1e-5  (small, pretrained)
  - classifier + proj head            : lr = 1e-4  (larger, random init)

Start:   python train.py
Resume:  python train.py --resume
"""

import yaml
import torch
import numpy as np
import argparse
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from tqdm import tqdm
from pathlib import Path

from data.dataset import (
    FaceForensicsDataset, load_records,
    split_records, make_weighted_sampler,
)
from models.cat_model import (
    CATModelWithSupCon, FocalLoss, SupConLoss, compute_loss,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

SUPCON_WARMUP_EPOCHS = 2
MAX_FAKE             = 2000


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state, save_dir, epoch, is_best=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    epoch_path = save_dir / f'epoch_{epoch:02d}.pth'
    torch.save(state, str(epoch_path))
    if is_best:
        torch.save(state, str(save_dir / 'best_model.pth'))
        print(f"  *** New best model saved (AUC: {state['val_auc']:.4f}) ***")
    for old in sorted(save_dir.glob('epoch_*.pth'))[:-3]:
        old.unlink()
    with open(save_dir / 'last_checkpoint.txt', 'w') as f:
        f.write(str(epoch_path))


def find_last_checkpoint(save_dir):
    save_dir = Path(save_dir)
    tracker  = save_dir / 'last_checkpoint.txt'
    if tracker.exists():
        p = Path(tracker.read_text().strip())
        if p.exists():
            return str(p)
    files = sorted(save_dir.glob('epoch_*.pth'))
    return str(files[-1]) if files else None


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    print(f"\nResuming from: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    scaler.load_state_dict(ckpt['scaler_state'])
    start_epoch = ckpt['epoch']
    best_auc    = ckpt.get('best_auc', ckpt.get('val_auc', 0.0))
    print(f"  Resumed at epoch {start_epoch} | Best AUC: {best_auc:.4f}")
    return start_epoch, best_auc


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for frames, clips, labels in loader:
        frames = frames.to(device, non_blocking=True)
        clips  = clips.to(device, non_blocking=True)
        logits = model(frames, clips)
        probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = (all_probs >= 0.5).astype(int)
    auc     = roc_auc_score(all_labels, all_probs)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1      = f1_score(all_labels, all_preds, zero_division=0)
    return auc, bal_acc, f1


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(resume=False, checkpoint_path=None):
    cfg    = yaml.safe_load(open('configs/config.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nDevice : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Data ─────────────────────────────────────────────────────────────────
    records_path = Path(cfg['data']['output_dir']) / 'records.json'
    if not records_path.exists():
        raise FileNotFoundError(
            f"records.json not found at {records_path}\n"
            f"Run: python data/preprocess.py")

    all_records = load_records(str(records_path), max_fake=MAX_FAKE)
    train_recs, val_recs, test_recs = split_records(
        all_records,
        train_ratio=cfg['data']['train_split'],
        val_ratio=cfg['data']['val_split'],
    )

    train_ds = FaceForensicsDataset(
        train_recs, split='train',
        n_frames=cfg['data']['num_frames'],
        img_size=cfg['data']['image_size'],
    )
    val_ds = FaceForensicsDataset(
        val_recs, split='val',
        n_frames=cfg['data']['num_frames'],
        img_size=cfg['data']['image_size'],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        sampler=make_weighted_sampler(train_recs),
        num_workers=cfg['hardware']['num_workers'],
        pin_memory=cfg['hardware']['pin_memory'],
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['hardware']['num_workers'],
        pin_memory=cfg['hardware']['pin_memory'],
        persistent_workers=True,
    )

    print(f"\nTrain : {len(train_ds)} samples  |  Val : {len(val_ds)} samples")

    # ── Model ────────────────────────────────────────────────────────────────
    model = CATModelWithSupCon(
        embed_dim=cfg['model']['embed_dim'],
        proj_dim=cfg['model'].get('proj_dim', 128),
        num_classes=2,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params : {trainable:,}")

    # ── Loss ─────────────────────────────────────────────────────────────────
    focal_loss_fn  = FocalLoss(
        alpha=cfg['model'].get('focal_alpha', 0.75),
        gamma=cfg['model'].get('focal_gamma', 2.0),
    )
    supcon_loss_fn = SupConLoss(
        temperature=cfg['model'].get('supcon_temperature', 0.1),
    )

    # ── Optimizer — separate LR for backbone vs head ──────────────────────────
    # Pretrained backbone layers get 10x smaller LR than randomly-init head
    backbone_params = []
    head_params     = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'vit_stream' in name or 'tsf_stream' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    base_lr      = cfg['training']['lr']          # 1e-4 for head
    backbone_lr  = base_lr * 0.1                  # 1e-5 for backbone

    optimizer = AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params,     'lr': base_lr},
    ], weight_decay=cfg['training']['weight_decay'])

    print(f"Backbone LR : {backbone_lr:.1e}  |  Head LR : {base_lr:.1e}")

    # ── Scheduler — linear warmup (2 epochs) then cosine decay ───────────────
    total_epochs  = cfg['training']['epochs']
    accum_steps   = cfg['training']['accumulate_grad_steps']
    steps_per_epoch = len(train_loader) // accum_steps
    total_steps   = steps_per_epoch * total_epochs
    warmup_steps  = steps_per_epoch * 2              # 2 epoch warmup

    warmup_sched = LinearLR(
        optimizer,
        start_factor=0.1,    # start at 10% of base_lr
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=base_lr * 0.01,   # decay to 1% of base_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps],
    )

    use_amp = cfg['training']['mixed_precision'] and torch.cuda.is_available()
    scaler  = GradScaler('cuda', enabled=use_amp)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_auc    = 0.0
    if resume:
        ckpt_path = checkpoint_path or find_last_checkpoint(cfg['logging']['save_dir'])
        if ckpt_path:
            start_epoch, best_auc = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, scaler, device)
        else:
            print("No checkpoint found — starting from scratch.")

    # ── TensorBoard ──────────────────────────────────────────────────────────
    save_dir = Path(cfg['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / 'tb_logs'))

    print(f"\n{'='*60}")
    print(f"Training epochs   : {start_epoch + 1} → {total_epochs}")
    print(f"Effective batch   : {cfg['training']['batch_size'] * accum_steps}")
    print(f"Fake cap          : {MAX_FAKE}")
    print(f"Scheduler         : LinearWarmup({warmup_steps} steps) → CosineDecay")
    print(f"Mixed precision   : {use_amp}")
    print(f"SupCon warmup     : first {SUPCON_WARMUP_EPOCHS} epochs focal-only")
    print(f"{'='*60}\n")

    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss = epoch_ce = epoch_sc = 0.0
        nan_batches = 0
        optimizer.zero_grad()

        use_supcon = (epoch >= SUPCON_WARMUP_EPOCHS)
        if not use_supcon:
            print(f"  [Warmup epoch {epoch+1}] focal loss only")

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1:02d}/{total_epochs}")

        for step, (frames, clips, labels) in pbar:
            frames = frames.to(device, non_blocking=True)
            clips  = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast('cuda', enabled=use_amp):
                logits, proj = model.forward_supcon(frames, clips)
                loss, ce_val, sc_val = compute_loss(
                    logits=logits,
                    labels=labels,
                    proj=proj if use_supcon else None,
                    focal_loss_fn=focal_loss_fn,
                    supcon_loss_fn=supcon_loss_fn if use_supcon else None,
                    alpha=cfg['model'].get('supcon_alpha', 0.6),
                )
                loss = loss / accum_steps

            if torch.isnan(loss):
                nan_batches += 1
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg['training']['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            actual_loss = loss.item() * accum_steps
            ce_item     = ce_val.item() if torch.is_tensor(ce_val) else float(ce_val)
            sc_item     = sc_val.item() if torch.is_tensor(sc_val) else float(sc_val)
            epoch_loss += actual_loss
            epoch_ce   += ce_item
            epoch_sc   += sc_item

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{actual_loss:.4f}',
                'ce':   f'{ce_item:.4f}',
                'sc':   f'{sc_item:.4f}' if use_supcon else 'off',
                'lr':   f'{current_lr:.2e}',
            })

            if step % cfg['logging']['log_every'] == 0:
                writer.add_scalar('train/loss',    actual_loss, global_step)
                writer.add_scalar('train/ce_loss', ce_item,     global_step)
                writer.add_scalar('train/sc_loss', sc_item,     global_step)
                writer.add_scalar('train/lr',      current_lr,  global_step)

        # ── Validation ────────────────────────────────────────────────────────
        val_auc, val_bal_acc, val_f1 = evaluate(model, val_loader, device)
        n        = max(len(train_loader) - nan_batches, 1)
        avg_loss = epoch_loss / n

        writer.add_scalar('val/auc',          val_auc,     epoch)
        writer.add_scalar('val/bal_acc',      val_bal_acc, epoch)
        writer.add_scalar('val/f1',           val_f1,      epoch)
        writer.add_scalar('train/epoch_loss', avg_loss,    epoch)

        print(f"\nEpoch {epoch+1:02d}/{total_epochs}")
        print(f"  Loss     : {avg_loss:.4f}  "
              f"(CE: {epoch_ce/n:.4f}  SC: {epoch_sc/n:.4f})")
        print(f"  Val AUC  : {val_auc:.4f}")
        print(f"  Bal Acc  : {val_bal_acc:.4f}")
        print(f"  F1       : {val_f1:.4f}")
        if nan_batches:
            print(f"  NaN batches skipped : {nan_batches}")

        is_best  = val_auc > best_auc
        best_auc = max(best_auc, val_auc)

        save_checkpoint({
            'epoch':           epoch + 1,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state':    scaler.state_dict(),
            'val_auc':         val_auc,
            'val_bal_acc':     val_bal_acc,
            'val_f1':          val_f1,
            'best_auc':        best_auc,
            'cfg':             cfg,
            'test_records':    test_recs,
        }, cfg['logging']['save_dir'], epoch + 1, is_best=is_best)

        print(f"  Checkpoint : epoch_{epoch+1:02d}.pth saved\n")

    writer.close()
    print(f"\n{'='*60}")
    print(f"Training complete!  Best Val AUC : {best_auc:.4f}")
    print(f"Next: python evaluate.py")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    train(resume=args.resume, checkpoint_path=args.checkpoint)