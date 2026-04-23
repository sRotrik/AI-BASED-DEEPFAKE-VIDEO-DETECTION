"""
inference.py
------------
Run the trained model on a single video file.
Ideal for demonstrating the system in your final review.

Usage:
    python inference.py --video /path/to/video.mp4
    python inference.py --video /path/to/video.mp4 --show_frames
"""

import argparse
import cv2
import torch
import numpy as np
import yaml
import time
from pathlib import Path
from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2

# FIX: use CATModelWithSupCon (not CATModel)
from models.cat_model import CATModelWithSupCon

THRESHOLD = 0.5308   # optimal threshold found during evaluation


# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path, cfg, device):
    model = CATModelWithSupCon(
        embed_dim=cfg['model']['embed_dim'],
        proj_dim=cfg['model'].get('proj_dim', 128),
        num_classes=2,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  Checkpoint epoch : {ckpt.get('epoch', '?')}")
    print(f"  Val AUC          : {ckpt.get('val_auc', 0):.4f}")
    return model


# ─────────────────────────────────────────────────────────────
# Video processing — matches dataset.py exactly
# ─────────────────────────────────────────────────────────────

def extract_faces_from_video(video_path, mtcnn, num_frames=16):
    """
    Extract num_frames+1 evenly-spaced face crops from video.
    +1 because we need n+1 frames to compute n frame differences.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps if fps > 0 else 0

    # need num_frames+1 raw frames for diff computation
    n_raw   = num_frames + 1
    indices = np.linspace(0, total - 1, n_raw, dtype=int)

    raw_frames  = []
    face_frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_frames.append(rgb)

        face = mtcnn(rgb)
        if face is not None:
            face_np = face.permute(1, 2, 0).numpy()
            face_np = ((face_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            face_frames.append(face_np)
        elif face_frames:
            face_frames.append(face_frames[-1])   # pad with last good face
        else:
            # fallback: resize whole frame
            resized = cv2.resize(rgb, (224, 224))
            face_frames.append(resized)

    cap.release()

    print(f"  Video    : {total} frames @ {fps:.1f} fps ({duration:.1f}s)")
    print(f"  Faces    : {len(face_frames)} / {n_raw} sampled frames")

    if len(face_frames) == 0:
        raise ValueError("No faces detected in this video.")

    # pad or trim to exactly n_raw
    while len(face_frames) < n_raw:
        face_frames.append(face_frames[-1])
    face_frames = face_frames[:n_raw]

    return face_frames, raw_frames


def preprocess(face_frames, image_size=224):
    """
    Returns:
      frame_t : (1, C, H, W)       — single frame for ViT
      clip_t  : (1, T, C, H, W)    — frame differences for TimeSformer
    Matches dataset.py preprocessing exactly.
    """
    # ── Single frame (middle) for ViT ────────────────────────
    frame_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    middle_face = face_frames[len(face_frames) // 2]
    frame_t     = frame_transform(image=middle_face)["image"]  # (C,H,W)
    frame_t     = frame_t.unsqueeze(0)                         # (1,C,H,W)

    # ── Frame differences for TimeSformer ────────────────────
    diff_transform = A.Compose([
        A.Resize(image_size, image_size),
        ToTensorV2(),
    ])
    diff_frames = []
    for t in range(len(face_frames) - 1):
        f1   = face_frames[t].astype(np.int16)
        f2   = face_frames[t + 1].astype(np.int16)
        diff = ((f2 - f1 + 255) / 2).astype(np.uint8)   # [0,255]
        diff_frames.append(diff_transform(image=diff)["image"])

    clip_t = torch.stack(diff_frames, dim=0).float() / 255.0   # [0,1]
    clip_t = (clip_t - 0.5) * 2.0                              # [-1,1]
    clip_t = clip_t.unsqueeze(0)                               # (1,T,C,H,W)

    return frame_t, clip_t


# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, frame_t, clip_t, device):
    frame_t = frame_t.to(device)
    clip_t  = clip_t.to(device)

    t0     = time.time()
    logits = model(frame_t, clip_t)       # dual-stream forward
    elapsed = time.time() - t0

    probs     = torch.softmax(logits, dim=-1)[0]
    prob_real = float(probs[0].cpu())
    prob_fake = float(probs[1].cpu())

    return prob_real, prob_fake, elapsed


# ─────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────

def print_result(video_path, prob_real, prob_fake, elapsed):
    verdict    = "FAKE" if prob_fake >= THRESHOLD else "REAL"
    confidence = prob_fake if verdict == "FAKE" else prob_real
    emoji      = "🚨" if verdict == "FAKE" else "✅"

    print(f"\n{'='*55}")
    print(f"  Video      : {Path(video_path).name}")
    print(f"  P(real)    : {prob_real:.4f}  ({prob_real*100:.1f}%)")
    print(f"  P(fake)    : {prob_fake:.4f}  ({prob_fake*100:.1f}%)")
    print(f"  Threshold  : {THRESHOLD}")
    print(f"  {emoji} Verdict : {verdict}  ({confidence*100:.1f}% confident)")
    print(f"  Latency    : {elapsed*1000:.1f} ms")

    # ASCII confidence bar
    bar_len = 40
    n_fake  = int(prob_fake * bar_len)
    n_real  = bar_len - n_fake
    bar     = f"  REAL [{'█'*n_real}{'░'*n_fake}] FAKE"
    print(bar)
    print(f"{'='*55}\n")


def show_faces_grid(face_frames, prob_fake, results_dir):
    """Save grid of extracted face crops + frame differences."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        n_frames = len(face_frames) - 1   # raw faces
        cols     = 4
        rows     = (n_frames + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
        axes = axes.flatten() if rows > 1 else [axes] * cols

        for i in range(n_frames):
            axes[i].imshow(face_frames[i])
            axes[i].set_title(f'Frame {i+1}', fontsize=8)
            axes[i].axis('off')
        for i in range(n_frames, len(axes)):
            axes[i].axis('off')

        verdict = 'FAKE 🚨' if prob_fake >= THRESHOLD else 'REAL ✅'
        color   = 'red' if prob_fake >= THRESHOLD else 'green'
        fig.suptitle(
            f'Extracted Face Crops — Verdict: {verdict}  '
            f'(P(fake)={prob_fake:.3f})',
            fontsize=13, fontweight='bold', color=color
        )
        plt.tight_layout()
        out_path = Path(results_dir) / 'inference_faces.png'
        plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
        print(f"  Face grid saved to {out_path}")
        plt.close()

        # Also save frame differences
        fig2, axes2 = plt.subplots(rows, cols, figsize=(12, rows * 3))
        axes2 = axes2.flatten() if rows > 1 else [axes2] * cols
        for i in range(n_frames):
            f1   = face_frames[i].astype(np.int16)
            f2   = face_frames[i + 1].astype(np.int16)
            diff = ((f2 - f1 + 255) / 2).astype(np.uint8)
            axes2[i].imshow(diff)
            axes2[i].set_title(f'Δ Frame {i+1}→{i+2}', fontsize=8)
            axes2[i].axis('off')
        for i in range(n_frames, len(axes2)):
            axes2[i].axis('off')
        fig2.suptitle('Frame Differences (TimeSformer Input) — '
                      'Brighter = More Change', fontsize=12)
        plt.tight_layout()
        diff_path = Path(results_dir) / 'inference_diffs.png'
        plt.savefig(str(diff_path), dpi=150, bbox_inches='tight')
        print(f"  Diff grid saved to {diff_path}")
        plt.close()

    except Exception as e:
        print(f"Could not save face grid: {e}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Deepfake detection inference')
    parser.add_argument('--video',      required=True,
                        help='Path to input video (.mp4)')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--config',     default='configs/config.yaml')
    parser.add_argument('--show_frames', action='store_true',
                        help='Save grid of extracted face crops + diffs')
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nDevice   : {device}")
    print(f"Video    : {args.video}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, cfg, device)

    # Face detector
    mtcnn = MTCNN(
        image_size=224, margin=20, min_face_size=80,
        thresholds=[0.6, 0.7, 0.7],
        device=device, keep_all=False, post_process=True,
    )

    # Extract faces
    print("\nExtracting faces...")
    face_frames, raw_frames = extract_faces_from_video(
        args.video, mtcnn, num_frames=cfg['data']['num_frames'])

    # Preprocess — dual stream
    print("Preprocessing...")
    frame_t, clip_t = preprocess(face_frames, cfg['data']['image_size'])

    # Predict
    prob_real, prob_fake, elapsed = predict(model, frame_t, clip_t, device)

    # Show result
    print_result(args.video, prob_real, prob_fake, elapsed)

    if args.show_frames:
        Path('results').mkdir(exist_ok=True)
        show_faces_grid(face_frames, prob_fake, 'results')


if __name__ == '__main__':
    main()