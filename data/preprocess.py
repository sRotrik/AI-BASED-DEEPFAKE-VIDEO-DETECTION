"""
preprocess.py — FaceForensics++ (flat folder structure)
--------------------------------------------------------
Your dataset structure (confirmed):
    FaceForensics++/
        original/               <- REAL videos
        DeepFakeDetection/      <- FAKE
        Deepfakes/              <- FAKE
        Face2Face/              <- FAKE
        FaceShifter/            <- FAKE
        FaceSwap/               <- FAKE
        NeuralTextures/         <- FAKE
        testvideos/             <- ignored
        csv file/               <- ignored

Run:
    python data/preprocess.py
"""

import cv2
import json
import numpy as np
import torch
import yaml
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from tqdm import tqdm
from facenet_pytorch import MTCNN


# ─────────────────────────────────────────────────────────────────────────────
# Exact folder → label mapping (from your confirmed scan)
# ─────────────────────────────────────────────────────────────────────────────

FOLDER_LABEL_MAP = {
    'original':           0,   # REAL
    'DeepFakeDetection':  1,   # FAKE
    'Deepfakes':          1,   # FAKE
    'Face2Face':          1,   # FAKE
    'FaceShifter':        1,   # FAKE
    'FaceSwap':           1,   # FAKE
    'NeuralTextures':     1,   # FAKE
}


# ─────────────────────────────────────────────────────────────────────────────
# Frame sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_frames(video_path, n):
    """Uniformly sample n frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Face detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_faces(frames, mtcnn):
    """Run MTCNN on each frame. Returns list of uint8 (H,W,C) arrays."""
    faces = []
    for frame in frames:
        face_tensor = mtcnn(frame)      # (C,H,W) float [-1,1] or None
        if face_tensor is not None:
            face_np = face_tensor.permute(1, 2, 0).numpy()
            face_np = ((face_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            faces.append(face_np)
    return faces


def pad_or_trim(faces, n):
    """Ensure exactly n faces by padding or trimming."""
    if not faces:
        return None
    while len(faces) < n:
        faces.append(faces[-1])
    return faces[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(root_dir, output_dir, num_frames=16):

    # ── GPU setup ─────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("WARNING : CUDA not available — using CPU (slow)")

    mtcnn = MTCNN(
        image_size=224, margin=20, min_face_size=80,
        thresholds=[0.6, 0.7, 0.7],
        device=device, keep_all=False, post_process=True
    )

    root        = Path(root_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Collect all video paths ───────────────────────────────────────────────
    all_videos = []
    for folder_name, label in FOLDER_LABEL_MAP.items():
        folder_path = root / folder_name
        if not folder_path.exists():
            print(f"WARNING : Folder not found, skipping → {folder_path}")
            continue
        videos = sorted(folder_path.glob('*.mp4'))
        print(f"  Found {len(videos):4d} videos in [{folder_name}]  label={label}")
        for v in videos:
            all_videos.append((v, label, folder_name))

    if not all_videos:
        print("\nERROR: No videos collected. Check root_dir in config.yaml.")
        print(f"  root_dir = {root_dir}")
        print(f"  Exists   = {root.exists()}")
        return []

    n_real = sum(1 for _, l, _ in all_videos if l == 0)
    n_fake = sum(1 for _, l, _ in all_videos if l == 1)
    print(f"\nTotal   : {len(all_videos)} videos  ({n_real} real, {n_fake} fake)")
    print(f"Ratio   : 1 real : {n_fake / max(n_real,1):.1f} fake")
    print(f"\nStarting face extraction (this takes 2-4 hours)...\n")

    # ── Process ───────────────────────────────────────────────────────────────
    records   = []
    skipped   = 0
    processed = 0

    for video_path, label, source in tqdm(all_videos, desc='Extracting faces'):
        try:
            frames = sample_frames(str(video_path), num_frames)
            if len(frames) < num_frames // 2:
                skipped += 1
                continue

            faces = detect_faces(frames, mtcnn)
            faces = pad_or_trim(faces, num_frames)
            if faces is None:
                skipped += 1
                continue

            # save as (T, H, W, C) uint8 numpy array
            save_dir  = output_path / source / video_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            save_file = str(save_dir / 'faces.npy')
            np.save(save_file, np.stack(faces))

            records.append({
                'path':   save_file,
                'label':  label,
                'source': source,
                'video':  video_path.stem
            })
            processed += 1

        except Exception as e:
            tqdm.write(f"  ERROR {video_path.name}: {e}")
            skipped += 1

    # ── Save records index ────────────────────────────────────────────────────
    records_file = output_path / 'records.json'
    with open(records_file, 'w') as f:
        json.dump(records, f, indent=2)

    n_real_out = sum(1 for r in records if r['label'] == 0)
    n_fake_out = sum(1 for r in records if r['label'] == 1)

    print(f"\n{'='*50}")
    print(f"Preprocessing complete.")
    print(f"  Processed : {processed}")
    print(f"  Skipped   : {skipped}")
    print(f"  Real      : {n_real_out}")
    print(f"  Fake      : {n_fake_out}")
    if n_real_out > 0:
        print(f"  Ratio     : 1 real : {n_fake_out / n_real_out:.1f} fake")
    print(f"  Saved to  : {records_file}")
    print(f"{'='*50}")
    print(f"\nNext step: python train.py")

    return records


if __name__ == '__main__':
    cfg = yaml.safe_load(open('configs/config.yaml'))
    build_dataset(
        root_dir=cfg['data']['root_dir'],
        output_dir=cfg['data']['output_dir'],
        num_frames=cfg['data']['num_frames']
    )