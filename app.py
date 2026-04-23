"""
app.py — Flask backend for the Deepfake Detection Web UI
---------------------------------------------------------
Usage:
    pip install Flask flask-cors
    python app.py

Opens at http://127.0.0.1:5000
"""

import os
import uuid
import tempfile
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.cat_model import CATModelWithSupCon

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

UPLOAD_FOLDER = Path(tempfile.gettempdir()) / 'deepfake_uploads'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Load model once at startup
# ─────────────────────────────────────────────────────────────

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5308

cfg   = yaml.safe_load(open('configs/config.yaml'))
model = CATModelWithSupCon(
    embed_dim=cfg['model']['embed_dim'],
    proj_dim=cfg['model'].get('proj_dim', 128),
).to(DEVICE)

ckpt_path = Path('checkpoints/best_model.pth')
ckpt      = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()

val_auc = ckpt.get('val_auc', 0)
print(f"[OK] Model loaded on {DEVICE} | Val AUC: {val_auc:.4f}")

mtcnn = MTCNN(
    image_size=224, margin=20, min_face_size=80,
    thresholds=[0.6, 0.7, 0.7],
    device=DEVICE, keep_all=False, post_process=True
)

N_FRAMES = cfg['data']['num_frames']  # 16

# ─────────────────────────────────────────────────────────────
# Preprocessing (mirrors demo.py exactly)
# ─────────────────────────────────────────────────────────────

frame_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def extract_frames(video_path, n=17):
    """Extract n uniformly spaced frames from video."""
    cap    = cv2.VideoCapture(str(video_path))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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


def detect_face(frame):
    """Run MTCNN on a frame, return uint8 (H,W,C) face crop or None."""
    face_tensor = mtcnn(frame)
    if face_tensor is None:
        return None
    face_np = face_tensor.permute(1, 2, 0).numpy()
    face_np = ((face_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return face_np


def preprocess_video(video_path):
    """
    Extract face crops, compute frame differences for TimeSformer,
    return (frame_tensor, clip_tensor) ready for model.
    """
    raw_frames = extract_frames(video_path, n=N_FRAMES + 1)
    if not raw_frames:
        return None, None, "Could not read video file."

    faces = []
    for f in raw_frames:
        face = detect_face(f)
        if face is not None:
            faces.append(face)
        elif faces:
            faces.append(faces[-1])
        else:
            resized = cv2.resize(f, (224, 224))
            faces.append(resized)

    if len(faces) < N_FRAMES + 1:
        while len(faces) < N_FRAMES + 1:
            faces.append(faces[-1])

    faces = faces[:N_FRAMES + 1]

    single_face = faces[N_FRAMES // 2]
    frame_t     = frame_transform(image=single_face)["image"]

    diff_transform = A.Compose([A.Resize(224, 224), ToTensorV2()])
    diff_frames = []
    for t in range(N_FRAMES):
        f1   = faces[t].astype(np.int16)
        f2   = faces[t + 1].astype(np.int16)
        diff = ((f2 - f1 + 255) / 2).astype(np.uint8)
        diff_frames.append(diff_transform(image=diff)["image"])

    clip_t = torch.stack(diff_frames, dim=0).float() / 255.0
    clip_t = (clip_t - 0.5) * 2.0

    return frame_t, clip_t, None


@torch.no_grad()
def run_inference(video_path):
    frame_t, clip_t, error = preprocess_video(video_path)
    if error:
        return None, error

    frames = frame_t.unsqueeze(0).to(DEVICE)
    clips  = clip_t.unsqueeze(0).to(DEVICE)

    logits = model(frames, clips)
    probs  = torch.softmax(logits, dim=-1)[0]
    p_fake = float(probs[1].cpu())
    p_real = float(probs[0].cpu())

    label = "FAKE" if p_fake >= THRESHOLD else "REAL"

    return {
        "label":     label,
        "p_fake":    round(p_fake * 100, 2),
        "p_real":    round(p_real * 100, 2),
        "threshold": round(THRESHOLD * 100, 2),
        "val_auc":   round(val_auc, 4),
        "device":    str(DEVICE),
    }, None


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save to temp file
    suffix = Path(file.filename).suffix or '.mp4'
    tmp_path = UPLOAD_FOLDER / f"{uuid.uuid4()}{suffix}"
    file.save(str(tmp_path))

    try:
        result, error = run_inference(str(tmp_path))
        if error:
            return jsonify({'error': error}), 500
        return jsonify(result)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.route('/api/status')
def status():
    return jsonify({
        'model': 'CATModelWithSupCon (ViT + TimeSformer)',
        'device': str(DEVICE),
        'val_auc': round(val_auc, 4),
        'threshold': THRESHOLD,
        'ready': True,
    })


if __name__ == '__main__':
    print("\n[STARTING] Deepfake Detection Web UI")
    print(f"   Open: http://127.0.0.1:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
