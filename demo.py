"""
demo.py — Gradio web demo for deepfake detection
-------------------------------------------------
Usage:
    pip install gradio --break-system-packages
    python demo.py

Opens at http://localhost:7860
"""

import cv2
import torch
import numpy as np
import yaml
import tempfile
import gradio as gr
from pathlib import Path
from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.cat_model import CATModelWithSupCon

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
ckpt      = torch.load(str(ckpt_path), map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Model loaded on {DEVICE} | Val AUC: {ckpt.get('val_auc', 0):.4f}")

mtcnn = MTCNN(
    image_size=224, margin=20, min_face_size=80,
    thresholds=[0.6, 0.7, 0.7],
    device=DEVICE, keep_all=False, post_process=True
)

N_FRAMES = cfg['data']['num_frames']  # 16

# ─────────────────────────────────────────────────────────────
# Preprocessing (matches dataset.py exactly)
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

    # Detect faces
    faces = []
    for f in raw_frames:
        face = detect_face(f)
        if face is not None:
            faces.append(face)
        elif faces:
            faces.append(faces[-1])   # pad with last good face
        else:
            # Resize whole frame as fallback
            resized = cv2.resize(f, (224, 224))
            faces.append(resized)

    if len(faces) < N_FRAMES + 1:
        while len(faces) < N_FRAMES + 1:
            faces.append(faces[-1])

    faces = faces[:N_FRAMES + 1]

    # Single frame for ViT (middle frame)
    single_face = faces[N_FRAMES // 2]
    frame_t     = frame_transform(image=single_face)["image"]  # (C,H,W)

    # Frame differences for TimeSformer
    diff_transform = A.Compose([A.Resize(224, 224), ToTensorV2()])
    diff_frames = []
    for t in range(N_FRAMES):
        f1   = faces[t].astype(np.int16)
        f2   = faces[t + 1].astype(np.int16)
        diff = ((f2 - f1 + 255) / 2).astype(np.uint8)
        diff_frames.append(diff_transform(image=diff)["image"])

    clip_t = torch.stack(diff_frames, dim=0).float() / 255.0
    clip_t = (clip_t - 0.5) * 2.0   # [-1, 1]

    return frame_t, clip_t, None


# ─────────────────────────────────────────────────────────────
# Prediction function
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(video_path):
    if video_path is None:
        return "Please upload a video.", None, None

    frame_t, clip_t, error = preprocess_video(video_path)
    if error:
        return f"Error: {error}", None, None

    # Add batch dimension
    frames = frame_t.unsqueeze(0).to(DEVICE)
    clips  = clip_t.unsqueeze(0).to(DEVICE)

    logits = model(frames, clips)
    probs  = torch.softmax(logits, dim=-1)[0]
    p_fake = float(probs[1].cpu())
    p_real = float(probs[0].cpu())

    label      = "FAKE" if p_fake >= THRESHOLD else "REAL"
    confidence = p_fake if label == "FAKE" else p_real

    # Result text
    emoji  = "🚨" if label == "FAKE" else "✅"
    result = f"{emoji} {label}  ({confidence*100:.1f}% confident)"

    # Confidence bar data for plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 1.8))
    colors = ['#1D9E75', '#D85A30']
    bars   = ax.barh(['Real', 'Fake'], [p_real * 100, p_fake * 100],
                     color=colors, height=0.5)
    ax.bar_label(bars, fmt='%.1f%%', padding=4, fontsize=12)
    ax.set_xlim([0, 110])
    ax.set_xlabel('Confidence (%)')
    ax.axvline(THRESHOLD * 100, color='black', lw=1,
               linestyle='--', label=f'Threshold {THRESHOLD*100:.0f}%')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Detection Result', fontsize=13, fontweight='bold')
    plt.tight_layout()

    return result, fig, f"P(fake)={p_fake:.4f}  |  P(real)={p_real:.4f}  |  Threshold={THRESHOLD}"


# ─────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────

with gr.Blocks(title="Deepfake Detector", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🎭 Deepfake Video Detector
    **Dual-Stream ViT + TimeSformer with Supervised Contrastive Learning**
    
    Upload a video and the model will classify it as **REAL** or **FAKE**.
    Uses spatial features (ViT) + temporal frame-difference features (TimeSformer).
    """)

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video", height=300)
            submit_btn  = gr.Button("🔍 Analyze Video",
                                    variant="primary", size="lg")

        with gr.Column(scale=1):
            result_text  = gr.Textbox(label="Prediction", lines=2,
                                      text_align="center",
                                      elem_id="result")
            result_plot  = gr.Plot(label="Confidence Scores")
            debug_text   = gr.Textbox(label="Raw scores", lines=1)

    gr.Markdown("""
    ---
    **Model:** CAT Fusion (ViT-Base + TimeSformer-Base) + SupCon Loss  
    **Dataset:** FaceForensics++ (6 manipulation methods)  
    **Val AUC:** 0.8806  |  **Threshold:** 0.5308
    """)

    submit_btn.click(
        fn=predict,
        inputs=[video_input],
        outputs=[result_text, result_plot, debug_text],
    )

    gr.Examples(
        examples=[
            [str(p)] for p in
            Path('FaceForensics++/testvideos').glob('*.mp4')
        ][:4] if Path('FaceForensics++/testvideos').exists() else [],
        inputs=video_input,
        label="Example videos from FF++ testset"
    )


if __name__ == '__main__':
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,        # set True to get a public URL
        inbrowser=True,     # auto-opens browser
    )