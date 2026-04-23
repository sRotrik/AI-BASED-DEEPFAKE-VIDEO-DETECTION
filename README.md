# 🎭 AI-Based Deepfake Video Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)
![Val AUC](https://img.shields.io/badge/Val%20AUC-0.8806-brightgreen)

**B.Tech Final Year Project | SRM Institute of Science and Technology, Kattankulathur**  
**Course: 21CSP302L | Department: CTECH | Batch: B305 | Semester 6 (2025–2026)**

</div>

---

## 🧠 Overview

This project presents a **dual-stream transformer-based deepfake video detection system** that simultaneously analyses **spatial** and **temporal** features to classify videos as real or manipulated.

Unlike single-frame approaches, our system:
- Uses a **Vision Transformer (ViT-Base)** to detect frame-level visual anomalies
- Uses a **TimeSformer** operating on **frame difference clips** to detect unnatural inter-frame motion
- Employs **Supervised Contrastive Learning** to sharpen the separation between real and fake feature embeddings
- Achieves a **Validation AUC of 0.8806** on FaceForensics++ across 6 manipulation methods

---

## 🏗️ Architecture

```
Input Video
    │
    ├── MTCNN Face Detector
    │
    ├── 16+1 uniformly sampled face crops
    │
    ├────────────────────┬────────────────────────────────────
    │                    │
    │  SPATIAL STREAM    │  TEMPORAL STREAM
    │                    │
    │  Middle face frame │  Frame Differences
    │  (C, H, W)        │  diff[t] = frame[t+1] - frame[t]
    │                    │  (T=16, C, H, W)
    │       ↓            │         ↓
    │  ViT-Base          │  TimeSformer-Base
    │  → 512-dim embed   │  → 512-dim embed
    │                    │
    └────────────────────┘
              │
        Concatenate → 1024-dim
              │
        LayerNorm → Linear → GELU → Dropout → Linear(2)
              │
       Softmax → [P(real), P(fake)]
              │
       FAKE if P(fake) ≥ 0.5308
```

---

## 📊 Results

| Metric | Value |
|---|---|
| **Validation AUC** | **0.8806** |
| **Balanced Accuracy** | 0.78 |
| **Optimal Threshold** | 0.5308 |
| ViT alone (ablation) | ~0.74 AUC |
| TimeSformer alone (ablation) | ~0.71 AUC |
| **Combined (CAT Model)** | **0.8806 AUC** |

### Per-Method AUC (FaceForensics++)
The model is evaluated across all 6 manipulation methods:
`Deepfakes` | `Face2Face` | `FaceSwap` | `FaceShifter` | `NeuralTextures` | `DeepFakeDetection`

See [`results/per_method_auc.png`](results/per_method_auc.png) and [`results/metrics.json`](results/metrics.json) for full breakdown.

---

## 📁 Project Structure

```
deepfake_detection/
│
├── models/
│   ├── cat_model.py          # CATModel + CATModelWithSupCon + FocalLoss + SupConLoss
│   ├── vit_stream.py         # ViT spatial stream
│   ├── timesformer_stream.py # TimeSformer temporal stream
│   └── supcon_loss.py        # Supervised Contrastive Loss
│
├── data/
│   ├── dataset.py            # FaceForensicsDataset — dual-stream (frame + diff clip)
│   ├── preprocess.py         # MTCNN face extraction → .npy files
│   └── processed/
│       └── records.json      # Dataset metadata (paths, labels, method sources)
│
├── configs/
│   └── config.yaml           # All hyperparameters
│
├── static/                   # Web UI (HTML/CSS/JS — dark glassmorphism)
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── results/                  # Evaluation figures and metrics
│   ├── metrics.json
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── tsne.png
│   ├── pr_curve.png
│   ├── score_distribution.png
│   └── per_method_auc.png
│
├── checkpoints/              # Model checkpoints (not tracked by Git — see below)
│   └── last_checkpoint.txt
│
├── train.py                  # Full training loop (resume-capable)
├── evaluate.py               # Full evaluation + publication-quality figures
├── inference.py              # CLI single-video inference
├── demo.py                   # Gradio web demo (localhost:7860)
├── app.py                    # Flask REST API backend
├── requirements.txt
└── .gitignore
```

> ⚠️ **Model checkpoints (`best_model.pth`, ~1.5 GB) and the FaceForensics++ dataset are NOT included in this repository** due to size constraints. See the **Setup** section below.

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sRotrik/AI-BASED-DEEPFAKE-VIDEO-DETECTION.git
cd AI-BASED-DEEPFAKE-VIDEO-DETECTION
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Download Pre-trained Model Checkpoint
> The `best_model.pth` file is not in the repository. Download it separately and place at:
```
checkpoints/best_model.pth
```

### 5. Configure `configs/config.yaml`
Update `data.root_dir` to point to your local FaceForensics++ dataset path.

---

## 🚀 Usage

### Command-Line Inference (Single Video)
```bash
python inference.py --video path/to/video.mp4
# With face grid visualization:
python inference.py --video path/to/video.mp4 --show_frames
```

**Example output:**
```
=======================================================
  Video      : sample.mp4
  P(real)    : 0.1243  (12.4%)
  P(fake)    : 0.8757  (87.6%)
  Threshold  : 0.5308
  🚨 Verdict : FAKE  (87.6% confident)
  Latency    : 843.2 ms
  REAL [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] FAKE
=======================================================
```

### Gradio Web Demo
```bash
python demo.py
# Opens at http://localhost:7860
```

### Flask Web Application (Full UI)
```bash
python app.py
# Opens at http://127.0.0.1:5000
```

### Training From Scratch
```bash
# First preprocess the dataset:
python data/preprocess.py

# Then train:
python train.py

# Resume from checkpoint:
python train.py --resume
```

### Evaluation
```bash
python evaluate.py
# Generates all figures in results/ folder
```

---

## 🧪 Key Technical Decisions

| Design Choice | Rationale |
|---|---|
| **Frame differences → TimeSformer** | Forces temporal stream to detect change artefacts, not natural motion |
| **Differential LRs** (backbone: 1e-5, head: 1e-4) | Prevents catastrophic forgetting of pre-trained ViT/TimeSformer weights |
| **Focal Loss (α=0.75)** | Handles 1:6 real:fake class imbalance |
| **SupCon Loss (τ=0.15)** | Tightens embedding clusters — reduces false positives on ambiguous samples |
| **Linear Warmup → CosineDecay** | Replaced OneCycleLR (caused LR spikes and classifier explosion) |
| **16 frames / video** | Standardises input shape; balances temporal coverage vs. compute cost |

---

## 📈 Evaluation Figures

| Figure | Description |
|---|---|
| [`roc_curve.png`](results/roc_curve.png) | ROC Curve — AUC 0.8806 |
| [`confusion_matrix.png`](results/confusion_matrix.png) | Confusion matrix at threshold 0.5308 |
| [`tsne.png`](results/tsne.png) | t-SNE embedding visualization (real vs fake clusters) |
| [`pr_curve.png`](results/pr_curve.png) | Precision-Recall Curve |
| [`score_distribution.png`](results/score_distribution.png) | P(fake) score distribution for real and fake samples |
| [`per_method_auc.png`](results/per_method_auc.png) | Per-method AUC across all 6 FF++ manipulation types |

---

## 📦 Dependencies

Key libraries:
- `torch >= 2.0.0` — model training and inference
- `transformers >= 4.35.0` — ViT and TimeSformer pretrained models
- `facenet-pytorch >= 2.5.3` — MTCNN face detection
- `albumentations >= 1.3.1` — image augmentation pipeline
- `opencv-python-headless >= 4.8.0` — video frame extraction
- `flask`, `flask-cors` — REST API backend
- `gradio` — web demo interface
- `scikit-learn`, `matplotlib`, `seaborn` — evaluation and plotting

See [`requirements.txt`](requirements.txt) for the full list.

---

## 👥 Team

| Name | Register No. | Role | Email |
|---|---|---|---|
| **Reshma K** | RA2311003011843 | Data pipeline, SupCon loss, Evaluation | rk6362@srmist.edu.in |
| **Srotrik Pradhan** | RA2311003011860 | Model architecture, Training infra, Web app | sp8087@srmist.edu.in |

**Supervisor:** Dr. Shyni Shajahan | Assistant Professor, CTECH Dept.  
SRM Institute of Science and Technology, Kattankulathur — 603203

---

## 📄 License

This project is developed for academic purposes under SRM IST (Course 21CSP302L).  
Code is released under the [MIT License](LICENSE).

---

## 🔗 References

- FaceForensics++: [Rössler et al., ICCV 2019](https://github.com/ondyari/FaceForensics)
- Vision Transformer: [Dosovitskiy et al., ICLR 2021](https://arxiv.org/abs/2010.11929)
- TimeSformer: [Bertasius et al., ICML 2021](https://arxiv.org/abs/2102.05095)
- Supervised Contrastive Learning: [Khosla et al., NeurIPS 2020](https://arxiv.org/abs/2004.11362)
- MTCNN: [Zhang et al., IEEE Signal Processing Letters 2016](https://arxiv.org/abs/1604.02878)
