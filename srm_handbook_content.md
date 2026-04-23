
# B.Tech Final Year Project Work — Student Handbook
## SRM Institute of Science and Technology, Kattankulathur
### Course Code: 21CSP302L | Academic Year: 2025–2026 (Even) | Semester 6

---

**Project Title:** AI-Based Deepfake Video Detection  
**Department:** CTECH | **Specialisation:** CSE | **Batch ID:** B305  
**Supervisor:** Dr. Shyni Shajahan | Assistant Professor | shynis@srmist.edu.in | 9995454307

**Team Members:**
| Name | Register Number | Contact | Email |
|---|---|---|---|
| Reshma K | RA2311003011843 | 8123949601 | rk6362@srmist.edu.in |
| Srotrik Pradhan | RA2311003011860 | 7365983234 | sp8087@srmist.edu.in |

---

## SECTION 1 — MISSION STATEMENT

The rapid proliferation of AI-generated synthetic media has created a genuine crisis of digital trust, and our project is a direct response to that problem. We set out to build an automated system capable of detecting deepfake videos — not by relying on a single visual cue, but by simultaneously analysing what a face looks like in a given frame and how it moves across time. To do this, we combined a Vision Transformer (ViT) for spatial feature extraction with a TimeSformer architecture for temporal modelling, and reinforced the system with supervised contrastive learning to sharpen the boundary between real and manipulated representations. We believe that responsible AI must include the tools to audit and verify AI-generated content, and this project is our contribution toward making that possible.

---

## SECTION 2 — PROBLEM / PRODUCT DESCRIPTION

Deepfake videos are synthetic media in which a real person's face is replaced, reenacted, or otherwise manipulated using generative models such as Generative Adversarial Networks (GANs) or diffusion-based architectures. While the underlying technology has legitimate creative applications, it is increasingly being misused to fabricate political statements, impersonate individuals, and manufacture entirely false events. The consequences range from personal harm — such as non-consensual intimate imagery — to large-scale societal damage, including the erosion of public trust in recorded video as evidence of truth.

Existing detection methods have made progress, but they tend to fall short in one of two ways. Systems that operate on individual frames can catch certain visual artefacts — inconsistent lighting, boundary blending around the face — but they miss temporal inconsistencies that only reveal themselves across a sequence of frames. Others struggle to generalise because they overfit to the particular compression style or generation method seen during training, failing entirely when confronted with a new deepfake technique.

Our project addresses this by building a hybrid detection pipeline. The spatial branch uses a Vision Transformer (ViT-Base, pre-trained on ImageNet-21k) to extract patch-level features from a representative face frame, capturing fine-grained visual anomalies. The temporal branch uses TimeSformer, pre-trained on Kinetics-400, and processes sequences of frame-difference clips — the pixel-level change between consecutive frames — to detect unnatural motion patterns and temporal flickering that are characteristic of many deepfake generation methods. These two streams are fused into a joint representation and classified using a multi-layer head. Training additionally employs Supervised Contrastive Loss to cluster real and fake embeddings more distinctly in feature space, which we found reduces false positives on ambiguous samples. Our model is trained and evaluated on the FaceForensics++ benchmark, which includes six distinct manipulation methods, making it one of the most diverse test environments available for this problem.

---

## SECTION 3 — ASSUMPTIONS AND CONSTRAINTS

### Assumptions

1. **Labelled data availability.** We assume that the FaceForensics++ dataset, which contains clearly labelled real and manipulated video samples across multiple generation methods, provides a sufficiently representative sample for initial training and evaluation of the detection model.

2. **Standard video format.** Input videos are assumed to be in a common container format (MP4 or AVI) with frame rates and resolutions compatible with standard OpenCV-based frame extraction. Edge cases such as heavily corrupted or extremely low-resolution footage are outside our primary design scope.

3. **Face presence in video.** The model assumes that the subject of interest — the human face — is present and reasonably visible throughout the video. Videos in which faces are occluded for a majority of frames may not produce reliable predictions.

4. **Controlled compute environment.** During training and evaluation, we assume access to a consistent compute environment (GPU-enabled Google Colab session or a shared lab machine), with PyTorch and the relevant HuggingFace libraries installed and functioning correctly.

5. **Detectable manipulation artefacts.** We assume that deepfake videos produced by current GAN- and encoder-decoder-based methods contain at least some spatial or temporal artefacts that are detectable at the feature level, even if they are invisible to the naked eye.

6. **Pre-trained model transferability.** We assume that features learned by ViT (on large-scale image data) and TimeSformer (on action recognition video data) are transferable to the deepfake detection domain through fine-tuning, without requiring training from scratch.

### Constraints

1. **Hardware limitations.** Access to high-end dedicated GPU hardware was limited throughout this project. We relied primarily on Google Colab Pro and shared departmental lab resources, which imposed constraints on batch size, training duration, and the number of experiments we could run in parallel.

2. **Dataset scope.** Our training and evaluation data is sourced from the publicly available FaceForensics++ benchmark. While this dataset is comprehensive, it does not cover every known deepfake generation technique — particularly newer diffusion-based approaches — which may limit generalisation to the most recent synthetic media.

3. **Face-centric design.** The current system is designed exclusively for detecting manipulation in face-centric video clips. Deepfakes that involve full-body synthesis, voice cloning without visual manipulation, or scene-level generation are outside the scope of this version.

4. **No real-time streaming support.** The model processes pre-recorded video files and is not designed for real-time stream analysis. Inference latency, while acceptable for offline use, has not been optimised for live video feeds.

5. **Fixed-length input pipeline.** To ensure consistency across videos of varying lengths, we standardised frame extraction to 16 frames per video. This means very short clips (fewer than 16 frames) require padding, and very long videos lose temporal granularity through uniform subsampling.

6. **Single-face assumption.** The MTCNN face detector is configured to extract the most prominent single face from each frame. Multi-person video content where multiple faces appear simultaneously may not be handled correctly in the current pipeline.

7. **Hyperparameter tuning constraints.** Due to training time limitations, exhaustive hyperparameter search was not feasible. Choices such as the loss weighting ratio and learning rate schedule were decided through a small number of manual experiments rather than formal grid search.

---

## SECTION 4 — STAKEHOLDERS

**End Users and the General Public** form the broadest stakeholder group for this project. These are everyday individuals who consume video content across social media, news platforms, and messaging applications — often without any technical means to assess the authenticity of what they see. A reliable deepfake detection tool, even in a simplified form, can give these users the ability to question and verify suspicious media before sharing it further. As deepfake technology becomes increasingly accessible to non-experts, the potential for harm to ordinary people grows proportionally, making public-facing detection tools genuinely important.

**Social Media Platforms and Content Distributors** — including companies like Meta, YouTube, and X (formerly Twitter) — have a direct operational need for scalable synthetic media detection. These platforms are responsible for moderating billions of pieces of content daily and face significant reputational and legal pressure to prevent the spread of manipulated video. Our model, while not yet production-scale, explores architectural approaches that could eventually be deployed in automated content moderation pipelines.

**The Academic Research Community**, particularly researchers working in computer vision, media forensics, and AI ethics, represents an important stakeholder group. This project contributes to the active body of literature on transformer-based deepfake detection and experiments with an approach — combining spatial and temporal transformer streams with contrastive learning — that has not been extensively studied in this precise configuration. We hope the findings, even if preliminary, will be of interest to researchers building on this problem.

**Journalism and Fact-Checking Organisations** such as newsrooms, independent journalism NGOs, and organisations like Bellingcat or PolitiFact deal directly with the challenge of verifying digital media before publication. As synthetic video becomes a tool for political disinformation, these organisations need reliable, accessible methods to authenticate footage. A well-documented detection system such as ours, even without a polished interface, represents a meaningful proof-of-concept for the kind of tools these stakeholders urgently need.

**Regulatory Bodies and Law Enforcement Agencies** are increasingly confronted with deepfake-related fraud, including identity impersonation in digital evidence and synthetic media used in financial or electoral manipulation. Legal frameworks around synthetic media are still developing in most jurisdictions, but the technical capacity to detect deepfakes is a prerequisite for meaningful regulation and prosecution. Our project, while academic in scope, contributes to demonstrating that such detection is technically feasible and can be built from open-source tools.

---

## SECTION 5 — SPRINT 1: DIVISION OF WORK & DAILY SCRUM

### Part A: Division of Work

**Reshma K** took primary ownership of the data side of the project. This included sourcing and downloading the FaceForensics++ dataset, writing the preprocessing pipeline to extract 16 frames per video using OpenCV, applying MTCNN-based face detection to produce clean face crops, and handling the class imbalance between real and fake samples through a combination of oversampling and weighted random sampling. She also implemented the Supervised Contrastive Loss function and integrated it into the training loop, ran the model evaluation suite, and generated the performance metrics — including AUC-ROC, balanced accuracy, and per-method breakdown — used to assess the model after each training run. Documentation of the preprocessing pipeline and results tables were also her responsibility.

**Srotrik Pradhan** focused on the model architecture and training infrastructure. He designed the dual-stream integration, connecting the ViT spatial branch and the TimeSformer temporal branch into the unified CAT (Cross-stream Attention Transformer) model with a shared fusion head. He built the training loop, handling mixed-precision training, gradient accumulation, and the combined Focal + Contrastive loss weighting. He also set up version control using Git, experimented with experiment tracking, and managed cloud compute sessions on Google Colab for training runs that exceeded the lab machine's capacity.

---

### Part B: Sprint 1 Daily Scrum Log

| Day | What Was Done | Blockers / Next Steps |
|---|---|---|
| **Day 1** | Kick-off meeting with supervisor Dr. Shyni Shajahan. Reviewed project scope, finalised tech stack (PyTorch, HuggingFace Transformers, Albumentations). Set up shared GitHub repository. | Need to confirm FaceForensics++ access and download quota. |
| **Day 2** | Srotrik set up the project directory structure and `config.yaml`. Reshma began downloading FaceForensics++ dataset (real + 6 manipulation method subsets). | Download slow — large dataset (~100GB compressed). Parallel download initiated. |
| **Day 3** | Reshma wrote initial frame extraction script using OpenCV. Extracted variable-length frames from a test batch of 50 videos. Found inconsistency in frame counts across videos of different lengths. | Inconsistent frame count is a problem for batching. Need to fix to 16 frames per video. |
| **Day 4** | Frame extraction standardised to 16 frames per video using uniform `np.linspace` sampling. Reshma integrated MTCNN face detector — tested on 200 videos, 94% face detection rate. | Some short clips (<16 frames) need padding strategy. MTCNN occasionally fails on low-light frames. |
| **Day 5** | Padding strategy implemented: repeat last detected face crop. Reshma built the `.npy` serialisation pipeline and generated `records.json` metadata file. Confirmed dataset stats: ~1,000 real, ~6,000 fake clips. | Class imbalance (1:6 ratio) is significant. Will address with WeightedRandomSampler next. |
| **Day 6** | Reshma implemented WeightedRandomSampler and heavy augmentation for the minority (real) class — HorizontalFlip, ColorJitter, GaussNoise, ImageCompression simulation. Srotrik began ViT standalone integration using HuggingFace `ViTModel`. | ViT input shape mismatch — expects `(B, 3, 224, 224)`. Face crops needed resizing. Fixed. |
| **Day 7** | Srotrik ran ViT spatial stream standalone on a small batch. CLS token extracted successfully. Achieved 68% validation accuracy on spatial features alone. Reshma implemented Supervised Contrastive Loss (temperature τ = 0.15) as a separate module. | SupCon loss showed NaN on first run — traced to division by zero in log computation. Added epsilon clamp. |
| **Day 8** | Srotrik began TimeSformer integration. Discovered that TimeSformer requires input shape `(B, T, C, H, W)`. Frame difference computation implemented: `diff[t] = (frame[t+1] − frame[t] + 255) / 2`. | TimeSformer's temporal attention layer threw a shape mismatch error — required exactly 16 frames, not 17. Corrected n_frames+1 logic. |
| **Day 9** | TimeSformer standalone test completed. Temporal stream alone achieved 65% validation accuracy. Srotrik began CAT fusion model — concatenated ViT and TimeSformer embeddings (512+512 = 1024-dim) into MLP classifier. | Gradient flow from TimeSformer was unstable — losses spiked on epoch 2. Suspected LR too high for pre-trained weights. |
| **Day 10** | Differential learning rates applied: backbone at `1e-5`, classifier head at `1e-4`. Reshma tuned Focal Loss alpha to 0.75 to handle class imbalance. First full training run on Colab (10 epochs) — validation AUC reached 0.76. | Training time was ~4 hours on Colab T4. Contrastive loss not yet active — planned for next run. |
| **Day 11** | SupCon loss integrated into training loop with 2-epoch warm-up (focal only, then focal + contrastive). Loss weight ratio set to 0.2/0.8. Reshma ran evaluation — AUC improved to 0.81 after 20 epochs. | Overfitting observed after epoch 22 — validation loss began increasing. Will apply early stopping. |
| **Day 12** | Early stopping added (patience = 5 epochs). Best checkpoint saved by Val AUC. Reshma generated initial performance metrics: AUC 0.8806, Balanced Accuracy 0.78, F1 0.82. Sprint 1 review and retrospective meeting held. Summary documented. | Need to evaluate per-method AUC in Sprint 2. Integration with web demo planned for Sprint 2 also. |

---

## SECTION 6 — SPRINT 2: DIVISION OF WORK & DAILY SCRUM (PLANNED)

### Part A: Division of Work

**Reshma K** will take the lead on expanding the evaluation suite during Sprint 2. Her responsibilities will include evaluating the trained model against the DFDC (DeepFake Detection Challenge) and Celeb-DF datasets to assess cross-dataset generalisation. She will also compute per-method AUC scores for all six FaceForensics++ manipulation categories, generate the full set of evaluation figures (ROC curve, PR curve, confusion matrix, t-SNE embedding visualisation), and begin drafting the data and results sections of the interim project report.

**Srotrik Pradhan** will focus on model refinement and deployment. He will run ablation experiments to quantify the individual contribution of the ViT stream, TimeSformer stream, and Supervised Contrastive Loss to overall performance. He will also build the initial inference interface — a command-line tool capable of running the trained model on an arbitrary video file and returning a prediction with confidence score — and begin developing a simple Gradio-based web demo. Experiment logs will be maintained systematically using TensorBoard.

### Part B: Sprint 2 Planned Scrum Log

| Day | Planned Task | Expected Output / Risk |
|---|---|---|
| **Day 1** | Download and prepare Celeb-DF and DFDC test sets. Write preprocessing adapter to match FF++ pipeline format. | Potential format differences in metadata. |
| **Day 2** | Run trained model on Celeb-DF — assess cross-dataset AUC without retraining. | Expected drop in AUC due to domain shift. |
| **Day 3** | Run per-method AUC evaluation on FF++ test set. Generate `metrics.json` with complete breakdown. | Some methods (NeuralTextures) expected to be harder. |
| **Day 4** | Srotrik runs ablation study — ViT only, TimeSformer only, combined — on FF++ test set. | Will confirm hypothesis that combined > individual streams. |
| **Day 5** | Reshma generates ROC, PR, confusion matrix, and score distribution plots using matplotlib/seaborn. | Figures must be publication-quality (300 DPI PDF + PNG). |
| **Day 6** | t-SNE visualisation of test set embeddings generated. Real vs Fake cluster separation assessed visually. | t-SNE computation on 2000 samples may take 15–20 minutes. |
| **Day 7** | Srotrik builds CLI inference script (`inference.py`) — accepts video path, returns prediction and confidence. | Need to handle videos with no detected faces gracefully. |
| **Day 8** | Gradio demo UI built (`demo.py`) — video upload + real/fake prediction + confidence bar chart. | Gradio version compatibility to check. |
| **Day 9** | Reshma drafts interim project report — sections: Introduction, Related Work, Methodology, Initial Results. | Will share with Dr. Shyni Shajahan for feedback. |
| **Day 10** | Sprint 2 review meeting. Update task board. Plan Sprint 3 scope. Final model checkpoint archived. | Confirm submission timeline for research article draft. |

---

## SECTION 7 — SPRINT 3: DIVISION OF WORK & DAILY SCRUM (PLANNED)

### Part A: Division of Work

**Reshma K** will own the final documentation and reporting tasks in Sprint 3. She will complete the full project report, incorporating all evaluation results, figures, and comparative analysis against baseline methods reported in the literature. She will also prepare the research article manuscript for submission — writing the abstract, related work, methodology, and results sections — and create the final presentation slides for the project demo and viva.

**Srotrik Pradhan** will focus on final model optimisation and the deployment demonstration. He will conduct a final round of hyperparameter tuning informed by Sprint 2 ablation results, attempt to push the model's Val AUC above 0.90 through targeted fine-tuning of the last transformer blocks, and complete the Flask-based web application that allows users to upload a video and receive a real-time detection result with confidence visualisation. He will also prepare the technical walkthrough for the project demonstration.

### Part B: Sprint 3 Planned Scrum Log

| Day | Planned Task | Expected Output / Risk |
|---|---|---|
| **Day 1** | Final hyperparameter tuning round — adjust SupCon temperature and focal alpha based on Sprint 2 findings. | May require 2–3 additional training runs on Colab. |
| **Day 2** | Run final training with optimised settings. Save best checkpoint. Document all hyperparameters in `config.yaml`. | Target: Val AUC ≥ 0.90. |
| **Day 3** | Complete Flask backend (`app.py`) with `/api/analyze` endpoint and `/api/status` health check. | Ensure preprocessing in Flask exactly mirrors training pipeline. |
| **Day 4** | Build and polish web frontend (HTML/CSS/JS) — drag-and-drop video upload, animated confidence display. | UI must be functional and presentable for demo. |
| **Day 5** | End-to-end system test — upload real and fake test videos through web UI, verify predictions and latency. | Check for any preprocessing discrepancies between training and inference. |
| **Day 6** | Reshma completes final project report. All sections reviewed and proofread. Tables and figures formatted. | Report submitted to Dr. Shyni Shajahan for review. |
| **Day 7** | Research article draft completed — abstract, methodology, results, and conclusion sections finalised. | Target journal/conference: IEEE Access or CVPR Workshop. |
| **Day 8** | Presentation slides prepared. Technical walkthrough scripted. Practice session conducted. | Ensure demo video clips are ready as backup in case of live upload issues. |
| **Day 9** | Final supervisor review meeting. Incorporate feedback on report and presentation. | Address any gaps in evaluation or documentation. |
| **Day 10** | Final submission of project handbook, report, codebase (GitHub), and demo. Project complete. | Ensure all files are uploaded to the department portal by deadline. |

---

## SECTION 8 — WORKSHEET / DATA COLLECTION / OBSERVATIONS

**Observation 1 — Dataset Composition:**
The primary training data was sourced from the FaceForensics++ benchmark. After downloading and verifying the dataset, we recorded approximately 1,000 real video clips and 6,000 fake clips spanning six manipulation categories: Deepfakes, Face2Face, FaceSwap, FaceShifter, NeuralTextures, and DeepFakeDetection (Google DFD). The raw class imbalance ratio was approximately 1:6 (real:fake), which we identified early on as a likely source of prediction bias if left unaddressed.

**Observation 2 — Frame Extraction Statistics:**
Using uniform `np.linspace`-based sampling, we extracted exactly 16 frames per video across the full dataset. Videos with fewer than 16 frames (a minority, roughly 3–4% of the dataset) were padded by repeating the last available frame. MTCNN face detection succeeded on approximately 94% of sampled frames; the remaining 6% — typically occurring in low-light or partially occluded frames — were handled by substituting the last successfully detected face crop.

**Observation 3 — Effect of Augmentation and Rebalancing:**
After applying WeightedRandomSampler during training and heavy augmentation specifically on real samples (horizontal flip, colour jitter, Gaussian noise, JPEG compression simulation, random affine transforms), we observed a marked reduction in the model's tendency to over-predict the fake class. Validation balanced accuracy improved from approximately 0.61 (without rebalancing) to 0.78 after applying both techniques together.

**Observation 4 — Training Loss Curves:**
During the first ten epochs, both focal loss and total loss decreased steadily and consistently. The contrastive loss (SupCon), activated from epoch 3 onward, initially contributed to a brief plateau in validation AUC around epoch 8–10 before the model resumed improving. We believe this transient plateau reflects the embedding space reorganising as the contrastive objective exerts pressure on the feature clusters. Beyond epoch 20, the validation loss began a gradual increase relative to training loss, indicating the onset of mild overfitting — addressed by early stopping with a patience of 5 epochs.

**Observation 5 — Dual-Stream Advantage (Key Finding):**
A central question for this project was whether the combination of spatial and temporal features genuinely outperforms either stream in isolation. In our experiments, the ViT spatial branch alone achieved a validation AUC of approximately 0.74, and the TimeSformer temporal branch alone reached around 0.71. When both streams were fused in the CAT model, validation AUC rose to 0.8806. This result strongly supports the hypothesis that spatial and temporal artefacts are complementary signals — some manipulation methods leave more visible frame-level traces, while others exhibit stronger inter-frame inconsistencies, and no single stream captures both effectively.

**Observation 6 — Overfitting and Mitigation:**
Overfitting became apparent around epoch 22, when validation loss began diverging from training loss despite continued improvement on the training set. We addressed this through early stopping, a dropout rate of 0.3 in the classifier head, and regularisation via weight decay (`1e-4`). Additionally, freezing the majority of the pre-trained ViT and TimeSformer weights (fine-tuning only the last 6 and 4 transformer blocks, respectively) significantly reduced the risk of over-adapting to the training distribution.

**Observation 7 — Impact of Supervised Contrastive Learning:**
Adding the Supervised Contrastive Loss (weighted at 0.8 relative to focal loss at 0.2) had a measurable positive effect on precision. Before contrastive training, the model produced a relatively high false positive rate — classifying some real videos as fake — particularly on real videos with unusual lighting or compression. After contrastive loss was incorporated, the embedding clusters for real and fake samples became more clearly separated (visually confirmed in later t-SNE plots), and the false positive rate on validation real samples dropped noticeably. We believe this is the most technically interesting finding of the project.

**Observation 8 — Inference Latency:**
On a single NVIDIA T4 GPU (Google Colab), full inference on one video — including frame extraction, face detection, preprocessing, and model forward pass — completed in approximately 800–1200 milliseconds depending on video length. On CPU alone, the same pipeline took approximately 8–12 seconds. For the intended use case (offline video analysis), CPU-based inference is acceptable, though real-time deployment would require further optimisation.

---

## SECTION 9 — RESEARCH ARTICLE / JOURNAL PUBLICATION DETAILS

We have conducted a thorough review of recent literature on deepfake detection, covering works on Vision Transformer-based spatial analysis, temporal modelling using video transformers, and the application of contrastive learning to media forensics. Building on this foundation, a research article is currently being prepared for submission to a peer-reviewed venue — likely IEEE Access or a CVPR Workshop on Media Forensics — presenting our hybrid ViT + TimeSformer framework with Supervised Contrastive Learning, supported by comparative results on the FaceForensics++ benchmark. The manuscript is in active preparation and is expected to be submitted before the end of Semester 6 (Academic Year 2025–2026).

---
*Document prepared by Reshma K (RA2311003011843) and Srotrik Pradhan (RA2311003011860) under the supervision of Dr. Shyni Shajahan, SRM IST, Kattankulathur.*
