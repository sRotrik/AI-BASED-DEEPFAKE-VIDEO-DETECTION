"""
cat_model.py — Dual-stream transformer deepfake detection
Changes in Option 5:
  - ViTStream      : freeze_except_last_n = 6  (was 4)
  - TimeSformerStream: freeze_except_last_n = 4  (was 2)
  More unfrozen layers = more expressive fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, TimesformerModel


# ─────────────────────────────────────────────────────────────
# Streams
# ─────────────────────────────────────────────────────────────

class ViTStream(nn.Module):
    def __init__(self, embed_dim: int = 256, freeze_except_last_n: int = 6):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        for param in self.vit.parameters():
            param.requires_grad = False
        # Unfreeze last N transformer blocks (ViT-Base has 12 blocks total)
        for i in range(freeze_except_last_n):
            for param in self.vit.encoder.layer[11 - i].parameters():
                param.requires_grad = True
        for param in self.vit.layernorm.parameters():
            param.requires_grad = True
        self.proj = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        out = self.vit(pixel_values=x)
        return self.proj(out.last_hidden_state[:, 0, :])


class TimeSformerStream(nn.Module):
    def __init__(self, embed_dim: int = 256, freeze_except_last_n: int = 4):
        super().__init__()
        self.tsf = TimesformerModel.from_pretrained(
            "facebook/timesformer-base-finetuned-k400"
        )
        for param in self.tsf.parameters():
            param.requires_grad = False
        # Unfreeze last N transformer blocks
        num_blocks = len(self.tsf.encoder.layer)
        for i in range(freeze_except_last_n):
            for param in self.tsf.encoder.layer[num_blocks - 1 - i].parameters():
                param.requires_grad = True
        for param in self.tsf.layernorm.parameters():
            param.requires_grad = True
        self.proj = nn.Sequential(
            nn.Linear(self.tsf.config.hidden_size, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        # x: (B, T, C, H, W) — HuggingFace TimeSformer expects exactly this
        x = x.float()  # survive autocast fp16
        out = self.tsf(pixel_values=x)
        return self.proj(out.last_hidden_state[:, 0, :])


# ─────────────────────────────────────────────────────────────
# Fusion heads
# ─────────────────────────────────────────────────────────────

class CATModel(nn.Module):
    def __init__(self, embed_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.vit_stream = ViTStream(embed_dim=embed_dim)
        self.tsf_stream = TimeSformerStream(embed_dim=embed_dim)

        fused_dim = embed_dim * 2

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, frames, clips):
        return self.classifier(self.get_embeddings(frames, clips))

    def get_embeddings(self, frames, clips):
        return torch.cat([self.vit_stream(frames),
                          self.tsf_stream(clips)], dim=1)


class CATModelWithSupCon(nn.Module):
    def __init__(self, embed_dim: int = 256, proj_dim: int = 128,
                 num_classes: int = 2):
        super().__init__()
        self.backbone  = CATModel(embed_dim=embed_dim, num_classes=num_classes)
        self.proj_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def forward(self, frames, clips):
        return self.backbone(frames, clips)

    def forward_supcon(self, frames, clips):
        emb    = self.backbone.get_embeddings(frames, clips)
        logits = self.backbone.classifier(emb)
        proj   = F.normalize(self.proj_head(emb.float()), dim=1)
        return logits, proj


# ─────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Per-class alpha Focal Loss.
    alpha = weight for class-0 (real/minority).
    gamma = 0 → plain weighted CE (safer for small datasets).
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        logits  = logits.clamp(-15.0, 15.0)
        ce      = F.cross_entropy(logits, targets, reduction="none")
        pt      = torch.exp(-ce)
        alpha_t = torch.where(
            targets == 0,
            torch.full_like(ce, self.alpha),
            torch.full_like(ce, 1.0 - self.alpha),
        )
        loss = alpha_t * (1 - pt) ** self.gamma * ce
        return loss.mean()


class SupConLoss(nn.Module):
    """Numerically stable Supervised Contrastive Loss."""
    def __init__(self, temperature: float = 0.15):
        super().__init__()
        self.temperature = max(temperature, 0.1)

    def forward(self, features, labels):
        features = features.float()
        device   = features.device
        B        = features.shape[0]

        features  = F.normalize(features, dim=1)
        labels    = labels.contiguous().view(-1, 1)
        pos_mask  = torch.eq(labels, labels.T).float().to(device)

        sim       = (torch.matmul(features, features.T) / self.temperature
                     ).clamp(-20.0, 20.0)
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        sim       = sim.masked_fill(self_mask, -20.0)
        pos_mask  = pos_mask.masked_fill(self_mask, 0.0)

        sim_max   = sim.detach().max(dim=1, keepdim=True).values
        exp_sim   = torch.exp(sim - sim_max)
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8) + sim_max
        log_prob  = sim - log_denom

        n_pos = pos_mask.sum(dim=1)
        valid = n_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -(pos_mask * log_prob).sum(dim=1)
        loss = loss[valid] / n_pos[valid]
        return loss.mean()


def compute_loss(logits, labels, proj=None, focal_loss_fn=None,
                 supcon_loss_fn=None, alpha: float = 0.2):
    if focal_loss_fn is None:
        focal_loss_fn = FocalLoss()

    focal = focal_loss_fn(logits, labels)

    if proj is not None and supcon_loss_fn is not None:
        supcon = supcon_loss_fn(proj, labels)
        if torch.isnan(supcon):
            return focal, focal, torch.tensor(0.0, device=logits.device)
        return alpha * focal + (1 - alpha) * supcon, focal, supcon

    return focal, focal, torch.tensor(0.0, device=logits.device)