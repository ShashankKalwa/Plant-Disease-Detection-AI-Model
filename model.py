# =============================================================================
# model.py — Model definitions: EfficientNetV2-S, MobileNetV2, CustomCNN
# =============================================================================
#
# Every model exposes .phase1_mode() and .phase2_mode() for two-phase training.
#   Phase 1 — freeze backbone, train head only (warm-up)
#   Phase 2 — unfreeze ALL layers, fine-tune with a small LR
#
# The build_model() factory returns whichever architecture is requested.
# =============================================================================

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

import config


# ── 1. EfficientNetV2-S (primary — targets 92%+) ─────────────────────────────

class PlantEfficientNetV2(nn.Module):
    """
    EfficientNet-V2-S backbone with a custom two-layer classifier head.

    Architecture:
        backbone.features → AdaptiveAvgPool2d(1,1) → Flatten →
        Dropout(0.3) → Linear(1280,512) → GELU →
        Dropout(0.15) → Linear(512, num_classes)

    Two-phase training:
        phase1_mode() — freeze backbone, train head only
        phase2_mode() — unfreeze everything for full fine-tuning
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES) -> None:
        super().__init__()

        # Load pretrained backbone — ImageNet-1K weights for V2-S
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_v2_s(weights=weights)

        # Extract the convolutional feature layers (everything except classifier)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Custom classifier head — deeper than default for better plant features
        in_features = 1280  # EfficientNetV2-S final conv output channels
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=config.DROPOUT_RATE),          # 0.3
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=config.DROPOUT_RATE / 2),      # 0.15
            nn.Linear(512, num_classes),
        )

        # Start in phase 1 mode (backbone frozen)
        self.phase1_mode()

    def phase1_mode(self) -> None:
        """Freeze ALL backbone layers — only the classifier head is trainable."""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Model] Phase 1 — Head only | "
              f"Trainable: {trainable:,} / {total:,} params")

    def phase2_mode(self) -> None:
        """Unfreeze ALL layers (backbone + head) for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Model] Phase 2 — Full fine-tune | "
              f"Trainable: {trainable:,} / {total:,} params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: features → avgpool → classifier."""
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


# ── 2. MobileNetV2 (lightweight fallback for CPU-constrained envs) ────────────

class PlantMobileNetV2(nn.Module):
    """
    MobileNetV2 backbone with custom two-layer classifier head.

    Fully unfrozen at 224×224 for best accuracy (no UNFREEZE_LAYERS cap).
    Kept as a fallback for CPU-constrained environments or fast prototyping.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES) -> None:
        super().__init__()

        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        self.features = backbone.features

        in_features = backbone.classifier[1].in_features  # 1280
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE / 2),
            nn.Linear(512, num_classes),
        )

        # Start in phase 1 mode
        self.phase1_mode()

    def phase1_mode(self) -> None:
        """Freeze backbone, train head only."""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Model] Phase 1 — Head only | "
              f"Trainable: {trainable:,} / {total:,} params")

    def phase2_mode(self) -> None:
        """Unfreeze ALL layers — no cap, no UNFREEZE_LAYERS."""
        for param in self.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Model] Phase 2 — Full fine-tune | "
              f"Trainable: {trainable:,} / {total:,} params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through features → classifier."""
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── 3. Custom Lightweight CNN (from-scratch baseline) ─────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU → optional MaxPool2d."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution block."""
        return self.block(x)


class CustomCNN(nn.Module):
    """
    5-block CNN trained from scratch (no pretrained weights).

    Input  : (B, 3, IMAGE_SIZE, IMAGE_SIZE)
    Output : (B, NUM_CLASSES)
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(512, num_classes),
        )

    def phase1_mode(self) -> None:
        """No-op for CustomCNN (no pretrained backbone to freeze)."""
        pass

    def phase2_mode(self) -> None:
        """No-op for CustomCNN (all layers are always trainable)."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: features → classifier."""
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(model_name: Optional[str] = None,
                num_classes: int = config.NUM_CLASSES) -> nn.Module:
    """
    Build and return the requested model architecture.

    Args:
        model_name: One of 'efficientnet_v2_s', 'mobilenet_v2', 'custom_cnn'.
                    Defaults to config.MODEL_NAME.
        num_classes: Number of output classes (default: 38).

    Returns:
        nn.Module with .phase1_mode() and .phase2_mode() methods.
    """
    if model_name is None:
        model_name = config.MODEL_NAME

    if model_name == "efficientnet_v2_s":
        model = PlantEfficientNetV2(num_classes)
    elif model_name == "mobilenet_v2":
        model = PlantMobileNetV2(num_classes)
    elif model_name == "custom_cnn":
        model = CustomCNN(num_classes)
    else:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            "Choose 'efficientnet_v2_s', 'mobilenet_v2', or 'custom_cnn'."
        )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_name}  |  "
          f"Total params: {total:,}  |  Trainable: {trainable:,}")
    return model
