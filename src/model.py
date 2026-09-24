"""
model.py
========
Phase 4 - CNN Architecture with Transfer Learning & Grad-CAM
Diabetic Retinopathy Stage Detection (Computer Vision Assignment)

This module implements a transfer learning architecture for 5-class diabetic
retinopathy (DR) grading and binary DR detection from retinal fundus photographs.

Supported Backbones:
  1. EfficientNet-B0 (Default / Recommended):
     - Parameters: ~5.3M (5x fewer than ResNet-50)
     - Key feature: Compound scaling + Squeeze-and-Excitation (SE) channel attention
     - Memory footprint: ~20 MB (ideal for Google Colab T4 GPU limits)
     - Medical imaging track record: State-of-the-art efficiency on APTOS 2019

  2. ResNet-50 (Alternative Baseline):
     - Parameters: ~25.6M
     - Key feature: Identity residual connections mitigating vanishing gradients
     - Standard baseline across literature (EyePACS, Messidor, APTOS)

Core Capabilities:
  - Pretrained ImageNet feature extraction with staged freezing/unfreezing
  - Modular classification head with Dropout (p=0.4, p=0.2) and BatchNorm
  - Unified output: 5-class DR staging AND binary DR presence from single forward pass
  - Integrated Grad-CAM class activation mapping for clinical interpretability
  - Fully portable: pure relative paths, zero hardcoded OS-specific paths

Author: Sanduni Herath
Repository: https://github.com/SanduniHerath/diabetic-retinopathy-cv-assignment
"""

import os
import sys
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
)

# ---------------------------------------------------------------------------
# Class definitions matching APTOS 2019 clinical scale
# ---------------------------------------------------------------------------
CLASS_NAMES: List[str] = [
    "No_DR",            # Grade 0: Normal retina, zero visible microaneurysms
    "Mild",             # Grade 1: Microaneurysms only
    "Moderate",         # Grade 2: More than microaneurysms, less than severe
    "Severe",           # Grade 3: >20 intraretinal haemorrhages, venous beading, IRMA
    "Proliferative_DR", # Grade 4: Neovascularisation, vitreous/preretinal haemorrhage
]
NUM_CLASSES: int = len(CLASS_NAMES)


# ===========================================================================
# 1. Transfer Learning Model Architecture
# ===========================================================================
class DRTransferModel(nn.Module):
    """
    CNN Transfer Learning model for Diabetic Retinopathy classification and staging.

    Satisfies both assignment objectives with a single unified architecture:
      - Objective A (Binary Detection): Normal (No_DR) vs Disease Present (DR Positive)
      - Objective B (Stage Grading): 5-class severity grading (Grades 0 to 4)

    Architecture:
      1. Pretrained Backbone (EfficientNet-B0 or ResNet-50)
      2. Adaptive Average Pooling (spatial dimension -> 1x1)
      3. Classification Head:
         Dropout(p=0.4) -> Linear(in_feat, hidden_dim) -> BatchNorm1d -> SiLU/ReLU
         -> Dropout(p=0.2) -> Linear(hidden_dim, num_classes) -> Raw Logits
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout_rate: float = 0.4,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone_name: str = backbone_name.lower()
        self.num_classes: int = num_classes
        self.dropout_rate: float = dropout_rate
        self.hidden_dim: int = hidden_dim

        # -------------------------------------------------------------------
        # Instantiate Backbone
        # -------------------------------------------------------------------
        if self.backbone_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model: Any = models.efficientnet_b0(weights=weights)

            # Features extractor (stem + 8 MBConv stages)
            self.backbone: nn.Sequential = base_model.features
            in_features: int = 1280
            if hasattr(base_model, "classifier"):
                cls_seq = getattr(base_model, "classifier")
                if len(cls_seq) > 1 and hasattr(cls_seq[1], "in_features"):
                    in_features = int(getattr(cls_seq[1], "in_features"))

            self.target_layer: nn.Module = self.backbone[-1]  # Conv2dNormActivation (1280 channels)

            # Custom classification head
            self.head: nn.Sequential = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(p=dropout_rate / 2.0),
                nn.Linear(hidden_dim, num_classes),
            )

        elif self.backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base_model_res: Any = models.resnet50(weights=weights)

            # Exclude original avgpool and fc
            self.backbone = nn.Sequential(
                base_model_res.conv1,
                base_model_res.bn1,
                base_model_res.relu,
                base_model_res.maxpool,
                base_model_res.layer1,
                base_model_res.layer2,
                base_model_res.layer3,
                base_model_res.layer4,
            )
            in_features = 2048
            if hasattr(base_model_res, "fc") and hasattr(base_model_res.fc, "in_features"):
                in_features = int(base_model_res.fc.in_features)

            self.target_layer = base_model_res.layer4[-1]  # Last Bottleneck block

            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate / 2.0),
                nn.Linear(hidden_dim, num_classes),
            )

        else:
            raise ValueError(
                f"Unsupported backbone: '{backbone_name}'. Choose 'efficientnet_b0' or 'resnet50'."
            )

        # Default initialization: Freeze all backbone parameters for Stage 1 feature extraction
        self.freeze_backbone()

    # -----------------------------------------------------------------------
    # Forward Pass
    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw unnormalized logits for each of the 5 classes.
        Raw logits are optimal for numerical stability with torch.nn.CrossEntropyLoss.
        """
        features: torch.Tensor = self.backbone(x)
        logits: torch.Tensor = self.head(features)
        return logits

    # -----------------------------------------------------------------------
    # Layer Freezing / Unfreezing Strategies (Transfer Learning Controls)
    # -----------------------------------------------------------------------
    def freeze_backbone(self) -> None:
        """
        Freeze all backbone layers (requires_grad = False).
        Used during Stage 1 (Warmup / Feature Extraction) to train only the
        randomly initialized classification head without destroying pretrained weights.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Head always remains trainable
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_all(self) -> None:
        """
        Unfreeze all backbone and head parameters for full end-to-end fine-tuning.
        """
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_top_blocks(self, num_blocks: int = 2) -> None:
        """
        Selectively unfreeze only the top N convolutional blocks while keeping
        early general feature extractors (edges, simple textures) frozen.

        For EfficientNet-B0:
          - Total 8 feature stages (indices 0 to 7, plus final conv stage 8).
          - num_blocks=2 unfreezes stage 7 (MBConv6) and stage 8 (Conv2d 1280).
        For ResNet-50:
          - 4 residual layers. num_blocks=1 unfreezes layer4, num_blocks=2 unfreezes layer3+layer4.
        """
        # First ensure all backbone parameters are frozen
        self.freeze_backbone()

        stages: List[nn.Module] = list(self.backbone.children())
        total_stages = len(stages)
        unfreeze_from = max(0, total_stages - num_blocks)

        for i in range(unfreeze_from, total_stages):
            for param in stages[i].parameters():
                param.requires_grad = True

        # Head remains trainable
        for param in self.head.parameters():
            param.requires_grad = True

    # -----------------------------------------------------------------------
    # Dual Requirement Prediction (Classification + Staging from 1 Model)
    # -----------------------------------------------------------------------
    def predict_stage_and_presence(
        self,
        x: torch.Tensor,
        dr_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Derives both assignment deliverables from a single forward pass:
          1. Binary DR Presence (Normal vs Diabetic Retinopathy)
          2. Multi-class DR Stage (No_DR, Mild, Moderate, Severe, Proliferative_DR)

        Mathematical Foundation:
          Given predicted class probabilities P = [p_0, p_1, p_2, p_3, p_4]:
          - Binary Presence:
              P(DR_Positive) = sum_{k=1}^4 p_k = 1.0 - p_0
              Decision: DR Present if P(DR_Positive) >= dr_threshold (default 0.5)
          - Multi-class Severity Stage:
              Stage_Grade = argmax_{k in {0..4}} p_k
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)

        p_no_dr = probabilities[:, 0]
        p_dr_positive = 1.0 - p_no_dr
        binary_pred = (p_dr_positive >= dr_threshold).long()

        predicted_stages = torch.argmax(probabilities, dim=1)

        results: Dict[str, Any] = {
            "logits": logits.cpu().numpy(),
            "probabilities": probabilities.cpu().numpy(),
            "p_dr_positive": p_dr_positive.cpu().numpy(),
            "binary_prediction": binary_pred.cpu().numpy(),  # 0: No DR, 1: DR Present
            "predicted_stage": predicted_stages.cpu().numpy(),
            "predicted_class_name": [CLASS_NAMES[idx] for idx in predicted_stages.cpu().tolist()],
        }
        return results

    # -----------------------------------------------------------------------
    # Parameter Inspection
    # -----------------------------------------------------------------------
    def get_parameter_counts(self) -> Dict[str, int]:
        """
        Returns exact counts of total, trainable, and frozen parameters.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": frozen,
        }

    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Returns the target convolutional layer for Grad-CAM gradient capture.
        """
        return self.target_layer


# ===========================================================================
# 2. Grad-CAM (Gradient-weighted Class Activation Mapping) Utility
# ===========================================================================
# Ensure src directory is present in sys.path for robust modular import
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from gradcam import GradCAM, generate_gradcam_heatmap


# ===========================================================================
# 3. Model Factory and Summary Reporter
# ===========================================================================
def build_model(
    backbone_name: str = "efficientnet_b0",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout_rate: float = 0.4,
    hidden_dim: int = 256,
) -> DRTransferModel:
    """
    Factory function to construct the DRTransferModel.
    """
    return DRTransferModel(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
    )


def format_model_summary(model: DRTransferModel, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> str:
    """
    Generates a detailed diagnostic summary string of the model architecture,
    layer breakdown, and parameter counts across Stage 1 and Stage 2 transfer learning.
    """
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append(f"  TRANSFER LEARNING MODEL ARCHITECTURE: {model.backbone_name.upper()}")
    lines.append("  Diabetic Retinopathy Stage Detection & Binary Classification")
    lines.append("=" * 80)

    # 1. High level config
    lines.append("\n[1] SPECIFICATIONS & HYPERPARAMETERS:")
    lines.append(f"  - Backbone Architecture    : {model.backbone_name}")
    lines.append(f"  - Pretrained Weights       : ImageNet-1K (Default)")
    lines.append(f"  - Input Dimensions         : {input_shape}")
    lines.append(f"  - Output Target Classes    : {model.num_classes} ({', '.join(CLASS_NAMES)})")
    lines.append(f"  - Primary Dropout Rate     : {model.dropout_rate}")
    lines.append(f"  - Projection Hidden Dim    : {model.hidden_dim}")
    lines.append(f"  - Grad-CAM Target Layer    : {model.get_gradcam_target_layer().__class__.__name__}")

    # 2. Stage 1 Parameter Counts (Frozen Backbone / Feature Extraction)
    model.freeze_backbone()
    p1 = model.get_parameter_counts()
    lines.append("\n[2] STAGE 1: FEATURE EXTRACTION (Base Layers Frozen)")
    lines.append(f"  - Total Parameters         : {p1['total_params']:,}")
    lines.append(f"  - Trainable (Head only)    : {p1['trainable_params']:,} ({p1['trainable_params']/p1['total_params']*100:.2f}%)")
    lines.append(f"  - Frozen (Backbone)        : {p1['frozen_params']:,} ({p1['frozen_params']/p1['total_params']*100:.2f}%)")

    # 3. Stage 2 Parameter Counts (Fine-Tuning Top Blocks)
    model.unfreeze_top_blocks(num_blocks=2)
    p2 = model.get_parameter_counts()
    lines.append("\n[3] STAGE 2: FINE-TUNING (Top 2 Blocks + Head Unfrozen)")
    lines.append(f"  - Total Parameters         : {p2['total_params']:,}")
    lines.append(f"  - Trainable Parameters     : {p2['trainable_params']:,} ({p2['trainable_params']/p2['total_params']*100:.2f}%)")
    lines.append(f"  - Frozen Parameters        : {p2['frozen_params']:,} ({p2['frozen_params']/p2['total_params']*100:.2f}%)")

    # 4. Stage 3 Parameter Counts (Full Unfreeze Reference)
    model.unfreeze_all()
    p3 = model.get_parameter_counts()
    lines.append("\n[4] FULL UNFREEZE REFERENCE (All Layers Trainable)")
    lines.append(f"  - Total Parameters         : {p3['total_params']:,}")
    lines.append(f"  - Trainable Parameters     : {p3['trainable_params']:,} (100.00%)")
    lines.append(f"  - Frozen Parameters        : 0 (0.00%)")

    # Reset back to Stage 1 default
    model.freeze_backbone()

    # 5. Classification Head Breakdown
    lines.append("\n[5] CLASSIFICATION HEAD LAYERS:")
    for idx, layer in enumerate(list(model.head.children())):
        num_params = sum(p.numel() for p in layer.parameters())
        lines.append(f"  [{idx}] {layer.__class__.__name__:<22} Parameters: {num_params:>10,}")

    # 6. Verification Forward Pass
    dummy_input = torch.randn(*input_shape)
    model.eval()
    with torch.no_grad():
        out_logits = model(dummy_input)
        preds = model.predict_stage_and_presence(dummy_input)

    lines.append("\n[6] FORWARD PASS & DUAL OBJECTIVE VERIFICATION:")
    lines.append(f"  - Input Tensor Shape       : {list(dummy_input.shape)}")
    lines.append(f"  - Output Logits Shape      : {list(out_logits.shape)}")
    sample_probs_arr: np.ndarray = np.asarray(preds["probabilities"][0])
    sample_p_dr_val: float = float(np.asarray(preds["p_dr_positive"])[0])
    lines.append(f"  - Sample Softmax Probs     : {np.round(sample_probs_arr, 4).tolist()}")
    lines.append(f"  - Sample Stage Prediction  : {preds['predicted_class_name'][0]} (Grade {preds['predicted_stage'][0]})")
    lines.append(f"  - Sample DR Binary Prob    : {sample_p_dr_val:.4f} (Positive: {bool(preds['binary_prediction'][0])})")

    # 7. Grad-CAM Verification
    try:
        heatmap = generate_gradcam_heatmap(model, dummy_input, target_class=1)
        lines.append(f"  - Grad-CAM Heatmap Shape   : {heatmap.shape} (Range: [{heatmap.min():.2f}, {heatmap.max():.2f}]) -> PASSED")
    except Exception as e:
        lines.append(f"  - Grad-CAM Verification    : FAILED ({e})")

    lines.append("=" * 80)
    return "\n".join(lines)


# ===========================================================================
# 4. Command-Line Entry Point
# ===========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DR Transfer Learning CNN Model and Inspect Architecture."
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
        choices=["efficientnet_b0", "resnet50"],
        help="CNN Backbone architecture to instantiate (default: efficientnet_b0).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout probability in classification head (default: 0.4).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Dimensionality of dense projection layer in head (default: 256).",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default=os.path.join("reports", "model_architecture", "model_summary.txt"),
        help="Relative path to save text model summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"\nBuilding Transfer Learning model with backbone: {args.backbone} ...")
    model = build_model(
        backbone_name=args.backbone,
        dropout_rate=args.dropout,
        hidden_dim=args.hidden_dim,
    )

    summary_str = format_model_summary(model)
    print(summary_str)

    # Save summary report
    out_path = args.output_summary
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary_str)
    print(f"\n[OK] Model summary written to: {out_path}\n")


if __name__ == "__main__":
    main()
