# =============================================================================
# evaluate.py — Test set evaluation with comprehensive metrics
# =============================================================================

import os
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)

import config
from utils import (
    compute_accuracy, compute_top5_accuracy,
    plot_per_class_accuracy, plot_confusion_matrix,
)


def evaluate(model: nn.Module, test_loader, label_map: dict,
             device: Optional[torch.device] = None,
             save_plots: bool = True) -> dict:
    """
    Evaluate the model on the test set.

    Computes:
        - Top-1 and Top-5 accuracy
        - Per-class precision, recall, F1
        - Macro and weighted F1
        - Confusion matrix

    Args:
        model: Trained model.
        test_loader: Test DataLoader.
        label_map: {class_name: index} mapping.
        device: Torch device. Auto-detects if None.
        save_plots: Whether to save confusion matrix & per-class accuracy plots.

    Returns:
        Dict with all metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []

    print("\n[Evaluate] Running evaluation on test set...")

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="  Testing", leave=False, ncols=100)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(outputs.cpu())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = torch.cat(all_logits, dim=0)

    # Class names in index order
    class_names = sorted(label_map, key=label_map.get)

    # ── Metrics ───────────────────────────────────────────────────────────────
    top1_acc = accuracy_score(all_labels, all_preds)

    # Top-5 accuracy
    top5_correct = 0
    for i, logit in enumerate(all_logits):
        top5_indices = logit.topk(min(5, logit.size(0))).indices.numpy()
        if all_labels[i] in top5_indices:
            top5_correct += 1
    top5_acc = top5_correct / len(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted",
                           zero_division=0)

    # Classification report
    report_str = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # ── Print Results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n  Top-1 Accuracy  : {top1_acc:.4f}  ({top1_acc * 100:.2f}%)")
    print(f"  Top-5 Accuracy  : {top5_acc:.4f}  ({top5_acc * 100:.2f}%)")
    print(f"  Macro F1        : {macro_f1:.4f}")
    print(f"  Weighted F1     : {weighted_f1:.4f}")
    print(f"\n{report_str}")

    # ── Save Plots ────────────────────────────────────────────────────────────
    if save_plots:
        os.makedirs(config.PLOT_SAVE_DIR, exist_ok=True)
        plot_confusion_matrix(cm, class_names, config.PLOT_SAVE_DIR)
        plot_per_class_accuracy(report_dict, config.PLOT_SAVE_DIR)

    return {
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
