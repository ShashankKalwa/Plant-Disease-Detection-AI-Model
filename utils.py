# =============================================================================
# utils.py — Utilities: checkpointing, plotting, model bundles, helpers
# =============================================================================
#
# Core utilities used across the entire pipeline:
#   • BestModelCheckpoint — saves weights when val_accuracy improves
#   • EarlyStopping      — stops training after patience epochs w/o improvement
#   • save_model_bundle / load_model_bundle — persist label_map + metadata
#   • plot_curves / plot_per_class_accuracy / plot_confusion_matrix
#   • normalise_class_name / format_prevention_list / format_cure_list
# =============================================================================

import json
import os
import re
import copy
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import numpy as np


# ── Best Model Checkpoint (tracks val_accuracy — higher is better) ────────────

class BestModelCheckpoint:
    """
    Saves the model state_dict whenever validation accuracy improves.

    Unlike loss-based checkpointing, this tracks val_accuracy so the
    checkpoint saved corresponds to the epoch with the highest accuracy
    on the validation set.
    """

    def __init__(self, save_path: str, verbose: bool = True) -> None:
        self.save_path = save_path
        self.verbose = verbose
        self.best_accuracy: float = 0.0
        self.best_state: Optional[dict] = None

    def step(self, val_accuracy: float, model: nn.Module) -> bool:
        """
        Check if val_accuracy improved. If so, deep-copy the state.

        Args:
            val_accuracy: Current epoch's validation accuracy (0.0–1.0).
            model: The model to checkpoint.

        Returns:
            True if this was a new best, False otherwise.
        """
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.best_state = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"  ✔ New best model saved (val_acc={val_accuracy:.4f})")
            return True
        return False

    def save(self) -> None:
        """Persist the best state_dict to disk."""
        if self.best_state is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(self.best_state, self.save_path)
            if self.verbose:
                print(f"  [Checkpoint] Saved best model to {self.save_path}")


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when validation accuracy stops improving.

    The counter increments each epoch where val_accuracy does not
    exceed (best_accuracy + min_delta). When counter reaches patience,
    step() returns True to signal the training loop to break.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4,
                 verbose: bool = True) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter: int = 0
        self.best_accuracy: float = 0.0
        self.should_stop: bool = False

    def step(self, val_accuracy: float) -> bool:
        """
        Check whether training should stop.

        Args:
            val_accuracy: Current epoch's validation accuracy (0.0–1.0).

        Returns:
            True if training should stop, False otherwise.
        """
        if val_accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = val_accuracy
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] No improvement for "
                      f"{self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print("  [EarlyStopping] Triggered — stopping training.")
        return self.should_stop


# ── Model Bundle Save / Load ─────────────────────────────────────────────────

def save_model_bundle(model: nn.Module, label_map: dict,
                      class_names: list, metadata: dict,
                      path: str) -> None:
    """
    Save a complete model bundle to a single .pth file.

    The bundle contains:
        • state_dict   — model weights
        • label_map    — {class_name: index} mapping
        • class_names  — ordered list of class names
        • metadata     — arbitrary dict (model_name, image_size, val_accuracy, etc.)

    This allows the API and predict.py to fully restore the model
    without needing access to the data/ directory.

    Args:
        model: Trained model.
        label_map: {class_name: index} mapping.
        class_names: Sorted list of class names.
        metadata: Additional info (model_name, image_size, val_accuracy, etc.).
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "label_map": label_map,
        "class_names": class_names,
        **metadata,
    }
    torch.save(bundle, path)
    print(f"[Bundle] Saved model bundle to {path}")


def load_model_bundle(path: str,
                      device: str = "cpu") -> tuple[dict, dict, list, dict]:
    """
    Load a model bundle from a .pth file.

    Args:
        path: Path to the .pth bundle file.
        device: Device to map tensors to ('cpu' or 'cuda').

    Returns:
        (state_dict, label_map, class_names, metadata)

    Raises:
        FileNotFoundError: If the bundle file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model bundle not found: {path}")

    bundle = torch.load(path, map_location=device, weights_only=False)
    state_dict = bundle.pop("state_dict")
    label_map = bundle.pop("label_map")
    class_names = bundle.pop("class_names")
    metadata = bundle  # everything remaining is metadata
    return state_dict, label_map, class_names, metadata


# ── Training History ──────────────────────────────────────────────────────────

def save_history(history: dict, path: str) -> None:
    """
    Save training history dict to JSON.

    Args:
        history: Dict with train_loss, val_loss, train_acc, val_acc lists.
        path: Output JSON file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[History] Saved to {path}")


def load_history(path: str) -> dict:
    """
    Load training history dict from JSON.

    Args:
        path: Path to the JSON history file.

    Returns:
        Dict with train_loss, val_loss, train_acc, val_acc lists.
    """
    with open(path, "r") as f:
        return json.load(f)


# ── Accuracy Helpers ──────────────────────────────────────────────────────────

def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute top-1 accuracy from logits and labels.

    Args:
        outputs: Model logits, shape (N, C).
        labels: Ground truth labels, shape (N,).

    Returns:
        Accuracy as a float in [0.0, 1.0].
    """
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).sum().item()  # type: ignore[union-attr]
    return correct / labels.size(0)


def compute_top5_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute top-5 accuracy from logits and labels.

    Args:
        outputs: Model logits, shape (N, C).
        labels: Ground truth labels, shape (N,).

    Returns:
        Top-5 accuracy as a float in [0.0, 1.0].
    """
    _, top5_preds = outputs.topk(5, dim=1)
    correct = top5_preds.eq(labels.unsqueeze(1).expand_as(top5_preds)).any(dim=1)
    return correct.sum().item() / labels.size(0)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_curves(history: dict, save_dir: str) -> None:
    """
    Plot and save training loss and accuracy curves.

    Args:
        history: Dict with train_loss, val_loss, train_acc, val_acc lists.
        save_dir: Directory to save the plot PNGs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # — Loss curve —
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved loss_curve.png")

    # — Accuracy curve —
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy", markersize=4)
    ax.plot(epochs, history["val_acc"], "r-o", label="Val Accuracy", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved accuracy_curve.png")


def plot_per_class_accuracy(report_dict: dict, save_dir: str) -> None:
    """
    Plot a horizontal bar chart of per-class accuracy (recall).

    Args:
        report_dict: Output of sklearn classification_report(output_dict=True).
        save_dir: Directory to save the plot PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # Extract per-class entries (skip summary rows)
    skip = {"accuracy", "macro avg", "weighted avg"}
    classes: list[str] = []
    accuracies: list[float] = []
    for cls_name, metrics in report_dict.items():
        if cls_name in skip:
            continue
        if isinstance(metrics, dict) and "recall" in metrics:
            classes.append(cls_name)
            accuracies.append(metrics["recall"])  # recall = per-class accuracy

    # Sort by accuracy ascending for the bar chart
    pairs = sorted(zip(accuracies, classes))
    accuracies, classes = zip(*pairs) if pairs else ([], [])

    fig, ax = plt.subplots(figsize=(12.0, max(8.0, len(classes) * 0.35)))
    colors = plt.cm.RdYlGn(np.array(accuracies))
    ax.barh(range(len(classes)), accuracies, color=colors)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Accuracy (Recall)")
    ax.set_title("Per-Class Accuracy")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "per_class_accuracy.png"), dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved per_class_accuracy.png")


def plot_confusion_matrix(cm: np.ndarray, class_names: list,
                          save_dir: str) -> None:
    """
    Plot and save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array of shape (num_classes, num_classes).
        class_names: List of class names in index order.
        save_dir: Directory to save the plot PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(14.0, len(class_names) * 0.5),
                                     max(12.0, len(class_names) * 0.45)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=5)
    ax.set_yticklabels(class_names, fontsize=5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved confusion_matrix.png")


# ── CSV / Disease Name Helpers ────────────────────────────────────────────────

def normalise_class_name(name: str) -> str:
    """
    Normalise a class folder name or CSV disease name for fuzzy matching.

    Strips underscores, triple-underscores, parentheses, commas, and
    collapses whitespace to produce a lowercase comparator string.

    Examples:
        'Tomato___Early_blight' → 'tomato early blight'
        'Tomato Early blight'   → 'tomato early blight'
        'Cherry_(including_sour)___Powdery_mildew' → 'cherry including sour powdery mildew'

    Args:
        name: Raw class name string.

    Returns:
        Normalised lowercase string.
    """
    s = name.strip()
    s = s.replace("___", " ").replace("__", " ").replace("_", " ")
    s = s.replace(",", "").replace("(", "").replace(")", "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def format_prevention_list(raw_string: str) -> list[str]:
    """
    Split semicolon-separated prevention methods into a clean list.

    Args:
        raw_string: Raw CSV cell value with ';'-delimited tips.

    Returns:
        List of stripped non-empty strings.
    """
    if not raw_string or not raw_string.strip():
        return []
    return [tip.strip() for tip in raw_string.split(";") if tip.strip()]


def format_cure_list(raw_string: str) -> list[str]:
    """
    Split semicolon-separated cure methods into a clean list.

    Args:
        raw_string: Raw CSV cell value with ';'-delimited steps.

    Returns:
        List of stripped non-empty strings.
    """
    if not raw_string or not raw_string.strip():
        return []
    return [step.strip() for step in raw_string.split(";") if step.strip()]
