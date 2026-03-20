# =============================================================================
# predict.py — Structured single & batch prediction with disease info
# =============================================================================

import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import config
from model import build_model
from dataset import load_disease_db, get_disease_info
from utils import load_model_bundle


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _get_inference_transform(image_size: Optional[int] = None):
    """Build the inference preprocessing pipeline."""
    if image_size is None:
        image_size = config.IMAGE_SIZE
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _load_model_and_maps(model_path: Optional[str] = None, device: Optional[str] = None):
    """
    Load a model bundle and return (model, label_map, class_names, metadata).
    """
    if model_path is None:
        model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pth")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state_dict, label_map, class_names, metadata = load_model_bundle(
        model_path, device)

    model_name = metadata.get("model_name", config.MODEL_NAME)
    num_classes = len(class_names)

    model = build_model(model_name, num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, label_map, class_names, metadata


# ── Single Prediction ────────────────────────────────────────────────────────

def predict_single(image_path: str,
                   label_map: Optional[dict] = None,
                   disease_db: Optional[dict] = None,
                   model_path: Optional[str] = None,
                   model: Optional[torch.nn.Module] = None,
                   class_names: Optional[list] = None,
                   device: Optional[str] = None) -> dict:
    """
    Predict disease for a single leaf image.

    Returns a structured dict:
        {
            predicted_class_raw: str,
            disease_name: str,
            confidence_pct: float,
            is_healthy: bool,
            plant: str,
            description: str,
            prevention: list[str],
            cure: list[str],
            top5: list[dict],
            inference_time_ms: float,
        }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model if not provided
    if model is None or class_names is None or label_map is None:
        model, label_map, class_names, metadata = _load_model_and_maps(
            model_path, device)

    # Load disease DB if not provided
    if disease_db is None:
        disease_db = load_disease_db()

    # Load and preprocess image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {
            "error": f"Could not open image: {e}",
            "predicted_class_raw": "",
            "disease_name": "",
            "confidence_pct": 0.0,
            "is_healthy": False,
            "plant": "",
            "description": "",
            "prevention": [],
            "cure": [],
            "top5": [],
            "inference_time_ms": 0.0,
        }

    transform = _get_inference_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Top-5
    top5_probs, top5_indices = probs.topk(min(5, len(class_names)))
    idx_to_class = {v: k for k, v in label_map.items()}

    top5_results = []
    for prob, idx in zip(top5_probs.cpu(), top5_indices.cpu()):
        cls_name = idx_to_class.get(idx.item(), f"class_{idx.item()}")
        info = get_disease_info(cls_name, disease_db)
        top5_results.append({
            "class": cls_name,
            "disease_name": info.get("disease_name", cls_name),
            "confidence_pct": round(prob.item() * 100, 2),
        })

    # Top-1
    pred_idx = top5_indices[0].item()
    pred_class = idx_to_class.get(pred_idx, f"class_{pred_idx}")
    pred_conf = round(top5_probs[0].item() * 100, 2)

    # Disease info
    info = get_disease_info(pred_class, disease_db)

    # Parse plant name
    parts = pred_class.split("___")
    plant = parts[0].replace("_", " ") if parts else "Unknown"

    is_healthy = "healthy" in pred_class.lower()

    return {
        "predicted_class_raw": pred_class,
        "disease_name": info.get("disease_name", pred_class),
        "confidence_pct": pred_conf,
        "is_healthy": is_healthy,
        "plant": plant,
        "description": info.get("description", ""),
        "prevention": info.get("prevention", []),
        "cure": info.get("cure", []),
        "top5": top5_results,
        "inference_time_ms": round(elapsed_ms, 2),
    }


# ── Batch Prediction ─────────────────────────────────────────────────────────

def predict_batch(image_paths: list,
                  model_path: Optional[str] = None,
                  disease_db: Optional[dict] = None,
                  device: Optional[str] = None) -> list:
    """
    Predict diseases for multiple leaf images.

    Args:
        image_paths: List of image file paths.
        model_path: Path to model bundle .pth file.
        disease_db: Pre-loaded disease database dict.
        device: 'cuda' or 'cpu'.

    Returns:
        List of prediction dicts (same schema as predict_single).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, label_map, class_names, metadata = _load_model_and_maps(
        model_path, device)

    if disease_db is None:
        disease_db = load_disease_db()

    results = []
    for path in image_paths:
        result = predict_single(
            path,
            label_map=label_map,
            disease_db=disease_db,
            model=model,
            class_names=class_names,
            device=device,
        )
        results.append(result)

    return results


# ── CLI helper ────────────────────────────────────────────────────────────────

def print_prediction(result: dict) -> None:
    """Pretty-print a prediction result dict."""
    print("\n" + "=" * 60)
    print(f"  🌿  Plant     : {result['plant']}")
    print(f"  🔬  Disease   : {result['disease_name']}")
    print(f"  📊  Confidence: {result['confidence_pct']}%")
    print(f"  ✅  Healthy   : {'Yes' if result['is_healthy'] else 'No'}")
    print(f"  ⏱️  Inference : {result['inference_time_ms']:.1f} ms")
    print("-" * 60)

    if result["description"]:
        print(f"\n  📋 Description:")
        print(f"     {result['description'][:200]}...")

    if result["prevention"]:
        print(f"\n  🛡️  Prevention:")
        for i, tip in enumerate(result["prevention"][:5], 1):
            print(f"     {i}. {tip}")

    if result["cure"]:
        print(f"\n  💊 Cure:")
        for i, step in enumerate(result["cure"][:5], 1):
            print(f"     {i}. {step}")

    if result["top5"]:
        print(f"\n  📊 Top-5 Predictions:")
        for item in result["top5"]:
            print(f"     {item['confidence_pct']:6.2f}%  {item['disease_name']}")

    print("=" * 60)
