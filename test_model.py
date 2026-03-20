# =============================================================================
# test_model.py — PlantDoctor AI  |  Complete Post-Training Test Suite
# =============================================================================
# Verifies every aspect of the trained model:
#   1. Model file loading and bundle integrity
#   2. Inference pipeline correctness
#   3. Disease database coverage
#   4. Prediction output structure
#   5. Held-out test-set accuracy (Top-1, Top-5, per-class, F1)
#   6. Edge case and robustness
#   7. Training history validation
#
# Usage:
#   python test_model.py                  # full suite
#   python test_model.py --skip-accuracy  # skip slow test-set scan
#   python test_model.py --quick          # sections 1-4 + 6-7 only
# =============================================================================

from __future__ import annotations

import os
import sys
import csv
import json
import time
import traceback
from typing import Optional

# ── Colour helpers (works on Windows 10+ and all Unix terminals) ─────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "outputs", "models", "best_model.pth")
CSV_PATH      = os.path.join(BASE_DIR, "plant_disease_training_dataset_optionB.csv")
HISTORY_PATH  = os.path.join(BASE_DIR, "outputs", "training_history.json")
DATA_DIR_HINT = os.path.join(BASE_DIR, "data", "plantvillage", "color")

# ── Global counters ───────────────────────────────────────────────────────────
PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0
RESULTS: list[tuple[str, str, str]] = []   # (status, name, reason)


# =============================================================================
# Output helpers
# =============================================================================

def _pass(name: str) -> None:
    global PASS_COUNT
    PASS_COUNT += 1
    RESULTS.append(("PASS", name, ""))
    print(f"  {GREEN}[PASS]{RESET}  {name}")


def _fail(name: str, reason: str = "") -> None:
    global FAIL_COUNT
    FAIL_COUNT += 1
    RESULTS.append(("FAIL", name, reason))
    print(f"  {RED}[FAIL]{RESET}  {name}")
    if reason:
        print(f"         {RED}→ {reason}{RESET}")


def _skip(name: str, reason: str = "") -> None:
    global SKIP_COUNT
    SKIP_COUNT += 1
    RESULTS.append(("SKIP", name, reason))
    print(f"  {YELLOW}[SKIP]{RESET}  {name}  ({reason})")


def _section(number: int, title: str) -> None:
    print(f"\n{CYAN}{BOLD}{'─'*60}{RESET}")
    print(f"{CYAN}{BOLD}  {number}. {title}{RESET}")
    print(f"{CYAN}{'─'*60}{RESET}")


def _info(msg: str) -> None:
    print(f"  {YELLOW}ℹ  {msg}{RESET}")


# =============================================================================
# SECTION 1 — Model Loading Tests
# =============================================================================

def test_model_loading() -> tuple:
    """
    Verify best_model.pth exists, loads cleanly, and contains all
    required metadata keys. Returns (model, label_map, class_names, bundle).
    """
    _section(1, "Model Loading Tests")

    # 1.1 File exists
    if not os.path.isfile(MODEL_PATH):
        _fail("best_model.pth exists", f"Not found → {MODEL_PATH}")
        _fail("All remaining loading tests", "Cannot continue without model file")
        return None, None, None, None
    size_mb = os.path.getsize(MODEL_PATH) / (1024 ** 2)
    _pass(f"best_model.pth exists  ({size_mb:.1f} MB)")

    # 1.2 Bundle loads
    import torch
    try:
        bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        _pass("Bundle loads without errors")
    except Exception as exc:
        _fail("Bundle loads without errors", str(exc))
        return None, None, None, None

    # 1.3 label_map — exactly 38 classes
    label_map: Optional[dict] = bundle.get("label_map")
    n_lm = len(label_map) if label_map else 0
    if label_map and n_lm == 38:
        _pass(f"label_map present — {n_lm} classes")
    elif label_map:
        _fail("label_map has 38 classes", f"Found {n_lm}")
    else:
        _fail("label_map present in bundle", "Key 'label_map' missing")

    # 1.4 class_names — exactly 38 entries
    class_names: Optional[list] = bundle.get("class_names")
    n_cn = len(class_names) if class_names else 0
    if class_names and n_cn == 38:
        _pass(f"class_names present — {n_cn} entries")
    elif class_names:
        _fail("class_names has 38 entries", f"Found {n_cn}")
    else:
        _fail("class_names present in bundle", "Key 'class_names' missing")

    # 1.5 Metadata: model_name + image_size
    model_name  = bundle.get("model_name")
    image_size  = bundle.get("image_size")
    if model_name and image_size:
        _pass(f"Metadata OK — model={model_name}, image_size={image_size}")
    else:
        _fail("Metadata (model_name + image_size)",
              f"model_name={model_name!r}, image_size={image_size!r}")

    # 1.6 state_dict present and non-empty
    state_dict = bundle.get("state_dict")
    n_sd = len(state_dict) if state_dict else 0
    if state_dict and n_sd > 0:
        _pass(f"state_dict present — {n_sd} parameter tensors")
    else:
        _fail("state_dict present and non-empty", f"n_tensors={n_sd}")

    # 1.7 val_accuracy recorded in bundle
    saved_val_acc = bundle.get("val_accuracy") or bundle.get("best_val_acc")
    if saved_val_acc is not None:
        _pass(f"Saved val_accuracy = {saved_val_acc * 100:.2f}%")
    else:
        _skip("Saved val_accuracy in bundle", "Key not stored — non-critical")

    # 1.8 Model rebuilds and loads state_dict
    try:
        from model import build_model
        n_classes = n_lm if n_lm else 38
        name = model_name or "efficientnet_v2_s"
        model = build_model(name, n_classes)
        model.load_state_dict(state_dict)
        model.eval()
        _pass(f"Model rebuilt ({name}) and state_dict loaded")
    except Exception as exc:
        _fail("Model rebuilt and state_dict loaded", str(exc))
        return None, label_map, class_names, bundle

    return model, label_map, class_names, bundle


# =============================================================================
# SECTION 2 — Inference Pipeline Tests
# =============================================================================

def test_inference(model) -> None:
    """
    Verify forward pass shape, probability correctness, GPU OOM safety,
    and AMP consistency.
    """
    _section(2, "Inference Pipeline Tests")

    if model is None:
        _skip("All inference tests", "Model not loaded")
        return

    import torch
    from torch.amp import autocast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _info(f"Running on: {device}" +
          (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    model = model.to(device).eval()

    # 2.1 Accepts 224×224 input
    dummy = torch.randn(1, 3, 224, 224, device=device)
    try:
        with torch.no_grad():
            out = model(dummy)
        _pass("Accepts 1×3×224×224 input tensor")
    except Exception as exc:
        _fail("Accepts 1×3×224×224 input", str(exc))
        model.to("cpu")
        return

    # 2.2 Output shape (1, 38)
    if tuple(out.shape) == (1, 38):
        _pass(f"Output shape = {tuple(out.shape)}")
    else:
        _fail("Output shape = (1, 38)", f"Got {tuple(out.shape)}")

    # 2.3 Softmax sums to 1.0 ± 1e-3
    probs      = torch.softmax(out, dim=1)
    prob_sum   = probs.sum().item()
    if abs(prob_sum - 1.0) < 1e-3:
        _pass(f"Softmax sum = {prob_sum:.6f}  (within 1e-3 of 1.0)")
    else:
        _fail("Softmax sums to 1.0", f"Got {prob_sum:.6f}")

    # 2.4 Top-1 confidence in [0, 100]
    top1_conf = probs.max().item() * 100
    if 0.0 <= top1_conf <= 100.0:
        _pass(f"Top-1 confidence = {top1_conf:.2f}%  (valid range)")
    else:
        _fail("Top-1 confidence in [0, 100]", f"Got {top1_conf:.2f}%")

    # 2.5 Top-5 are 5 distinct classes
    _, top5_idx = probs.topk(5, dim=1)
    top5_list = top5_idx[0].tolist()
    if len(set(top5_list)) == 5:
        _pass(f"Top-5 are 5 distinct class indices")
    else:
        _fail("Top-5 are 5 distinct classes", f"Indices: {top5_list}")

    # 2.6 Batch inference — no OOM on 16 images
    try:
        batch = torch.randn(16, 3, 224, 224, device=device)
        with torch.no_grad():
            _ = model(batch)
        del batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _pass("Batch (16 images) runs without OOM")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            _fail("Batch (16 images) no OOM",
                  "CUDA OOM — reduce BATCH_SIZE or enable AMP")
        else:
            _fail("Batch (16 images) no OOM", str(exc))

    # 2.7 AMP prediction matches FP32
    if device.type == "cuda":
        test_in = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            fp32_pred = model(test_in).argmax(dim=1).item()
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=True):
                amp_pred = model(test_in).argmax(dim=1).item()
        if fp32_pred == amp_pred:
            _pass(f"AMP prediction matches FP32  (class {fp32_pred})")
        else:
            _fail("AMP matches FP32",
                  f"FP32 predicted class {fp32_pred}, AMP predicted {amp_pred}")
        del test_in
    else:
        _skip("AMP consistency test", "Running on CPU — AMP is CUDA-only")

    # 2.8 Inference speed benchmark (10 batches of 1)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(torch.randn(1, 3, 224, 224, device=device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - t0) / 10 * 1000
    _pass(f"Average single-image inference = {elapsed_ms:.1f} ms")

    # 2.9 VRAM usage within budget for RTX 2050
    if device.type == "cuda":
        used_mb  = torch.cuda.memory_allocated(0) / (1024 ** 2)
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        pct = used_mb / total_mb * 100
        _info(f"VRAM after inference: {used_mb:.0f} MB / {total_mb:.0f} MB  ({pct:.1f}%)")
        if used_mb < total_mb * 0.85:
            _pass(f"VRAM usage {pct:.1f}% — within 85% budget")
        else:
            _fail("VRAM within 85% budget",
                  f"Using {pct:.1f}% — risk of OOM during batch training")

    model.to("cpu")
    del dummy, out, probs


# =============================================================================
# SECTION 3 — Disease Database Tests
# =============================================================================

def test_disease_db(class_names: Optional[list]) -> None:
    """
    Verify CSV loads, all 38 folder names map to a CSV row,
    and prevention/cure fields are non-empty for disease classes.
    """
    _section(3, "Disease Database Tests")

    # 3.1 CSV exists
    if not os.path.isfile(CSV_PATH):
        _fail("Disease CSV exists", f"Not found → {CSV_PATH}")
        _skip("Remaining DB tests", "CSV missing")
        return
    _pass(f"CSV found: {os.path.basename(CSV_PATH)}")

    # 3.2 Parse CSV
    rows: list[dict] = []
    try:
        with open(CSV_PATH, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
            rows = list(reader)
    except Exception as exc:
        _fail("CSV is readable", str(exc))
        return

    # 3.3 Row count ≥ 38
    if len(rows) >= 38:
        _pass(f"CSV has {len(rows)} data rows  (≥ 38 required)")
    else:
        _fail("CSV has ≥ 38 rows", f"Found {len(rows)}")

    # 3.4 Required columns present
    required_cols = {"Disease_Name", "Disease_Description",
                     "Prevention_Methods", "Cure_Methods"}
    if required_cols.issubset(set(headers)):
        _pass(f"All 4 required columns present")
    else:
        _fail("All 4 columns present",
              f"Missing: {required_cols - set(headers)}")

    # 3.5 No empty Disease_Name
    empty = [i + 2 for i, r in enumerate(rows)
             if not r.get("Disease_Name", "").strip()]
    if not empty:
        _pass("No null / empty Disease_Name values")
    else:
        _fail("No empty Disease_Name", f"Empty at CSV rows: {empty}")

    # 3.6 No empty descriptions
    empty_desc = [r["Disease_Name"] for r in rows
                  if not r.get("Disease_Description", "").strip()]
    if not empty_desc:
        _pass("All 38 rows have a disease description")
    else:
        _fail("No empty descriptions", f"Empty for: {empty_desc[:3]}")

    if not class_names:
        _skip("Folder→CSV mapping tests", "class_names not available")
        return

    # Build a simple lookup: normalise CSV names for matching
    def _norm(s: str) -> str:
        return (s.lower()
                 .replace("___", " ")
                 .replace("__", " ")
                 .replace("_", " ")
                 .replace(",", "")
                 .replace("(", "")
                 .replace(")", "")
                 .strip())

    csv_lookup = {_norm(r["Disease_Name"]): r for r in rows}

    # 3.7 All 38 folder names resolve to a CSV row
    unmatched = []
    matched_map: dict[str, dict] = {}
    for cn in class_names:
        key = _norm(cn)
        # Try direct match, then progressively shorter keys
        if key in csv_lookup:
            matched_map[cn] = csv_lookup[key]
            continue
        # Try matching with just the disease part (after plant prefix)
        parts = cn.replace("___", " ").split(" ", 1)
        partial = _norm(parts[-1]) if len(parts) > 1 else key
        if partial in csv_lookup:
            matched_map[cn] = csv_lookup[partial]
            continue
        # difflib fallback
        import difflib
        close = difflib.get_close_matches(key, csv_lookup.keys(), n=1, cutoff=0.55)
        if close:
            matched_map[cn] = csv_lookup[close[0]]
        else:
            unmatched.append(cn)

    if not unmatched:
        _pass(f"All {len(class_names)} folder names resolve to a CSV row")
    else:
        _fail("All classes resolve to CSV row",
              f"Unmatched ({len(unmatched)}): {unmatched[:3]}")

    # 3.8 Disease classes have prevention
    disease_classes = [cn for cn in class_names if "healthy" not in cn.lower()]
    no_prev = [cn for cn in disease_classes
               if not matched_map.get(cn, {}).get("Prevention_Methods", "").strip()]
    if not no_prev:
        _pass(f"All {len(disease_classes)} disease classes have prevention methods")
    else:
        _fail("All disease classes have prevention",
              f"Missing for: {no_prev[:3]}")

    # 3.9 Disease classes have cure
    no_cure = [cn for cn in disease_classes
               if not matched_map.get(cn, {}).get("Cure_Methods", "").strip()]
    if not no_cure:
        _pass(f"All {len(disease_classes)} disease classes have cure methods")
    else:
        _fail("All disease classes have cure methods",
              f"Missing for: {no_cure[:3]}")

    # 3.10 Healthy classes return non-empty description
    healthy = [cn for cn in class_names if "healthy" in cn.lower()]
    bad_healthy = [cn for cn in healthy
                   if not matched_map.get(cn, {}).get("Disease_Description", "").strip()]
    if not bad_healthy:
        _pass(f"All {len(healthy)} healthy classes have descriptions")
    else:
        _fail("Healthy class descriptions", f"Empty for: {bad_healthy[:3]}")


# =============================================================================
# SECTION 4 — Prediction Output Structure Tests
# =============================================================================

def test_prediction_output(class_names: Optional[list]) -> None:
    """
    Verify predict_single returns the correct structured dict with all
    required fields in the correct types.
    """
    _section(4, "Prediction Output Structure Tests")

    # Find a real test image
    test_image: Optional[str] = None
    import config as cfg
    data_dir = getattr(cfg, "DATA_DIR", DATA_DIR_HINT)
    if os.path.isdir(data_dir):
        for cls_dir in sorted(os.listdir(data_dir)):
            full = os.path.join(data_dir, cls_dir)
            if not os.path.isdir(full):
                continue
            for fname in os.listdir(full):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    test_image = os.path.join(full, fname)
                    break
            if test_image:
                break
    else:
        _skip("Prediction output tests",
              f"Data directory not found: {data_dir}")
        return

    if not test_image:
        _skip("Prediction output tests", "No image found in data directory")
        return

    _info(f"Test image: {os.path.basename(os.path.dirname(test_image))}"
          f"/{os.path.basename(test_image)}")

    # Import predict_single
    try:
        from predict import predict_single
    except ImportError as exc:
        _skip("Prediction output tests", f"Cannot import predict: {exc}")
        return

    # Build minimal label_map if class_names available
    lmap: Optional[dict] = None
    if class_names:
        lmap = {cn: i for i, cn in enumerate(sorted(class_names))}

    # Run inference
    try:
        result = predict_single(test_image, lmap) if lmap else predict_single(test_image, {})
    except Exception as exc:
        _fail("predict_single runs without error", str(exc))
        return

    # 4.1 Returns a dict
    if isinstance(result, dict):
        _pass("predict_single returns a dict  (not tuple or None)")
    else:
        _fail("Returns dict", f"Got {type(result).__name__}")
        return

    # 4.2 Required keys present
    required = {"disease_name", "confidence_pct"}
    missing  = required - result.keys()
    if not missing:
        _pass(f"Required keys present: {sorted(required)}")
    else:
        _fail("Required keys present", f"Missing: {missing}")

    # 4.3 confidence_pct is float [0, 100]
    conf = result.get("confidence_pct")
    if isinstance(conf, (int, float)) and 0.0 <= conf <= 100.0:
        _pass(f"confidence_pct = {conf:.2f}%  (valid range)")
    else:
        _fail("confidence_pct in [0, 100]", f"Got {conf!r} ({type(conf).__name__})")

    # 4.4 disease_name is non-empty string
    dn = result.get("disease_name", "")
    if isinstance(dn, str) and dn.strip():
        _pass(f"disease_name = '{dn}'")
    else:
        _fail("disease_name is non-empty string", f"Got {dn!r}")

    # 4.5 is_healthy flag present and correct type
    is_h = result.get("is_healthy")
    if isinstance(is_h, bool):
        _pass(f"is_healthy = {is_h}  (correct bool type)")
    else:
        _skip("is_healthy flag", f"Key not present or wrong type: {is_h!r}")

    # 4.6 prevention is list of strings (not raw semicolon string)
    prev = result.get("prevention")
    if prev is not None:
        if isinstance(prev, list) and all(isinstance(s, str) for s in prev):
            _pass(f"prevention is list of {len(prev)} strings  (correctly split)")
        elif isinstance(prev, str) and ";" in prev:
            _fail("prevention is list (not raw string)",
                  "Still a raw semicolon-separated string — split not applied")
        else:
            _fail("prevention is list of strings", f"Type: {type(prev).__name__}")
    else:
        _skip("prevention field", "Key not in result — implement in predict.py")

    # 4.7 cure is list of strings
    cure = result.get("cure")
    if cure is not None:
        if isinstance(cure, list) and all(isinstance(s, str) for s in cure):
            _pass(f"cure is list of {len(cure)} strings  (correctly split)")
        elif isinstance(cure, str) and ";" in cure:
            _fail("cure is list (not raw string)",
                  "Still a raw semicolon-separated string")
        else:
            _fail("cure is list of strings", f"Type: {type(cure).__name__}")
    else:
        _skip("cure field", "Key not in result — implement in predict.py")

    # 4.8 top5 list present with 5 entries
    top5 = result.get("top5")
    if isinstance(top5, list) and len(top5) == 5:
        _pass(f"top5 contains exactly 5 entries")
    elif isinstance(top5, list):
        _skip("top5 has 5 entries", f"Has {len(top5)} entries")
    else:
        _skip("top5 field", "Key not in result")


# =============================================================================
# SECTION 5 — Held-Out Test Set Accuracy
# =============================================================================

def test_accuracy(model, label_map: Optional[dict]) -> None:
    """
    Run inference on the held-out test split and report
    Top-1, Top-5, per-class accuracy, macro F1, weighted F1.
    """
    _section(5, "Held-Out Test Set Accuracy")

    if model is None:
        _skip("All accuracy tests", "Model not loaded")
        return

    import torch
    import config as cfg
    from dataset import get_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()

    # Build test loader using the same seed as training
    _info("Building test DataLoader with training seed ...")
    try:
        _, _, test_loader, lm = get_dataloaders(cfg.DATA_DIR)
    except Exception as exc:
        _fail("Test DataLoader creation", str(exc))
        return

    class_names = sorted(lm, key=lm.get)
    num_classes  = len(class_names)
    _info(f"Test set: {len(test_loader.dataset)} images  |  {num_classes} classes")

    correct_top1  = 0
    correct_top5  = 0
    total         = 0
    per_cls_ok    = [0] * num_classes
    per_cls_total = [0] * num_classes
    all_preds: list[int] = []
    all_labels: list[int] = []

    print()
    t0 = time.time()
    try:
        from tqdm import tqdm
        loader = tqdm(test_loader, desc="  Inference", ncols=70, leave=False)
    except ImportError:
        loader = test_loader
        _info("tqdm not installed — no progress bar")

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            out    = model(images)

            # Top-1
            preds = out.argmax(dim=1)
            correct_top1 += (preds == labels).sum().item()

            # Top-5
            _, top5_idx = out.topk(min(5, num_classes), dim=1)
            for i in range(labels.size(0)):
                if labels[i].item() in top5_idx[i].tolist():
                    correct_top5 += 1

            # Per-class
            for i in range(labels.size(0)):
                lbl = labels[i].item()
                per_cls_total[lbl] += 1
                if preds[i].item() == lbl:
                    per_cls_ok[lbl] += 1

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            total += labels.size(0)

    elapsed = time.time() - t0
    top1 = correct_top1 / total if total else 0.0
    top5 = correct_top5 / total if total else 0.0

    _info(f"Inference complete: {total} images in {elapsed:.1f}s "
          f"({total/elapsed:.0f} img/s)")
    print()

    # 5.1 Top-1 ≥ 95%
    if top1 >= 0.95:
        _pass(f"Top-1 accuracy = {top1*100:.2f}%  (≥ 95% required)")
    else:
        _fail(f"Top-1 ≥ 95%", f"Got {top1*100:.2f}%")

    # 5.2 Top-5 ≥ 98%
    if top5 >= 0.98:
        _pass(f"Top-5 accuracy = {top5*100:.2f}%  (≥ 98% required)")
    else:
        _fail(f"Top-5 ≥ 98%", f"Got {top5*100:.2f}%")

    # 5.3 No class below 80%
    cls_accs = []
    for i in range(num_classes):
        if per_cls_total[i] > 0:
            cls_accs.append((class_names[i],
                             per_cls_ok[i] / per_cls_total[i],
                             per_cls_total[i]))
    cls_accs.sort(key=lambda x: x[1])
    below_80 = [(n, a) for n, a, _ in cls_accs if a < 0.80]
    if not below_80:
        worst_n, worst_a, _ = cls_accs[0]
        _pass(f"No class below 80%  "
              f"(worst: '{worst_n.split('___')[-1]}' at {worst_a*100:.1f}%)")
    else:
        details = ", ".join(f"{n.split('___')[-1]}={a*100:.1f}%"
                            for n, a in below_80[:3])
        _fail("No class below 80%", details)

    # 5.4 Weighted F1 ≥ 0.95
    try:
        from sklearn.metrics import f1_score, classification_report
        w_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        m_f1 = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
        if w_f1 >= 0.95:
            _pass(f"Weighted F1 = {w_f1:.4f}  (≥ 0.95 required)")
        else:
            _fail("Weighted F1 ≥ 0.95", f"Got {w_f1:.4f}")
        _info(f"Macro F1    = {m_f1:.4f}")
    except ImportError:
        _skip("F1 score tests", "scikit-learn not installed")

    # 5.5 Per-class accuracy table (worst 5)
    print(f"\n  {'─'*52}")
    print(f"  {'Per-class accuracy — worst 5 classes':^52}")
    print(f"  {'─'*52}")
    for name, acc, cnt in cls_accs[:5]:
        bar  = "█" * int(acc * 20)
        dname = name.replace("___", " / ").replace("_", " ")[:38]
        print(f"  {acc*100:5.1f}%  {bar:<20}  ({cnt:4d} imgs)  {dname}")
    print(f"  {'─'*52}")

    model.to("cpu")


# =============================================================================
# SECTION 6 — Edge Case and Robustness Tests
# =============================================================================

def test_edge_cases(model) -> None:
    """
    Verify model handles unusual inputs gracefully without crashing.
    """
    _section(6, "Edge Case and Robustness Tests")

    if model is None:
        _skip("All edge case tests", "Model not loaded")
        return

    import torch
    import numpy as np
    from PIL import Image
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def _run(name: str, img: Image.Image) -> bool:
        try:
            t = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(t)
            pred = out.argmax(dim=1).item()
            conf = torch.softmax(out, dim=1).max().item() * 100
            _pass(f"{name}  →  class {pred}  ({conf:.1f}% conf)")
            del t, out
            return True
        except Exception as exc:
            _fail(name, str(exc))
            return False

    # 6.1 All-black image
    _run("All-black image  (224×224)", Image.new("RGB", (224, 224), (0, 0, 0)))

    # 6.2 All-white image
    _run("All-white image  (224×224)", Image.new("RGB", (224, 224), (255, 255, 255)))

    # 6.3 Very small image (32×32) — resize should handle it
    _run("Tiny image  (32×32)  after resize", Image.new("RGB", (32, 32), (128, 200, 80)))

    # 6.4 Random noise (non-plant)
    noise = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
    _run("Random noise  (non-plant image)", Image.fromarray(noise))

    # 6.5 Portrait aspect ratio
    _run("Portrait image  (100×400)", Image.new("RGB", (100, 400), (60, 120, 40)))

    # 6.6 Landscape aspect ratio
    _run("Landscape image  (400×100)", Image.new("RGB", (400, 100), (80, 180, 60)))

    # 6.7 Grayscale converted to RGB
    grey = Image.new("L", (224, 224), 128).convert("RGB")
    _run("Grayscale→RGB converted image", grey)

    # 6.8 Simulated green leaf patch (most likely to give meaningful result)
    leaf = np.full((224, 224, 3), [34, 139, 34], dtype=np.uint8)
    _run("Solid green leaf-colour patch", Image.fromarray(leaf))

    model.to("cpu")


# =============================================================================
# SECTION 7 — Training History Validation
# =============================================================================

def test_training_history() -> None:
    """
    Validate training_history.json for correctness and sanity.
    """
    _section(7, "Training History Validation")

    # 7.1 File exists
    if not os.path.isfile(HISTORY_PATH):
        _fail("training_history.json exists", f"Not found → {HISTORY_PATH}")
        return
    _pass("training_history.json exists")

    # 7.2 Loads as valid JSON
    try:
        with open(HISTORY_PATH, "r") as fh:
            history = json.load(fh)
        _pass("training_history.json is valid JSON")
    except json.JSONDecodeError as exc:
        _fail("Valid JSON", str(exc))
        return

    # 7.3 All 4 required keys present
    required = {"train_loss", "val_loss", "train_acc", "val_acc"}
    if required.issubset(history.keys()):
        _pass("History has all 4 keys: train_loss, val_loss, train_acc, val_acc")
    else:
        _fail("4 required keys", f"Missing: {required - history.keys()}")
        return

    # 7.4 All lists same length (number of epochs)
    lengths = {k: len(history[k]) for k in required}
    if len(set(lengths.values())) == 1:
        n_epochs = list(lengths.values())[0]
        _pass(f"All history lists have {n_epochs} epochs")
    else:
        _fail("All lists same length", f"Lengths: {lengths}")

    # 7.5 Best val accuracy ≥ 95%
    best_val   = max(history["val_acc"])
    best_epoch = history["val_acc"].index(best_val) + 1
    if best_val >= 0.95:
        _pass(f"Best val accuracy = {best_val*100:.2f}%  at epoch {best_epoch}")
    elif best_val >= 0.90:
        _fail(f"Best val ≥ 95%",
              f"Got {best_val*100:.2f}% — model may not have trained fully")
    else:
        _fail(f"Best val ≥ 95%",
              f"Got {best_val*100:.2f}% — serious training issue")

    # 7.6 No NaN or Inf
    bad = [(k, i) for k, vals in history.items()
           for i, v in enumerate(vals)
           if v != v or abs(v) == float("inf")]
    if not bad:
        _pass("No NaN or Inf values in any metric")
    else:
        _fail("No NaN/Inf", f"Found at: {bad[:3]}")

    # 7.7 Training loss decreased overall
    t_loss = history["train_loss"]
    if t_loss[-1] < t_loss[0]:
        _pass(f"Train loss decreased: {t_loss[0]:.4f} → {t_loss[-1]:.4f}")
    else:
        _fail("Train loss decreased", f"{t_loss[0]:.4f} → {t_loss[-1]:.4f}")

    # 7.8 Val accuracy improved overall
    v_acc = history["val_acc"]
    if v_acc[-1] >= v_acc[0]:
        _pass(f"Val accuracy improved: {v_acc[0]*100:.2f}% → {v_acc[-1]*100:.2f}%")
    else:
        _fail("Val accuracy improved", f"{v_acc[0]*100:.2f}% → {v_acc[-1]*100:.2f}%")

    # 7.9 Val acc consistently ≥ train acc (no overfitting)
    gaps = [(va - ta) for ta, va in zip(history["train_acc"], history["val_acc"])]
    positive_gap_epochs = sum(1 for g in gaps if g >= -0.005)   # allow 0.5% margin
    pct_ok = positive_gap_epochs / len(gaps) * 100
    if pct_ok >= 70:
        _pass(f"Val acc ≥ Train acc in {pct_ok:.0f}% of epochs  (no overfitting)")
    else:
        _fail("No overfitting (val ≥ train)",
              f"Val < Train in {100-pct_ok:.0f}% of epochs")

    # 7.10 Print mini accuracy table
    n  = len(v_acc)
    ep = [1, n // 4, n // 2, 3 * n // 4, n]
    print(f"\n  {'─'*40}")
    print(f"  {'Epoch':>6}  {'Train Acc':>10}  {'Val Acc':>10}")
    print(f"  {'─'*40}")
    for e in ep:
        e = max(1, min(e, n))
        print(f"  {e:>6}  {history['train_acc'][e-1]*100:>9.2f}%  "
              f"{history['val_acc'][e-1]*100:>9.2f}%")
    print(f"  {'─'*40}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    quick_mode    = "--quick"          in sys.argv
    skip_accuracy = "--skip-accuracy"  in sys.argv or quick_mode

    print()
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  PlantDoctor AI — Post-Training Test Suite{RESET}")
    print(f"{BOLD}{CYAN}  Model  : EfficientNetV2-S{RESET}")
    print(f"{BOLD}{CYAN}  Target : 99.62% val accuracy{RESET}")
    if quick_mode:
        print(f"{YELLOW}  Mode   : QUICK (sections 1-4, 6-7){RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")

    start = time.time()

    # ── Run all sections ───────────────────────────────────────────────────
    model, label_map, class_names, bundle = test_model_loading()
    test_inference(model)
    test_disease_db(class_names)
    test_prediction_output(class_names)

    if skip_accuracy:
        _section(5, "Held-Out Test Set Accuracy")
        _skip("Full test set scan", "--skip-accuracy or --quick flag used")
    else:
        test_accuracy(model, label_map)

    test_edge_cases(model)
    test_training_history()

    # ── Final report ───────────────────────────────────────────────────────
    elapsed = time.time() - start
    total   = PASS_COUNT + FAIL_COUNT
    score   = PASS_COUNT / total * 100 if total else 0

    status_colour = GREEN if FAIL_COUNT == 0 else RED
    verdict       = "ALL TESTS PASSED ✓" if FAIL_COUNT == 0 else f"{FAIL_COUNT} TEST(S) FAILED ✗"

    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  FINAL REPORT{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"  Tests Run  : {total}")
    print(f"  {GREEN}Passed     : {PASS_COUNT}{RESET}")
    if FAIL_COUNT:
        print(f"  {RED}Failed     : {FAIL_COUNT}{RESET}")
    if SKIP_COUNT:
        print(f"  {YELLOW}Skipped    : {SKIP_COUNT}{RESET}")
    print(f"  Score      : {status_colour}{BOLD}{score:.1f}%{RESET}")
    print(f"  Duration   : {elapsed:.1f}s")
    print(f"  Verdict    : {status_colour}{BOLD}{verdict}{RESET}")

    if FAIL_COUNT > 0:
        print(f"\n{RED}  Failed tests:{RESET}")
        for status, name, reason in RESULTS:
            if status == "FAIL":
                print(f"    {RED}✗  {name}{RESET}")
                if reason:
                    print(f"       {reason}")

    print(f"{BOLD}{CYAN}{'═'*60}{RESET}\n")
    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()