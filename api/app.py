# =============================================================================
# api/app.py — FastAPI REST application for PlantDoctor AI
# =============================================================================

import io
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from model import build_model
from predict import predict_single, _get_inference_transform
from utils import load_model_bundle
from api.schemas import (
    PredictionResponse, BatchPredictionResponse,
    DiseaseRecord, HealthResponse, Top5Item,
)
from api.disease_db import DiseaseDatabase


# ── Global state ──────────────────────────────────────────────────────────────
_state = {
    "model": None,
    "label_map": None,
    "class_names": None,
    "disease_db": None,
    "device": None,
    "transform": None,
    "start_time": None,
}


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and disease DB on startup, clean up on shutdown."""
    print("[API] Starting PlantDoctor AI API...")
    _state["start_time"] = time.time()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _state["device"] = device
    print(f"[API] Device: {device}")

    # Load model bundle
    model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pth")
    if os.path.isfile(model_path):
        state_dict, label_map, class_names, metadata = load_model_bundle(
            model_path, device)
        model_name = metadata.get("model_name", config.MODEL_NAME)
        num_classes = len(class_names)

        model = build_model(model_name, num_classes)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Warm up with a dummy tensor
        dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
        with torch.no_grad():
            model(dummy)

        _state["model"] = model
        _state["label_map"] = label_map
        _state["class_names"] = class_names
        print(f"[API] Model loaded: {model_name} ({num_classes} classes)")
    else:
        print(f"[WARNING] No model found at {model_path}")
        print("[WARNING] API will start but predictions will fail.")
        print("[WARNING] Train a model first: python main.py --mode train")

    # Load disease database
    db = DiseaseDatabase(config.CSV_PATH, config.DISEASE_DB_JSON)
    _state["disease_db"] = db
    print(f"[API] Disease DB loaded: {len(db.all())} entries")

    # Inference transform
    _state["transform"] = _get_inference_transform()

    print("[API] ✅ PlantDoctor AI API ready!")

    yield  # App runs here

    # Cleanup
    _state["model"] = None
    print("[API] Shutdown complete.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PlantDoctor AI",
    description="Plant disease detection API with EfficientNetV2-S",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins (frontend will tighten this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _check_model_loaded():
    """Raise 503 if model isn't loaded."""
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first: "
                   "python main.py --mode train"
        )


async def _read_image(file: UploadFile) -> Image.Image:
    """Read and validate an uploaded image file."""
    # Check content type
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {content_type}. "
                   "Please upload a JPEG or PNG image."
        )

    # Read and check size
    contents = await file.read()
    if len(contents) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {config.MAX_UPLOAD_MB} MB"
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Please upload a valid JPEG or PNG."
        )

    return image


def _predict_from_image(image: Image.Image) -> dict:
    """Run prediction on a PIL Image and return structured result."""
    import torch.nn.functional as F

    model = _state["model"]
    label_map = _state["label_map"]
    class_names = _state["class_names"]
    disease_db = _state["disease_db"]
    device = _state["device"]
    transform = _state["transform"]

    tensor = transform(image).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    idx_to_class = {v: k for k, v in label_map.items()}

    top5_probs, top5_indices = probs.topk(min(5, len(class_names)))
    top5_results = []
    for prob, idx in zip(top5_probs.cpu(), top5_indices.cpu()):
        cls_name = idx_to_class.get(idx.item(), f"class_{idx.item()}")
        info = disease_db.get(cls_name)
        top5_results.append({
            "class_raw": cls_name,
            "disease_name": info.get("disease_name", cls_name),
            "confidence_pct": round(prob.item() * 100, 2),
        })

    pred_idx = top5_indices[0].item()
    pred_class = idx_to_class.get(pred_idx, f"class_{pred_idx}")
    pred_conf = round(top5_probs[0].item() * 100, 2)

    info = disease_db.get(pred_class)
    parts = pred_class.split("___")
    plant = parts[0].replace("_", " ") if parts else "Unknown"
    is_healthy = "healthy" in pred_class.lower()

    # Warning for low confidence
    warning = None
    if pred_conf < 50:
        warning = ("Low confidence prediction. The image may not be a plant "
                   "leaf or may be of poor quality.")

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
        "warning": warning,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - _state["start_time"] if _state["start_time"] else 0
    return HealthResponse(
        status="ok",
        model=config.MODEL_NAME,
        classes=config.NUM_CLASSES,
        uptime_seconds=round(uptime, 1),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Predict plant disease from a leaf image.

    Accepts JPEG/PNG image file. Returns structured prediction with
    disease name, confidence, description, prevention tips, and cure methods.
    """
    _check_model_loaded()
    image = await _read_image(file)
    result = _predict_from_image(image)
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_endpoint(files: list[UploadFile] = File(...)):
    """
    Predict plant disease for multiple leaf images (up to 10).

    Accepts multiple JPEG/PNG image files.
    """
    _check_model_loaded()

    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images per batch request."
        )

    results = []
    total_start = time.perf_counter()

    for file in files:
        image = await _read_image(file)
        result = _predict_from_image(image)
        results.append(PredictionResponse(**result))

    total_ms = (time.perf_counter() - total_start) * 1000

    return BatchPredictionResponse(
        results=results,
        total_inference_time_ms=round(total_ms, 2),
    )


@app.get("/diseases", response_model=list[DiseaseRecord])
async def list_diseases():
    """Return all diseases in the knowledge base."""
    db = _state["disease_db"]
    if db is None:
        raise HTTPException(status_code=503, detail="Disease database not loaded.")
    entries = db.all()
    return [DiseaseRecord(**e) for e in entries]


@app.get("/diseases/{disease_name}", response_model=DiseaseRecord)
async def get_disease(disease_name: str):
    """Look up a disease by name (supports fuzzy matching)."""
    db = _state["disease_db"]
    if db is None:
        raise HTTPException(status_code=503, detail="Disease database not loaded.")

    results = db.search(disease_name)
    if not results:
        raise HTTPException(status_code=404,
                            detail=f"Disease not found: {disease_name}")
    return DiseaseRecord(**results[0])
