# 🌿 PlantDoctor AI

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25-success)
![Classes](https://img.shields.io/badge/Classes-38-orange)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ShashankKalwa/PlantDoctor_AI-Disease_Detection_API)

**Production-grade plant disease detection** powered by EfficientNetV2-S with a FastAPI REST API, disease knowledge base from CSV, and mobile model export (ONNX + TFLite).

> 38-class classifier on the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) · Target accuracy: **90–95%** · Inference: **< 300ms CPU**

---

## 🌐 Live Demo

The API is deployed and running on **Hugging Face Spaces**:

🔗 **[PlantDoctor AI — Live API](https://huggingface.co/spaces/ShashankKalwa/PlantDoctor_AI-Disease_Detection_API)**

You can test the API directly without any local setup by clicking the link above and using the interactive demo.

---

## 🏗️ Project Structure

```
plantdoctor_ai/
├── config.py                 # Central configuration
├── dataset.py                # Data loading, transforms, disease DB
├── model.py                  # EfficientNetV2-S / MobileNetV2 / CustomCNN
├── train.py                  # Two-phase training loop
├── evaluate.py               # Test evaluation + metrics
├── predict.py                # Structured prediction with disease info
├── utils.py                  # Checkpointing, plotting, helpers
├── main.py                   # CLI entrypoint
├── api/
│   ├── app.py                # FastAPI application
│   ├── schemas.py            # Pydantic v2 models
│   └── disease_db.py         # Disease database with fuzzy matching
├── export/
│   ├── export_onnx.py        # ONNX export + validation
│   └── export_tflite.py      # TFLite export (FP32 + INT8)
├── scripts/
│   └── download_dataset.py   # Kaggle dataset downloader
├── data/plantvillage/color/   # 38 class sub-folders <actual dataset here>
├── outputs/
│   ├── models/               # Saved model checkpoints
│   └── plots/                # Training curves, confusion matrix
├── plant_disease_training_dataset_optionB.csv
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## 🚀 Quick Start (Local)

### 1. Navigate to project root
```bash
cd path/to/project
```

### 2. Create & activate virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt

# For GPU (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision
```

### 4. Verify dataset
```bash
# Check 38 class folders exist:
# Windows:
dir /b data\plantvillage\color | find /c /v ""
# Linux:
ls data/plantvillage/color | wc -l
```

### 5. Train the model
```bash
python main.py --mode train
```

### 6. Evaluate on test set
```bash
python main.py --mode evaluate
```

### 7. Predict on a single image
```bash
python main.py --mode predict --image path/to/leaf.jpg
```

### 8. Start the API server
```bash
python main.py --mode api
```

### 9. Export to ONNX + TFLite
```bash
python main.py --mode export
```

### 10. Full pipeline (train → evaluate → export)
```bash
python main.py --mode all
```

---

## 🐳 Docker Quick Start

```bash
# Build and run
docker-compose up --build

# API is available at http://localhost:8000
```

---

## 🔌 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model": "efficientnet_v2_s", "classes": 38, "uptime_seconds": 42.5}
```

### Predict Disease
```bash
curl -X POST http://localhost:8000/predict -F "file=@leaf.jpg"
```
```json
{
  "predicted_class_raw": "Tomato___Early_blight",
  "disease_name": "Tomato Early blight",
  "confidence_pct": 94.7,
  "is_healthy": false,
  "plant": "Tomato",
  "description": "Early blight, caused by Alternaria solani...",
  "prevention": [
    "Rotate crops for 2-3 years away from solanaceous hosts...",
    "Use disease-free seed/transplants..."
  ],
  "cure": [
    "Remove lower infected leaves...",
    "Apply labeled protectant fungicides..."
  ],
  "top5": [
    {"class": "Tomato___Early_blight", "disease_name": "Tomato Early blight", "confidence_pct": 94.7},
    {"class": "Tomato___Septoria_leaf_spot", "disease_name": "Tomato Septoria leaf spot", "confidence_pct": 3.1}
  ],
  "inference_time_ms": 45.2,
  "warning": null
}
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "files=@leaf1.jpg" -F "files=@leaf2.jpg"
```

### List All Diseases
```bash
curl http://localhost:8000/diseases
```

### Search Disease
```bash
curl http://localhost:8000/diseases/tomato%20blight
```

---

## 🧠 Model Architecture

**EfficientNetV2-S** with two-phase fine-tuning:

| Phase | Layers Trained | LR | Epochs | Scheduler |
|-------|---------------|-----|--------|-----------|
| 1 (Warm-up) | Head only | 3e-4 | 5 | ReduceLROnPlateau |
| 2 (Fine-tune) | All layers | 1e-5 | 30 | CosineAnnealingLR |

**Custom head**: AdaptiveAvgPool → Dropout(0.3) → Linear(1280,512) → GELU → Dropout(0.15) → Linear(512,38)

---

## 🎯 Accuracy Tuning

| Setting | Effect |
|---------|--------|
| `PHASE2_EPOCHS = 40` | More training time → higher accuracy |
| `LR_PHASE2 = 5e-6` | Slower fine-tuning → more stable |
| `DROPOUT_RATE = 0.2` | Less regularisation → higher capacity |
| `IMAGE_SIZE = 256` | Higher resolution → better features |

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `outputs/models/best_model.pth` | Best weights (highest val accuracy) |
| `outputs/models/final_model.pth` | Weights at last epoch |
| `outputs/training_history.json` | Loss & accuracy per epoch |
| `outputs/plots/loss_curve.png` | Train/val loss graph |
| `outputs/plots/accuracy_curve.png` | Train/val accuracy graph |
| `outputs/plots/confusion_matrix.png` | Per-class confusion matrix |
| `outputs/plots/per_class_accuracy.png` | Per-class accuracy bar chart |
| `outputs/models/best_model.onnx` | ONNX export |
| `outputs/models/model_fp32.tflite` | TFLite FP32 |
| `outputs/models/model_int8.tflite` | TFLite INT8 quantised |

---

## 🔧 Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'config'` | Run from project root (folder containing main.py) |
| `FileNotFoundError` on data directory | Verify `DATA_DIR` in config.py. Check dataset placement. |
| CUDA out of memory | Reduce `BATCH_SIZE` in config.py to 16 or 8 |
| Training accuracy stuck below 50% | Increase epochs or check dataset integrity |
| API returns 503 | Train a model first: `python main.py --mode train` |
| Low confidence predictions | Image may not be a plant leaf or is poor quality |

---

## 📋 Supported Plant Diseases (38 Classes)

Apple (4), Blueberry (1), Cherry (2), Corn (4), Grape (4), Orange (1), Peach (2), Pepper Bell (2), Potato (3), Raspberry (1), Soybean (1), Squash (1), Strawberry (2), Tomato (10)

---

## ⚙️ Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.1
- ~54,000 images in PlantVillage color subset
- ~4 GB RAM (CPU) or 4 GB VRAM (GPU) for training

```