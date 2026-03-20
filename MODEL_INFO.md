# PlantDoctor AI -- Model Card

## Project Overview

PlantDoctor AI is a production-grade plant disease detection system that classifies leaf images into 38 disease/healthy categories across 14 crop species. The system uses a fine-tuned EfficientNetV2-S backbone with a custom classifier head, trained in two phases on the PlantVillage dataset.

## Model Architecture

| Component | Detail |
|---|---|
| **Architecture** | EfficientNetV2-S |
| **Pretrained Weights** | ImageNet-1K (torchvision IMAGENET1K_V1) |
| **Backbone Output** | 1280-dim feature vector |
| **Classifier Head** | Flatten -> Dropout(0.3) -> Linear(1280,512) -> GELU -> Dropout(0.15) -> Linear(512,38) |
| **Total Parameters** | ~21.5 million |
| **Input Size** | 224 x 224 x 3 (RGB) |
| **Output** | 38-class softmax probabilities |

### Two-Phase Training Strategy

**Phase 1 -- Head Warm-Up (5 epochs)**
- Backbone layers: FROZEN (ImageNet weights preserved)
- Only classifier head trained (~680K params)
- Optimizer: AdamW, lr=3e-4
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=2)

**Phase 2 -- Full Fine-Tune (29 epochs, early stopping at 7 patience)**
- ALL layers unfrozen (~21.5M params)
- Differential LR: backbone=1e-5, head=3e-4
- Scheduler: CosineAnnealingLR (T_max=30, eta_min=1e-7)
- EarlyStopping: patience=7 on val_accuracy

## Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 224 x 224 |
| Batch Size | 16 (effective 32 via 2x gradient accumulation) |
| Weight Decay | 1e-4 |
| Label Smoothing | 0.1 |
| AMP (Mixed Precision) | Enabled (FP16) |
| cuDNN Benchmark | Enabled |
| Channels-Last | Enabled |
| Random Seed | 42 |
| Data Splits | 70% train / 15% val / 15% test (stratified) |
| Class Balancing | WeightedRandomSampler |

## Dataset

| Stat | Value |
|---|---|
| Dataset | PlantVillage (color subset) |
| Total Images | 54,306 |
| Classes | 38 |
| Crop Species | 14 |
| Train Split | ~38,014 images |
| Val Split | ~8,146 images |
| Test Split | ~8,146 images |
| Format | JPEG/PNG, variable resolution |
| Augmentation (train) | RandomResizedCrop, HFlip, VFlip, Rotation(30), ColorJitter, GaussianBlur, Grayscale |
| Augmentation (val/test) | Resize(256) -> CenterCrop(224) |

## Final Results

| Metric | Value |
|---|---|
| **Best Val Accuracy** | **99.74%** |
| Final Train Accuracy | 99.67% |
| Final Val Accuracy | 99.70% |
| Final Train Loss | 0.7063 (label smoothing inflated -- see note) |
| Final Val Loss | 0.6874 |
| Total Epochs | 34 (5 Phase 1 + 29 Phase 2) |
| Original Target | 92% |
| Target Exceeded By | +7.74 percentage points |

> **Note on Loss Values:** The train loss of 0.71 appears high relative to 99.7% accuracy. This is expected and correct. Label smoothing (0.1) redistributes 10% of the target probability mass across all 38 classes, creating a theoretical minimum loss of ~0.66 even with perfect predictions. The model is performing optimally.

## Training Progression (Key Epochs)

| Epoch | Phase | Train Acc | Val Acc | Train Loss | Val Loss |
|---|---|---|---|---|---|
| 1 | Phase 1 | 66.11% | 80.16% | 1.8517 | 1.5789 |
| 5 | Phase 1 | 79.26% | 85.30% | 1.3632 | 1.2187 |
| 6 | Phase 2 | 88.83% | 95.53% | 1.1070 | 0.9059 |
| 10 | Phase 2 | 97.91% | 99.08% | 0.8098 | 0.7614 |
| 15 | Phase 2 | 99.08% | 99.50% | 0.7472 | 0.7126 |
| 20 | Phase 2 | 99.36% | 99.56% | 0.7250 | 0.6998 |
| 25 | Phase 2 | 99.62% | 99.70% | 0.7116 | 0.6897 |
| 30 | Phase 2 | 99.63% | 99.72% | 0.7082 | 0.6897 |
| 34 | Phase 2 | 99.67% | 99.70% | 0.7063 | 0.6874 |

## Hardware and Training Time

| Component | Detail |
|---|---|
| GPU | NVIDIA GeForce RTX 2050 (4GB VRAM) |
| CPU | Intel Core i5-12450H (12th Gen) |
| OS | Windows 11 |
| CUDA | 12.1 |
| PyTorch | 2.5.1+cu121 |
| Training Duration | ~26 hours |

## Complete List of 38 Classes

| # | Class Name | Type |
|---|---|---|
| 1 | Apple___Apple_scab | Disease |
| 2 | Apple___Black_rot | Disease |
| 3 | Apple___Cedar_apple_rust | Disease |
| 4 | Apple___healthy | Healthy |
| 5 | Blueberry___healthy | Healthy |
| 6 | Cherry_(including_sour)___Powdery_mildew | Disease |
| 7 | Cherry_(including_sour)___healthy | Healthy |
| 8 | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | Disease |
| 9 | Corn_(maize)___Common_rust_ | Disease |
| 10 | Corn_(maize)___Northern_Leaf_Blight | Disease |
| 11 | Corn_(maize)___healthy | Healthy |
| 12 | Grape___Black_rot | Disease |
| 13 | Grape___Esca_(Black_Measles) | Disease |
| 14 | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | Disease |
| 15 | Grape___healthy | Healthy |
| 16 | Orange___Haunglongbing_(Citrus_greening) | Disease |
| 17 | Peach___Bacterial_spot | Disease |
| 18 | Peach___healthy | Healthy |
| 19 | Pepper,_bell___Bacterial_spot | Disease |
| 20 | Pepper,_bell___healthy | Healthy |
| 21 | Potato___Early_blight | Disease |
| 22 | Potato___Late_blight | Disease |
| 23 | Potato___healthy | Healthy |
| 24 | Raspberry___healthy | Healthy |
| 25 | Soybean___healthy | Healthy |
| 26 | Squash___Powdery_mildew | Disease |
| 27 | Strawberry___Leaf_scorch | Disease |
| 28 | Strawberry___healthy | Healthy |
| 29 | Tomato___Bacterial_spot | Disease |
| 30 | Tomato___Early_blight | Disease |
| 31 | Tomato___Late_blight | Disease |
| 32 | Tomato___Leaf_Mold | Disease |
| 33 | Tomato___Septoria_leaf_spot | Disease |
| 34 | Tomato___Spider_mites Two-spotted_spider_mite | Disease |
| 35 | Tomato___Target_Spot | Disease |
| 36 | Tomato___Tomato_Yellow_Leaf_Curl_Virus | Disease |
| 37 | Tomato___Tomato_mosaic_virus | Disease |
| 38 | Tomato___healthy | Healthy |

## How to Use the Model

### Loading the trained model

```python
import torch
from model import build_model
from utils import load_model_bundle

# Load the full bundle
path = "outputs/models/best_model.pth"
state_dict, label_map, class_names, metadata = load_model_bundle(path, "cuda")

# Rebuild the model
model = build_model(metadata["model_name"], len(class_names))
model.load_state_dict(state_dict)
model.eval()
model.to("cuda")
```

### Running predictions

```bash
python main.py --mode predict --image path/to/leaf.jpg
```

### Running the test suite

```bash
python test_model.py                 # Full test (includes accuracy test)
python test_model.py --skip-accuracy # Quick test (skips ~8000 image inference)
```

### Running evaluation

```bash
python main.py --mode evaluate       # Generates confusion matrix and per-class plots
```

## Known Limitations

1. **Domain-specific**: Trained only on PlantVillage data -- may not generalize to field photos with complex backgrounds, lighting, or multiple leaves.
2. **38-class fixed**: Cannot detect diseases outside the 38 trained classes. Out-of-distribution images will receive a confident but incorrect prediction.
3. **Single leaf**: Expects a single leaf image. Multi-leaf or whole-plant images may reduce accuracy.
4. **Label smoothing loss**: Loss values appear higher than expected due to label smoothing=0.1. This is by design and does not indicate poor performance.
5. **Windows NUM_WORKERS=0**: DataLoader runs in main process on Windows to avoid shared memory crashes. This does not affect training accuracy.

## Version History

| Version | Date | Changes |
|---|---|---|
| 2.0.0 | 2026-03-14 | EfficientNetV2-S, two-phase training, AMP, 99.74% val acc |
| 1.0.0 | Previous | MobileNetV2, single-phase, ~70-80% accuracy |
