# -*- coding: utf-8 -*-
# =============================================================================
# config.py -- Central configuration for PlantDoctor AI
# =============================================================================
#
# Every training, evaluation, API, and export parameter is defined here.
# No other file should hard-code paths, hyperparameters, or model choices.
#
# GPU Profile: NVIDIA RTX 2050 | 4GB VRAM | Ampere (GA107)
# =============================================================================

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# -- Paths ------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data", "plantvillage", "color")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
MODEL_SAVE_DIR  = os.path.join(OUTPUT_DIR, "models")
PLOT_SAVE_DIR   = os.path.join(OUTPUT_DIR, "plots")

# -- Disease CSV & Cache ----------------------------------------------------
CSV_PATH        = os.path.join(BASE_DIR, "plant_disease_training_dataset_optionB.csv")
DISEASE_DB_JSON = os.path.join(OUTPUT_DIR, "disease_db.json")

# -- Export Paths ------------------------------------------------------------
ONNX_MODEL_PATH   = os.path.join(MODEL_SAVE_DIR, "best_model.onnx")
TFLITE_FP32_PATH  = os.path.join(MODEL_SAVE_DIR, "model_fp32.tflite")
TFLITE_INT8_PATH  = os.path.join(MODEL_SAVE_DIR, "model_int8.tflite")

# -- Dataset -----------------------------------------------------------------
IMAGE_SIZE      = 224          # EfficientNetV2-S native input -- do NOT reduce
NUM_CLASSES     = 38           # PlantVillage has 38 disease/species classes
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15
RANDOM_SEED     = 42

# -- Training -- Phase 1 (head only, backbone frozen) -----------------------
PHASE1_EPOCHS   = 5
LR_PHASE1       = 3e-4

# -- Training -- Phase 2 (full fine-tune, all layers) -----------------------
PHASE2_EPOCHS   = 30
LR_PHASE2       = 1e-5

# -- Training -- Common -----------------------------------------------------
BATCH_SIZE          = 16       # Physical batch -- fits in 4GB VRAM with AMP
ACCUMULATION_STEPS  = 2        # Effective batch = 16 x 2 = 32
WEIGHT_DECAY        = 1e-4
LABEL_SMOOTHING     = 0.1
PATIENCE            = 7        # early-stopping patience (epochs)

# -- Model -------------------------------------------------------------------
# Options: "efficientnet_v2_s" | "mobilenet_v2" | "custom_cnn"
MODEL_NAME      = "efficientnet_v2_s"
DROPOUT_RATE    = 0.3

# -- GPU Enforcement ---------------------------------------------------------
DEVICE          = "cuda"       # Target device -- enforced by validate_gpu()
FORCE_GPU       = True         # If True, crash with clear error when GPU missing

# -- GPU Optimizations -- RTX 2050 (4GB VRAM, Ampere) -----------------------
USE_AMP             = True     # Automatic Mixed Precision -- 2-3x speed, ~40% VRAM saved
CUDNN_BENCHMARK     = True     # cuDNN auto-tuning -- 10-20% faster convolutions
USE_CHANNELS_LAST   = True     # NHWC memory layout -- 5-15% faster on Ampere

# -- DataLoader --------------------------------------------------------------
NUM_WORKERS     = 0            # 0 for Windows (avoids shared memory crash)
PIN_MEMORY      = True         # Faster CPU->GPU tensor transfer

# -- API ---------------------------------------------------------------------
API_HOST        = os.getenv("API_HOST", "0.0.0.0")
API_PORT        = int(os.getenv("API_PORT", "8000"))
MAX_UPLOAD_MB   = 10


# -- GPU Validation ----------------------------------------------------------

def validate_gpu():
    """
    Validate that a CUDA-capable GPU is available for training.

    When FORCE_GPU is True (default), this function will raise a
    RuntimeError with exact pip fix commands if no GPU is found.
    When FORCE_GPU is False, it falls back to CPU with a loud warning.

    Returns:
        torch.device -- either 'cuda' or 'cpu' (only if FORCE_GPU=False).

    Raises:
        RuntimeError: If GPU is not found and FORCE_GPU is True.
    """
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cuda_ver = torch.version.cuda or "unknown"
        print(f"[GPU] OK - GPU detected: {gpu_name}")
        print(f"[GPU]    VRAM: {vram_gb:.1f} GB | CUDA: {cuda_ver}")
        print(f"[GPU]    PyTorch: {torch.__version__}")
        return torch.device("cuda")

    # GPU NOT found
    if FORCE_GPU:
        error_msg = (
            "\n"
            "=" * 66 + "\n"
            "  GPU NOT FOUND -- Training cannot start\n"
            "=" * 66 + "\n"
            "\n"
            "  torch.cuda.is_available() returned False.\n"
            "  PlantDoctor AI requires a CUDA GPU for training.\n"
            "\n"
            "  FIX -- run these commands in order:\n"
            "\n"
            "  Step 1: Uninstall CPU-only PyTorch\n"
            "    pip uninstall torch torchvision torchaudio -y\n"
            "\n"
            "  Step 2: Install CUDA 12.1 build\n"
            "    pip install torch torchvision\n"
            "      --index-url https://download.pytorch.org/whl/cu121\n"
            "\n"
            "  Step 3: Verify\n"
            '    python -c "import torch; print(torch.cuda.is_available())"\n'
            "    Expected output: True\n"
            "\n"
            "  If CUDA 12.1 does not work, try CUDA 11.8:\n"
            "    pip install torch torchvision\n"
            "      --index-url https://download.pytorch.org/whl/cu118\n"
            "\n"
            "  If torch.cuda.is_available() is still False:\n"
            "    - Run: nvidia-smi\n"
            "    - If not found -- install GPU driver from nvidia.com/drivers\n"
            "    - Restart terminal/IDE after installing drivers\n"
            "\n"
            "  Run: python gpu_check.py    for full diagnostics\n"
            "\n"
            "=" * 66 + "\n"
        )
        raise RuntimeError(error_msg)

    # FORCE_GPU is False -- reluctantly fall back to CPU
    print("\n" + "!" * 65)
    print("  WARNING: GPU not found -- falling back to CPU")
    print("  WARNING: Training will be 7-10x slower!")
    print("  WARNING: Set FORCE_GPU = True in config.py to enforce GPU")
    print("!" * 65 + "\n")
    return torch.device("cpu")
