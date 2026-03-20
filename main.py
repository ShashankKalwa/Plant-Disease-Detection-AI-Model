# =============================================================================
# main.py — PlantDoctor AI CLI entrypoint
# =============================================================================
#
# GPU enforcement: validate_gpu() is called as the VERY FIRST operation
# before any training code runs. If GPU is missing, it crashes with
# exact pip fix commands — never silently falls back to CPU.
# =============================================================================

import argparse
import os
import sys

import config


def _print_banner(mode: str) -> None:
    """
    Print a startup banner with GPU info, model config, and current mode.

    Called AFTER validate_gpu() succeeds, so GPU info is guaranteed available.

    Args:
        mode: The current operation mode (train, evaluate, predict, etc.).
    """
    import torch

    gpu_name = "CPU"
    vram_str = "N/A"
    cuda_str = "N/A"
    amp_str = "Disabled"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        vram_str = f"{vram_gb:.1f} GB"
        cuda_str = torch.version.cuda or "unknown"
    if config.USE_AMP and torch.cuda.is_available():
        amp_str = "Enabled (FP16 mixed precision)"

    eff_batch = config.BATCH_SIZE * config.ACCUMULATION_STEPS

    print()
    print("=" * 60)
    print("  🌿  PlantDoctor AI  v2.0.0")
    print("=" * 60)
    print(f"  Device     : {gpu_name}")
    print(f"  VRAM       : {vram_str}")
    print(f"  CUDA       : {cuda_str}")
    print(f"  PyTorch    : {torch.__version__}")
    print(f"  Mode       : {mode}")
    print(f"  Model      : {config.MODEL_NAME}")
    print(f"  Classes    : {config.NUM_CLASSES}")
    print(f"  Image size : {config.IMAGE_SIZE} × {config.IMAGE_SIZE}")
    print(f"  Batch size : {config.BATCH_SIZE} (effective {eff_batch} "
          f"via {config.ACCUMULATION_STEPS}× gradient accumulation)")
    print(f"  AMP        : {amp_str}")

    csv_status = "✅ Found" if os.path.isfile(config.CSV_PATH) else "❌ Missing"
    print(f"  CSV        : {csv_status}")
    print(f"  Data dir   : {config.DATA_DIR}")
    print("=" * 60)
    print()


def _ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.PLOT_SAVE_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def _build_disease_db() -> dict:
    """Load or build the disease database from CSV."""
    from dataset import load_disease_db
    return load_disease_db(config.CSV_PATH, config.DISEASE_DB_JSON)


def mode_train() -> None:
    """Train the model (two-phase training)."""
    from dataset import get_dataloaders, build_label_map
    from model import build_model
    from train import train

    _ensure_dirs()
    disease_db_path = config.DISEASE_DB_JSON
    _build_disease_db()

    label_map = build_label_map(config.DATA_DIR)
    train_loader, val_loader, test_loader, label_map = get_dataloaders(
        config.DATA_DIR, label_map)

    model = build_model(config.MODEL_NAME, len(label_map))
    history = train(model, train_loader, val_loader, label_map, disease_db_path)

    print(f"\n[Main] Training complete!")
    print(f"[Main] Best val accuracy: {max(history['val_acc']):.4f}")


def mode_evaluate() -> None:
    """Evaluate the trained model on the test set."""
    import torch
    from dataset import get_dataloaders, build_label_map
    from model import build_model
    from evaluate import evaluate
    from utils import load_model_bundle

    model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pth")
    if not os.path.isfile(model_path):
        print(f"[ERROR] No trained model found at {model_path}")
        print("[ERROR] Run training first: python main.py --mode train")
        sys.exit(1)

    device = config.validate_gpu()
    state_dict, label_map, class_names, metadata = load_model_bundle(
        model_path, str(device))

    model_name = metadata.get("model_name", config.MODEL_NAME)
    model = build_model(model_name, len(class_names))
    model.load_state_dict(state_dict)

    _, _, test_loader, _ = get_dataloaders(config.DATA_DIR, label_map)

    results = evaluate(model, test_loader, label_map, save_plots=True)
    print(f"\n[Main] Top-1 Accuracy: {results['top1_accuracy']:.4f}")
    print(f"[Main] Top-5 Accuracy: {results['top5_accuracy']:.4f}")


def mode_predict(image_path: str) -> None:
    """Predict disease for a single leaf image."""
    from predict import predict_single, print_prediction

    if not os.path.isfile(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    disease_db = _build_disease_db()
    result = predict_single(image_path, disease_db=disease_db)
    print_prediction(result)


def mode_api() -> None:
    """Launch the FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        print("[ERROR] uvicorn not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)

    print(f"[Main] Starting API server on {config.API_HOST}:{config.API_PORT}")
    uvicorn.run(
        "api.app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        log_level="info",
    )


def mode_export() -> None:
    """Export model to ONNX and TFLite formats."""
    model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pth")
    if not os.path.isfile(model_path):
        print(f"[ERROR] No trained model found at {model_path}")
        print("[ERROR] Run training first: python main.py --mode train")
        sys.exit(1)

    # ONNX export
    print("\n[Main] Exporting to ONNX...")
    try:
        from export.export_onnx import export_onnx
        export_onnx(model_path)
    except ImportError as e:
        print(f"[WARNING] ONNX export skipped: {e}")
        print("[WARNING] Install: pip install onnx onnxruntime")

    # TFLite export
    print("\n[Main] Exporting to TFLite...")
    try:
        from export.export_tflite import export_tflite
        export_tflite(model_path)
    except ImportError as e:
        print(f"[WARNING] TFLite export skipped: {e}")
        print("[WARNING] Install required packages for TFLite conversion.")


def main() -> None:
    """Main entry point with CLI argument parsing and GPU enforcement."""
    parser = argparse.ArgumentParser(
        description="PlantDoctor AI — Plant Disease Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  train      Two-phase training (head warm-up + full fine-tune)
  evaluate   Evaluate trained model on test set
  predict    Predict disease for a single leaf image
  api        Start the FastAPI REST server
  export     Export model to ONNX and TFLite formats
  all        Run train → evaluate → export
        """,
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["train", "evaluate", "predict", "api", "export", "all"],
        help="Operation mode",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to leaf image (required for --mode predict)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model architecture (efficientnet_v2_s, mobilenet_v2, custom_cnn)",
    )

    args = parser.parse_args()

    # Override model if specified
    if args.model:
        config.MODEL_NAME = args.model

    # ── GPU ENFORCEMENT — FIRST THING BEFORE ANYTHING ELSE ────────────────
    # For modes that need GPU (train, evaluate, all), validate GPU first.
    # API/predict/export can work on CPU, so they skip enforcement.
    gpu_modes = {"train", "evaluate", "all"}
    if args.mode in gpu_modes:
        device = config.validate_gpu()
        # If we reach here, GPU is confirmed active.

    _print_banner(args.mode)

    if args.mode == "train":
        mode_train()
    elif args.mode == "evaluate":
        mode_evaluate()
    elif args.mode == "predict":
        if not args.image:
            print("[ERROR] --image is required for predict mode.")
            print("Usage: python main.py --mode predict --image path/to/leaf.jpg")
            sys.exit(1)
        mode_predict(args.image)
    elif args.mode == "api":
        mode_api()
    elif args.mode == "export":
        mode_export()
    elif args.mode == "all":
        print("[Main] Running full pipeline: train → evaluate → export\n")
        mode_train()
        mode_evaluate()
        mode_export()
        print("\n[Main] ✅ Full pipeline complete!")


if __name__ == "__main__":
    main()
