# =============================================================================
# export/export_tflite.py — Export trained model to TFLite format
# =============================================================================

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from model import build_model
from utils import load_model_bundle


def export_tflite(model_path: Optional[str] = None,
                  fp32_path: Optional[str] = None,
                  int8_path: Optional[str] = None) -> tuple:
    """
    Export the trained PlantDoctor model to TFLite format.

    Attempts multiple conversion paths in order:
        1. ai_edge_torch (Google's native PyTorch→TFLite converter)
        2. onnx → tensorflow → tflite (via onnx-tf)

    Produces two variants:
        - FP32 (full precision)
        - INT8 (quantised for mobile)

    Args:
        model_path: Path to model bundle .pth file.
        fp32_path: Output path for FP32 TFLite model.
        int8_path: Output path for INT8 TFLite model.

    Returns:
        Tuple of (fp32_path, int8_path) or (None, None) on failure.
    """
    if model_path is None:
        model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pth")
    if fp32_path is None:
        fp32_path = config.TFLITE_FP32_PATH
    if int8_path is None:
        int8_path = config.TFLITE_INT8_PATH

    os.makedirs(os.path.dirname(fp32_path), exist_ok=True)

    # Load model
    import torch
    device = "cpu"
    state_dict, label_map, class_names, metadata = load_model_bundle(
        model_path, device)

    model_name = metadata.get("model_name", config.MODEL_NAME)
    image_size = metadata.get("image_size", config.IMAGE_SIZE)
    num_classes = len(class_names)

    model = build_model(model_name, num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    # Try ai_edge_torch first
    try:
        return _export_via_ai_edge_torch(model, image_size, fp32_path, int8_path)
    except ImportError:
        print("[TFLite] ai_edge_torch not available, trying onnx-tf path...")
    except Exception as e:
        print(f"[TFLite] ai_edge_torch failed: {e}")

    # Fallback: onnx → tensorflow → tflite
    try:
        return _export_via_onnx_tf(model, image_size, fp32_path, int8_path)
    except ImportError:
        print("[TFLite] onnx-tf not available.")
    except Exception as e:
        print(f"[TFLite] onnx-tf conversion failed: {e}")

    print("\n[TFLite] ❌ TFLite export failed. Install one of:")
    print("  pip install ai-edge-torch")
    print("  pip install onnx-tf tensorflow")
    return None, None


def _export_via_ai_edge_torch(model, image_size, fp32_path, int8_path):
    """Export using Google's ai_edge_torch (native PyTorch→TFLite)."""
    import torch
    import ai_edge_torch

    print("[TFLite] Using ai_edge_torch for conversion...")

    dummy_input = (torch.randn(1, 3, image_size, image_size),)

    # FP32 export
    edge_model = ai_edge_torch.convert(model, dummy_input)
    edge_model.export(fp32_path)
    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    print(f"[TFLite] FP32 model: {fp32_path} ({fp32_size:.1f} MB)")

    # INT8 quantisation
    try:
        from ai_edge_torch.quantize import quant_recipes
        quant_config = quant_recipes.full_int8_dynamic_recipe()
        edge_model_q = ai_edge_torch.convert(
            model, dummy_input, quant_config=quant_config)
        edge_model_q.export(int8_path)
        int8_size = os.path.getsize(int8_path) / (1024 * 1024)
        print(f"[TFLite] INT8 model: {int8_path} ({int8_size:.1f} MB)")
    except Exception as e:
        print(f"[TFLite] INT8 quantisation failed: {e}")
        print("[TFLite] FP32 model exported successfully.")

    return fp32_path, int8_path


def _export_via_onnx_tf(model, image_size, fp32_path, int8_path):
    """Fallback: Export via ONNX → TensorFlow → TFLite."""
    import torch
    import numpy as np
    import tempfile

    onnx_path = config.ONNX_MODEL_PATH
    if not os.path.isfile(onnx_path):
        print("[TFLite] ONNX model not found, exporting first...")
        from export.export_onnx import export_onnx
        export_onnx()

    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    print("[TFLite] Converting ONNX → TensorFlow → TFLite...")

    # ONNX → TF SavedModel
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_dir = os.path.join(tmpdir, "saved_model")
        tf_rep.export_graph(saved_model_dir)

        # TF → TFLite FP32
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()

        with open(fp32_path, "wb") as f:
            f.write(tflite_model)
        fp32_size = len(tflite_model) / (1024 * 1024)
        print(f"[TFLite] FP32 model: {fp32_path} ({fp32_size:.1f} MB)")

        # TF → TFLite INT8
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset for calibration
        def representative_dataset():
            for _ in range(100):
                data = np.random.randn(1, image_size, image_size, 3).astype(
                    np.float32)
                yield [data]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        try:
            tflite_model_q = converter.convert()
            with open(int8_path, "wb") as f:
                f.write(tflite_model_q)
            int8_size = len(tflite_model_q) / (1024 * 1024)
            print(f"[TFLite] INT8 model: {int8_path} ({int8_size:.1f} MB)")
        except Exception as e:
            print(f"[TFLite] INT8 quantisation failed: {e}")

    return fp32_path, int8_path


if __name__ == "__main__":
    export_tflite()
