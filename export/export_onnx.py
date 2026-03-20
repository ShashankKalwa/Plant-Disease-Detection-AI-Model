# =============================================================================
# export/export_onnx.py — Export trained model to ONNX format
# =============================================================================

import os
import sys
from typing import Optional

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from model import build_model
from utils import load_model_bundle


def export_onnx(model_path: Optional[str] = None, output_path: Optional[str] = None) -> str:
    """
    Export the trained PlantDoctor model to ONNX format.

    Args:
        model_path: Path to model bundle .pth file.
        output_path: Output path for .onnx file.

    Returns:
        Path to the exported ONNX file.
    """
    if model_path is None:
        model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pth")
    if output_path is None:
        output_path = config.ONNX_MODEL_PATH

    device = "cpu"  # Export on CPU for compatibility

    # Load model bundle
    print(f"[ONNX] Loading model from {model_path}")
    state_dict, label_map, class_names, metadata = load_model_bundle(
        model_path, device)

    model_name = metadata.get("model_name", config.MODEL_NAME)
    image_size = metadata.get("image_size", config.IMAGE_SIZE)
    num_classes = len(class_names)

    model = build_model(model_name, num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # Export
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[ONNX] Exporting to {output_path}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["leaf_image"],
        output_names=["class_logits"],
        dynamic_axes={
            "leaf_image": {0: "batch_size"},
            "class_logits": {0: "batch_size"},
        },
        do_constant_folding=True,
    )

    # Validate
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[ONNX] Model validation passed ✅")
    except ImportError:
        print("[ONNX] onnx package not available for validation")

    # Compare PyTorch vs ONNX outputs
    try:
        import onnxruntime as ort

        ort_session = ort.InferenceSession(output_path)
        torch_output = model(dummy_input).detach().numpy()
        ort_output = ort_session.run(
            None, {"leaf_image": dummy_input.numpy()})[0]

        max_diff = np.max(np.abs(torch_output - ort_output))
        print(f"[ONNX] PyTorch vs ONNX max diff: {max_diff:.6e}")

        if max_diff < 1e-4:
            print("[ONNX] Outputs match ✅")
        else:
            print(f"[WARNING] Output difference ({max_diff:.6e}) exceeds 1e-4")
    except ImportError:
        print("[ONNX] onnxruntime not available for output comparison")

    # Print file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[ONNX] Model size: {size_mb:.1f} MB")
    print(f"[ONNX] Export complete: {output_path}")

    return output_path


if __name__ == "__main__":
    export_onnx()
