# -*- coding: utf-8 -*-
# =============================================================================
# gpu_check.py -- Standalone GPU Diagnostic for PlantDoctor AI
# =============================================================================
#
# Run this BEFORE training to confirm your GPU setup is correct.
# This file has ZERO project imports -- it is fully self-contained.
#
# Usage:  python gpu_check.py
# =============================================================================

import sys


def _pass(msg):
    """Print a pass check with message."""
    print(f"  [PASS] {msg}")


def _fail(msg, fix):
    """Print a fail check with message and exact fix, then exit."""
    print(f"  [FAIL] {msg}")
    print()
    print("  -- HOW TO FIX ------------------------------------------------")
    for line in fix.strip().split("\n"):
        print(f"  |  {line.strip()}")
    print("  ---------------------------------------------------------------")
    print()
    sys.exit(1)


def check_python_version():
    """Check 1: Python version >= 3.10."""
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major >= 3 and v.minor >= 10:
        _pass(f"Python {ver_str}")
    else:
        _fail(
            f"Python {ver_str} -- requires >= 3.10",
            "Download Python 3.10+ from https://www.python.org/downloads/\n"
            "Re-create your virtual environment with the new Python."
        )


def check_pytorch_installed():
    """Check 2: PyTorch is installed and version printed."""
    try:
        import torch
        _pass(f"PyTorch {torch.__version__} installed")
    except ImportError:
        _fail(
            "PyTorch is NOT installed",
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        )


def check_pytorch_cuda_build():
    """Check 3: PyTorch build contains +cu (CUDA build, not CPU-only)."""
    import torch
    version = torch.__version__
    if "+cu" in version:
        _pass(f"PyTorch is CUDA build ({version})")
    else:
        _fail(
            f"PyTorch is CPU-only build ({version}) -- no CUDA support",
            "pip uninstall torch torchvision torchaudio -y\n"
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            "\n"
            "If CUDA 12.1 fails, try CUDA 11.8:\n"
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        )


def check_cuda_available():
    """Check 4: torch.cuda.is_available() returns True."""
    import torch
    if torch.cuda.is_available():
        cuda_ver = torch.version.cuda or "unknown"
        _pass(f"CUDA available (CUDA {cuda_ver})")
    else:
        _fail(
            "torch.cuda.is_available() returned False",
            "1. Run: nvidia-smi\n"
            "   -> If 'command not found': install NVIDIA GPU driver from\n"
            "     https://www.nvidia.com/drivers\n"
            "   -> Note the CUDA Version in top-right corner\n"
            "\n"
            "2. Re-install PyTorch with matching CUDA:\n"
            "   pip uninstall torch torchvision torchaudio -y\n"
            "   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            "\n"
            "3. Restart your terminal/IDE after installing drivers."
        )


def check_gpu_name():
    """Check 5: GPU name is printed (should show RTX 2050 or similar)."""
    import torch
    name = torch.cuda.get_device_name(0)
    _pass(f"GPU found: {name}")


def check_vram():
    """Check 6: VRAM is printed (should show ~4.0 GB for RTX 2050)."""
    import torch
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    _pass(f"VRAM: {vram_gb:.2f} GB ({props.name})")


def check_gpu_tensor_ops():
    """Check 7: Create tensor on CUDA, do math, sync -- proves GPU is active."""
    import torch
    try:
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        del a, b, c
        torch.cuda.empty_cache()
        _pass("GPU tensor operations work (1024x1024 matmul)")
    except RuntimeError as e:
        _fail(
            f"GPU tensor operation failed: {e}",
            "Your GPU driver may be incompatible with this CUDA version.\n"
            "Run: nvidia-smi\n"
            "Ensure CUDA Version >= 12.1 (or install cu118 PyTorch build).\n"
            "Update GPU drivers from https://www.nvidia.com/drivers"
        )


def check_amp_autocast():
    """Check 8: AMP autocast works -- confirms FP16 operations function."""
    import torch
    from torch.amp import autocast
    try:
        with autocast(device_type="cuda", enabled=True):
            x = torch.randn(32, 3, 224, 224, device="cuda")
            w = torch.randn(16, 3, 3, 3, device="cuda")
            y = torch.nn.functional.conv2d(x, w, padding=1)
            assert y.dtype == torch.float16, f"Expected fp16, got {y.dtype}"
        torch.cuda.synchronize()
        del x, w, y
        torch.cuda.empty_cache()
        _pass("AMP autocast works (FP16 operations confirmed)")
    except Exception as e:
        _fail(
            f"AMP autocast failed: {e}",
            "Your GPU may not support FP16. Check that you have an\n"
            "Ampere, Turing, or newer GPU architecture.\n"
            "If this fails, set USE_AMP = False in config.py"
        )


def check_torchvision_efficientnet():
    """Check 9: torchvision installed and EfficientNetV2-S weights available."""
    try:
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        _pass("torchvision OK -- EfficientNet_V2_S_Weights available")
    except ImportError as e:
        _fail(
            f"torchvision or EfficientNetV2-S weights not found: {e}",
            "pip install torchvision>=0.16 --index-url https://download.pytorch.org/whl/cu121"
        )


def main():
    """Run all 9 GPU diagnostic checks in order."""
    print()
    print("=" * 60)
    print("  PlantDoctor AI -- GPU Diagnostic Report")
    print("=" * 60)
    print()

    check_python_version()
    check_pytorch_installed()
    check_pytorch_cuda_build()
    check_cuda_available()
    check_gpu_name()
    check_vram()
    check_gpu_tensor_ops()
    check_amp_autocast()
    check_torchvision_efficientnet()

    print()
    print("=" * 60)
    print("  ALL 9 CHECKS PASSED -- Ready to train on GPU!")
    print("=" * 60)
    print()
    print("  Next step:")
    print("    python main.py --mode train")
    print()
    sys.exit(0)


if __name__ == "__main__":
    main()
