# =============================================================================
# train.py — Two-phase training loop for PlantDoctor AI
# =============================================================================
#
# GPU-optimised for NVIDIA RTX 2050 (4GB VRAM, Ampere):
#   • Automatic Mixed Precision (AMP)  — 2-3× speed, ~40% less VRAM
#   • Gradient Accumulation            — effective batch=32 from physical batch=16
#   • cuDNN Benchmark                  — 10-20% faster convolutions
#   • Channels-Last Memory             — 5-15% faster on Ampere tensor cores
#   • GPU Warmup                       — dummy forward pass before real training
#   • Per-epoch VRAM monitoring        — live GPU memory usage logging
#   • Inter-phase cache clearing       — prevents memory fragmentation
#
# Phase 1: Head warm-up (backbone frozen)
#   • AdamW, lr=3e-4, ReduceLROnPlateau
#   • 5 epochs — head learns to use frozen ImageNet features
#
# Phase 2: Full fine-tune (all layers unfrozen)
#   • AdamW with differential LR: backbone=1e-5, head=3e-4
#   • CosineAnnealingLR smoothly decays to eta_min=1e-7
#   • EarlyStopping with patience=7
#   • Up to 30 epochs
#
# BestModelCheckpoint tracks val_accuracy (higher = better).
# save_model_bundle embeds label_map + class_names + metadata into .pth.
# =============================================================================

import gc
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import config
from utils import (
    BestModelCheckpoint, EarlyStopping, save_model_bundle,
    save_history, plot_curves,
)

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def _log_gpu_memory(tag: str = "") -> None:
    """
    Print current GPU memory usage stats.

    Args:
        tag: Label for the log line (e.g. 'epoch 5', 'after cache clear').
    """
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  [GPU {tag}] VRAM: {alloc:.2f} GB allocated | "
              f"{reserved:.2f} GB reserved | {total:.1f} GB total")


def _gpu_warmup(model: nn.Module, device: torch.device) -> None:
    """
    Run a dummy forward pass on the GPU to verify it works before real data.

    This catches CUDA errors early (before DataLoader spins up) and
    warms up cuDNN auto-tuner if benchmark mode is enabled.

    Args:
        model: The model (already on device).
        device: The torch device.
    """
    if device.type != "cuda":
        return

    print("[GPU] Running warmup forward pass...")
    dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    if config.USE_CHANNELS_LAST:
        dummy = dummy.to(memory_format=torch.channels_last)

    try:
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=config.USE_AMP):
                _ = model(dummy)
        torch.cuda.synchronize()
        del dummy
        torch.cuda.empty_cache()
        print("[GPU] ✅ Warmup complete — GPU is active and responding")
        _log_gpu_memory("after warmup")
    except RuntimeError as e:
        raise RuntimeError(
            f"[GPU] ❌ GPU warmup FAILED: {e}\n\n"
            "This usually means:\n"
            "  1. GPU driver is outdated — update from nvidia.com/drivers\n"
            "  2. Another process is using all VRAM — close other GPU apps\n"
            "  3. CUDA version mismatch — run: python gpu_check.py\n"
        ) from e


def _run_epoch(model: nn.Module, loader, criterion: nn.Module,
               optimizer: Optional[optim.Optimizer],
               device: torch.device, is_train: bool, desc: str,
               scaler: Optional[GradScaler] = None,
               accumulation_steps: int = 1,
               use_amp: bool = False,
               use_channels_last: bool = False) -> tuple[float, float]:
    """
    Run one epoch of training or validation with AMP and gradient accumulation.

    Args:
        model: The neural network.
        loader: DataLoader for the current split.
        criterion: Loss function.
        optimizer: Optimizer (unused during validation).
        device: Torch device.
        is_train: True for training, False for validation.
        desc: Description for the tqdm progress bar.
        scaler: GradScaler for AMP (pass None to disable).
        accumulation_steps: Number of mini-batches to accumulate before stepping.
        use_amp: Whether to use autocast for mixed precision.
        use_channels_last: Whether to convert inputs to channels-last format.

    Returns:
        (avg_loss, accuracy) for the epoch.
    """
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    batch_idx = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, ncols=110)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Channels-last for Ampere tensor core acceleration
            if use_channels_last:
                images = images.to(memory_format=torch.channels_last)

            # Forward pass with optional AMP autocast
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Scale loss for gradient accumulation
                if is_train and accumulation_steps > 1:
                    loss = loss / accumulation_steps

            if is_train:
                # Backward pass with AMP scaling
                scaler.scale(loss).backward()  # type: ignore[union-attr]

                # Only step optimizer every accumulation_steps mini-batches
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)  # type: ignore[union-attr]
                    scaler.update()  # type: ignore[union-attr]
                    optimizer.zero_grad(set_to_none=True)

            # Track metrics (un-scale the loss for logging)
            batch_loss = loss.item()
            if is_train and accumulation_steps > 1:
                batch_loss *= accumulation_steps
            running_loss += batch_loss * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()  # type: ignore[union-attr]
            total += labels.size(0)

            pbar.set_postfix(loss=f"{batch_loss:.4f}",
                             acc=f"{correct / total:.4f}")

        # Flush any remaining accumulated gradients at end of epoch
        if is_train and (batch_idx + 1) % accumulation_steps != 0:
            scaler.step(optimizer)  # type: ignore[union-attr]
            scaler.update()  # type: ignore[union-attr]
            optimizer.zero_grad(set_to_none=True)

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train(model: nn.Module, train_loader, val_loader,
          label_map: dict,
          disease_db_path: Optional[str] = None) -> dict:
    """
    Two-phase training targeting 92%+ accuracy, optimised for RTX 2050.

    GPU Optimizations applied:
        • GPU warmup — dummy forward pass before real data
        • AMP (autocast + GradScaler) — 2-3× speed, ~40% VRAM saved
        • Gradient accumulation (2×) — effective batch=32 from physical batch=16
        • cuDNN benchmark — 10-20% faster convolutions
        • Channels-last format — 5-15% faster on Ampere
        • Per-epoch VRAM monitoring — live memory usage logging
        • Inter-phase cache clearing — prevents fragmentation

    Phase 1 — Head Warm-Up:
        Backbone frozen → only head params trained → AdamW lr=3e-4
        ReduceLROnPlateau on val_accuracy → 5 epochs

    Phase 2 — Full Fine-Tune:
        All layers unfrozen → differential LR (backbone=1e-5, head=3e-4)
        CosineAnnealingLR (T_max=30, eta_min=1e-7) → up to 30 epochs
        EarlyStopping patience=7

    Args:
        model: nn.Module with phase1_mode() and phase2_mode() methods.
        train_loader: Training DataLoader (with WeightedRandomSampler).
        val_loader: Validation DataLoader.
        label_map: {class_name: index} mapping.
        disease_db_path: Path to disease DB JSON (stored in bundle metadata).

    Returns:
        History dict with train_loss, val_loss, train_acc, val_acc per epoch.
    """
    # ── Resolve device ────────────────────────────────────────────────────────
    device = config.validate_gpu()

    # ── GPU optimizations ─────────────────────────────────────────────────────
    use_amp = config.USE_AMP and device.type == "cuda"
    use_channels_last = config.USE_CHANNELS_LAST and device.type == "cuda"
    accumulation_steps = config.ACCUMULATION_STEPS

    if config.CUDNN_BENCHMARK and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("[Train] cuDNN benchmark mode ENABLED")

    # Move model to device, then convert to channels-last
    model = model.to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("[Train] Channels-last memory format ENABLED")

    # AMP scaler — enabled=False gracefully degrades on CPU
    scaler = GradScaler(device=device.type, enabled=use_amp)
    if use_amp:
        print("[Train] Automatic Mixed Precision (AMP) ENABLED")
    print(f"[Train] Gradient accumulation: {accumulation_steps} steps "
          f"(effective batch={config.BATCH_SIZE * accumulation_steps})")

    # ── GPU Warmup — BEFORE any DataLoader iteration ─────────────────────────
    _gpu_warmup(model, device)

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.PLOT_SAVE_DIR, exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    best_ckpt = BestModelCheckpoint(
        os.path.join(config.MODEL_SAVE_DIR, "best_model_weights.pth"))
    early_stop = EarlyStopping(patience=config.PATIENCE)

    # TensorBoard writer (optional)
    writer = None
    if HAS_TENSORBOARD:
        try:
            writer = SummaryWriter(os.path.join(config.OUTPUT_DIR, "tb_logs"))
        except Exception:
            writer = None

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    class_names = sorted(label_map, key=label_map.get)
    total_start = time.time()
    global_epoch = 0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 1 — Head Only (Warm-up)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 70)
    print("  PHASE 1 — Training classifier head only (backbone frozen)")
    print("=" * 70)
    model.phase1_mode()

    optimizer_p1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR_PHASE1,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler_p1 = ReduceLROnPlateau(optimizer_p1, mode="max",
                                      factor=0.5, patience=2)

    for epoch in range(1, config.PHASE1_EPOCHS + 1):
        global_epoch += 1
        print(f"\n[Phase 1] Epoch [{epoch:02d}/{config.PHASE1_EPOCHS:02d}]")
        _log_gpu_memory(f"epoch {global_epoch} start")

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer_p1, device,
            is_train=True, desc=f"  Train {epoch}",
            scaler=scaler, accumulation_steps=accumulation_steps,
            use_amp=use_amp, use_channels_last=use_channels_last)
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device,
            is_train=False, desc=f"  Val   {epoch}",
            use_amp=use_amp, use_channels_last=use_channels_last)

        scheduler_p1.step(val_acc)
        best_ckpt.step(val_acc, model)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        current_lr = optimizer_p1.param_groups[0]["lr"]
        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")
        print(f"  LR: {current_lr:.2e}")

        if writer:
            writer.add_scalars("Loss", {"train": train_loss, "val": val_loss},
                               global_epoch)
            writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc},
                               global_epoch)

    # ── Clear GPU cache between phases ────────────────────────────────────────
    del optimizer_p1, scheduler_p1
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print("\n[Train] GPU cache cleared between phases")
    _log_gpu_memory("after cache clear")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 2 — Full Fine-tune
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 70)
    print("  PHASE 2 — Full fine-tuning (ALL layers unfrozen)")
    print("=" * 70)
    model.phase2_mode()

    # Differential learning rates: backbone at LR_PHASE2, head at LR_PHASE1
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer_p2 = optim.AdamW([
        {"params": backbone_params, "lr": config.LR_PHASE2},
        {"params": head_params,     "lr": config.LR_PHASE1},
    ], weight_decay=config.WEIGHT_DECAY)

    scheduler_p2 = CosineAnnealingLR(
        optimizer_p2, T_max=config.PHASE2_EPOCHS, eta_min=1e-7)

    # Reset early stopping for phase 2
    early_stop = EarlyStopping(patience=config.PATIENCE)

    for epoch in range(1, config.PHASE2_EPOCHS + 1):
        global_epoch += 1
        print(f"\n[Phase 2] Epoch [{epoch:02d}/{config.PHASE2_EPOCHS:02d}]")
        _log_gpu_memory(f"epoch {global_epoch} start")

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer_p2, device,
            is_train=True, desc=f"  Train {epoch}",
            scaler=scaler, accumulation_steps=accumulation_steps,
            use_amp=use_amp, use_channels_last=use_channels_last)
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device,
            is_train=False, desc=f"  Val   {epoch}",
            use_amp=use_amp, use_channels_last=use_channels_last)

        scheduler_p2.step()
        best_ckpt.step(val_acc, model)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        current_lr = scheduler_p2.get_last_lr()[0]
        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")
        print(f"  LR: {current_lr:.2e}")

        if writer:
            writer.add_scalars("Loss", {"train": train_loss, "val": val_loss},
                               global_epoch)
            writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc},
                               global_epoch)

        if early_stop.step(val_acc):
            print(f"\n[Train] Early stopping triggered at Phase 2 epoch {epoch}.")
            break

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Save & Finalize
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elapsed = time.time() - total_start

    # Restore the best model weights before saving
    if best_ckpt.best_state is not None:
        model.load_state_dict(best_ckpt.best_state)

    # Save best model as a full bundle (state_dict + label_map + metadata)
    bundle_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pth")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    save_model_bundle(model, label_map, class_names, {
        "val_accuracy": best_ckpt.best_accuracy,
        "epoch": global_epoch,
        "model_name": config.MODEL_NAME,
        "image_size": config.IMAGE_SIZE,
        "disease_db_path": disease_db_path or config.DISEASE_DB_JSON,
        "training_time_seconds": elapsed,
        "gpu": gpu_name,
        "amp_enabled": use_amp,
    }, bundle_path)

    # Also save the final epoch model
    final_path = os.path.join(config.MODEL_SAVE_DIR, "final_model.pth")
    save_model_bundle(model, label_map, class_names, {
        "val_accuracy": history["val_acc"][-1],
        "epoch": global_epoch,
        "model_name": config.MODEL_NAME,
        "image_size": config.IMAGE_SIZE,
    }, final_path)

    # Save history JSON
    history_path = os.path.join(config.OUTPUT_DIR, "training_history.json")
    save_history(history, history_path)

    # Plot training curves
    plot_curves(history, config.PLOT_SAVE_DIR)

    if writer:
        writer.close()

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total time     : {elapsed / 60:.1f} minutes")
    print(f"  Best val acc   : {best_ckpt.best_accuracy:.4f} "
          f"({best_ckpt.best_accuracy * 100:.2f}%)")
    print(f"  Best model     : {bundle_path}")
    print(f"  Training curves: {config.PLOT_SAVE_DIR}")
    print(f"  GPU            : {gpu_name}")
    if use_amp:
        print(f"  AMP            : ENABLED (saved ~40% VRAM)")

    # Peak VRAM usage
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  Peak VRAM      : {peak_gb:.2f} GB / {total_gb:.1f} GB")

    print(f"{'=' * 70}\n")

    return history
