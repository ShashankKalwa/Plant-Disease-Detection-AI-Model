# =============================================================================
# dataset.py — Data loading, transforms, disease DB, class-balanced sampling
# =============================================================================
#
# Handles everything data-related:
#   • PlantVillageDataset — reads class sub-folders as (image, label) pairs
#   • get_transforms()   — strong augmentation for train, clean for val/test
#   • get_dataloaders()   — stratified 70/15/15 split + WeightedRandomSampler
#   • load_disease_db()   — reads CSV into a normalised lookup dict
#   • get_disease_info()  — fuzzy-matches a folder name to a CSV disease row
#
# DataLoader: pin_memory + persistent_workers + prefetch_factor tuned for
# RTX 2050 (4GB VRAM, Ampere) via config.
# =============================================================================

import csv
import json
import os
import random
import difflib
from collections import Counter, defaultdict
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np

import config
from utils import normalise_class_name, format_prevention_list, format_cure_list


# ── Label Map ─────────────────────────────────────────────────────────────────

def build_label_map(data_dir: str) -> dict:
    """
    Scan the data directory and build a {class_name: index} mapping.
    Each sub-folder is treated as one class, sorted alphabetically
    for deterministic indexing.

    Args:
        data_dir: Path to the directory containing class sub-folders.

    Returns:
        Dict mapping class folder name to integer index.
    """
    classes = sorted(
        entry.name
        for entry in os.scandir(data_dir)
        if entry.is_dir()
    )
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    print(f"[Dataset] Found {len(label_map)} classes in {data_dir}")
    return label_map


# ── Dataset Class ─────────────────────────────────────────────────────────────

class PlantVillageDataset(Dataset):
    """
    Reads PlantVillage-style directory layout where each sub-folder
    is a class containing JPEG/PNG leaf images.

    Layout:
        data_dir/
            Apple___Apple_scab/
            Apple___Black_rot/
            ...
    """

    def __init__(self, data_dir: str, label_map: dict,
                 transform: Optional[transforms.Compose] = None) -> None:
        self.data_dir = data_dir
        self.label_map = label_map
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        for class_name, label_idx in label_map.items():
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(class_dir, fname), label_idx)
                    )

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Load and transform one image, return (tensor, label)."""
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Return a blank image if the file is corrupt
            image = Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE))
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """
    Build the augmentation pipelines.

    Returns:
        (train_transform, val_transform) — strong augmentation for
        training and clean resize/crop for validation and test.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, val_transform


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders(data_dir: str,
                    label_map: Optional[dict] = None
                    ) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create stratified train/val/test DataLoaders with class-balanced sampling.

    Splitting strategy:
        70% train / 15% val / 15% test — stratified by class so every
        class is fairly represented in each split.

    The train loader uses WeightedRandomSampler to counter class imbalance
    (e.g. Tomato healthy ~5000 vs Apple Cedar rust ~275).

    DataLoader tuning (RTX 2050):
        • BATCH_SIZE=16 physically, paired with ACCUMULATION_STEPS=2 in train.py
        • pin_memory=True — faster CPU→GPU transfer
        • persistent_workers=True — avoids respawning workers each epoch
        • prefetch_factor=2 — pre-loads next batch while GPU computes

    Args:
        data_dir: Path to the directory containing class sub-folders.
        label_map: Optional pre-built {class_name: index} map.

    Returns:
        (train_loader, val_loader, test_loader, label_map)
    """
    if label_map is None:
        label_map = build_label_map(data_dir)

    train_tf, val_tf = get_transforms()

    # Build full dataset to discover all samples and their labels
    full_dataset = PlantVillageDataset(data_dir, label_map, transform=None)
    n_total = len(full_dataset)

    if n_total == 0:
        raise RuntimeError(f"No images found in {data_dir}. "
                           "Check that class sub-folders contain .jpg/.png files.")

    # ── Stratified split ──────────────────────────────────────────────────────
    all_labels = [s[1] for s in full_dataset.samples]
    label_to_indices = defaultdict(list)
    for idx, lbl in enumerate(all_labels):
        label_to_indices[lbl].append(idx)

    train_idx = []
    val_idx = []
    test_idx = []
    rng = random.Random(config.RANDOM_SEED)

    for lbl, indices in label_to_indices.items():
        indices_copy = list(indices)
        rng.shuffle(indices_copy)
        n = len(indices_copy)
        n_train = int(n * config.TRAIN_SPLIT)
        n_val = int(n * config.VAL_SPLIT)
        train_idx.extend(indices_copy[:n_train])  # type: ignore[index]
        val_idx.extend(indices_copy[n_train:n_train + n_val])  # type: ignore[index]
        test_idx.extend(indices_copy[n_train + n_val:])  # type: ignore[index]

    # Print per-class sample counts
    class_names = {v: k for k, v in label_map.items()}
    train_label_counts = Counter([all_labels[i] for i in train_idx])  # type: ignore[call-overload]
    print(f"\n[Dataset] Per-class train sample counts:")
    for lbl_idx in sorted(train_label_counts.keys()):
        print(f"  {class_names[lbl_idx]}: {train_label_counts[lbl_idx]}")

    # ── WeightedRandomSampler for class balance ───────────────────────────────
    train_labels = [all_labels[i] for i in train_idx]
    class_counts = Counter(train_labels)  # type: ignore[call-overload]
    total_train = len(train_labels)
    class_weights = {int(c): total_train / count for c, count in class_counts.items()}
    sample_weights = [class_weights[int(lbl)] for lbl in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    train_dataset = PlantVillageDataset(data_dir, label_map, transform=train_tf)
    val_dataset = PlantVillageDataset(data_dir, label_map, transform=val_tf)
    test_dataset = PlantVillageDataset(data_dir, label_map, transform=val_tf)

    use_cuda = torch.cuda.is_available()
    num_workers = config.NUM_WORKERS
    pin_memory = config.PIN_MEMORY and use_cuda

    # Common DataLoader kwargs for optimal GPU utilization
    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        common_kwargs["persistent_workers"] = True
        common_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        **common_kwargs,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        **common_kwargs,
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_idx),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        **common_kwargs,
    )

    print(f"\n[Dataset] Classes  : {len(label_map)}")
    print(f"[Dataset] Total    : {n_total}")
    print(f"[Dataset] Train    : {len(train_idx)}")
    print(f"[Dataset] Val      : {len(val_idx)}")
    print(f"[Dataset] Test     : {len(test_idx)}")
    print(f"[DataLoader] batch_size={config.BATCH_SIZE} | "
          f"workers={num_workers} | pin_memory={pin_memory}")

    return train_loader, val_loader, test_loader, label_map


# ── Disease Database from CSV ─────────────────────────────────────────────────

def load_disease_db(csv_path: Optional[str] = None,
                    cache_path: Optional[str] = None) -> dict:
    """
    Read the disease CSV and build a normalised lookup dictionary.
    Caches the result as JSON for fast reloading.

    The CSV must have columns:
        Disease_Name, Disease_Description, Prevention_Methods, Cure_Methods

    Args:
        csv_path: Path to the CSV file. Defaults to config.CSV_PATH.
        cache_path: Path to the JSON cache. Defaults to config.DISEASE_DB_JSON.

    Returns:
        Dict mapping normalised_key → {
            disease_name, description, prevention: list, cure: list
        }
    """
    if csv_path is None:
        csv_path = config.CSV_PATH
    if cache_path is None:
        cache_path = config.DISEASE_DB_JSON

    # Try loading from cache if newer than CSV
    if os.path.isfile(cache_path):
        csv_mtime = os.path.getmtime(csv_path) if os.path.isfile(csv_path) else 0
        cache_mtime = os.path.getmtime(cache_path)
        if cache_mtime > csv_mtime:
            with open(cache_path, "r", encoding="utf-8") as f:
                db = json.load(f)
            print(f"[DiseaseDB] Loaded {len(db)} entries from cache")
            return db

    if not os.path.isfile(csv_path):
        print(f"[WARNING] Disease CSV not found: {csv_path}")
        return {}

    db: dict[str, dict] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Disease_Name", "").strip()
            if not name:
                continue
            key = normalise_class_name(name)
            db[key] = {
                "disease_name": name,
                "description": row.get("Disease_Description", "").strip(),
                "prevention": format_prevention_list(
                    row.get("Prevention_Methods", "")),
                "cure": format_cure_list(row.get("Cure_Methods", "")),
            }

    # Write cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    print(f"[DiseaseDB] Built {len(db)} entries from CSV, cached to {cache_path}")
    return db


def get_disease_info(class_name: str, disease_db: dict) -> dict:
    """
    Fuzzy-match a folder class name (e.g. 'Tomato___Early_blight') to
    the disease database and return structured disease info.

    Matching strategy (in priority order):
        1. Exact match after normalisation
        2. Substring containment (either direction)
        3. difflib.get_close_matches fallback (cutoff=0.5)
        4. Graceful fallback for healthy / unknown classes

    Args:
        class_name: The folder name from PlantVillage (e.g. 'Tomato___Early_blight').
        disease_db: Dict returned by load_disease_db().

    Returns:
        Dict with keys: disease_name, description, prevention, cure.
    """
    normalised = normalise_class_name(class_name)

    # 1. Exact match
    if normalised in disease_db:
        return disease_db[normalised]

    # 2. Substring containment
    all_keys = list(disease_db.keys())
    for key in all_keys:
        if normalised in key or key in normalised:
            return disease_db[key]

    # 3. difflib fuzzy match
    matches = difflib.get_close_matches(normalised, all_keys, n=1, cutoff=0.5)
    if matches:
        return disease_db[matches[0]]

    # 4. Fallback for healthy classes
    if "healthy" in normalised:
        plant = normalised.split("healthy")[0].strip()
        return {
            "disease_name": f"{plant.title()} Healthy",
            "description": "Plant appears healthy with no disease symptoms.",
            "prevention": ["Maintain good sanitation and balanced nutrition."],
            "cure": ["No treatment needed."],
        }

    # Unknown class
    return {
        "disease_name": class_name.replace("___", " ").replace("_", " "),
        "description": "No information available for this condition.",
        "prevention": [],
        "cure": [],
    }
