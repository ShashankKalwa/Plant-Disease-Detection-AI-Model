# =============================================================================
# scripts/download_dataset.py — Download PlantVillage dataset from Kaggle
# =============================================================================

import os
import sys
import shutil
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


from typing import Optional


def download_dataset(target_dir: Optional[str] = None) -> None:
    """
    Download and extract the PlantVillage dataset from Kaggle.

    Requires the Kaggle CLI to be installed and configured:
        pip install kaggle
        export KAGGLE_USERNAME=your_username
        export KAGGLE_KEY=your_api_key

    Falls back to manual download instructions if Kaggle CLI is unavailable.
    """
    if target_dir is None:
        target_dir = os.path.join(config.BASE_DIR, "data", "plantvillage")

    os.makedirs(target_dir, exist_ok=True)

    # Check if dataset already exists
    color_dir = os.path.join(target_dir, "color")
    if os.path.isdir(color_dir) and len(os.listdir(color_dir)) >= 30:
        print(f"[Dataset] Dataset already exists at {color_dir}")
        print(f"[Dataset] Found {len(os.listdir(color_dir))} class folders")
        return

    # Try Kaggle CLI
    try:
        import kaggle
        print("[Dataset] Downloading PlantVillage dataset from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "abdallahalidev/plantvillage-dataset",
            path=target_dir,
            unzip=True,
        )
        print("[Dataset] Download complete!")
        _reorganise(target_dir)
        return
    except ImportError:
        print("[Dataset] Kaggle CLI not installed.")
    except Exception as e:
        print(f"[Dataset] Kaggle download failed: {e}")

    # Manual instructions
    print("\n" + "=" * 65)
    print("  MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 65)
    print()
    print("  1. Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    print("  2. Click 'Download' (requires Kaggle account)")
    print(f"  3. Extract the ZIP file to: {target_dir}")
    print("  4. Ensure the structure is:")
    print(f"     {target_dir}/")
    print("       color/")
    print("         Apple___Apple_scab/")
    print("         Apple___Black_rot/")
    print("         ... (38 folders)")
    print("       grayscale/  (optional)")
    print("       segmented/  (optional)")
    print()
    print("  OR install Kaggle CLI:")
    print("     pip install kaggle")
    print("     Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
    print("     Then re-run this script.")
    print("=" * 65)


def _reorganise(target_dir: str):
    """
    After Kaggle download, the dataset may be nested inside
    extra directories. This function finds and moves the color/
    grayscale/segmented folders to the correct location.
    """
    expected_subsets = ["color", "grayscale", "segmented"]

    for subset in expected_subsets:
        subset_path = os.path.join(target_dir, subset)
        if os.path.isdir(subset_path):
            continue

        # Search for the subset in nested directories
        for root, dirs, _ in os.walk(target_dir):
            if subset in dirs:
                source = os.path.join(root, subset)
                if source != subset_path:
                    print(f"[Dataset] Moving {source} → {subset_path}")
                    shutil.move(source, subset_path)
                break

    # Validate
    color_dir = os.path.join(target_dir, "color")
    if os.path.isdir(color_dir):
        n_classes = len([d for d in os.listdir(color_dir)
                         if os.path.isdir(os.path.join(color_dir, d))])
        print(f"[Dataset] Validated: {n_classes} class folders in color/")
    else:
        print("[WARNING] color/ directory not found after reorganisation")


if __name__ == "__main__":
    download_dataset()
