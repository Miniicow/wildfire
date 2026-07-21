"""
diagnose_dfire_labels.py

Quick diagnostic: inspects a sample of images and their label files to
verify why the SAM pipeline is not finding any 'fire' class boxes.
"""

import logging
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _find_matching_images(images_dir: Path, base_name: str) -> List[Path]:
    """
    Finds image files matching a given base name (any extension) inside
    a directory.

    Args:
        images_dir: Directory to search in.
        base_name: File stem to match (without extension).

    Returns:
        List of matching image paths.
    """
    return list(images_dir.glob(f"{base_name}.*"))


def diagnose_labels(dfire_dir: str, n_samples: int = 5) -> None:
    """
    Prints the raw content of the first N label files found in the
    train/labels folder, along with the corresponding image path, to
    manually verify the annotation format and path mapping.

    Args:
        dfire_dir: Root path of the D-Fire dataset (e.g. the kagglehub
            path, containing 'data/train/images' and 'data/train/labels').
        n_samples: Number of label files to inspect.

    Raises:
        FileNotFoundError: If the expected train/labels folder doesn't exist.
    """
    root = Path(dfire_dir)
    train_labels_dir = root / "data" / "train" / "labels"
    train_images_dir = root / "data" / "train" / "images"

    logger.info(
        "Labels folder exists?: %s -> %s", train_labels_dir.is_dir(), train_labels_dir
    )
    logger.info(
        "Images folder exists?: %s -> %s", train_images_dir.is_dir(), train_images_dir
    )

    if not train_labels_dir.is_dir():
        raise FileNotFoundError(f"Labels folder not found: {train_labels_dir}")

    all_label_files = list(train_labels_dir.glob("*.txt"))
    sample_labels = sorted(all_label_files)[:n_samples]

    logger.info("Total .txt files in labels/: %d", len(all_label_files))
    logger.info("Showing the first %d:", len(sample_labels))

    for label_file in sample_labels:
        content = label_file.read_text().strip()
        base_name = label_file.stem
        matching_images = _find_matching_images(train_images_dir, base_name)

        print(f"--- {label_file.name} ---")
        print(f"Raw content:\n{content if content else '(empty file)'}")
        print(f"Matching image found: {matching_images}")
        print()


if __name__ == "__main__":
    diagnose_labels("/kaggle/input/smoke-fire-detection-yolo", n_samples=5)