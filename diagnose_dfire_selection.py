"""
diagnose_dfire_selection.py

Replicates the exact file selection logic used in `process_dfire_dataset`
to identify why the first N images were not generating any valid fire
bounding boxes.
"""

import logging
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DFIRE_FIRE_CLASS_ID = 1
IMAGE_EXTENSIONS = ("*.jpg", "*.png")


def _resolve_label_path(image_path: Path) -> Path:
    """
    Determines the label (.txt) path corresponding to an image by
    replacing the 'images' segment of its path with 'labels', following
    the standard YOLO folder convention.

    Args:
        image_path: Path to the image.

    Returns:
        Candidate path to the .txt label file (may not exist; the
        caller must check with `.is_file()`).
    """
    label_parts = ["labels" if part.lower() == "images" else part for part in image_path.parts]
    return Path(*label_parts).with_suffix(".txt")


def _count_fire_boxes(label_path: Path) -> int:
    """
    Counts how many bounding boxes of the fire class appear in a YOLO
    label file.

    Args:
        label_path: Path to the YOLO .txt annotation file.

    Returns:
        Number of lines whose class ID matches the fire class.
    """
    content = label_path.read_text().strip().splitlines()
    return sum(1 for line in content if line.strip().startswith(f"{DFIRE_FIRE_CLASS_ID} "))


def _list_all_images(root: Path) -> List[Path]:
    """
    Lists all images in the dataset matching the supported extensions.

    Args:
        root: Root directory to search recursively.

    Returns:
        Sorted list of image paths.
    """
    return sorted(p for pattern in IMAGE_EXTENSIONS for p in root.rglob(pattern))


def diagnose_selection(dfire_dir: str, n_samples: int = 20) -> None:
    """
    Replicates the image listing logic from `process_dfire_dataset`
    (same order, same limit) and prints, for each image, whether its
    label was found and how many fire class boxes (class_id == 1) it
    contains.

    Args:
        dfire_dir: Root path of the D-Fire dataset.
        n_samples: Number of images to inspect (same value used in
            `process_dfire_dataset`).

    Raises:
        FileNotFoundError: If the dataset directory doesn't exist.
    """
    root = Path(dfire_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dfire_dir}")

    all_images = _list_all_images(root)
    sample_images = all_images[:n_samples]

    logger.info("Total images found in the whole dataset: %d", len(all_images))
    logger.info("Showing the first %d (same order as the pipeline):", len(sample_images))

    for img_path in sample_images:
        label_path = _resolve_label_path(img_path)
        label_exists = label_path.is_file()
        fire_box_count = _count_fire_boxes(label_path) if label_exists else 0

        print(
            f"{img_path.relative_to(root)} | label exists: {label_exists} | "
            f"fire boxes: {fire_box_count}"
        )


if __name__ == "__main__":
    diagnose_selection("/kaggle/input/smoke-fire-detection-yolo", n_samples=20)