"""
verify_sam_masks.py

Generates a comparison grid (original image + SAM mask overlay) for a
set of samples, allowing visual verification of pseudo-mask quality
before using them to train the segmentation model.
"""

import logging
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OVERLAY_COLOR_RGB = (255, 0, 0)  # red overlay for the fire mask
OVERLAY_ALPHA = 0.4


def _parse_polygon(line: str, width: int, height: int) -> np.ndarray:
    """
    Parses a single YOLO-segmentation label line into pixel-coordinate
    polygon points.

    Args:
        line: Raw label line ("class_id x1 y1 x2 y2 ... xn yn",
            normalized coordinates).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Array of shape (N, 2) with polygon points in pixel coordinates.
    """
    coords = list(map(float, line.strip().split()[1:]))
    points = [
        [int(coords[i] * width), int(coords[i + 1] * height)]
        for i in range(0, len(coords), 2)
    ]
    return np.array(points, dtype=np.int32)


def _load_sample_with_overlay(image_path: Path, labels_dir: Path) -> np.ndarray:
    """
    Loads an image and blends it with a semi-transparent overlay of its
    corresponding segmentation mask, if a label file exists.

    Args:
        image_path: Path to the image file.
        labels_dir: Directory containing the corresponding label files.

    Returns:
        RGB image blended with the mask overlay.
    """
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    label_path = labels_dir / f"{image_path.stem}.txt"
    overlay = image.copy()

    if label_path.is_file():
        for line in label_path.read_text().strip().splitlines():
            polygon = _parse_polygon(line, width, height)
            cv2.fillPoly(overlay, [polygon], color=OVERLAY_COLOR_RGB)

    return cv2.addWeighted(image, 1 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0)


def visualize_samples(
    dataset_dir: str,
    n_samples: int = 9,
    columns: int = 3,
) -> None:
    """
    Displays a grid of images from the generated dataset with their
    segmentation mask overlaid in semi-transparent red, for quick visual
    inspection of SAM pseudo-mask quality.

    Args:
        dataset_dir: Root path of the generated YOLO-seg dataset (must
            contain 'images' and 'labels' subfolders).
        n_samples: Number of images to display.
        columns: Number of columns in the display grid.

    Raises:
        FileNotFoundError: If no images are found in the dataset.
    """
    images_dir = Path(dataset_dir) / "images"
    labels_dir = Path(dataset_dir) / "labels"

    image_files = sorted(images_dir.glob("*.jpg"))[:n_samples]
    if not image_files:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    n_rows = (len(image_files) + columns - 1) // columns
    _, axes = plt.subplots(n_rows, columns, figsize=(columns * 5, n_rows * 5))
    axes = np.array(axes).reshape(-1)

    for idx, image_path in enumerate(image_files):
        blended = _load_sample_with_overlay(image_path, labels_dir)
        axes[idx].imshow(blended)
        axes[idx].set_title(image_path.stem, fontsize=9)
        axes[idx].axis("off")

    for j in range(len(image_files), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_samples("/content/dfire_yolo_seg", n_samples=9)