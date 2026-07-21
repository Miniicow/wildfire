"""
download_dfire.py

Downloads the D-Fire dataset (images + YOLO-format detection
annotations) from its Kaggle mirror, for later conversion to
segmentation via SAM (Segment Anything Model).

NOTE: the official GitHub repository (gaia-solutions-on-demand/DFireDataset)
only contains code and documentation; the actual images are hosted on
OneDrive or on this Kaggle mirror, which is much simpler to download
from Colab.

D-Fire class convention (confirmed by the dataset documentation):
    class 0 = smoke
    class 1 = fire
"""

import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DFIRE_FIRE_CLASS_ID = 1
KAGGLE_DATASET_SLUG = "sayedgamal99/smoke-fire-detection-yolo"


def download_dfire(destination: Optional[str] = None) -> str:
    """
    Downloads the D-Fire mirror hosted on Kaggle
    ('sayedgamal99/smoke-fire-detection-yolo'), ready to use without
    needing OneDrive.

    Args:
        destination: Ignored when using kagglehub (kagglehub manages its
            own local cache). Kept for signature compatibility.

    Returns:
        Local path where kagglehub stored the downloaded dataset.

    Raises:
        RuntimeError: If the download fails (no connection, dataset
            moved, or missing Kaggle credentials).
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Run: !pip install -q kagglehub"
        ) from exc

    try:
        dataset_path = kagglehub.dataset_download(KAGGLE_DATASET_SLUG)
    except Exception as exc:
        raise RuntimeError(f"Failed to download D-Fire from Kaggle: {exc}") from exc

    logger.info("D-Fire (Kaggle mirror) downloaded to: %s", dataset_path)
    return dataset_path


def count_fire_images(dataset_dir: str) -> int:
    """
    Counts how many images have at least one 'fire' class annotation
    (class_id == 1), which are the only ones relevant for our flame
    segmentation pipeline.

    Args:
        dataset_dir: Root path of the downloaded D-Fire dataset.

    Returns:
        Number of images with at least one annotated fire instance.
    """
    root = Path(dataset_dir)
    label_files = list(root.rglob("*.txt"))

    count = 0
    for label_file in label_files:
        try:
            content = label_file.read_text().strip().splitlines()
        except OSError:
            continue
        if any(line.strip().startswith(f"{DFIRE_FIRE_CLASS_ID} ") for line in content):
            count += 1

    logger.info("Images with at least one 'fire' instance: %d", count)
    return count


if __name__ == "__main__":
    dataset_path = download_dfire()
    count_fire_images(dataset_path)