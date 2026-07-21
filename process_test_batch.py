"""
process_test_batch.py

Processes a full batch of test images with the trained wildfire severity
model, generating a summary table (FP, level, alert) and saving the
annotated images — ready for presentation/delivery.
"""

import logging
from pathlib import Path
from typing import Dict, List

from inference import FireSeverityEstimator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _list_images(images_dir: Path) -> List[Path]:
    """
    Lists all supported image files inside a directory, sorted by name.

    Args:
        images_dir: Directory to search in.

    Returns:
        Sorted list of image paths.
    """
    return sorted(f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


def process_batch(
    weights_path: str,
    images_dir: str,
    annotated_output_dir: str,
    conf_threshold: float = 0.35,
) -> List[Dict]:
    """
    Runs the full pipeline (segmentation + FP + severity + alert) over
    every image in a folder, saving annotated versions and returning a
    tabular summary.

    Args:
        weights_path: Path to the trained checkpoint (best.pt).
        images_dir: Folder containing the test images.
        annotated_output_dir: Folder where images with the overlaid
            mask and FP/level text will be saved.
        conf_threshold: Confidence threshold for inference.

    Returns:
        List of dictionaries with the results for each image.

    Raises:
        FileNotFoundError: If the images folder doesn't exist or is empty.
    """
    images_path = Path(images_dir)
    if not images_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {images_dir}")

    images = _list_images(images_path)
    if not images:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    Path(annotated_output_dir).mkdir(parents=True, exist_ok=True)

    estimator = FireSeverityEstimator(weights_path=weights_path, conf_threshold=conf_threshold)

    summary = []
    print(f"{'Image':<25}{'FP (%)':<10}{'Level':<12}{'Instances':<12}")
    print("-" * 60)

    for img_path in images:
        result = estimator.estimate(str(img_path))

        annotated_output_path = str(Path(annotated_output_dir) / f"annotated_{img_path.name}")
        estimator.visualize(str(img_path), result, output_path=annotated_output_path)

        print(
            f"{img_path.name:<25}{result.fire_percentage:<10}"
            f"{result.severity_level:<12}{result.num_instances:<12}"
        )

        summary.append({
            "image": img_path.name,
            "fire_percentage": result.fire_percentage,
            "severity": result.severity_level,
            "instances": result.num_instances,
            "mean_confidence": result.mean_confidence,
            "alert_message": estimator.generate_alert_message(result),
        })

    logger.info("Annotated images saved to: %s", annotated_output_dir)
    return summary


if __name__ == "__main__":
    summary = process_batch(
        weights_path="/content/drive/MyDrive/wildfire_runs/yolo26n_seg_v2_combined/weights/best.pt",
        images_dir="/content/test_images",
        annotated_output_dir="/content/test_images_annotated",
    )

    print("\n--- Full alert messages ---")
    for entry in summary:
        print(f"\n{entry['image']}:")
        print(entry["alert_message"])