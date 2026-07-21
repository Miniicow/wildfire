"""
combine_datasets.py

Combines the original Roboflow dataset (201 images, manually annotated
masks) with the D-Fire dataset processed via SAM (5,822 images,
pseudo-masks), generating a final train/valid/test structure ready
for training YOLOv26n-seg.

Partition strategy:
    - The original Roboflow partition is kept as-is (70/20/10), since
      its masks are of higher confidence (manual annotation).
    - D-Fire is partitioned with the same proportions (70/20/10)
      randomly, since its pseudo-masks are of medium confidence (SAM).
"""

import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SPLIT_RATIOS: Tuple[float, float, float] = (0.7, 0.2, 0.1)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _list_yolo_seg_pairs(directory: str) -> List[Tuple[Path, Path]]:
    """
    Lists all valid (image, label) pairs inside a folder containing
    'images' and 'labels' subfolders.

    Args:
        directory: Root path containing 'images/' and 'labels/'.

    Returns:
        List of (image_path, label_path) tuples for pairs where both
        files exist.

    Raises:
        FileNotFoundError: If the expected subfolders do not exist.
    """
    images_dir = Path(directory) / "images"
    labels_dir = Path(directory) / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected 'images/' and 'labels/' folders inside: {directory}"
        )

    pairs = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.is_file():
            pairs.append((img_path, label_path))

    return pairs


def _copy_pairs(
    pairs: List[Tuple[Path, Path]],
    dest_images: Path,
    dest_labels: Path,
    prefix: str = "",
) -> None:
    """
    Copies a list of (image, label) pairs to the destination folders,
    optionally prepending a prefix to the filename to avoid collisions
    between different sources (e.g. 'dfire_' vs 'roboflow_').

    Args:
        pairs: List of (image_path, label_path) tuples.
        dest_images: Destination folder for images.
        dest_labels: Destination folder for labels.
        prefix: Prefix to prepend to the output filename.

    Raises:
        OSError: If a file cannot be copied (e.g. disk full, permissions).
    """
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in pairs:
        new_name = f"{prefix}{img_path.stem}"
        try:
            shutil.copy(img_path, dest_images / f"{new_name}{img_path.suffix}")
            shutil.copy(label_path, dest_labels / f"{new_name}.txt")
        except OSError as exc:
            raise OSError(f"Failed to copy pair '{img_path.name}': {exc}") from exc


def _resolve_roboflow_split_names(roboflow_root: Path) -> Dict[str, str]:
    """
    Determines the actual folder names used by the Roboflow export for
    each split, accounting for the common 'val' vs 'valid' naming
    inconsistency.

    Args:
        roboflow_root: Root path of the Roboflow dataset.

    Returns:
        Mapping from canonical split name ('train', 'valid', 'test') to
        the actual folder name found on disk.
    """
    split_names = {"train": "train", "valid": "valid", "test": "test"}
    if (roboflow_root / "val").is_dir() and not (roboflow_root / "valid").is_dir():
        split_names["valid"] = "val"
    return split_names


def _copy_roboflow_dataset(roboflow_dir: str, output_dir: str) -> int:
    """
    Copies the Roboflow dataset into the output structure, preserving
    its original train/valid/test partition.

    Args:
        roboflow_dir: Root path of the Roboflow dataset.
        output_dir: Root path of the combined output dataset.

    Returns:
        Total number of image/label pairs copied.
    """
    roboflow_root = Path(roboflow_dir)
    split_names = _resolve_roboflow_split_names(roboflow_root)

    total_copied = 0
    for canonical_split, actual_folder in split_names.items():
        source = roboflow_root / actual_folder
        if not source.is_dir():
            logger.warning("Roboflow split '%s' not found, skipping.", actual_folder)
            continue

        pairs = _list_yolo_seg_pairs(str(source))
        _copy_pairs(
            pairs,
            Path(output_dir) / canonical_split / "images",
            Path(output_dir) / canonical_split / "labels",
            prefix="rf_",
        )
        total_copied += len(pairs)
        logger.info("Roboflow -> %s: %d pairs copied", canonical_split, len(pairs))

    return total_copied


def _split_and_copy_dfire(
    dfire_sam_dir: str,
    output_dir: str,
    split_ratios: Tuple[float, float, float],
    seed: int,
) -> int:
    """
    Randomly partitions the D-Fire+SAM dataset into train/valid/test and
    copies each split into the output structure.

    Args:
        dfire_sam_dir: Root path of the processed D-Fire+SAM dataset
            (unpartitioned 'images/' and 'labels/' subfolders).
        output_dir: Root path of the combined output dataset.
        split_ratios: (train, val, test) proportions. Must sum to 1.0.
        seed: Random seed for reproducible partitioning.

    Returns:
        Total number of image/label pairs copied.

    Raises:
        FileNotFoundError: If no valid pairs are found in dfire_sam_dir.
    """
    pairs = _list_yolo_seg_pairs(dfire_sam_dir)
    if not pairs:
        raise FileNotFoundError(f"No valid image/label pairs found in: {dfire_sam_dir}")

    rng = random.Random(seed)
    rng.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])

    splits = {
        "train": pairs[:n_train],
        "valid": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    total_copied = 0
    for split_name, split_pairs in splits.items():
        _copy_pairs(
            split_pairs,
            Path(output_dir) / split_name / "images",
            Path(output_dir) / split_name / "labels",
            prefix="dfire_",
        )
        total_copied += len(split_pairs)
        logger.info("D-Fire+SAM -> %s: %d pairs copied", split_name, len(split_pairs))

    return total_copied


def _write_data_yaml(output_dir: str, class_names: List[str] = None) -> Path:
    """
    Writes the Ultralytics-compatible data.yaml file for the combined
    dataset.

    Args:
        output_dir: Root path of the combined output dataset.
        class_names: List of class names. Defaults to ['fire'].

    Returns:
        Path to the written data.yaml file.
    """
    class_names = class_names or ["fire"]
    yaml_content = (
        f"train: {output_dir}/train/images\n"
        f"val: {output_dir}/valid/images\n"
        f"test: {output_dir}/test/images\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    yaml_path = Path(output_dir) / "data.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path


def combine_datasets(
    roboflow_dir: str,
    dfire_sam_dir: str,
    output_dir: str,
    dfire_split_ratios: Tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 42,
) -> None:
    """
    Combines the Roboflow dataset (already partitioned into
    train/valid/test) with the D-Fire+SAM dataset (unpartitioned),
    generating the final combined structure ready for Ultralytics.

    Args:
        roboflow_dir: Root path of the Roboflow dataset, with
            train/, valid/ (or val/), test/ subfolders, each containing
            images/ and labels/.
        dfire_sam_dir: Root path of the processed D-Fire+SAM dataset
            ('images/' and 'labels/' subfolders, unpartitioned).
        output_dir: Path where the final combined dataset
            (train/valid/test) will be built.
        dfire_split_ratios: (train, val, test) proportions used to
            randomly split D-Fire. Must sum to 1.0.
        seed: Random seed for reproducible partitioning.

    Raises:
        ValueError: If dfire_split_ratios does not sum to 1.0.
        FileNotFoundError: If either input path does not exist or
            contains no valid data.
    """
    if abs(sum(dfire_split_ratios) - 1.0) > 1e-6:
        raise ValueError("dfire_split_ratios must sum to 1.0")

    if not Path(roboflow_dir).is_dir():
        raise FileNotFoundError(f"Roboflow directory not found: {roboflow_dir}")
    if not Path(dfire_sam_dir).is_dir():
        raise FileNotFoundError(f"D-Fire+SAM directory not found: {dfire_sam_dir}")

    total_roboflow = _copy_roboflow_dataset(roboflow_dir, output_dir)
    total_dfire = _split_and_copy_dfire(dfire_sam_dir, output_dir, dfire_split_ratios, seed)
    yaml_path = _write_data_yaml(output_dir)

    total_combined = total_roboflow + total_dfire
    logger.info("Combined dataset generated at: %s", output_dir)
    logger.info(
        "Total Roboflow: %d | Total D-Fire+SAM: %d | Total combined: %d",
        total_roboflow, total_dfire, total_combined,
    )
    logger.info("data.yaml written to: %s", yaml_path)


if __name__ == "__main__":
    combine_datasets(
        roboflow_dir="/content/Fire-and-Smoke-Segmentation-1",
        dfire_sam_dir="/content/drive/MyDrive/dfire_yolo_seg",
        output_dir="/content/firequant_dataset_v2",
    )