"""
generate_result_charts.py

Generates result charts for the wildfire severity estimation
presentation/paper:
1. Training metric evolution per epoch (from the results.csv that
   Ultralytics saves automatically).
2. Fire Percentage per test image, colored by severity level.
3. Qualitative Original -> Mask -> Level sequence for representative images.
4. Severity level distribution across the test set.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEVERITY_COLORS: Dict[str, str] = {
    "Low": "#2e7d32",
    "Medium": "#f9a825",
    "High": "#ef6c00",
    "Critical": "#b71c1c",
}

SEVERITY_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "Low": (0.0, 3.0),
    "Medium": (3.0, 10.0),
    "High": (10.0, 25.0),
    "Critical": (25.0, 100.0),
}

MASK_OVERLAY_COLOR_BGR = (203, 27, 106)  # purple/blue, high contrast against fire


def _load_results_csv(run_path: str) -> pd.DataFrame:
    """
    Loads and cleans the results.csv file generated automatically by
    Ultralytics for a given training run.

    Args:
        run_path: Path to the training run folder (e.g.
            '.../wildfire_runs/yolo26n_seg_v2_combined').

    Returns:
        DataFrame with stripped column names.

    Raises:
        FileNotFoundError: If results.csv is not found at the given path.
    """
    csv_path = Path(run_path) / "results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"results.csv not found at: {run_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _classify_severity(fire_percentage: float) -> str:
    """
    Assigns a discrete severity level based on the Fire Percentage.

    Args:
        fire_percentage: FP value in the range [0, 100].

    Returns:
        Severity label: 'Low', 'Medium', 'High', or 'Critical'.
    """
    for level, (lower_bound, upper_bound) in SEVERITY_THRESHOLDS.items():
        if lower_bound <= fire_percentage < upper_bound:
            return level
    return "Critical"


def plot_training_evolution(
    run_path: str,
    output_path: str = "training_evolution.png",
) -> None:
    """
    Plots the evolution of mAP50 (box and mask) and the main training
    losses across epochs, from the results.csv generated automatically
    by Ultralytics.

    Args:
        run_path: Path to the training run folder.
        output_path: Name of the image file to generate.

    Raises:
        FileNotFoundError: If results.csv is not found at the given path.
    """
    df = _load_results_csv(run_path)

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    box_map_cols = [c for c in df.columns if "mAP50(B)" in c or "mAP50(B" in c]
    mask_map_cols = [c for c in df.columns if "mAP50(M)" in c or "mAP50(M" in c]

    if box_map_cols:
        axes[0].plot(df["epoch"], df[box_map_cols[0]], label="mAP50 (Box)", color="#1565c0")
    if mask_map_cols:
        axes[0].plot(df["epoch"], df[mask_map_cols[0]], label="mAP50 (Mask)", color="#c62828")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("mAP50")
    axes[0].set_title("mAP50 Evolution During Training")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    box_loss_cols = [c for c in df.columns if "box_loss" in c and "train" in c]
    seg_loss_cols = [c for c in df.columns if "seg_loss" in c and "train" in c]
    cls_loss_cols = [c for c in df.columns if "cls_loss" in c and "train" in c]

    if box_loss_cols:
        axes[1].plot(df["epoch"], df[box_loss_cols[0]], label="Box Loss", color="#2e7d32")
    if seg_loss_cols:
        axes[1].plot(df["epoch"], df[seg_loss_cols[0]], label="Seg Loss", color="#6a1b9a")
    if cls_loss_cols:
        axes[1].plot(df["epoch"], df[cls_loss_cols[0]], label="Cls Loss", color="#e65100")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Loss Evolution")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Chart saved to: %s", output_path)


def plot_fire_percentage_per_image(
    summary: List[Dict],
    output_path: str = "fire_percentage_per_image.png",
) -> None:
    """
    Generates a bar chart of the Fire Percentage per test image, colored
    by the assigned severity level.

    Args:
        summary: List of dictionaries with per-image results (the output
            of `process_batch()` in process_test_batch.py). Each entry
            must have the keys 'image', 'fire_percentage', 'severity'.
        output_path: Name of the image file to generate.

    Raises:
        ValueError: If `summary` is empty.
    """
    if not summary:
        raise ValueError("The results summary is empty.")

    names = [entry["image"] for entry in summary]
    fp_values = [entry["fire_percentage"] for entry in summary]
    colors = [SEVERITY_COLORS.get(entry["severity"], "#757575") for entry in summary]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, fp_values, color=colors)

    for bar, entry in zip(bars, summary):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{entry['fire_percentage']}%\n{entry['severity']}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel("Test image")
    ax.set_ylabel("Fire Percentage (%)")
    ax.set_title("Fire Percentage and Severity Level per Test Image")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Chart saved to: %s", output_path)


def print_final_metrics_table(run_path: str) -> None:
    """
    Extracts and prints, in a clean table format, the metrics from the
    last epoch recorded in results.csv, ready to copy into a document.

    Args:
        run_path: Path to the training run folder.

    Raises:
        FileNotFoundError: If results.csv is not found at the given path.
    """
    df = _load_results_csv(run_path)
    last_row = df.iloc[-1]

    print("=== Final Training Metrics ===")
    for column in df.columns:
        if "metrics" in column or "loss" in column:
            print(f"{column:<30}: {last_row[column]:.4f}")


def plot_qualitative_sequence(
    weights_path: str,
    image_paths: List[str],
    conf_threshold: float = 0.15,
    output_path: str = "qualitative_sequence.png",
) -> None:
    """
    Generates a figure with the Original -> Mask -> Level sequence for a
    small set of example images (ideally one per representative severity
    level: Low, Medium/High, Critical).

    Args:
        weights_path: Path to the trained checkpoint (best.pt).
        image_paths: List of paths to the example images to display
            (3 recommended: one per relevant severity range).
        conf_threshold: Confidence threshold for inference.
        output_path: Name of the image file to generate.

    Raises:
        FileNotFoundError: If the checkpoint or any image doesn't exist.
    """
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

    model = YOLO(weights_path)
    n_images = len(image_paths)

    _, axes = plt.subplots(n_images, 2, figsize=(9, n_images * 4.2))
    if n_images == 1:
        axes = axes.reshape(1, 2)

    for idx, img_path in enumerate(image_paths):
        if not Path(img_path).is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")

        frame = cv2.imread(str(img_path))
        height, width = frame.shape[:2]
        total_pixels = height * width

        result = model.predict(source=frame, conf=conf_threshold, verbose=False)[0]

        mask = np.zeros((height, width), dtype=np.uint8)
        if result.masks is not None:
            for m in result.masks.data.cpu().numpy():
                resized_mask = cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                mask = np.logical_or(mask, resized_mask).astype(np.uint8)

        fire_pixels = int(np.count_nonzero(mask))
        fp = round((fire_pixels / total_pixels) * 100, 2) if total_pixels > 0 else 0.0
        level = _classify_severity(fp)

        overlay = frame.copy()
        overlay[mask == 1] = MASK_OVERLAY_COLOR_BGR
        blended = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

        axes[idx, 0].imshow(frame_rgb)
        axes[idx, 0].set_title(f"Original — {Path(img_path).stem}", fontsize=9)
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(blended_rgb)
        axes[idx, 1].set_title(f"Mask | FP: {fp}% | Level: {level}", fontsize=9)
        axes[idx, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Chart saved to: %s", output_path)


def plot_severity_distribution(
    summary: List[Dict],
    output_path: str = "severity_distribution.png",
) -> None:
    """
    Generates a bar chart with the distribution of severity levels
    assigned across the test image set.

    Args:
        summary: List of dictionaries with per-image results (the output
            of `process_batch()` in process_test_batch.py). Each entry
            must have the key 'severity'.
        output_path: Name of the image file to generate.

    Raises:
        ValueError: If `summary` is empty.
    """
    if not summary:
        raise ValueError("The results summary is empty.")

    level_order = ["Low", "Medium", "High", "Critical"]
    counts = {level: 0 for level in level_order}
    for entry in summary:
        counts[entry["severity"]] = counts.get(entry["severity"], 0) + 1

    total = len(summary)
    values = [counts[level] for level in level_order]
    percentages = [round((v / total) * 100, 1) for v in values]
    colors = [SEVERITY_COLORS[level] for level in level_order]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(level_order, values, color=colors)

    for bar, value, pct in zip(bars, values, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{value} ({pct}%)",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel("Number of images")
    ax.set_title("Distribution of Assigned Severity Levels")
    ax.set_ylim(0, max(values) + 2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Chart saved to: %s", output_path)


if __name__ == "__main__":
    RUN_PATH = "/content/drive/MyDrive/wildfire_runs/yolo26n_seg_v2_combined"

    plot_training_evolution(RUN_PATH)
    print_final_metrics_table(RUN_PATH)

    # For the FP-per-image chart, use the 'summary' already generated
    # with process_test_batch.py:
    #
    # from process_test_batch import process_batch
    # summary = process_batch(...)
    # plot_fire_percentage_per_image(summary)
