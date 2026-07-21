"""
train_model.py

Training script for wildfire semantic segmentation using YOLOv26n-seg
(Ultralytics) on the combined Roboflow + D-Fire dataset.

Designed to run on Google Colab. Includes a reinforced data
augmentation configuration to mitigate the overfitting observed in the
earlier phase of the project (210 images from a single source video).
"""

import logging
import os
from pathlib import Path
from typing import Dict

import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PROJECT = "/content/drive/MyDrive/wildfire_runs"


def verify_environment() -> None:
    """
    Checks GPU availability before starting training.

    Raises:
        RuntimeError: If no GPU is detected in the Colab runtime.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. Go to Runtime > Change runtime type > "
            "Hardware accelerator > GPU (T4 or higher)."
        )
    logger.info("GPU detected: %s", torch.cuda.get_device_name(0))


def _resolve_resume_checkpoint(output_project: str, experiment_name: str) -> str:
    """
    Checks whether a 'last.pt' checkpoint from a previously interrupted
    run already exists (e.g. due to a Colab disconnection), so training
    can resume from there instead of starting from scratch.

    Args:
        output_project: Root folder where results are saved.
        experiment_name: Subfolder name for this run.

    Returns:
        Path to the checkpoint to resume from if one exists, otherwise
        an empty string.
    """
    checkpoint_path = os.path.join(output_project, experiment_name, "weights", "last.pt")
    if os.path.isfile(checkpoint_path):
        logger.info("Previous checkpoint found at: %s", checkpoint_path)
        logger.info("Resuming training from where it left off...")
        return checkpoint_path
    return ""


def train_model(
    data_yaml: str,
    base_model: str = "yolo26n-seg.pt",
    epochs: int = 60,
    imgsz: int = 640,
    batch: int = 48,
    output_project: str = DEFAULT_OUTPUT_PROJECT,
    experiment_name: str = "yolo26n_seg_v1",
    cache: str = "ram",
) -> YOLO:
    """
    Trains a YOLOv26n-seg model on the fire segmentation dataset,
    applying a reinforced data augmentation policy to combat the
    overfitting observed when testing the model on external scenarios.

    Args:
        data_yaml: Path to the data.yaml file for the dataset.
        base_model: Starting pretrained checkpoint (transfer learning
            from COCO). Use the nano (n) variant for efficiency.
        epochs: Maximum number of training epochs.
        imgsz: Input image size (square resolution).
        batch: Batch size. Adjust based on available VRAM in Colab.
        output_project: Root folder where results are saved.
        experiment_name: Subfolder name for this run.
        cache: Image caching mode ('ram', 'disk', or False).

    Returns:
        YOLO instance with the trained weights loaded (best.pt).

    Raises:
        FileNotFoundError: If data_yaml doesn't exist at the given path.
        RuntimeError: If training fails due to a framework error.
    """
    if not Path(data_yaml).is_file():
        raise FileNotFoundError(f"data.yaml file not found at: {data_yaml}")

    resume_checkpoint = _resolve_resume_checkpoint(output_project, experiment_name)
    should_resume = bool(resume_checkpoint)
    if should_resume:
        base_model = resume_checkpoint

    try:
        model = YOLO(base_model)

        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=output_project,
            name=experiment_name,
            resume=should_resume,
            exist_ok=True,

            # -----------------------------------------------------------
            # General regularization (key given the observed overfitting)
            # -----------------------------------------------------------
            patience=15,          # more aggressive: stops earlier if it stalls
            cache=cache,          # 'ram' caches decoded images, avoiding re-reads each epoch
            dropout=0.1,          # additional regularization in the head
            weight_decay=0.0008,  # penalizes large weights (default 0.0005)
            cos_lr=True,          # cosine learning rate scheduler
            lr0=0.001,            # conservative initial LR (small dataset)
            optimizer="AdamW",

            # -----------------------------------------------------------
            # Reinforced data augmentation — CRITICAL for generalization
            # A single source video implies low variability in lighting,
            # angle, and background; compensated with aggressive augmentation.
            # -----------------------------------------------------------
            hsv_h=0.02,        # hue variation (simulates different cameras)
            hsv_s=0.7,         # saturation variation (different lighting)
            hsv_v=0.5,         # brightness variation (day/night, exposure)
            degrees=10.0,      # random rotation (+/- degrees)
            translate=0.15,    # random translation
            scale=0.6,         # random scaling (simulates different distances)
            shear=5.0,         # slight shear/deformation
            perspective=0.0005,
            flipud=0.1,        # occasional vertical flip
            fliplr=0.5,        # horizontal flip (fire has no fixed orientation)
            mosaic=1.0,        # 4-image mosaic: forces learning varied context
            mixup=0.15,        # mixes two images: reduces scene memorization
            copy_paste=0.3,    # pastes fire instances onto different backgrounds (seg)
            erasing=0.2,       # random erasing: simulates partial occlusions

            # -----------------------------------------------------------
            # Other
            # -----------------------------------------------------------
            val=True,
            plots=True,
            save=True,
            save_period=5,        # checkpoint every 5 epochs (was 10): less loss on disconnect
            device=0,
            workers=8,
            seed=42,
            verbose=True,
        )

        logger.info("Training finished.")
        logger.info("Best weights saved to: %s/%s/weights/best.pt", output_project, experiment_name)
        return model

    except Exception as exc:
        raise RuntimeError(f"Error during training: {exc}") from exc


def validate_model(model: YOLO, data_yaml: str, split: str = "test") -> Dict[str, float]:
    """
    Validates the trained model on the specified split (test by
    default) and returns the key metrics (mask mAP, precision, recall).

    Args:
        model: Already-trained YOLO instance (or loaded from best.pt).
        data_yaml: Path to the dataset's data.yaml.
        split: Split to evaluate ('val' or 'test').

    Returns:
        Dictionary with the main segmentation metrics.

    Raises:
        RuntimeError: If validation fails.
    """
    try:
        metrics = model.val(data=data_yaml, split=split, plots=True)
        summary = {
            "mAP50_mask": metrics.seg.map50,
            "mAP50-95_mask": metrics.seg.map,
            "precision_mask": metrics.seg.mp,
            "recall_mask": metrics.seg.mr,
        }
        logger.info("Validation metrics (segmentation):")
        for key, value in summary.items():
            logger.info("  %s: %.4f", key, value)
        return summary
    except Exception as exc:
        raise RuntimeError(f"Error during validation: {exc}") from exc


if __name__ == "__main__":
    # Adjust this path to the combined dataset (Roboflow + D-Fire+SAM).
    DATA_YAML = "/content/wildfire_dataset_v2/data.yaml"

    verify_environment()
    # With ~6,000 images (30x more than the previous 210-image run), the
    # number of epochs is reduced: there are many more iterations per
    # epoch, so fewer epochs already cover a comparable or greater
    # learning volume.
    trained_model = train_model(
        data_yaml=DATA_YAML,
        epochs=80,
        batch=32,  # increase batch size if the GPU allows it, larger dataset
        experiment_name="yolo26n_seg_v2_combined",
    )
    validate_model(trained_model, data_yaml=DATA_YAML, split="test")
