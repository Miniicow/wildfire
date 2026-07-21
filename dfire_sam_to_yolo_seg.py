"""
dfire_sam_to_yolo_seg.py

Converts bounding box annotations from the D-Fire dataset (class 'fire')
into pixel-level segmentation masks using the Segment Anything Model
(SAM), and exports the result in YOLO-segmentation format.

Methodology: for each 'fire' bounding box, the box is used as a
geometric prompt for SAM, which returns the mask of the object contained
within that region. This approach follows the same methodology used in
the "Boreal Forest Fire" dataset (Pesonen et al., 2025) to generate
smoke masks from human-provided boxes.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DFIRE_FIRE_CLASS_ID = 1  # 'fire' class id in the original D-Fire annotations
OUTPUT_FIRE_CLASS_ID = 0  # class id used in the final YOLO-seg dataset
MIN_CONTOUR_AREA = 20
IMAGE_EXTENSIONS = ("*.jpg", "*.png")
PROGRESS_LOG_INTERVAL = 50


def load_sam_model(
    checkpoint_path: str = "/content/sam_vit_b_01ec64.pth",
    model_type: str = "vit_b",
) -> SamPredictor:
    """
    Loads the pretrained SAM model and returns a SamPredictor ready to
    generate masks from prompts (boxes or points).

    Args:
        checkpoint_path: Path to the SAM .pth checkpoint. If it doesn't
            exist, it must be downloaded beforehand (see usage
            instructions at the bottom of this file).
        model_type: SAM variant ('vit_b', 'vit_l', 'vit_h'). 'vit_b' is
            the lightest, recommended for free Colab tiers.

    Returns:
        SamPredictor instance with the model loaded on GPU if available.

    Raises:
        FileNotFoundError: If the checkpoint doesn't exist at the given path.
    """
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"SAM checkpoint not found at: {checkpoint_path}\n"
            "Download it with:\n"
            "!wget https://dl.fbaipublicfiles.com/segment_anything/"
            "sam_vit_b_01ec64.pth -P /content/"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)

    logger.info("SAM (%s) loaded on: %s", model_type, device)
    return SamPredictor(sam)


def read_fire_boxes_yolo(
    label_path: str,
    img_width: int,
    img_height: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Reads a YOLO detection annotation file and returns the 'fire' class
    boxes converted to pixel coordinates (x1, y1, x2, y2).

    Args:
        label_path: Path to the YOLO .txt annotation file.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        List of (x1, y1, x2, y2) boxes in pixels, fire class only.

    Raises:
        FileNotFoundError: If the label file doesn't exist.
    """
    if not Path(label_path).is_file():
        raise FileNotFoundError(f"Annotation file not found: {label_path}")

    boxes = []
    for line in Path(label_path).read_text().strip().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        class_id = int(parts[0])
        if class_id != DFIRE_FIRE_CLASS_ID:
            continue

        x_center, y_center, rel_width, rel_height = map(float, parts[1:5])
        x1 = int((x_center - rel_width / 2) * img_width)
        y1 = int((y_center - rel_height / 2) * img_height)
        x2 = int((x_center + rel_width / 2) * img_width)
        y2 = int((y_center + rel_height / 2) * img_height)
        boxes.append((max(x1, 0), max(y1, 0), min(x2, img_width), min(y2, img_height)))

    return boxes


def generate_mask_with_sam(
    predictor: SamPredictor,
    image_rgb: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Generates a combined binary mask using SAM, guided by one or more
    fire bounding boxes.

    Args:
        predictor: SamPredictor instance ready to process the image
            (the method internally calls set_image).
        image_rgb: Image in RGB format (not BGR) as a NumPy array.
        boxes: List of (x1, y1, x2, y2) boxes in pixels.

    Returns:
        Binary mask (uint8, 0/255) of the same size as the image,
        combining all detected instances.

    Raises:
        RuntimeError: If SAM fails to generate a mask for any box.
    """
    height, width = image_rgb.shape[:2]
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    predictor.set_image(image_rgb)

    for box in boxes:
        box_array = np.array(box)
        try:
            masks, scores, _ = predictor.predict(
                box=box_array[None, :],
                multimask_output=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to generate SAM mask for box {box}: {exc}") from exc

        instance_mask = masks[0].astype(np.uint8) * 255
        combined_mask = np.maximum(combined_mask, instance_mask)

    return combined_mask


def mask_to_yolo_seg(mask: np.ndarray, class_id: int = OUTPUT_FIRE_CLASS_ID) -> List[str]:
    """
    Converts a binary mask into YOLO-segmentation annotation lines.

    Args:
        mask: Binary mask (0/255).
        class_id: Class ID to assign in the output.

    Returns:
        List of lines in "class_id x1 y1 x2 y2 ... xn yn" format
        (normalized coordinates).
    """
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        epsilon = 0.003 * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified_contour) < 3:
            continue

        normalized_points = []
        for point in simplified_contour.reshape(-1, 2):
            normalized_points.extend([f"{point[0] / width:.6f}", f"{point[1] / height:.6f}"])
        lines.append(f"{class_id} " + " ".join(normalized_points))

    return lines


def _resolve_label_path(image_path: Path) -> Optional[Path]:
    """
    Determines the label (.txt) file path corresponding to an image,
    supporting two common dataset structure conventions:

    1. Label in the same folder as the image (same name, .txt).
    2. Label in a sibling 'labels/' folder (standard YOLO convention,
       where images live in 'images/' and labels in 'labels/' with the
       same subfolder structure).

    Args:
        image_path: Path to the image.

    Returns:
        Candidate path to the .txt file, or None if neither convention
        yields an existing file.
    """
    same_folder_candidate = image_path.with_suffix(".txt")
    if same_folder_candidate.is_file():
        return same_folder_candidate

    parts = list(image_path.parts)
    label_parts = ["labels" if part.lower() == "images" else part for part in parts]
    labels_folder_candidate = Path(*label_parts).with_suffix(".txt")
    if labels_folder_candidate.is_file():
        return labels_folder_candidate

    return None


def _collect_fire_images(root_dir: Path, max_images: Optional[int]) -> List[Path]:
    """
    Collects image paths that have at least one 'fire' class instance
    in their corresponding label file, stopping early once `max_images`
    candidates are found.

    Args:
        root_dir: Root directory to search for images.
        max_images: Optional cap on the number of candidates to collect.

    Returns:
        List of image paths with at least one fire annotation.
    """
    all_images = sorted(
        p for pattern in IMAGE_EXTENSIONS for p in root_dir.rglob(pattern)
    )

    fire_images = []
    for img_path in all_images:
        label_path = _resolve_label_path(img_path)
        if label_path is None:
            continue
        content = label_path.read_text().strip().splitlines()
        if any(line.strip().startswith(f"{DFIRE_FIRE_CLASS_ID} ") for line in content):
            fire_images.append(img_path)
        if max_images and len(fire_images) >= max_images:
            break

    return fire_images


def process_dfire_dataset(
    dfire_dir: str,
    output_dir: str,
    predictor: SamPredictor,
    max_images: Optional[int] = None,
) -> None:
    """
    Processes all D-Fire images with at least one fire instance,
    generating their pseudo-masks via SAM and exporting the
    image+annotation pair in YOLO-seg format.

    Args:
        dfire_dir: Root path of the downloaded D-Fire dataset.
        output_dir: Path where the YOLO-seg dataset will be built.
        predictor: Already-loaded SamPredictor instance.
        max_images: Optional cap on the number of images to process
            (useful for quick tests before running on the full dataset).

    Raises:
        FileNotFoundError: If no images or labels are found.
    """
    root = Path(dfire_dir)
    all_images = sorted(p for pattern in IMAGE_EXTENSIONS for p in root.rglob(pattern))
    if not all_images:
        raise FileNotFoundError(f"No images found in: {dfire_dir}")

    fire_images = _collect_fire_images(root, max_images)

    processed, skipped, already_existed = 0, 0, 0

    for img_path in fire_images:
        base_name = img_path.stem
        output_img = Path(output_dir) / "images" / f"{base_name}.jpg"
        output_label = Path(output_dir) / "labels" / f"{base_name}.txt"

        if output_img.is_file() and output_label.is_file():
            already_existed += 1
            continue

        label_path = _resolve_label_path(img_path)

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            skipped += 1
            continue

        height, width = image_bgr.shape[:2]
        fire_boxes = read_fire_boxes_yolo(str(label_path), width, height)

        if not fire_boxes:
            skipped += 1
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = generate_mask_with_sam(predictor, image_rgb, fire_boxes)
        yolo_lines = mask_to_yolo_seg(mask)

        if not yolo_lines:
            skipped += 1
            continue

        output_img.parent.mkdir(parents=True, exist_ok=True)
        output_label.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_img), image_bgr)
        output_label.write_text("\n".join(yolo_lines))

        processed += 1
        if processed % PROGRESS_LOG_INTERVAL == 0:
            logger.info(
                "Processed: %d | Skipped: %d | Already existed: %d",
                processed, skipped, already_existed,
            )

    logger.info(
        "Done. Total processed: %d | Skipped: %d | Already existed (resumed): %d",
        processed, skipped, already_existed,
    )
    logger.info("Dataset generated at: %s", output_dir)


if __name__ == "__main__":
    sam_predictor = load_sam_model(checkpoint_path="/content/sam_vit_b_01ec64.pth")

    process_dfire_dataset(
        dfire_dir="/content/DFireDataset",
        output_dir="/content/dfire_yolo_seg",
        predictor=sam_predictor,
        max_images=20,
    )