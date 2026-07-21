"""
inference.py

Inference module for the wildfire severity estimation system. Applies
the trained YOLOv26n-seg model to new images (outside the training
dataset), extracts the binary fire mask, computes the Fire Percentage
(FP) according to:

    FP = (P_fire / P_total) * 100

and classifies fire severity into discrete levels (Low, Medium, High,
Critical) to trigger early alerts.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEVERITY_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "Low": (0.0, 3.0),
    "Medium": (3.0, 10.0),
    "High": (10.0, 25.0),
    "Critical": (25.0, 100.0),
}

SEVERITY_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "Low": (0, 200, 0),
    "Medium": (0, 165, 255),
    "High": (0, 0, 255),
    "Critical": (0, 0, 139),
}

ALERT_TEMPLATES: Dict[str, str] = {
    "Low": "🟢 LOW LEVEL — Fire detected ({fp}% of area). Routine monitoring.",
    "Medium": "🟡 MEDIUM LEVEL — Fire expanding ({fp}% of area). Active surveillance recommended.",
    "High": "🟠 HIGH ALERT — Significant fire ({fp}% of area). Notify emergency responders.",
    "Critical": "🔴 CRITICAL ALERT — Large-scale fire ({fp}% of area). Immediate emergency protocol activation.",
}


@dataclass
class FireEstimationResult:
    """
    Encapsulates the fire quantification result for a single image.

    Attributes:
        fire_pixels: Number of pixels classified as fire.
        total_pixels: Total number of pixels in the image (height x width).
        fire_percentage: Percentage of the area occupied by fire (FP).
        severity_level: Severity level assigned based on FP.
        num_instances: Number of detected fire instances/regions.
        mean_confidence: Average confidence of the detections (0-1).
    """
    fire_pixels: int
    total_pixels: int
    fire_percentage: float
    severity_level: str
    num_instances: int
    mean_confidence: float


class FireSeverityEstimator:
    """
    Wraps the trained YOLOv26n-seg model and exposes methods to estimate
    the Fire Percentage and severity level on individual images or frames.
    """

    def __init__(self, weights_path: str, fire_class_id: int = 0, conf_threshold: float = 0.35):
        """
        Initializes the estimator by loading the trained weights.

        Args:
            weights_path: Path to the trained checkpoint (best.pt).
            fire_class_id: 'fire' class ID as defined in data.yaml
                (defaults to 0 for a single-class dataset).
            conf_threshold: Minimum confidence threshold for a detection
                to be considered valid.

        Raises:
            FileNotFoundError: If the weights file doesn't exist.
        """
        if not Path(weights_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

        self.model = YOLO(weights_path)
        self.fire_class_id = fire_class_id
        self.conf_threshold = conf_threshold

    @staticmethod
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

    @staticmethod
    def generate_alert_message(result: "FireEstimationResult") -> str:
        """
        Generates the early-alert message corresponding to the detected
        severity level, for notifying emergency responders.

        Args:
            result: Result already computed by `estimate()`.

        Returns:
            Formatted alert message, ready to display in console, send
            as a notification, or write to a log.
        """
        template = ALERT_TEMPLATES.get(result.severity_level, "Unknown level")
        return template.format(fp=result.fire_percentage)

    def _load_frame(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Loads an input image, accepting either a file path or an
        already-decoded NumPy array.

        Args:
            image: Path to an image file, or a NumPy array (BGR format,
                as returned by OpenCV) already loaded in memory.

        Returns:
            The image as a BGR NumPy array.

        Raises:
            FileNotFoundError: If `image` is a path that doesn't exist.
            ValueError: If the image couldn't be read/decoded, or if the
                input type is unsupported.
        """
        if isinstance(image, str):
            if not Path(image).is_file():
                raise FileNotFoundError(f"Image not found: {image}")
            frame = cv2.imread(image)
            if frame is None:
                raise ValueError(f"OpenCV could not decode the image: {image}")
            return frame

        if isinstance(image, np.ndarray):
            return image

        raise ValueError("The 'image' parameter must be a path (str) or a NumPy array.")

    def estimate(self, image: Union[str, np.ndarray]) -> FireEstimationResult:
        """
        Runs segmentation inference on an image and computes the Fire
        Percentage (FP) via direct pixel counting on the combined binary
        mask of all fire instances.

        Args:
            image: Path to an image file, or a NumPy array (BGR format,
                as returned by OpenCV) already loaded in memory.

        Returns:
            FireEstimationResult instance with the computed metrics.

        Raises:
            FileNotFoundError: If `image` is a path that doesn't exist.
            ValueError: If the image couldn't be read/decoded.
            RuntimeError: If model inference fails.
        """
        frame = self._load_frame(image)
        height, width = frame.shape[:2]
        total_pixels = height * width

        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Error during model inference: {exc}") from exc

        result = results[0]

        combined_mask = np.zeros((height, width), dtype=np.uint8)
        num_instances = 0
        confidences: List[float] = []

        if result.masks is not None and result.boxes is not None:
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            raw_masks = result.masks.data.cpu().numpy()

            for idx, class_id in enumerate(classes):
                if int(class_id) != self.fire_class_id:
                    continue

                instance_mask = raw_masks[idx]
                resized_mask = cv2.resize(
                    instance_mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                combined_mask = np.logical_or(combined_mask, resized_mask).astype(np.uint8)

                num_instances += 1
                confidences.append(float(confs[idx]))

        fire_pixels = int(np.count_nonzero(combined_mask))
        fire_percentage = (fire_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
        severity = self._classify_severity(fire_percentage)
        mean_confidence = float(np.mean(confidences)) if confidences else 0.0

        return FireEstimationResult(
            fire_pixels=fire_pixels,
            total_pixels=total_pixels,
            fire_percentage=round(fire_percentage, 2),
            severity_level=severity,
            num_instances=num_instances,
            mean_confidence=round(mean_confidence, 3),
        )

    def visualize(
        self,
        image: Union[str, np.ndarray],
        result: FireEstimationResult,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generates a visualization with the fire mask overlay and a text
        overlay showing the FP and severity level.

        Args:
            image: Path or array of the original image.
            result: Result already computed by `estimate()`.
            output_path: If provided, saves the annotated image to this path.

        Returns:
            Annotated image as a NumPy array (BGR).

        Raises:
            ValueError: If the image couldn't be loaded.
        """
        if isinstance(image, str):
            frame = cv2.imread(image)
            if frame is None:
                raise ValueError(f"Could not load the image: {image}")
        else:
            frame = image.copy()

        results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
        annotated = results[0].plot()

        color = SEVERITY_COLORS_BGR.get(result.severity_level, (255, 255, 255))

        text = f"FP: {result.fire_percentage}%  |  Level: {result.severity_level}"
        cv2.putText(
            annotated, text, (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA,
        )

        if output_path:
            cv2.imwrite(output_path, annotated)

        return annotated


if __name__ == "__main__":
    WEIGHTS_PATH = "runs/yolo26n_seg_v1/weights/best.pt"
    TEST_IMAGE = "/content/new_image.jpg"

    estimator = FireSeverityEstimator(weights_path=WEIGHTS_PATH, conf_threshold=0.35)
    result = estimator.estimate(TEST_IMAGE)

    logger.info("Fire pixels:          %d", result.fire_pixels)
    logger.info("Total pixels:         %d", result.total_pixels)
    logger.info("Fire Percentage (FP): %s%%", result.fire_percentage)
    logger.info("Severity level:       %s", result.severity_level)
    logger.info("Instances detected:   %d", result.num_instances)
    logger.info("Mean confidence:      %s", result.mean_confidence)

    print()
    print(estimator.generate_alert_message(result))

    estimator.visualize(TEST_IMAGE, result, output_path="/content/annotated_result.jpg")