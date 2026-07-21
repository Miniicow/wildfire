"""
webcam_demo.py

Live demo of the wildfire severity system using the computer's webcam.
Processes each frame in real time, computes the Fire Percentage (FP),
displays the severity level and corresponding alert overlaid on the
video, and generates a temporal evolution chart (FP vs. Time) when
finished.

Controls during the demo:
    - Press 'q' to exit and generate the final chart.
    - Point the camera at a photo/video of fire on another screen
      (phone, another monitor) to simulate a fire scene.
"""

import logging
import platform
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AUDIO_AVAILABLE = platform.system() == "Windows"
if AUDIO_AVAILABLE:
    import winsound

SEVERITY_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "Low": (0.0, 3.0),
    "Medium": (3.0, 10.0),
    "High": (10.0, 25.0),
    "Critical": (25.0, 100.0),
}

SEVERITY_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "Low": (0, 200, 0),
    "Medium": (0, 165, 255),
    "High": (0, 100, 255),
    "Critical": (0, 0, 200),
}

ALERT_TEMPLATES: Dict[str, str] = {
    "Low": "LOW LEVEL - Fire detected ({fp}% of area). Routine monitoring.",
    "Medium": "MEDIUM LEVEL - Fire expanding ({fp}% of area). Active surveillance.",
    "High": "HIGH ALERT - Significant fire ({fp}% of area). Notify emergency responders.",
    "Critical": "CRITICAL ALERT - Large-scale fire ({fp}% of area). Emergency protocol.",
}

ALARM_COOLDOWN_SECONDS = 2.0
ALARM_FREQUENCY_HZ = 1000
ALARM_DURATION_MS = 400
ALERT_BORDER_COLOR_BGR = (0, 0, 255)


def play_alarm_in_background(
    frequency: int = ALARM_FREQUENCY_HZ,
    duration_ms: int = ALARM_DURATION_MS,
) -> None:
    """
    Plays an alarm beep using winsound (native to Windows, no extra
    library required), on a separate thread so it doesn't block the
    main video loop.

    Args:
        frequency: Tone frequency in Hz.
        duration_ms: Tone duration in milliseconds.
    """
    if not AUDIO_AVAILABLE:
        return
    threading.Thread(target=winsound.Beep, args=(frequency, duration_ms), daemon=True).start()


def generate_alert_message(level: str, fp: float) -> str:
    """
    Generates the text alert message corresponding to the detected
    severity level.

    Args:
        level: Severity level ('Low', 'Medium', 'High', 'Critical').
        fp: Associated Fire Percentage value.

    Returns:
        Formatted alert message.
    """
    return ALERT_TEMPLATES.get(level, "Unknown level").format(fp=fp)


def classify_severity(fire_percentage: float) -> str:
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


def compute_fp_and_mask(result, height: int, width: int) -> Tuple[float, np.ndarray]:
    """
    Computes the Fire Percentage and the combined binary mask from a
    single frame's inference result.

    Args:
        result: Ultralytics Results object for one frame.
        height: Frame height in pixels.
        width: Frame width in pixels.

    Returns:
        Tuple (fire_percentage, binary_mask).
    """
    total_pixels = height * width
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    if result.masks is not None:
        for mask in result.masks.data.cpu().numpy():
            resized_mask = cv2.resize(
                mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            )
            combined_mask = np.logical_or(combined_mask, resized_mask).astype(np.uint8)

    fire_pixels = int(np.count_nonzero(combined_mask))
    fp = round((fire_pixels / total_pixels) * 100, 2) if total_pixels > 0 else 0.0
    return fp, combined_mask


def _draw_alert_overlay(frame: np.ndarray, height: int, width: int) -> None:
    """
    Draws a blinking red border and an "ALERT" banner on the frame,
    in place, to reinforce a High/Critical severity alert.

    Args:
        frame: Frame to draw on (modified in place).
        height: Frame height in pixels.
        width: Frame width in pixels.
    """
    is_blink_on = int(time.time() * 4) % 2 == 0
    if is_blink_on:
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), ALERT_BORDER_COLOR_BGR, 12)
    cv2.putText(
        frame, "!! ALERT !!", (width // 2 - 100, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1, ALERT_BORDER_COLOR_BGR, 3, cv2.LINE_AA,
    )


def run_webcam_demo(
    weights_path: str,
    conf_threshold: float = 0.15,
    camera_id: int = 0,
    chart_output_path: str = "fp_evolution_over_time.png",
) -> List[Tuple[float, float]]:
    """
    Runs the live demo: opens the camera, processes each frame with the
    model, displays the mask/FP/alert overlay, and records the (time, FP)
    history to plot when finished.

    Args:
        weights_path: Path to the trained checkpoint (best.pt).
        conf_threshold: Confidence threshold for inference.
        camera_id: Camera index to use (0 = default camera).
        chart_output_path: Path where the final FP vs. time chart will
            be saved.

    Returns:
        List of (time_seconds, fire_percentage) tuples recorded during
        the demo.

    Raises:
        FileNotFoundError: If the checkpoint doesn't exist.
        RuntimeError: If the camera can't be opened.
    """
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

    model = YOLO(weights_path)

    capture = cv2.VideoCapture(camera_id)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera with index {camera_id}")

    history: List[Tuple[float, float]] = []
    start_time = time.time()
    last_alarm_time = 0.0

    if not AUDIO_AVAILABLE:
        logger.warning(
            "Audible alarm not available on this OS (winsound is Windows-only). "
            "Only the visual alert will be shown."
        )

    logger.info("Live demo started. Press 'q' in the video window to exit.")

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                logger.warning("Could not read a frame from the camera.")
                break

            height, width = frame.shape[:2]
            result = model.predict(source=frame, conf=conf_threshold, verbose=False)[0]

            fp, mask = compute_fp_and_mask(result, height, width)
            level = classify_severity(fp)
            color = SEVERITY_COLORS_BGR[level]

            elapsed_time = round(time.time() - start_time, 2)
            history.append((elapsed_time, fp))

            overlay = frame.copy()
            overlay[mask == 1] = color
            final_frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

            is_alarm_active = level in ("High", "Critical")
            if is_alarm_active and (time.time() - last_alarm_time) > ALARM_COOLDOWN_SECONDS:
                last_alarm_time = time.time()
                logger.info("[ALARM] %s", generate_alert_message(level, fp))
                play_alarm_in_background()

            if is_alarm_active:
                _draw_alert_overlay(final_frame, height, width)

            text = f"FP: {fp}%  |  Level: {level}"
            cv2.putText(final_frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            cv2.putText(
                final_frame, "Press 'q' to exit", (15, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow("Wildfire Severity - Live Demo", final_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()

    if history:
        _plot_fp_evolution(history, chart_output_path)

    return history


def _plot_fp_evolution(history: List[Tuple[float, float]], output_path: str) -> None:
    """
    Generates and saves the Fire Percentage vs. Time chart from the
    history recorded during the demo.

    Args:
        history: List of (time_seconds, fire_percentage) tuples.
        output_path: Path of the image file to generate.
    """
    times = [entry[0] for entry in history]
    fp_values = [entry[1] for entry in history]

    plt.figure(figsize=(10, 5))
    plt.plot(times, fp_values, color="#c62828", linewidth=1.5)
    plt.fill_between(times, fp_values, color="#c62828", alpha=0.15)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Fire Percentage (%)")
    plt.title("Fire Percentage Evolution During the Live Demo")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Temporal evolution chart saved to: %s", output_path)


if __name__ == "__main__":
    run_webcam_demo(
        weights_path="best.pt",  # adjust to your local downloaded model path
        conf_threshold=0.15,
    )