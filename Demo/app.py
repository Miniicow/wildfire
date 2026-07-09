"""
app.py

Simple web interface for FireQuant built with Streamlit. Lets the user
upload an image, runs the full pipeline (YOLOv26n-seg segmentation, Fire
Percentage calculation, severity classification, alert generation), and
displays the result interactively.

"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


SEVERITY_THRESHOLDS = {
    "Low": (0.0, 3.0),
    "Medium": (3.0, 10.0),
    "High": (10.0, 25.0),
    "Critical": (25.0, 100.0),
}


SEVERITY_COLORS_RGB = {
    "Low": (65, 105, 225),
    "Medium": (106, 27, 154),
    "High": (156, 39, 176),  
    "Critical": (74, 20, 140), 
}

ALERT_MESSAGES = {
    "Low": "🟢 LOW LEVEL — Fire detected ({fp}% of area). Routine monitoring.",
    "Medium": "🟡 MEDIUM LEVEL — Fire expanding ({fp}% of area). Active surveillance recommended.",
    "High": "🟠 HIGH ALERT — Significant fire ({fp}% of area). Notify emergency responders.",
    "Critical": "🔴 CRITICAL ALERT — Large-scale fire ({fp}% of area). Immediate emergency protocol activation.",
}


@st.cache_resource
def load_model(weights_path: str) -> YOLO:
    """
    Loads the trained YOLOv26n-seg model. Cached with st.cache_resource
    so the model isn't reloaded on every user interaction.

    Args:
        weights_path: Path to the trained checkpoint (best.pt).

    Returns:
        YOLO instance ready for inference.

    Raises:
        FileNotFoundError: If the checkpoint doesn't exist.
    """
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")
    return YOLO(weights_path)


def classify_severity(fire_percentage: float) -> str:
    """
    Assigns a discrete severity level based on the Fire Percentage.

    Args:
        fire_percentage: FP value in the range [0, 100].

    Returns:
        Severity label.
    """
    for level, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= fire_percentage < hi:
            return level
    return "Critical"


def process_image(model: YOLO, pil_image: Image.Image, conf_threshold: float):
    """
    Runs full inference on an image and returns the results along with
    the annotated image.

    Args:
        model: Already-loaded YOLO instance.
        pil_image: Input image in PIL format.
        conf_threshold: Confidence threshold for inference.

    Returns:
        Tuple (annotated_image_rgb, fire_percentage, level, num_instances).

    Raises:
        RuntimeError: If inference fails.
    """
    try:
        frame = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        height, width = frame.shape[:2]
        total_pixels = height * width

        result = model.predict(source=frame, conf=conf_threshold, verbose=False)[0]

        combined_mask = np.zeros((height, width), dtype=np.uint8)
        num_instances = 0
        if result.masks is not None:
            for m in result.masks.data.cpu().numpy():
                m_resized = cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                combined_mask = np.logical_or(combined_mask, m_resized).astype(np.uint8)
                num_instances += 1

        fire_pixels = int(np.count_nonzero(combined_mask))
        fp = round((fire_pixels / total_pixels) * 100, 2) if total_pixels > 0 else 0.0
        level = classify_severity(fp)
        color_rgb = SEVERITY_COLORS_RGB[level]
        color_bgr = color_rgb[::-1]

        overlay = frame.copy()
        overlay[combined_mask == 1] = color_bgr
        blended = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

        return blended_rgb, fp, level, num_instances

    except Exception as exc:
        raise RuntimeError(f"Error while processing the image: {exc}") from exc


def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(page_title="FireQuant", page_icon="🔥", layout="centered")

    st.title("🔥 FireQuant")
    st.caption("Intelligent Monitoring and Analytical Quantification of Wildfires")

    with st.sidebar:
        st.header("Settings")
        weights_path = st.text_input("Model path (best.pt)", value="best.pt")
        conf_threshold = st.slider("Confidence threshold", min_value=0.05, max_value=0.9, value=0.15, step=0.05)
        st.markdown("---")
        st.subheader("Severity thresholds")
        for level, (lo, hi) in SEVERITY_THRESHOLDS.items():
            st.markdown(f"**{level}**: {lo}% – {hi}%")

    uploaded_file = st.file_uploader("Upload a wildfire image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)

        try:
            model = load_model(weights_path)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()

        with st.spinner("Analyzing image..."):
            try:
                annotated_image, fp, level, num_instances = process_image(model, pil_image, conf_threshold)
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original image")
            st.image(pil_image, use_container_width=True)
        with col2:
            st.subheader("Fire segmentation")
            st.image(annotated_image, use_container_width=True)

        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        c1.metric("Fire Percentage (FP)", f"{fp}%")
        c2.metric("Severity level", level)
        c3.metric("Instances detected", num_instances)

        message = ALERT_MESSAGES.get(level, "Unknown level").format(fp=fp)

        if level in ("High", "Critical"):
            st.error(message)
        elif level == "Medium":
            st.warning(message)
        else:
            st.success(message)

    else:
        st.info("Upload an image to start the analysis.")


if __name__ == "__main__":
    main()
