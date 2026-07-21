# Wildfire Fire Level Estimation

An intelligent monitoring and analytical quantification system for wildfires based on computer vision. The project uses **YOLOv26n-seg** to segment fire at the pixel level, calculate the percentage of area occupied by flames (*Fire Percentage*), and classify fire severity into discrete levels (Low, Medium, High, Critical), generating automatic real-time alerts.

## What does this project do?

```
Image / Video → Segmentation (YOLOv26n-seg) → Fire mask →
Pixel count → Fire Percentage (FP) → Severity level → Alert
```

**Core formula:**
```
FP = (Fire pixels / Total pixels in the image) × 100
```

## Repository structure and execution environment

⚠️ **Important:** this project was developed across two different environments, and not every script runs in the same place. Before running any file, check which environment it was designed for:

| Category | Recommended environment | Why |
|---|---|---|
| **Dataset download and preparation** scripts | Google Colab | They need to download large datasets (D-Fire, ~5.3GB) and run SAM on GPU; they rely on `google.colab.drive` for persistence in Google Drive |
| **Training** scripts | Google Colab (GPU) | The model was trained in Colab (T4 → L4) due to the GPU requirement; the code itself has no Colab-exclusive dependency, but training on a local CPU would be extremely slow |
| **Demos and inference** (`app.py`, `webcam_demo.py`) | **Local** (your computer) | Designed to run locally: the Streamlit app and the webcam demo need direct access to hardware (webcam) or a local server, and don't work the same way inside Colab |
| `inference.py` | Both | Environment-agnostic; works the same in Colab or locally, as long as it points to the correct path for `best.pt` |

### About Google Drive paths

Several download/training scripts (`download_dfire.py`, `combine_datasets.py`, `train_model.py`) use paths like `/content/drive/MyDrive/...`. **These paths only exist inside a Google Colab session with Drive mounted** (`from google.colab import drive; drive.mount('/content/drive')`). If you copy these scripts to your local computer, you'll need to:
1. Remove or comment out the `drive.mount(...)` part.
2. Replace the `/content/drive/MyDrive/...` paths with a local path of your choice (e.g. `C:/Users/your_user/wildfire_data/...`).

## File descriptions

| File | Environment | Function |
|---|---|---|
| `main.ipynb` | Colab | Main notebook where the full flow (dataset download, dataset combination, training, and results evaluation) is executed end to end. |
| `download_dfire.py` | Colab | Downloads the D-Fire dataset from its Kaggle mirror. |
| `dfire_sam_to_yolo_seg.py` | Colab (GPU) | Generates pseudo-segmentation masks for D-Fire using SAM (Segment Anything Model), guided by the original bounding boxes. |
| `diagnose_dfire_labels.py` | Colab | Debugging script: prints the raw content of a sample of label files to verify annotation format and image/label path mapping. |
| `diagnose_dfire_selection.py` | Colab | Debugging script: replicates the image selection logic used in the processing pipeline to verify why certain images were or weren't picked up. |
| `verify_sam_masks.py` | Colab | Visual inspection of SAM-generated mask quality before using them in training. |
| `combine_datasets.py` | Colab | Combines the manually-annotated Roboflow dataset with the D-Fire+SAM dataset into a single YOLO-seg training structure. |
| `train_model.py` | Colab (GPU) | Trains YOLOv26n-seg on the combined dataset, with a reinforced data augmentation strategy to mitigate overfitting. |
| `inference.py` | Colab or Local | Inference pipeline: loads the model, calculates the Fire Percentage, classifies severity, and generates the alert message. |
| `process_test_batch.py` | Colab or Local | Runs inference over a full folder of test images and generates a summary table + annotated images. |
| `generate_result_charts.py` | Colab or Local | Generates training evolution charts (mAP, losses), Fire Percentage per image, the qualitative Original→Mask→Level sequence, and the severity distribution chart. |
| `app.py` | **Local** | Streamlit web application: upload an image and visualize the segmentation, Fire Percentage, severity level, and alert. |
| `webcam_demo.py` | **Local** | Real-time demo using the computer's webcam: processes live video, calculates FP per frame, triggers a visual and audible alarm at high severity levels, and plots the temporal evolution when finished. |
| `best.pt` | Both | Weights of the already-trained YOLOv26n-seg model, ready for inference. |

## How to run the demos (local)

### Requirements
```bash
pip install ultralytics opencv-python matplotlib streamlit
```
On Windows, the audible alarm in `webcam_demo.py` uses `winsound` (already included with Python, no extra installation needed). On other operating systems, the audible alarm is automatically disabled and only the visual alert remains.

### Web app (Streamlit)
```bash
python -m streamlit run app.py
```
Opens at `http://localhost:8501`. Upload an image and view the result.

### Live webcam demo
```bash
python webcam_demo.py
```
Point the camera at a fire image or video. Press `q` to exit and generate the temporal evolution chart.

Both scripts expect to find `best.pt` in the same folder (or you can specify the path from the app's sidebar, or by editing the corresponding variable in the webcam script).

## How to test the model with new images

`process_test_batch.py` expects a folder of test images **separate from the training dataset folders** (to honestly evaluate generalization, not images the model already saw). You need to create this folder and populate it with images before running the script.

### In Google Colab
The code to upload images and create this folder is already included in `main.ipynb` (test image upload section).

### Locally
Simply create a folder (e.g. `test_images/`) on your computer and copy the images you want to test into it (`.jpg`, `.jpeg`, or `.png` formats). Then point `images_dir` in `process_test_batch.py` to that path.

In both cases, the number of images is flexible (the project was validated with 18), but it's recommended to use varied images (day/night, different distances, different fire intensity levels) to get a representative evaluation.

## Dataset and methodology

- **Roboflow** (201 images): original dataset, manually annotated with segmentation masks.
- **D-Fire + SAM** (5,822 images): public fire detection dataset (originally only bounding boxes); segmentation masks were automatically generated using SAM (Segment Anything Model, Meta AI), using the boxes as geometric guidance.
- **Combined total:** 6,023 images, used to retrain YOLOv26n-seg and mitigate the overfitting observed with the original single-video dataset.

## Results

Evaluated on the independent test set (589 of 603 valid images, never used for training or hyperparameter selection):

| Metric (mask, test set) | Value |
|---|---|
| Precision | 0.686 |
| Recall | 0.510 |
| mAP50 | 0.574 |
| mAP50-95 | 0.287 |

## Severity thresholds

| Level | Fire Percentage range |
|---|---|
| Low | 0.0% – 3.0% |
| Medium | 3.0% – 10.0% |
| High | 10.0% – 25.0% |
| Critical | 25.0% – 100.0% |

## Known limitations

- The model tends to underestimate the fire area in large-scale panoramic scenes, since the training dataset is mostly composed of fire seen at medium distance (a *domain gap* documented in the accompanying paper).
- SAM-generated pseudo-masks have not been quantitatively validated (IoU) against manual annotation.
- The temporal analysis module for real video (Fire Percentage evolution over an actual wildfire sequence, beyond the webcam demo) is still under development.
- D-Fire represents ~96.7% of the combined dataset versus ~3.3% for Roboflow, introducing a potential bias toward D-Fire's visual style that was not corrected via oversampling.

## Note on AI assistance

Parts of this project — including code translation to English, docstring writing, refactoring for coding best practices, and editing of this README — were developed with the assistance of an AI tool. All methodological decisions, experiments, and results were reviewed and validated by the author.

## Credits

- Segmentation architecture: Jocher, G., Qiu, J., Liu, M., Lyu, S., Akyon, F. C., & Kalfaoglu, M. E. (2026). *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models*. [arXiv:2606.03748](https://arxiv.org/abs/2606.03748)
- Pseudo-mask generation: Kirillov, A., Mintun, E., Ravi, N., et al. (2023). *Segment Anything*. Proceedings of ICCV. [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- D-Fire dataset: de Venâncio, P. V. A. B., Lisboa, A. C., & Barbosa, A. V. (2022). *An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices*. Neural Computing and Applications, 34(18), 15349–15368. Accessed via a public Kaggle mirror (`sayedgamal99/smoke-fire-detection-yolo`).
- Original manually-annotated dataset: [Fire and Smoke Segmentation Dataset](https://universe.roboflow.com/roboflow-universe-projects/fire-and-smoke-segmentation), Roboflow Universe Projects, 2026.
