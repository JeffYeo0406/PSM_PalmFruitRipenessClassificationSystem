# Palm Fruit Ripeness Classification System (CNN-MobileNetV2)

## 📌 Project Overview
This project addresses the critical need for objective, non-destructive ripeness grading in the palm oil industry. Transitioning from the initial PSM1 Proposal (which explored AlexNet/ResNet), this final implementation utilizes a highly optimized **MobileNetV2** architecture to achieve high-accuracy classification suitable for real-time deployment on edge hardware.

* **Aim:** Automate the classification of palm fruit into distinct ripeness categories.
* **Target Classes:** Underripe, Ripe, and Overripe.
* **Key Achievement:** Current best model achieves **~92.8% accuracy** with a **0.928 Macro F1-score**.

---

## 🏗️ System Architecture & Pipeline
The system follows a rigorous deep learning pipeline designed to prevent data leakage and ensure model generalization.

### 1. Data Engineering
* **Split Strategy:** Stratified 80/20 Training/Validation split with a strictly held-out Test set.
* **Pipeline:** `tf.data` API implementation with `parallel_map` and `prefetch` for optimal GPU/CPU throughput.
* **Preprocessing:** Images are resized to 224×224 and normalized to the [-1, 1] range via the MobileNetV2 `preprocess_input` standard.

### 2. Model Configuration
* **Backbone:** MobileNetV2 (Pre-trained on ImageNet).
* **Regularization:** In-model Data Augmentation (Random Flip, Rotation, Zoom) + Dropout (0.3) + Batch Normalization.

### 3. Training Phases
* **Warm-up:** Frozen backbone, Adam optimizer (η = 1e-4).
* **Fine-tuning:** Top 30 layers unfrozen, reduced learning rate (η = 5e-6).

---

## 📊 Performance & Validation
The model is evaluated using a dedicated test set that was never seen during the training or validation cycles.

### Latest Experimental Results (Run: 20260311_170850)
* **Accuracy:** 92.78%
* **Macro Precision:** 0.9296
* **Macro Recall:** 0.9278
* **Macro F1-Score:** 0.9282

**Artifacts Generated:**
* `confusion_matrix_*.png`: Visualizes class-wise misclassifications.
* `classification_metrics_*.png`: Provides a per-class breakdown of performance.
* `classification_report_*.xlsx`: Tabular classification report and confusion-matrix sheets for formal analysis/export.
* `Pipeline_Demonstration_Full_Report.docx`: End-to-end pipeline report with embedded figures and summary.
* `experiment_log.csv`: Comprehensive history of hyperparameter runs, configurations, and outcomes.

---

## 🚀 Deployment (Edge AI)
The system is designed for the **Raspberry Pi 4B (4GB)** using the **Camera Module 3**.

### Conversion & Optimization
The pipeline automatically converts the best `.h5` checkpoint into three TensorFlow Lite formats:
* **FP32:** Baseline precision.
* **FP16:** Recommended for Pi 4/5 CPU (optimized for ARMv8).
* **INT8:** Full integer quantization for maximum speed (requires representative dataset).

### Hardware Setup
* **Runtime:** LiteRT (formerly `tflite-runtime`).
* **Performance Target:** 2–4 FPS on Raspberry Pi CPU.

### Flask API on Raspberry Pi
Use the lightweight Flask service to accept multipart image uploads from a phone and return the top-1 ripeness label.

1) Export TFLite models (dev machine)
	- `python scripts/convert_tflite.py --h5 models/palm_ripeness_best_20260311_190150.h5 --rep-data /path/to/train --output-dir models [--labels labels.json]`
	- Outputs: `*_fp32.tflite`, `*_float16.tflite`, and `*_int8.tflite` (INT8 always produced; labels auto-detected from rep-data if not provided) plus a timestamped labels JSON and manifest.

2) Install Raspberry Pi runtime deps
	- `pip install -r requirements-pi.txt`

3) Start the API
	- `MODEL_PATH=/home/pi/models/palm_ripeness_best_<ts>_int8.tflite LABELS_PATH=/home/pi/models/labels_<ts>.json python api/app.py`
	- If `MODEL_PATH`/`LABELS_PATH` are not set, the API auto-discovers the newest matching artifacts in `models/`.
	- Compatibility fallback is built in: legacy `saved_models/` and root-level paths are also supported automatically.
	- Endpoints: `/health` (GET), `/classify` (POST multipart form with `file` field), `/result/<request_id>` (GET).
	- If artifacts are missing, `/health` returns `"status": "degraded"` and `/classify` returns HTTP 503 until model files are available.

4) Call from a phone or client
	- `curl -X POST -F "file=@/path/to/fruit.jpg" http://<pi-ip>:5000/classify`
	- JSON response: `{ "request_id": "ab12cd34", "label": "Ripe", "probability": 0.94, "index": 1, "latency_ms": 320.5, "runs": 3, "result_path": "/result/ab12cd34" }`
	- Optional fetch-by-id: `curl http://<pi-ip>:5000/result/ab12cd34`

5) CLI inference (no Flask)
	- `python pi_inference.py --model models/palm_ripeness_best_<ts>_int8.tflite --labels models/labels_<ts>.json --image sample.jpg --runs 5 --warmup 1`

### Pre-Inference Input Gate
Before ripeness classification, the runtime performs a quick hard-reject gate to avoid invalid predictions:

- Rejects low-quality captures (too small, blurry, low-contrast, over/under-exposed).
- Optionally rejects likely non-palm images using a lightweight binary TFLite model.
- Applies consistently to both API (`/classify`) and CLI (`pi_inference.py`).

API behavior:
- Rejected inputs return HTTP 422 with a structured reason (`error_code`, `error`, `hint`, and `details`).

CLI behavior:
- Rejected inputs print a structured JSON error and exit with non-zero status.

Environment variables
- `MODEL_PATH` and `LABELS_PATH`: override model/labels used by the API and CLI defaults.
- `WARMUP_RUNS`, `RUNS`: control warmup and timing iterations.
- `MAX_UPLOAD_MB`: upload size guardrail for the API (default 5 MB).
- `RESULT_TTL_SECONDS`: in-memory retention duration for `/result/<request_id>` entries (default 1800).
- `PORT`: HTTP port (default 5000).
- `MIN_IMAGE_WIDTH`, `MIN_IMAGE_HEIGHT`: minimum accepted image dimensions for quality gate.
- `MIN_ASPECT_RATIO`, `MAX_ASPECT_RATIO`: accepted aspect-ratio window.
- `MIN_BRIGHTNESS`, `MAX_BRIGHTNESS`: brightness range used to reject over/under-exposed captures.
- `MIN_CONTRAST_STD`: minimum grayscale standard deviation (low values are rejected as low contrast).
- `MIN_SHARPNESS`: minimum edge-strength score used to reject blurry images.
- `ENABLE_PALM_BINARY_GATE`: enable optional palm-vs-nonpalm gate (`true/false`).
- `PALM_BINARY_MODEL_PATH`: path to optional palm-presence binary TFLite model.
- `PALM_BINARY_THRESHOLD`: minimum palm probability required to continue to ripeness classification.
- `PALM_BINARY_PALM_INDEX`: palm class index for multi-logit binary outputs (default 1).
 
Security note: API is intended for LAN use; do not expose publicly without HTTPS and authentication.

---

## 📈 Comparison: Proposal vs. Implementation

| Feature | PSM1 Proposal | Final Implementation |
| :--- | :--- | :--- |
| **Model** | AlexNet / ResNet50 | **MobileNetV2** (Optimized for Edge) |
| **Hardware** | RPi 4 / OV5647 | **RPi 4B / Camera Module 3** |
| **Evaluation** | Basic Accuracy | **Macro F1 / Confusion Matrix / CSV Logging** |
| **Quantization** | Conceptual | **FP16 & INT8 Fully Integrated** |
| **Classification Scope** | 4 classes (Unripe, Underripe, Ripe, Overripe) | **3 classes** (Optimized for field usability) |
| **Validation** | Basic metrics | **Comprehensive automated reporting** |

---

## 🛠️ Future Improvements
1. **System Integration & Architecture Expansion:** Future iterations will focus on seamless end-to-end edge integration by connecting the Raspberry Pi inference pipeline to a lightweight mobile application for real-time, on-site monitoring. Additionally, alternative model architectures such as **YOLO** (for bounding-box object detection) or **ResNet50** will be benchmarked against the current MobileNetV2 baseline to evaluate accuracy-latency trade-offs.
2. **Dataset Expansion & MPOB Standard Alignment:** To fully align with the Malaysian Palm Oil Board (MPOB) official grading standards, the dataset will be expanded and re-annotated to support the comprehensive 4-class format: **Unripe, Underripe, Ripe, and Overripe**. This will enhance the system's industrial applicability and grading resolution.

---

## 📂 Repository Structure

### Current Structure (Latest, Full Migration)
```text
PSM_PalmFruitRipenessClassificationSystem/
├── api/
│   └── app.py
├── examples/
│   └── test_api.py
├── scripts/
│   └── convert_tflite.py
├── notebooks/
│   ├── Test1.ipynb
│   ├── Deployment.ipynb
│   └── Pipeline_Demo.ipynb
├── models/
│   ├── palm_ripeness_best_20260311_170850.h5
│   ├── palm_ripeness_best_20260311_170850_float32.tflite
│   ├── palm_ripeness_best_20260311_170850_float16.tflite
│   └── labels_20260311_170850.json
├── reports/
│   ├── experiment_log.csv
│   ├── confusion_matrix_20260311_170850.png
│   ├── classification_metrics_20260311_170850.png
│   ├── classification_report_20260311_170850.xlsx
│   └── Pipeline_Demonstration_Full_Report.docx
├── saved_models/                 # Optional legacy location (supported by compatibility resolver)
├── path_compat.py               # Tiny path-compatibility resolver for legacy/new paths
├── pi_inference.py
├── requirements-dev.txt
├── requirements-pi.txt
└── README.md
```

### Compatibility Note
This full migration remains safe for existing workflows:

- Canonical artifact locations are now `notebooks/`, `models/`, and `reports/`.
- A tiny compatibility layer (`path_compat.py`) allows runtime code to resolve legacy paths (`saved_models/` and previous root-style paths) automatically.
- Users can switch in one step by using the new canonical `models/` paths in commands; old path conventions continue to resolve when present.

### Target Structure (Recommended for Full Migration)
This repository now follows the target layout below:

```text
PSM_PalmFruitRipenessClassificationSystem/
├── api/
│   └── app.py
├── scripts/
│   └── convert_tflite.py
├── notebooks/
│   ├── Test1.ipynb
│   ├── Deployment.ipynb
│   └── Pipeline_Demo.ipynb
├── models/
│   ├── palm_ripeness_best_*.h5
│   ├── palm_ripeness_best_*_float32.tflite
│   ├── palm_ripeness_best_*_float16.tflite
│   └── labels_*.json
├── reports/
│   ├── experiment_log.csv
│   ├── classification_metrics_*.png
│   ├── confusion_matrix_*.png
│   ├── classification_report_*.xlsx
│   └── Pipeline_Demonstration_Full_Report.docx
├── examples/
│   └── test_api.py
├── pi_inference.py
├── requirements-dev.txt
├── requirements-pi.txt
└── README.md
```

This structure separates model artifacts, evaluation outputs, notebooks, and production code, making maintenance and deployment workflows easier.
