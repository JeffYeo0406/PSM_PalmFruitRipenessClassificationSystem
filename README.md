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
* `experiment_log.csv`: Comprehensive history of every hyperparameter trial.

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
```text
├── saved_models/
│   ├── palm_ripeness_best_*.h5
│   ├── palm_ripeness_best_*.tflite
│   ├── labels_*.json
│   ├── confusion_matrix_*.png
│   ├── classification_metrics_*.png
│   └── experiment_log.csv
├── Test1.ipynb          # Core training, fine-tuning, and evaluation notebook
├── Deployment.ipynb     # TFLite conversion and Pi packaging logic
└── pi_inference.py      # Optimized script for edge execution
