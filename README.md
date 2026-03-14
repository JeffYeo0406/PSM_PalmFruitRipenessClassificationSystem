# PSM_PalmFruitRipenessClassificationSystem
PSM project ongoing developement

## Overview
- End-to-end transfer learning with MobileNetV2 for three ripeness classes (Underripe, Ripe, Overripe).
- Clean splits: train/validation from the training set; test set held out for final evaluation only.
- Deploys to Raspberry Pi 4B via TensorFlow Lite (float32 and float16), with optional INT8.

## Architecture
- **Data layer (tf.data):** `image_dataset_from_directory` with `validation_split`; `preprocess_input` scales to [-1, 1]; map + prefetch for throughput.
- **Backbone:** MobileNetV2 (ImageNet weights), frozen during warm-up; train-time augmentation (RandomFlip, RandomRotation, RandomZoom).
- **Head:** GlobalAveragePooling2D → Dropout(0.3) → Dense(128, ReLU) → BatchNorm → Dropout(0.3) → Dense(num_classes, softmax).
- **Training loop:** Adam 1e-4 for warm-up; 1e-5 for fine-tuning (top ~30 layers unfrozen); callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint; optional smoke test.
- **Evaluation:** Held-out test set once; metrics = accuracy, per-class precision/recall/F1, confusion matrix; exports PNG plots and Excel report.
- **Experiment logging:** Appends run metadata/metrics to `experiment_log.csv` (mode, epochs, batch, val split, LRs, backbone freeze state, dataset sizes, accuracy/F1).
- **Per-file analysis:** Helper builds `(image, one_hot, filepath)` datasets for per-image inspection; normalizes paths for Windows.

## Training and Inference Flow
1) Configure paths, image size, batch size, val split, seeds, and run flags.
2) Ingest and split training data; build train/val datasets; load optional test dataset.
3) Preprocess with MobileNetV2 `preprocess_input`; map and prefetch.
4) Build model with augmentation + frozen MobileNetV2 + lightweight head.
5) Train with callbacks; optionally fine-tune top backbone layers at lower LR.
6) Evaluate on the held-out test set; generate reports/plots; log to CSV.
7) Deploy: convert best `.h5` to TFLite (float32, float16; optional INT8 with representative data); emit `labels_*.json` and Pi inference script.

## Deployment (Raspberry Pi 4B)
- Use float16 TFLite for best speed/accuracy on CPU.
- INT8 needs a representative dataset; enable `DO_INT8=True` and supply calibration samples.
- Pi script prefers LiteRT; falls back to tflite-runtime; preprocessing matches training (`preprocess_input`, 224x224 RGB).

## Quality and Reproducibility
- Data leakage avoided via train/val split and single-use test set.
- Validation-driven callbacks and augmentation curb overfitting.
- Timestamped artifacts and CSV logs maintain run traceability.
- For stricter repeatability, set global seeds (Python, NumPy, TF) and enable deterministic cuDNN where needed.
