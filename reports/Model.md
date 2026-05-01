# Model Selection and Comparison Plan for PSM

## 1. Purpose of This Document
This file defines the selected CNN candidates, justification for each model, and an execution-ready plan to:
- integrate each model into the current TensorFlow to TFLite pipeline,
- benchmark them fairly on the same dataset/protocol,
- choose one final deployment model for Raspberry Pi/mobile runtime.

This is a planning and decision document, not only a theory summary.

## 2. Current System Baseline (Measured)
Current production baseline is MobileNetV2 with measured results from project artifacts:
- Test accuracy: 92.78% (167/180)
- Macro F1: 0.9282
- INT8 test accuracy: 92.22% (166/180)
- INT8 drop vs FP32: 0.60%
- INT8 model size: 2.76 MB

Existing system integration references:
- Training history: reports/experiment_log.csv
- Conversion flow: scripts/convert_tflite.py
- Validation flow: scripts/validate_tflite.py
- Runtime and API integration: scripts/pi_inference.py and api/app.py

## 3. Why These Four Models Are Selected
The four models cover the required accuracy-efficiency spectrum for edge AI deployment.

| Model | Selection role | Why selected | Main risk |
|---|---|---|---|
| MobileNetV2 | Baseline anchor | Already proven and integrated in this repository | May be outperformed by newer architectures |
| MobileNetV3 | Balanced candidate | Better latency-accuracy tradeoff with SE and h-swish | Needs careful preprocessing alignment |
| EfficientNetB0 | Accuracy-oriented candidate | Strong feature extraction and often high top-1 accuracy | Higher compute and possible Pi latency increase |
| ShuffleNetV2 | Efficiency-oriented candidate | Designed for low memory access cost and fast edge throughput | Not currently implemented in this repo pipeline |

## 4. Model-by-Model Summary and Justification

### 4.1 MobileNetV2
Purpose in this PSM:
- Benchmark model and compatibility reference.

Technical strengths:
- Lightweight depthwise separable convolution design.
- Already validated in this system with stable metrics and TFLite deployment.

Limitations:
- Does not include newer architecture optimizations such as SE attention.

Justification:
- Required as control model for fair comparison because all other candidates must beat or match it on targeted metrics.

### 4.2 MobileNetV3
Purpose in this PSM:
- Primary candidate for final deployment.

Technical strengths:
- Hardware-aware NAS design.
- Includes Squeeze-and-Excitation blocks and hard-swish activation.

Limitations:
- Input preprocessing must be validated carefully to avoid mismatch during training and TFLite inference.

Justification:
- Most likely to improve real-time edge performance while keeping model size and integration complexity manageable.

### 4.3 EfficientNetB0
Purpose in this PSM:
- High-accuracy reference model.

Technical strengths:
- Compound scaling (depth, width, resolution) gives a strong accuracy profile.

Limitations:
- More compute-intensive than MobileNet variants and may increase latency on Raspberry Pi CPU.

Justification:
- Needed to evaluate whether accuracy gains justify runtime overhead for practical field deployment.

### 4.4 ShuffleNetV2
Purpose in this PSM:
- Runtime efficiency challenger.

Technical strengths:
- Optimized for low memory access cost and efficient channel flow.

Limitations:
- Not yet integrated in current repository; implementation path must be finalized before experiments.

Justification:
- Important to test a model explicitly designed for constrained hardware efficiency, especially for thermal and sustained runtime behavior.

## 5. Integration Plan with Current System

### 5.1 Common Integration Target
All candidate models must end in the same artifact/runtime interfaces:
- H5 checkpoint output
- TFLite exports: fp32, float16, int8
- labels timestamp json and manifest json
- API compatibility through MODEL_PATH and LABELS_PATH
- Inference compatibility with scripts/pi_inference.py and api/app.py

### 5.2 Existing Pipeline Mapping
Current integration pipeline to reuse for every model:
1. Train model backbone with same dataset split policy.
2. Export best checkpoint to h5.
3. Convert using scripts/convert_tflite.py.
4. Validate using scripts/validate_tflite.py.
5. Deploy selected TFLite artifact via API and CLI smoke checks.
6. Log experiment outcome to reports/experiment_log.csv and summary tables in this file.

### 5.3 Preprocessing Compatibility Rule
For each backbone, preprocessing must be explicitly verified during:
- dataset pipeline,
- TFLite representative dataset generation,
- runtime preprocessing in scripts/pi_inference.py.

No model is accepted if preprocessing assumptions differ between train, convert, and inference paths.

## 6. Fair Comparison Protocol

### 6.1 Fixed Experimental Settings (Must Be Identical)
- Same train/validation/test split and random seed
- Same input image size
- Same augmentation policy
- Same batch size
- Same optimizer and loss configuration
- Same warm-up/fine-tuning policy
- Same conversion options for fp32/float16/int8
- Same test dataset and evaluation script

### 6.2 Mandatory Metrics to Record
- Accuracy
- Macro precision
- Macro recall
- Macro F1
- TFLite file size (MB)
- Inference latency on Raspberry Pi (ms/image)
- Throughput (FPS)
- Thermal observation under continuous run

### 6.3 Acceptance Gates
- INT8 accuracy drop should be below 2% relative drop from corresponding fp32 model
- API and CLI inference must run without runtime errors
- Model must be compatible with current labels/manifest handling

## 7. Step-by-Step Implementation Guide per Model

### 7.1 MobileNetV2 (Baseline Re-run)
1. Reuse current MobileNetV2 training template and fixed protocol settings.
2. Train warm-up phase and optional fine-tune phase using the same data split.
3. Save best checkpoint as palm_ripeness_best_<timestamp>.h5.
4. Convert with scripts/convert_tflite.py to fp32, float16, int8 artifacts.
5. Validate with scripts/validate_tflite.py against test dataset.
6. Deploy int8 artifact via MODEL_PATH and LABELS_PATH and run smoke test through api/app.py and scripts/pi_inference.py.
7. Record final metrics in reports/experiment_log.csv and section 8 table.

### 7.2 MobileNetV3
Execution status (April 2026): completed with 7-flow reproduction.

1. Kept backbone at MobileNetV3Small and matched baseline protocol using `scripts/run_mobilenetv3_repro.py`.
2. Ran all 7 historical flow profiles top-to-bottom and logged results to `reports/experiment_log_mobilenetv3_repro.csv`.
3. Selected best profile by accuracy (tie-breaker macro F1): `05_full_train_e30`.
4. Best checkpoint selected for conversion: `saved_models/palm_ripeness_best_20260420_185817.h5`.
5. Primary `scripts/convert_tflite.py` deserialization path was blocked by Keras 3 and legacy H5 config incompatibility for this checkpoint.
6. Applied fallback conversion path with `scripts/extract_and_convert.py` (rebuild architecture + load weights by name) to generate fp32/float16/int8 + labels + manifest.
7. Path A validation (`models/tflite_manifest_20260420_200930.json`): FP32 88.89%, INT8 82.78%, relative drop 6.88% -> FAIL.
8. Path B applied deterministic balanced PTQ calibration (target 500 / actual 501 / seed 42), then revalidated (`models/tflite_manifest_20260421_022121.json`): FP32 88.89%, INT8 84.44%, relative drop 5.00% -> FAIL.
9. Path C executed QAT fine-tuning and QAT-export validation, then measured: FP32 93.33%, INT8 57.22%, relative drop 38.69% -> FAIL. Transient failed-attempt QAT artifacts were later cleaned up.
10. Additional QAT conversion attempt through `scripts/convert_tflite.py` failed in this environment with `Could not locate class 'Functional'` (`reports/qat_convert_20260421_0328.log`).
11. Branch decision: treat MobileNetV3 INT8 as non-effective in current stack and use FP16 artifact for deployment path.
12. Selected MobileNetV3 deployment artifact: `models/palm_ripeness_best_20260421_022121_float16.tflite` with `models/labels_20260421_022121.json`.

#### 7.2.1 Preprocessing ablation — detailed findings and recommendations

Summary of findings:
- A controlled ablation study (see `preprocessing_ablation_study.docx` and `reports/experiment_log_mobilenetv3_repro.csv`) demonstrated that applying both the MobileNetV3 built-in preprocessing layer and an external `preprocess_input_mobilenet_v3()` step in the dataset pipeline produces effectively doubled input scaling. The canonical symptom was the smoke test result: `01_smoke_test` produced 0.0938 accuracy when the preprocessing mismatch was present, whereas properly aligned runs (single preprocessing source) restored expected reproduction accuracies (best profile `05_full_train_e30` at 0.8889).

Root cause analysis:
- MobileNetV3 backbones can include a preprocessing layer that scales inputs internally to the model's expected range. If the dataset pipeline also calls `preprocess_input_*` before batching, the same transformation is applied twice (scale/shift), shifting inputs outside the distribution seen during training and causing catastrophic quality loss on validation.
- This mismatch affects training, representative dataset generation for PTQ calibration, and runtime inference consistency — all three must agree.

Quantitative impact (observed):
- Double-preprocessing produced a near-random smoke-test accuracy (0.0938) on the small verification run; with the preprocessing layer disabled (i.e., `include_preprocessing_layer=False`) and dataset preprocessing performed exactly once, reproduction runs recovered expected metrics (profile `05_full_train_e30` at 0.8889). INT8 conversion still failed quality gates across multiple PTQ/QAT strategies, so FP16 remains the chosen deployment format for MobileNetV3.

Recommendations and action items:
1. Pipeline invariant: choose one preprocessing source and enforce it project-wide for a given `preprocess_family`:
	- If `include_preprocessing_layer=True` in `build_model()`, then remove external `preprocess_input_*` calls from `make_datasets()`.
	- If `include_preprocessing_layer=False`, keep dataset-level `preprocess_input_*` and ensure conversion rep-dataset uses the same function.
2. Convert-time alignment: always pass `--preprocess-family mobilenet_v3` (or equivalent) to `scripts/convert_tflite.py` so the representative dataset generation matches the model's expected preprocessing.
3. Runtime alignment: ensure `api/app.py` and `scripts/pi_inference.py` read the model manifest or config and apply the same preprocessing family before inference.
4. Quick code pointer: update `scripts/run_mobilenetv3_repro.py` and `make_datasets()` to accept and propagate `include_preprocessing_layer` so dataset preprocessing toggles together with the model build option (avoid unilateral toggles).
5. Logging: add an explicit `notes` field (already present) that records the `include_preprocessing_layer` boolean and the `preprocess_family` used for every run and conversion manifest.

Implication for model selection:
- The preprocessing mismatch explains the severe early failures seen during some conversion/validation steps. After enforcing a single preprocessing source, the MobileNetV3 reproduction results are consistent and competitive; however, INT8 quantization quality remains unacceptable under the current toolchain and calibration paths, so FP16 is the pragmatic deployment choice while further INT8 work proceeds.

#### 7.2.2 MobileNetV3 implementation — final status and artifacts

**Implementation status:** Complete (April 2026)

The MobileNetV3 track is now fully implemented with:
- 7-flow reproduction completed and logged
- Preprocessing ablation study completed and documented
- Multi-path INT8 quantization explored (PTQ, balanced PTQ, QAT)
- Final artifacts generated and validated

**Primary deployment artifact (FP16):**
- Model: `models/palm_ripeness_best_20260421_022121_float16.tflite` (2.01 MB)
- Labels: `models/labels_20260421_022121.json`
- Manifest: `models/tflite_manifest_20260421_022121.json`
- FP32 accuracy: 88.89% (160/180)
- FP16 accuracy: equivalent to FP32 (no quantization-induced accuracy loss)
- Rationale: FP16 provides ~2x size reduction vs FP32 with no accuracy penalty, making it the optimal choice given INT8 quality gate failures.

**Alternative INT8 artifact (if accuracy drop gate is ignored):**
- Model: `models/palm_ripeness_best_20260421_022121_int8.tflite` (1.32 MB)
- INT8 accuracy: 84.44% (152/180)
- Relative INT8 drop: 5.00% (exceeds 2% gate threshold)
- Size advantage: ~34% smaller than FP16 (1.32 MB vs 2.01 MB)
- Use case: Deploy only when storage/memory constraints are critical and the ~4.5% absolute accuracy loss is acceptable for the application.

**Decision summary:**
- MobileNetV3 FP16 is the **recommended production artifact** for this model track.
- MobileNetV3 INT8 is **available as an alternative** for scenarios where the accuracy drop is tolerable.
- Runtime integration (API and CLI) defaults to the FP16 artifact.
- Future work: explore advanced quantization techniques (e.g., per-channel quantization, mixed precision) to potentially improve INT8 quality.

### 7.3 EfficientNetB0
**Implementation complete (April 2026).** Full 7-flow reproduction executed with same profile settings as MobileNetV3.

**Primary deployment artifact (FP16 recommended over INT8):**
- Best profile: `05_full_train_e30` (30 epochs warm-up, no fine-tuning) — accuracy 87.78%
- Checkpoint: `saved_models/palm_ripeness_best_20260423_172644.h5`
- TFLite artifacts: `models/palm_ripeness_best_20260423_214317_{fp32,float16,int8}.tflite`
- FP32 accuracy: 86.67% (156/180)
- FP16 accuracy: equivalent to FP32 (no quantization-induced accuracy loss)
- **INT8 gate FAIL:** Relative drop 6.41% (exceeds 2% threshold) — INT8 accuracy 81.11% (146/180)
- Rationale: FP16 recommended as primary artifact (no accuracy penalty). INT8 available only if ~5.5% absolute accuracy drop is acceptable and size saving is critical.
- Reproduction log: `reports/experiment_log_efficientnetb0_repro.csv`
- Best-run summary: `reports/efficientnetb0_repro_best_run.json`

**Notable observations:**
- EfficientNetB0 achieves 87.78% (best profile) vs MobileNetV3's 88.89% — slightly lower accuracy
- Fine-tuning profiles (06, 07) underperformed vs warm-up-only (05), suggesting EfficientNetB0 may overfit more easily with unfrozen backbone
- INT8 quantization degraded more severely for EfficientNetB0 (6.41% relative drop) than MobileNetV3 (5.00% relative drop)
- Larger model size than MobileNetV3 — trade-off between accuracy and latency on Pi should be evaluated

### 7.4 ShuffleNetV2
**Implementation complete (April 2026).** ONNX conversion path via PyTorch/timm → ONNX → onnxsim → onnx2tf → TFLite.

1. ✅ Finalized implementation: PyTorch/timm training + ONNX conversion path (Option B).
2. ✅ Output converted to TFLite and integrated with current API/runtime contracts.
3. ✅ Matched same training/evaluation protocol as other models (30 epochs + 15 fine-tune).
4. ✅ Export artifacts with standard naming pattern.
5. ✅ Validated FP32 and INT8 — INT8 accuracy drop: **62.80% (FAIL)**.
6. 🔲 Benchmark edge latency and thermal stability (pending).
7. 🔲 Compare efficiency gains against integration complexity (pending).
8. 🔲 Document implementation overhead as part of final model decision.

**Deployment artifact:** `models/palm_ripeness_best_20260427_162727_float16.tflite` (FP16, 2.888 MB)
**FP32 accuracy:** 91.11%
**INT8 gate:** ❌ FAIL — onnx2tf INT8 pipeline is systematically broken for PyTorch backbones.

## 8. Performance Tracking Tables

### 8.1 Measured Results Table (Fill During Experiments)
| Model | Accuracy | Macro F1 | INT8 Accuracy | INT8 Drop | INT8 Size (MB) | Pi Latency (ms) | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| MobileNetV2 | 92.78% | 0.9282 | 92.22% | 0.60% | 2.76 | To update | Current measured baseline |
| MobileNetV3 | 88.89% | 0.8872 | 84.44% | 5.00% | 1.32 | To update | **Implementation complete.** Primary: FP16 (`palm_ripeness_best_20260421_022121_float16.tflite`). Alternative: INT8 available if 5% drop acceptable. Multi-path INT8 attempts: PTQ 6.88%, balanced PTQ 5.00%, QAT 38.69% — all exceed 2% gate. |
| EfficientNetB0 | 87.78% | 0.8743 | 81.11% | 6.41% | To update | To update | **Implementation complete.** Primary: FP16. INT8 gate FAIL (6.41% relative drop). Best profile: 05_full_train_e30 (87.78%). 7-flow reproduction completed April 2026. |
| ShuffleNetV2 | 91.11% | 0.9085 | 33.89% | 62.80% | 1.633 | To update | **Implementation complete (onnx2tf).** Primary: FP16 (`palm_ripeness_best_20260427_162727_float16.tflite`, 2.888 MB). INT8 gate FAIL (62.80% relative drop — catastrophic onnx2tf INT8 pipeline failure). FP32 Acc: 91.11%, FP16 equivalent. PyTorch/timm → ONNX → onnx_sim → onnx2tf → TFLite. Preprocessing: `imagenet_timm`. |
| ResNet18 | 89.44% | 0.8905 | 35.56% | 60.25% | 11.319 | To update | **Implementation complete (onnx2tf).** Primary: FP16 (`palm_ripeness_best_20260428_193229_float16.tflite`, 22.491 MB). INT8 gate FAIL (60.25% relative drop — catastrophic onnx2tf INT8 pipeline failure). FP32 Acc: 89.44%, FP16 equivalent. PyTorch/torchvision → ONNX → onnx_sim → onnx2tf → TFLite. Preprocessing: `imagenet_torchvision`. |

### 8.1.1 MobileNetV3 7-Flow Reproduction Summary (April 2026)

| Profile | Run Mode | Accuracy | Macro F1 |
|---|---|---:|---:|
| 01_smoke_test | smoke_test | 0.0938 | 0.0857 |
| 02_full_train_e10 | full_train | 0.8389 | 0.8360 |
| 03_full_train_e10_ft5 | full_train + fine_tune | 0.8444 | 0.8426 |
| 04_full_train_e10_ft15 | full_train + fine_tune | 0.8778 | 0.8750 |
| 05_full_train_e30 | full_train | 0.8889 | 0.8872 |
| 06_full_train_e30_ft5 | full_train + fine_tune | 0.8278 | 0.8269 |
| 07_full_train_e30_ft15 | full_train + fine_tune | 0.8778 | 0.8770 |

Reference artifacts:
- Reproduction log: `reports/experiment_log_mobilenetv3_repro.csv`
- Best-run summary: `reports/mobilenetv3_repro_best_run.json`

### 8.1.2 EfficientNetB0 7-Flow Reproduction Summary (April 2026)

| Profile | Run Mode | Accuracy | Macro F1 |
|---|---|---:|---:|
| 01_smoke_test | smoke_test | 0.0000 | 0.0000 |
| 02_full_train_e10 | full_train | 0.8333 | 0.8306 |
| 03_full_train_e10_ft5 | full_train + fine_tune | 0.8667 | 0.8635 |
| 04_full_train_e10_ft15 | full_train + fine_tune | 0.8722 | 0.8690 |
| 05_full_train_e30 | full_train | 0.8778 | 0.8743 |
| 06_full_train_e30_ft5 | full_train + fine_tune | 0.8167 | 0.8161 |
| 07_full_train_e30_ft15 | full_train + fine_tune | 0.8722 | 0.8682 |

Reference artifacts:
- Reproduction log: `reports/experiment_log_efficientnetb0_repro.csv`
- Best-run summary: `reports/efficientnetb0_repro_best_run.json`

### 8.1.3 ShuffleNetV2_x1_0 — ONNX Conversion Summary (April 2026)

| Metric | Value |
|--------|-------|
| **Backbone** | ShuffleNetV2_x1_0 (PyTorch/timm) |
| **Pipeline** | PyTorch/timm → ONNX → onnxsim → onnx2tf → TFLite |
| **Preprocessing** | `imagenet_timm` |
| **Run mode** | full_train + fine_tune (30 epochs + 15 fine-tune) |
| **Batch size** | 32 |
| **FP32 Accuracy** | 91.11% (164/180) |
| **Macro F1** | 0.9085 |
| **INT8 Accuracy** | 33.89% (61/180) |
| **INT8 Relative Drop** | **62.80%** ❌ FAIL |
| **Deployment Format** | **FP16** (fallback) |
| **FP32 Size** | 5.641 MB |
| **FP16 Size** | 2.888 MB |
| **INT8 Size** | 1.633 MB |
| **Deployment Artifact** | `models/palm_ripeness_best_20260427_162727_float16.tflite` |
| **INT8 Gate** | ❌ FAIL — catastrophic onnx2tf INT8 pipeline failure |

**Decision:** FP16 artifact recommended for deployment (equivalent accuracy to FP32, 2.888 MB). INT8 pipeline is broken for onnx2tf path — cannot produce usable INT8 TFLite from PyTorch backbones via this path until conversion quality is resolved.

Reference artifacts:
- Training log: `notebooks/saved_models/experiment_log_onnxS.csv`
- Best model: `saved_models/palm_ripeness_finetuned_20260427_162727.pt`

### 8.1.4 ResNet18 — ONNX Conversion Summary (April 2026)

| Metric | Value |
|--------|-------|
| **Backbone** | ResNet18 (PyTorch/torchvision) |
| **Pipeline** | PyTorch/torchvision → ONNX → onnxsim → onnx2tf → TFLite |
| **Preprocessing** | `imagenet_torchvision` |
| **Run mode** | full_train + fine_tune (30 epochs + 15 fine-tune) |
| **Batch size** | 32 |
| **FP32 Accuracy** | 89.44% (161/180) |
| **Macro F1** | 0.8905 |
| **INT8 Accuracy** | 35.56% (64/180) |
| **INT8 Relative Drop** | **60.25%** ❌ FAIL |
| **Deployment Format** | **FP16** (fallback) |
| **FP32 Size** | 44.966 MB |
| **FP16 Size** | 22.491 MB |
| **INT8 Size** | 11.319 MB |
| **Deployment Artifact** | `models/palm_ripeness_best_20260428_193229_float16.tflite` |
| **INT8 Gate** | ❌ FAIL — catastrophic onnx2tf INT8 pipeline failure |

**Decision:** FP16 artifact recommended for deployment. ResNet18 is significantly larger (22.491 MB FP16) than MobileNetV2/V3 due to its deeper architecture. INT8 conversion shares the same onnx2tf pipeline bug seen with ShuffleNetV2.

Reference artifacts:
- Training log: `notebooks/saved_models/experiment_log_onnx.csv`
- Best model: `saved_models/palm_ripeness_finetuned_20260428_193229.pt`

### 8.2 Estimated Ranges (Planning Only, Not Final Evidence)
| Model | Estimated accuracy range | Estimated latency trend vs MobileNetV2 | Confidence |
|---|---|---|---|
| MobileNetV3 | 92% to 94% | Faster or similar | Medium |
| EfficientNetB0 | 93% to 95% | Slower | Medium |
| ShuffleNetV2 | 91% to 93.5% | Faster | Low to medium |

Note:
- Estimated ranges are planning assumptions only.
- Final selection must use measured values from section 8.1.

## 9. Final Model Decision Rule
Use weighted scoring after all experiments are complete:
- Accuracy and Macro F1: 40%
- Edge latency and FPS: 30%
- Model size and memory efficiency: 20%
- Integration complexity and runtime stability: 10%

Final selected model must satisfy all:
- Pass INT8 drop gate (<2% relative drop), or document an explicit FP16 fallback decision when INT8 repeatedly fails technical/quality gates
- Stable API and CLI runtime behavior
- Practical edge deployment performance for Raspberry Pi/mobile use case

## 10. Action Checklist
1. Lock protocol settings and seed before training new backbones.
2. Implement and train MobileNetV3 first (highest priority candidate).
3. Run EfficientNetB0 benchmark next for accuracy ceiling reference.
4. Finalize ShuffleNetV2 implementation path and run efficiency benchmark.
5. Fill measured table and apply weighted decision rule.
6. Freeze final deployment model and update production documentation.

## 11. Command Templates for Execution
Use these templates for each candidate model run so execution stays consistent.

### 11.1 Convert Trained H5 to TFLite Artifacts
```bash
python scripts/convert_tflite.py \
	--h5 models/palm_ripeness_best_<timestamp>.h5 \
	--rep-data <path_to_training_data_root> \
	--output-dir models
```

Expected outputs:
- models/palm_ripeness_best_<new_timestamp>_fp32.tflite
- models/palm_ripeness_best_<new_timestamp>_float16.tflite
- models/palm_ripeness_best_<new_timestamp>_int8.tflite
- models/labels_<new_timestamp>.json
- models/tflite_manifest_<new_timestamp>.json

### 11.2 Validate FP32 vs INT8 Accuracy Drop
```bash
python scripts/validate_tflite.py \
	--model-fp32 models/palm_ripeness_best_<ts>_fp32.tflite \
	--model-int8 models/palm_ripeness_best_<ts>_int8.tflite \
	--labels models/labels_<ts>.json \
	--data-dir <path_to_test_data_root>
```

Pass condition:
- Relative INT8 drop < 2%

### 11.3 API Runtime Integration Test
```bash
MODEL_PATH=models/palm_ripeness_best_<ts>_<variant>.tflite \
LABELS_PATH=models/labels_<ts>.json \
python api/app.py
```

Current MobileNetV3 decision uses `<variant>=float16`.

Health check:
```bash
curl http://127.0.0.1:5000/health
```

Inference check:
```bash
curl -X POST -F "file=@<sample_image>.jpg" http://127.0.0.1:5000/classify
```

### 11.4 CLI Latency Sampling
```bash
python scripts/pi_inference.py \
	--model models/palm_ripeness_best_<ts>_<variant>.tflite \
	--labels models/labels_<ts>.json \
	--image <sample_image>.jpg \
	--warmup 1 \
	--runs 10
```

Current MobileNetV3 decision uses `<variant>=float16`.

### 11.5 Experiment Logging Rule
After each model run:
1. Append run metrics to reports/experiment_log.csv.
2. Update section 8.1 measured table in this file.
3. Add short notes on integration issues (if any), especially preprocessing mismatches or runtime compatibility errors.
