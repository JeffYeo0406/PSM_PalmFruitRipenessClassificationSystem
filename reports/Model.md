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
1. Replace backbone with MobileNetV3 variant (Small preferred first).
2. Verify model-specific preprocessing configuration and keep it consistent in conversion and runtime.
3. Train with the exact same protocol used for baseline.
4. Export best h5 checkpoint.
5. Convert to fp32/float16/int8 using the same conversion script and representative dataset policy.
6. Validate with the same test dataset and script.
7. Integrate into API runtime and compare latency/accuracy versus MobileNetV2.
8. Update results tables with measured values and decision notes.

### 7.3 EfficientNetB0
1. Replace backbone with EfficientNetB0 while retaining classifier head policy as close as possible.
2. Confirm preprocessing compatibility before full training.
3. Train under identical protocol constraints.
4. Export best h5 checkpoint.
5. Convert to fp32/float16/int8 with same conversion settings.
6. Validate on the same test set.
7. Run runtime benchmark on Raspberry Pi to quantify latency and thermal behavior.
8. Record whether accuracy gain compensates for runtime cost.

### 7.4 ShuffleNetV2
1. Finalize implementation path:
	- Option A: TensorFlow-compatible ShuffleNetV2 implementation.
	- Option B: PyTorch training path with additional conversion workflow.
2. Ensure output can be converted to TFLite and integrated with current API/runtime contracts.
3. Match the same training/evaluation protocol as other models.
4. Export artifacts to the same naming pattern.
5. Validate fp32/int8 and quantify accuracy drop.
6. Benchmark edge latency and thermal stability.
7. Compare efficiency gains against integration complexity.
8. Document implementation overhead as part of final model decision.

## 8. Performance Tracking Tables

### 8.1 Measured Results Table (Fill During Experiments)
| Model | Accuracy | Macro F1 | INT8 Accuracy | INT8 Drop | INT8 Size (MB) | Pi Latency (ms) | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| MobileNetV2 | 92.78% | 0.9282 | 92.22% | 0.60% | 2.76 | To update | Current measured baseline |
| MobileNetV3 | 88.33% | 0.8976 | 82.78% | 6.29% | 1.26 | 287.45 | Measured in latest run; INT8 drop currently fails <2% gate |
| EfficientNetB0 | To update | To update | To update | To update | To update | To update | Pending run |
| ShuffleNetV2 | To update | To update | To update | To update | To update | To update | Pending implementation path |

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
- Pass INT8 drop gate (<2% relative drop)
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
MODEL_PATH=models/palm_ripeness_best_<ts>_int8.tflite \
LABELS_PATH=models/labels_<ts>.json \
python api/app.py
```

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
	--model models/palm_ripeness_best_<ts>_int8.tflite \
	--labels models/labels_<ts>.json \
	--image <sample_image>.jpg \
	--warmup 1 \
	--runs 10
```

### 11.5 Experiment Logging Rule
After each model run:
1. Append run metrics to reports/experiment_log.csv.
2. Update section 8.1 measured table in this file.
3. Add short notes on integration issues (if any), especially preprocessing mismatches or runtime compatibility errors.
