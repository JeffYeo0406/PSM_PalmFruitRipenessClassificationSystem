
import argparse
import importlib
import io
import json
import mimetypes
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from path_compat import resolve_artifact, resolve_path
from inference_db import build_stage_rows, init_inference_db, log_inference_event

# Prefer LiteRT (ai_edge_litert); fall back to tflite_runtime.
Interpreter = None
for _module_name, _label in (
    ("ai_edge_litert.interpreter", "ai_edge_litert"),
    ("tflite_runtime.interpreter", "tflite_runtime"),
):
    try:
        _module = importlib.import_module(_module_name)
        Interpreter = getattr(_module, "Interpreter")
        print(f"Using {_label} Interpreter")
        break
    except (ModuleNotFoundError, AttributeError):
        continue

if Interpreter is None:
    raise ImportError(
        "No supported TFLite runtime found. Install ai-edge-litert (preferred) or tflite-runtime."
    )


IMG_SIZE = (224, 224)
SUPPORTED_PREPROCESS_FAMILIES = {"mobilenet_v2", "mobilenet_v3", "none"}


def _preprocess_mobilenet_v2(arr: np.ndarray) -> np.ndarray:
    """MobileNetV2 preprocessing without TensorFlow runtime dependency."""
    return (arr.astype(np.float32) / 127.5) - 1.0


def _preprocess_mobilenet_v3(arr: np.ndarray) -> np.ndarray:
    """MobileNetV3 default path with in-model preprocessing enabled."""
    return arr.astype(np.float32)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_preprocess_family(raw_value: Optional[str], default: str = "mobilenet_v2") -> str:
    value = (raw_value or default).strip().lower()
    aliases = {
        "mv2": "mobilenet_v2",
        "mv3": "mobilenet_v3",
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_PREPROCESS_FAMILIES:
        raise ValueError(
            f"Unsupported MODEL_PREPROCESS_FAMILY '{raw_value}'. "
            f"Expected one of: {sorted(SUPPORTED_PREPROCESS_FAMILIES)}"
        )
    return value


DEFAULT_MODEL = resolve_artifact(
    env_value=os.getenv("MODEL_PATH"),
    default_candidates=[
        "models/palm_ripeness_best_int8.tflite",
        "saved_models/palm_ripeness_best_int8.tflite",
        "palm_ripeness_best_int8.tflite",
    ],
    glob_patterns=[
        "models/palm_ripeness_best_*_int8.tflite",
        "models/palm_ripeness_best_*_float16.tflite",
        "models/palm_ripeness_best_*_float32.tflite",
        "saved_models/palm_ripeness_best_*_int8.tflite",
        "saved_models/palm_ripeness_best_*_float16.tflite",
        "saved_models/palm_ripeness_best_*_float32.tflite",
        "palm_ripeness_best_*_int8.tflite",
        "palm_ripeness_best_*_float16.tflite",
        "palm_ripeness_best_*_float32.tflite",
    ],
    allow_missing_default=True,
)
DEFAULT_LABELS = resolve_artifact(
    env_value=os.getenv("LABELS_PATH"),
    default_candidates=[
        "models/labels.json",
        "saved_models/labels.json",
        "labels.json",
    ],
    glob_patterns=[
        "models/labels_*.json",
        "saved_models/labels_*.json",
        "labels_*.json",
    ],
    allow_missing_default=True,
)
DEFAULT_WARMUP = int(os.getenv("WARMUP_RUNS", "1"))
DEFAULT_RUNS = int(os.getenv("RUNS", "3"))
MODEL_PREPROCESS_FAMILY = _normalize_preprocess_family(
    os.getenv("MODEL_PREPROCESS_FAMILY"),
    default="mobilenet_v2",
)
MAX_FILE_BYTES = 10 * 1024 * 1024  # guardrail for API uploads

# Quick image gate (hard reject) before ripeness inference
MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "96"))
MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "96"))
MIN_ASPECT_RATIO = float(os.getenv("MIN_ASPECT_RATIO", "0.4"))
MAX_ASPECT_RATIO = float(os.getenv("MAX_ASPECT_RATIO", "2.5"))
MIN_BRIGHTNESS = float(os.getenv("MIN_BRIGHTNESS", "20"))
MAX_BRIGHTNESS = float(os.getenv("MAX_BRIGHTNESS", "235"))
MIN_CONTRAST_STD = float(os.getenv("MIN_CONTRAST_STD", "12"))
MIN_SHARPNESS = float(os.getenv("MIN_SHARPNESS", "6"))

# Stage 1 gate: palm/non-palm check is enabled by default.
ENABLE_PALM_BINARY_GATE = _env_bool("ENABLE_PALM_BINARY_GATE", default=True)
PALM_BINARY_THRESHOLD = float(os.getenv("PALM_BINARY_THRESHOLD", "0.60"))
PALM_BINARY_PALM_INDEX = int(os.getenv("PALM_BINARY_PALM_INDEX", "1"))
# Most binary gate models trained with scripts/train_binary_gate.py include
# MobileNetV2 preprocessing inside the model graph, so raw RGB should be fed.
# Set PALM_BINARY_APPLY_PREPROCESS=true only for legacy models that expect
# external [-1, 1] preprocessing before inference.
PALM_BINARY_APPLY_PREPROCESS = _env_bool("PALM_BINARY_APPLY_PREPROCESS", default=False)
PALM_BINARY_MODEL_PATH = resolve_artifact(
    env_value=os.getenv("PALM_BINARY_MODEL_PATH"),
    default_candidates=[
        "models/palm_presence_binary.tflite",
        "models/palm_detector_binary.tflite",
        "saved_models/palm_presence_binary.tflite",
    ],
    glob_patterns=[
        "models/*palm*binary*.tflite",
        "models/*palm*detector*.tflite",
        "saved_models/*palm*binary*.tflite",
    ],
    allow_missing_default=False,
)


@dataclass
class InterpreterBundle:
    interpreter: Any
    input_details: dict
    output_details: dict


class InputValidationError(ValueError):
    def __init__(self, code: str, message: str, hint: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "error": self.message,
            "error_code": self.code,
        }
        if self.hint:
            payload["hint"] = self.hint
        if self.details:
            payload["details"] = self.details
        return payload


_palm_binary_bundle: Optional[InterpreterBundle] = None
_palm_binary_bundle_path: Optional[str] = None


def load_labels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, list):
        raise ValueError("labels file must contain a JSON list of class names")
    return labels


def load_interpreter(model_path: str) -> InterpreterBundle:
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return InterpreterBundle(interpreter, input_details, output_details)


def _get_model_input_hw(input_details: dict) -> tuple[int, int]:
    shape = input_details.get("shape", [1, IMG_SIZE[0], IMG_SIZE[1], 3])
    if len(shape) >= 4 and int(shape[1]) > 0 and int(shape[2]) > 0:
        return int(shape[1]), int(shape[2])
    return IMG_SIZE


def _load_palm_binary_bundle() -> Optional[InterpreterBundle]:
    global _palm_binary_bundle, _palm_binary_bundle_path

    if not ENABLE_PALM_BINARY_GATE:
        return None

    if not PALM_BINARY_MODEL_PATH:
        raise RuntimeError(
            "Palm binary gate is enabled but no binary model was found. Set PALM_BINARY_MODEL_PATH or add a matching model in models/."
        )
    if not os.path.exists(PALM_BINARY_MODEL_PATH):
        raise RuntimeError(
            f"Palm binary gate is enabled but model was not found at {PALM_BINARY_MODEL_PATH}."
        )

    if _palm_binary_bundle is not None and _palm_binary_bundle_path == PALM_BINARY_MODEL_PATH:
        return _palm_binary_bundle

    _palm_binary_bundle = load_interpreter(PALM_BINARY_MODEL_PATH)
    _palm_binary_bundle_path = PALM_BINARY_MODEL_PATH
    return _palm_binary_bundle


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return float(1 / (1 + z))
    z = np.exp(x)
    return float(z / (1 + z))


def _run_palm_binary_gate(arr_rgb: np.ndarray) -> Optional[float]:
    bundle = _load_palm_binary_bundle()
    if bundle is None:
        return None

    target_h, target_w = _get_model_input_hw(bundle.input_details)
    img = Image.fromarray(arr_rgb.astype(np.uint8)).resize((target_w, target_h))
    arr = np.array(img, dtype=np.float32)
    if PALM_BINARY_APPLY_PREPROCESS:
        arr = _preprocess_mobilenet_v2(arr)
    batch = np.expand_dims(arr, axis=0)
    batch = _quantize_input(batch, bundle.input_details)

    bundle.interpreter.set_tensor(bundle.input_details["index"], batch)
    bundle.interpreter.invoke()
    raw = bundle.interpreter.get_tensor(bundle.output_details["index"])
    logits = _dequantize_output(raw, bundle.output_details).reshape(-1)

    if logits.size == 1:
        value = float(logits[0])
        if 0.0 <= value <= 1.0:
            return value
        return _sigmoid(value)

    probs = _normalize_probs(logits)
    palm_index = min(max(PALM_BINARY_PALM_INDEX, 0), probs.size - 1)
    return float(probs[palm_index])


def ensure_input_gate_ready() -> None:
    """Fail fast when mandatory palm/non-palm gate cannot be loaded."""
    _load_palm_binary_bundle()


def _validate_image_gate(arr_rgb: np.ndarray) -> Dict[str, float]:
    height, width = arr_rgb.shape[:2]
    aspect = width / max(height, 1)
    gray = arr_rgb.mean(axis=2)
    brightness = float(np.mean(gray))
    contrast_std = float(np.std(gray))
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    sharpness = float(np.mean(np.abs(dx)) + np.mean(np.abs(dy)))

    metrics = {
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(aspect),
        "brightness": brightness,
        "contrast_std": contrast_std,
        "sharpness": sharpness,
    }

    # Stage 1: semantic palm/non-palm gate (mandatory by default).
    palm_score = _run_palm_binary_gate(arr_rgb)
    if palm_score is not None:
        metrics["palm_score"] = float(palm_score)
        if palm_score < PALM_BINARY_THRESHOLD:
            raise InputValidationError(
                code="not_palm_fruit",
                message="Image likely does not contain palm fruit.",
                hint="Capture only palm fruit bunches in frame.",
                details=metrics,
            )

    # Stage 2: quality checks.
    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        raise InputValidationError(
            code="low_resolution",
            message="Image is too small for reliable ripeness prediction.",
            hint=f"Use a clearer image at least {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT} pixels.",
            details=metrics,
        )

    if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
        raise InputValidationError(
            code="bad_aspect_ratio",
            message="Image aspect ratio is outside supported range.",
            hint="Center one fruit bunch and avoid extreme panoramic crops.",
            details=metrics,
        )

    if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS:
        raise InputValidationError(
            code="bad_exposure",
            message="Image exposure is too dark or too bright.",
            hint="Capture under balanced lighting and avoid strong glare/shadows.",
            details=metrics,
        )

    if contrast_std < MIN_CONTRAST_STD:
        raise InputValidationError(
            code="low_contrast",
            message="Image contrast is too low for reliable prediction.",
            hint="Increase lighting contrast and avoid foggy or washed-out captures.",
            details=metrics,
        )

    if sharpness < MIN_SHARPNESS:
        raise InputValidationError(
            code="blurry_image",
            message="Image appears blurry or out of focus.",
            hint="Refocus camera and keep the fruit steady before capture.",
            details=metrics,
        )

    return metrics


def get_input_gate_config() -> Dict[str, Any]:
    return {
        "enabled": True,
        "pipeline_order": ["palm_non_palm", "quality", "ripeness"],
        "min_image_width": MIN_IMAGE_WIDTH,
        "min_image_height": MIN_IMAGE_HEIGHT,
        "min_aspect_ratio": MIN_ASPECT_RATIO,
        "max_aspect_ratio": MAX_ASPECT_RATIO,
        "min_brightness": MIN_BRIGHTNESS,
        "max_brightness": MAX_BRIGHTNESS,
        "min_contrast_std": MIN_CONTRAST_STD,
        "min_sharpness": MIN_SHARPNESS,
        "palm_binary_gate_required": ENABLE_PALM_BINARY_GATE,
        "palm_binary_gate_enabled": ENABLE_PALM_BINARY_GATE,
        "palm_binary_model": PALM_BINARY_MODEL_PATH,
        "palm_binary_model_exists": bool(PALM_BINARY_MODEL_PATH and os.path.exists(PALM_BINARY_MODEL_PATH)),
        "palm_binary_threshold": PALM_BINARY_THRESHOLD,
        "palm_binary_palm_index": PALM_BINARY_PALM_INDEX,
        "palm_binary_apply_preprocess": PALM_BINARY_APPLY_PREPROCESS,
        "model_preprocess_family": MODEL_PREPROCESS_FAMILY,
    }


def _apply_model_preprocess(arr: np.ndarray, preprocess_family: str) -> np.ndarray:
    family = _normalize_preprocess_family(preprocess_family, default=MODEL_PREPROCESS_FAMILY)
    if family == "mobilenet_v2":
        return _preprocess_mobilenet_v2(arr)
    if family == "mobilenet_v3":
        return _preprocess_mobilenet_v3(arr)
    return arr.astype(np.float32)


def preprocess_image_bytes(image_bytes: bytes, preprocess_family: str = MODEL_PREPROCESS_FAMILY) -> np.ndarray:
    if len(image_bytes) > MAX_FILE_BYTES:
        raise ValueError("image payload is too large")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    raw_arr = np.array(img, dtype=np.uint8)
    _validate_image_gate(raw_arr)
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = _apply_model_preprocess(arr, preprocess_family)
    return np.expand_dims(arr, axis=0)


def _extract_image_dimensions(image_bytes: bytes) -> Tuple[Optional[int], Optional[int]]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return int(img.width), int(img.height)
    except Exception:  # noqa: BLE001
        return None, None


def _quantize_input(batch: np.ndarray, input_details: dict) -> np.ndarray:
    scale, zero_point = input_details.get("quantization", (0.0, 0))
    dtype = input_details["dtype"]
    if scale and dtype != np.float32:
        batch = batch / scale + zero_point
        batch = np.clip(np.rint(batch), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    else:
        batch = batch.astype(dtype)
    return batch


def _dequantize_output(output: np.ndarray, output_details: dict) -> np.ndarray:
    scale, zero_point = output_details.get("quantization", (0.0, 0))
    dtype = output_details["dtype"]
    if scale and dtype != np.float32:
        return scale * (output.astype(np.float32) - zero_point)
    return output.astype(np.float32)


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    """Ensure probabilities are well-formed even with quantization rounding."""
    probs = np.clip(probs, 1e-9, None).astype(np.float32)
    total = probs.sum()
    if total <= 0:
        # Fallback to softmax if the sum is degenerate
        probs = np.exp(probs - np.max(probs))
        total = probs.sum()
    return probs / total


def predict_bytes(
    image_bytes: bytes,
    bundle: InterpreterBundle,
    labels: List[str],
    warmup: int = DEFAULT_WARMUP,
    runs: int = DEFAULT_RUNS,
    preprocess_family: str = MODEL_PREPROCESS_FAMILY,
):
    batch = preprocess_image_bytes(image_bytes, preprocess_family=preprocess_family)
    batch = _quantize_input(batch, bundle.input_details)

    for _ in range(max(0, warmup)):
        bundle.interpreter.set_tensor(bundle.input_details["index"], batch)
        bundle.interpreter.invoke()

    timings = []
    probs = None
    for _ in range(max(1, runs)):
        start = time.time()
        bundle.interpreter.set_tensor(bundle.input_details["index"], batch)
        bundle.interpreter.invoke()
        raw = bundle.interpreter.get_tensor(bundle.output_details["index"])
        probs = _normalize_probs(_dequantize_output(raw, bundle.output_details)[0])
        timings.append((time.time() - start) * 1000)

    top1 = int(np.argmax(probs))
    return {
        "label": labels[top1] if top1 < len(labels) else str(top1),
        "index": top1,
        "probability": float(probs[top1]),
        "avg_ms": float(np.mean(timings)),
        "runs": max(1, runs),
    }


def predict_file(
    image_path: str,
    bundle: InterpreterBundle,
    labels: List[str],
    warmup: int = DEFAULT_WARMUP,
    runs: int = DEFAULT_RUNS,
    preprocess_family: str = MODEL_PREPROCESS_FAMILY,
):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return predict_bytes(
        image_bytes,
        bundle,
        labels,
        warmup=warmup,
        runs=runs,
        preprocess_family=preprocess_family,
    )


def main():
    parser = argparse.ArgumentParser(description="Run palm fruit ripeness inference (TFLite/LiteRT)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to TFLite/LiteRT model")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Path to labels.json (list of class names)")
    parser.add_argument("--image", required=True, help="Path to image for inference")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup iterations before timing")
    parser.add_argument(
        "--preprocess-family",
        default=MODEL_PREPROCESS_FAMILY,
        choices=sorted(SUPPORTED_PREPROCESS_FAMILIES),
        help="Model input preprocessing family.",
    )
    args = parser.parse_args()

    model_path = resolve_path(args.model) or args.model
    labels_path = resolve_path(args.labels) or args.labels
    image_path = resolve_path(args.image) or args.image
    request_uid = f"cli-{time.strftime('%Y%m%d%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    image_name = os.path.basename(image_path)
    image_mime_type = mimetypes.guess_type(image_path)[0]

    try:
        init_inference_db()
    except Exception:  # noqa: BLE001
        # DB logging is best effort and must not break CLI behavior.
        pass

    try:
        labels = load_labels(labels_path)
        bundle = load_interpreter(model_path)
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_width_px, image_height_px = _extract_image_dimensions(image_bytes)

        result = predict_bytes(
            image_bytes,
            bundle,
            labels,
            warmup=args.warmup,
            runs=args.runs,
            preprocess_family=args.preprocess_family,
        )
        stage_rows = build_stage_rows(
            gate_enabled=ENABLE_PALM_BINARY_GATE,
            request_outcome_tag="accepted",
            prediction=result,
            binary_threshold=PALM_BINARY_THRESHOLD,
            inference_latency_ms=result["avg_ms"],
        )
        log_inference_event(
            source_tag="cli",
            request_uid=request_uid,
            model_path=model_path,
            labels_path=labels_path,
            binary_model_path=PALM_BINARY_MODEL_PATH,
            request_outcome_tag="accepted",
            stage_rows=stage_rows,
            image_name=image_name,
            image_mime_type=image_mime_type,
            image_size_bytes=len(image_bytes),
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            warmup_runs=args.warmup,
            timed_runs=args.runs,
            inference_latency_ms=result["avg_ms"],
            raw_result=result,
        )
        print(json.dumps(result, indent=2))
    except InputValidationError as exc:
        image_size_bytes = None
        image_width_px, image_height_px = None, None
        image_bytes = None
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_size_bytes = len(image_bytes)
            image_width_px, image_height_px = _extract_image_dimensions(image_bytes)
        except Exception:  # noqa: BLE001
            pass

        stage_rows = build_stage_rows(
            gate_enabled=ENABLE_PALM_BINARY_GATE,
            request_outcome_tag="gate_rejected",
            error_code_tag=exc.code,
            error_message=exc.message,
            hint_message=exc.hint,
            details=exc.details,
            binary_threshold=PALM_BINARY_THRESHOLD,
        )
        log_inference_event(
            source_tag="cli",
            request_uid=request_uid,
            model_path=model_path,
            labels_path=labels_path,
            binary_model_path=PALM_BINARY_MODEL_PATH,
            request_outcome_tag="gate_rejected",
            stage_rows=stage_rows,
            error_code_tag=exc.code,
            error_message=exc.message,
            hint_message=exc.hint,
            image_name=image_name,
            image_mime_type=image_mime_type,
            image_size_bytes=image_size_bytes,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            warmup_runs=args.warmup,
            timed_runs=args.runs,
            raw_error=exc.to_dict(),
        )
        print(json.dumps(exc.to_dict(), indent=2))
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        image_size_bytes = None
        image_width_px, image_height_px = None, None
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_size_bytes = len(image_bytes)
            image_width_px, image_height_px = _extract_image_dimensions(image_bytes)
        except Exception:  # noqa: BLE001
            pass

        stage_rows = build_stage_rows(
            gate_enabled=ENABLE_PALM_BINARY_GATE,
            request_outcome_tag="runtime_error",
            error_code_tag="unexpected_error",
            error_message=str(exc),
            binary_threshold=PALM_BINARY_THRESHOLD,
        )
        log_inference_event(
            source_tag="cli",
            request_uid=request_uid,
            model_path=model_path,
            labels_path=labels_path,
            binary_model_path=PALM_BINARY_MODEL_PATH,
            request_outcome_tag="runtime_error",
            stage_rows=stage_rows,
            error_code_tag="unexpected_error",
            error_message=str(exc),
            image_name=image_name,
            image_mime_type=image_mime_type,
            image_size_bytes=image_size_bytes,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            warmup_runs=args.warmup,
            timed_runs=args.runs,
            raw_error={"error": str(exc)},
        )
        print(json.dumps({"error": str(exc)}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
