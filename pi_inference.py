
import argparse
import io
import json
import os
import time
from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image
from path_compat import resolve_artifact

# Prefer LiteRT (ai_edge_litert); fall back to tflite_runtime, then legacy tf.lite
try:
    from ai_edge_litert.interpreter import LiteRTInterpreter as Interpreter
    print("Using LiteRTInterpreter (ai_edge_litert)")
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        print("Using tflite_runtime Interpreter")
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
            print("Using tf.lite.python Interpreter (legacy)")
        except ImportError as e:
            raise ImportError(
                "No LiteRT/tflite interpreter found. Install ai-edge-litert (preferred) or tflite-runtime."
            ) from e

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def _resolve_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT_DIR, p)


IMG_SIZE = (224, 224)
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
MAX_FILE_BYTES = 10 * 1024 * 1024  # guardrail for API uploads


@dataclass
class InterpreterBundle:
    interpreter: Interpreter
    input_details: dict
    output_details: dict


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


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    if len(image_bytes) > MAX_FILE_BYTES:
        raise ValueError("image payload is too large")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


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
):
    batch = preprocess_image_bytes(image_bytes)
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
):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return predict_bytes(image_bytes, bundle, labels, warmup=warmup, runs=runs)


def main():
    parser = argparse.ArgumentParser(description="Run palm fruit ripeness inference (TFLite/LiteRT)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to TFLite/LiteRT model")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Path to labels.json (list of class names)")
    parser.add_argument("--image", required=True, help="Path to image for inference")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup iterations before timing")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    bundle = load_interpreter(args.model)
    result = predict_file(args.image, bundle, labels, warmup=args.warmup, runs=args.runs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
