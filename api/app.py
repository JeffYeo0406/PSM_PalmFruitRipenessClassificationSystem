import logging
import os
import sys
import threading
import time
import uuid
from typing import Set

from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

try:
    from flask_cors import CORS
except ImportError:  # pragma: no cover - optional dependency
    CORS = None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.pi_inference import (  # noqa: E402
    DEFAULT_LABELS,
    DEFAULT_MODEL,
    DEFAULT_RUNS,
    DEFAULT_WARMUP,
    ensure_input_gate_ready,
    InputValidationError,
    get_input_gate_config,
    load_interpreter,
    load_labels,
    predict_bytes,
)
from path_compat import resolve_artifact  # noqa: E402

def _resolve_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT_DIR, p)


def _resolve_model_path() -> str:
    resolved = resolve_artifact(
        env_value=os.getenv("MODEL_PATH"),
        default_candidates=[
            os.path.relpath(DEFAULT_MODEL, ROOT_DIR) if os.path.isabs(DEFAULT_MODEL) else DEFAULT_MODEL,
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
    return resolved


def _resolve_labels_path() -> str:
    resolved = resolve_artifact(
        env_value=os.getenv("LABELS_PATH"),
        default_candidates=[
            os.path.relpath(DEFAULT_LABELS, ROOT_DIR) if os.path.isabs(DEFAULT_LABELS) else DEFAULT_LABELS,
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
    return resolved


MODEL_PATH = _resolve_model_path()
LABELS_PATH = _resolve_labels_path()
WARMUP_RUNS = int(os.getenv("WARMUP_RUNS", str(DEFAULT_WARMUP)))
RUNS = int(os.getenv("RUNS", str(DEFAULT_RUNS)))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
RESULT_TTL_SECONDS = int(os.getenv("RESULT_TTL_SECONDS", "1800"))
ALLOWED_MIMETYPES: Set[str] = {
    m.strip().lower() for m in os.getenv("ALLOWED_MIMETYPES", "image/jpeg,image/png").split(",") if m.strip()
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

if CORS:
    CORS(app)

labels = []
bundle = None
model_load_error = None
interpreter_lock = threading.Lock()
results_store = {}
results_lock = threading.Lock()


def _load_runtime_artifacts():
    global MODEL_PATH, LABELS_PATH, labels, bundle

    MODEL_PATH = _resolve_model_path()
    LABELS_PATH = _resolve_labels_path()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Set MODEL_PATH env var.")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}. Set LABELS_PATH env var.")

    labels = load_labels(LABELS_PATH)
    bundle = load_interpreter(MODEL_PATH)
    ensure_input_gate_ready()


def _ensure_runtime_ready():
    global model_load_error

    if bundle is not None and labels:
        model_load_error = None
        return True

    try:
        _load_runtime_artifacts()
        model_load_error = None
        return True
    except Exception as exc:  # noqa: BLE001
        model_load_error = str(exc)
        logging.warning("Runtime artifacts are not ready: %s", model_load_error)
        return False


def _purge_expired_results(now=None):
    now = now or time.time()
    expired_ids = []
    for req_id, entry in results_store.items():
        if entry["expires_at"] <= now:
            expired_ids.append(req_id)
    for req_id in expired_ids:
        del results_store[req_id]


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_):
    return jsonify({"error": "file too large"}), 413


@app.route("/health", methods=["GET"])
def health():
    with interpreter_lock:
        ready = _ensure_runtime_ready()

    with results_lock:
        _purge_expired_results()
        stored_count = len(results_store)

    return jsonify(
        {
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "model": MODEL_PATH,
            "labels": LABELS_PATH,
            "classes": labels,
            "runtime_error": model_load_error,
            "input_gate": get_input_gate_config(),
            "stored_results": stored_count,
            "result_ttl_seconds": RESULT_TTL_SECONDS,
        }
    )


@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "missing file field"}), 400
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "empty file"}), 400

    content_type = (file.mimetype or "").lower()
    if ALLOWED_MIMETYPES and content_type not in ALLOWED_MIMETYPES:
        return jsonify({"error": f"unsupported content-type: {content_type}"}), 415

    image_bytes = file.read()
    req_id = uuid.uuid4().hex[:8]
    api_start = time.perf_counter()
    try:
        with interpreter_lock:
            if not _ensure_runtime_ready():
                return jsonify({"error": model_load_error or "model runtime unavailable"}), 503

            result = predict_bytes(
                image_bytes,
                bundle=bundle,
                labels=labels,
                warmup=WARMUP_RUNS,
                runs=RUNS,
            )
    except InputValidationError as exc:
        logging.info("req=%s rejected by input gate: %s", req_id, exc.message)
        payload = {
            "request_id": req_id,
            **exc.to_dict(),
        }
        return jsonify(payload), 422
    except RuntimeError as exc:
        logging.warning("req=%s runtime unavailable: %s", req_id, exc)
        return jsonify({"request_id": req_id, "error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        logging.warning("req=%s classify failed: %s", req_id, exc)
        return jsonify({"request_id": req_id, "error": str(exc)}), 400

    api_latency = (time.perf_counter() - api_start) * 1000
    logging.info(
        "req=%s label=%s prob=%.4f inf_ms=%.1f api_ms=%.1f size_bytes=%d",
        req_id,
        result["label"],
        result["probability"],
        result["avg_ms"],
        api_latency,
        len(image_bytes),
    )

    response_data = {
        "request_id": req_id,
        "label": result["label"],
        "probability": result["probability"],
        "index": result["index"],
        "latency_ms": result["avg_ms"],
        "api_ms": api_latency,
        "runs": result["runs"],
    }

    with results_lock:
        _purge_expired_results()
        results_store[req_id] = {
            "result": response_data,
            "expires_at": time.time() + RESULT_TTL_SECONDS,
        }

    response_data["result_path"] = f"/result/{req_id}"
    return jsonify(response_data)


@app.route("/result/<request_id>", methods=["GET"])
def result_by_id(request_id):
    with results_lock:
        _purge_expired_results()
        entry = results_store.get(request_id)
        if entry is None:
            return jsonify({"error": "result not found or expired"}), 404
        return jsonify(entry["result"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
