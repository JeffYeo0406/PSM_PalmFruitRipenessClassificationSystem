import io
import json as _json
import logging
import os
import sqlite3 as _sqlite3
import sys
import threading
import time
import uuid
from typing import Optional, Set, Tuple

from flask import Flask, jsonify, request
from PIL import Image
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
    ENABLE_PALM_BINARY_GATE,
    ensure_input_gate_ready,
    InputValidationError,
    get_input_gate_config,
    load_interpreter,
    load_labels,
    MODEL_PREPROCESS_FAMILY,
    _normalize_preprocess_family,
    PALM_BINARY_MODEL_PATH,
    PALM_BINARY_THRESHOLD,
    predict_bytes,
)
from inference_db import build_stage_rows, init_inference_db, log_inference_event  # noqa: E402
from path_compat import resolve_artifact  # noqa: E402

def _resolve_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT_DIR, p)


def _resolve_model_path() -> str:
    resolved = resolve_artifact(
        env_value=os.getenv("MODEL_PATH"),
        default_candidates=[
            os.path.relpath(DEFAULT_MODEL, ROOT_DIR) if os.path.isabs(DEFAULT_MODEL) else DEFAULT_MODEL,
            "models/palm_ripeness_best_20260421_022121_float16.tflite",
            "models/palm_ripeness_best_int8.tflite",
            "saved_models/palm_ripeness_best_int8.tflite",
            "palm_ripeness_best_int8.tflite",
        ],
        glob_patterns=[
            "models/palm_ripeness_best_*_float16.tflite",
            "models/palm_ripeness_best_*_float32.tflite",
            "models/palm_ripeness_best_*_int8.tflite",
            "saved_models/palm_ripeness_best_*_float16.tflite",
            "saved_models/palm_ripeness_best_*_float32.tflite",
            "saved_models/palm_ripeness_best_*_int8.tflite",
            "palm_ripeness_best_*_float16.tflite",
            "palm_ripeness_best_*_float32.tflite",
            "palm_ripeness_best_*_int8.tflite",
        ],
        allow_missing_default=True,
    )
    return resolved


def _resolve_labels_path() -> str:
    resolved = resolve_artifact(
        env_value=os.getenv("LABELS_PATH"),
        default_candidates=[
            os.path.relpath(DEFAULT_LABELS, ROOT_DIR) if os.path.isabs(DEFAULT_LABELS) else DEFAULT_LABELS,
            "models/labels_20260421_022121.json",
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


def _extract_image_dimensions(image_bytes: bytes) -> Tuple[Optional[int], Optional[int]]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return int(img.width), int(img.height)
    except Exception:  # noqa: BLE001
        return None, None


try:
    init_inference_db()
except Exception as exc:  # noqa: BLE001
    logging.warning("Inference DB init failed, continuing without DB logging: %s", exc)


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


def _db():
    """Open a read-only WAL connection to inference_log.db."""
    from inference_db import resolve_db_path
    conn = _sqlite3.connect(resolve_db_path(), timeout=5.0)
    conn.row_factory = _sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


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
    image_width_px, image_height_px = _extract_image_dimensions(image_bytes)
    image_name = file.filename or None
    req_id = uuid.uuid4().hex[:8]
    api_start = time.perf_counter()

    try:
        with interpreter_lock:
            if not _ensure_runtime_ready():
                error_message = model_load_error or "model runtime unavailable"
                api_latency = (time.perf_counter() - api_start) * 1000
                stage_rows = build_stage_rows(
                    gate_enabled=ENABLE_PALM_BINARY_GATE,
                    request_outcome_tag="runtime_error",
                    error_code_tag="runtime_unavailable",
                    error_message=error_message,
                    binary_threshold=PALM_BINARY_THRESHOLD,
                )
                log_inference_event(
                    source_tag="api",
                    request_uid=req_id,
                    model_path=MODEL_PATH,
                    labels_path=LABELS_PATH,
                    binary_model_path=PALM_BINARY_MODEL_PATH,
                    request_outcome_tag="runtime_error",
                    stage_rows=stage_rows,
                    error_code_tag="runtime_unavailable",
                    error_message=error_message,
                    image_name=image_name,
                    image_mime_type=content_type,
                    image_size_bytes=len(image_bytes),
                    image_width_px=image_width_px,
                    image_height_px=image_height_px,
                    warmup_runs=WARMUP_RUNS,
                    timed_runs=RUNS,
                    api_latency_ms=api_latency,
                    http_status_code=503,
                    raw_error={"error": error_message},
                )
                return jsonify({"error": error_message}), 503

            # Read optional per-request preprocess family sent by web app.
            # Falls back to module-level MODEL_PREPROCESS_FAMILY if not provided.
            req_preprocess_family = request.form.get("preprocess_family", "").strip()
            try:
                effective_preprocess_family = (
                    _normalize_preprocess_family(req_preprocess_family)
                    if req_preprocess_family
                    else MODEL_PREPROCESS_FAMILY
                )
            except ValueError:
                effective_preprocess_family = MODEL_PREPROCESS_FAMILY

            result = predict_bytes(
                image_bytes,
                bundle=bundle,
                labels=labels,
                warmup=WARMUP_RUNS,
                runs=RUNS,
                preprocess_family=effective_preprocess_family,
            )
    except InputValidationError as exc:
        logging.info("req=%s rejected by input gate: %s", req_id, exc.message)
        api_latency = (time.perf_counter() - api_start) * 1000
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
            source_tag="api",
            request_uid=req_id,
            model_path=MODEL_PATH,
            labels_path=LABELS_PATH,
            binary_model_path=PALM_BINARY_MODEL_PATH,
            request_outcome_tag="gate_rejected",
            stage_rows=stage_rows,
            error_code_tag=exc.code,
            error_message=exc.message,
            hint_message=exc.hint,
            image_name=image_name,
            image_mime_type=content_type,
            image_size_bytes=len(image_bytes),
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            warmup_runs=WARMUP_RUNS,
            timed_runs=RUNS,
            api_latency_ms=api_latency,
            http_status_code=422,
            raw_error=exc.to_dict(),
        )
        payload = {
            "request_id": req_id,
            **exc.to_dict(),
        }
        return jsonify(payload), 422
    except RuntimeError as exc:
        logging.warning("req=%s runtime unavailable: %s", req_id, exc)
        api_latency = (time.perf_counter() - api_start) * 1000
        stage_rows = build_stage_rows(
            gate_enabled=ENABLE_PALM_BINARY_GATE,
            request_outcome_tag="runtime_error",
            error_code_tag="runtime_unavailable",
            error_message=str(exc),
            binary_threshold=PALM_BINARY_THRESHOLD,
        )
        log_inference_event(
            source_tag="api",
            request_uid=req_id,
            model_path=MODEL_PATH,
            labels_path=LABELS_PATH,
            binary_model_path=PALM_BINARY_MODEL_PATH,
            request_outcome_tag="runtime_error",
            stage_rows=stage_rows,
            error_code_tag="runtime_unavailable",
            error_message=str(exc),
            image_name=image_name,
            image_mime_type=content_type,
            image_size_bytes=len(image_bytes),
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            warmup_runs=WARMUP_RUNS,
            timed_runs=RUNS,
            api_latency_ms=api_latency,
            http_status_code=503,
            raw_error={"error": str(exc)},
        )
        return jsonify({"request_id": req_id, "error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        logging.warning("req=%s classify failed: %s", req_id, exc)
        api_latency = (time.perf_counter() - api_start) * 1000
        stage_rows = build_stage_rows(
            gate_enabled=ENABLE_PALM_BINARY_GATE,
            request_outcome_tag="input_error",
            error_code_tag="unexpected_error",
            error_message=str(exc),
            binary_threshold=PALM_BINARY_THRESHOLD,
        )
        log_inference_event(
            source_tag="api",
            request_uid=req_id,
            model_path=MODEL_PATH,
            labels_path=LABELS_PATH,
            binary_model_path=PALM_BINARY_MODEL_PATH,
            request_outcome_tag="input_error",
            stage_rows=stage_rows,
            error_code_tag="unexpected_error",
            error_message=str(exc),
            image_name=image_name,
            image_mime_type=content_type,
            image_size_bytes=len(image_bytes),
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            warmup_runs=WARMUP_RUNS,
            timed_runs=RUNS,
            api_latency_ms=api_latency,
            http_status_code=400,
            raw_error={"error": str(exc)},
        )
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

    success_stage_rows = build_stage_rows(
        gate_enabled=ENABLE_PALM_BINARY_GATE,
        request_outcome_tag="accepted",
        prediction=result,
        binary_threshold=PALM_BINARY_THRESHOLD,
        inference_latency_ms=result["avg_ms"],
    )
    log_inference_event(
        source_tag="api",
        request_uid=req_id,
        model_path=MODEL_PATH,
        labels_path=LABELS_PATH,
        binary_model_path=PALM_BINARY_MODEL_PATH,
        request_outcome_tag="accepted",
        stage_rows=success_stage_rows,
        image_name=image_name,
        image_mime_type=content_type,
        image_size_bytes=len(image_bytes),
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        warmup_runs=WARMUP_RUNS,
        timed_runs=RUNS,
        inference_latency_ms=result["avg_ms"],
        api_latency_ms=api_latency,
        http_status_code=200,
        raw_result=result,
    )

    with results_lock:
        _purge_expired_results()
        results_store[req_id] = {
            "result": response_data,
            "expires_at": time.time() + RESULT_TTL_SECONDS,
        }

    response_data["result_path"] = f"/result/{req_id}"
    return jsonify(response_data)


# ── GET /history ───────────────────────────────────────────────────────────
#
# Queries v_request_pipeline_trace joined with model_registry for the model
# path, plus per-request pipeline_stages row details.
#
# Query params:
#   filter   — 'all' | model_path fragment | source_tag ('api','cli')
#   page     — 1-based page number (default 1)
#   per_page — rows per page (default 50, max 100)

@app.route("/history", methods=["GET"])
def history():
    filter_val = request.args.get("filter", "all").strip()
    page       = max(1, int(request.args.get("page", 1)))
    per_page   = min(100, max(1, int(request.args.get("per_page", 50))))
    offset     = (page - 1) * per_page

    source_tags = {"api", "cli"}
    by_source = filter_val in source_tags

    try:
        with _db() as conn:
            if filter_val == "all":
                where, params = "1=1", []
            elif by_source:
                where, params = "ir.source_tag = ?", [filter_val]
            else:
                where, params = "mr.model_path LIKE ?", [f"%{filter_val}%"]

            total = conn.execute(
                f"""SELECT COUNT(*)
                    FROM inference_requests ir
                    JOIN model_registry mr ON mr.model_registry_id = ir.model_registry_id
                    WHERE {where}""",
                params,
            ).fetchone()[0]

            rows = conn.execute(
                f"""SELECT
                        vt.request_uid, vt.source_tag,
                        vt.request_outcome_tag, vt.error_code_tag,
                        mr.model_path,
                        vt.image_name, vt.image_size_bytes,
                        vt.image_width_px, vt.image_height_px,
                        vt.inference_latency_ms, vt.api_latency_ms,
                        vt.http_status_code, vt.created_at_utc,
                        vt.binary_gate_outcome, vt.quality_gate_outcome,
                        vt.ripeness_stage_outcome,
                        vt.predicted_label, vt.top1_probability,
                        vt.inference_id
                    FROM v_request_pipeline_trace vt
                    JOIN model_registry mr ON mr.model_registry_id = vt.model_registry_id
                    WHERE {where}
                    ORDER BY vt.created_at_utc DESC
                    LIMIT ? OFFSET ?""",
                params + [per_page, offset],
            ).fetchall()

            records = []
            for row in rows:
                stage_rows = conn.execute(
                    """SELECT stage_tag, stage_order, stage_outcome_tag,
                             palm_score, binary_threshold, binary_gate_pass,
                             aspect_ratio, brightness, contrast_std, sharpness,
                             quality_gate_pass,
                             predicted_label, predicted_index, top1_probability,
                             stage_latency_ms,
                             stage_error_code_tag, stage_error_message, stage_hint_message
                       FROM pipeline_stages
                       WHERE inference_id = ?
                       ORDER BY stage_order ASC""",
                    [row["inference_id"]],
                ).fetchall()

                stages = [
                    {
                        "stage_tag":            s["stage_tag"],
                        "stage_order":          s["stage_order"],
                        "stage_outcome_tag":    s["stage_outcome_tag"],
                        "palm_score":           s["palm_score"],
                        "binary_threshold":     s["binary_threshold"],
                        "binary_gate_pass":     s["binary_gate_pass"],
                        "aspect_ratio":         s["aspect_ratio"],
                        "brightness":           s["brightness"],
                        "contrast_std":         s["contrast_std"],
                        "sharpness":            s["sharpness"],
                        "quality_gate_pass":    s["quality_gate_pass"],
                        "predicted_label":      s["predicted_label"],
                        "top1_probability":     s["top1_probability"],
                        "stage_latency_ms":     s["stage_latency_ms"],
                        "stage_error_code_tag": s["stage_error_code_tag"],
                        "stage_error_message":  s["stage_error_message"],
                        "stage_hint_message":   s["stage_hint_message"],
                    }
                    for s in stage_rows
                ]

                records.append({
                    "request_uid":           row["request_uid"],
                    "source_tag":            row["source_tag"],
                    "request_outcome_tag":   row["request_outcome_tag"],
                    "error_code_tag":        row["error_code_tag"],
                    "model_path":            row["model_path"],
                    "image_name":            row["image_name"],
                    "image_size_bytes":      row["image_size_bytes"],
                    "image_width_px":        row["image_width_px"],
                    "image_height_px":       row["image_height_px"],
                    "inference_latency_ms":  row["inference_latency_ms"],
                    "api_latency_ms":        row["api_latency_ms"],
                    "http_status_code":      row["http_status_code"],
                    "created_at_utc":        row["created_at_utc"],
                    "binary_gate_outcome":   row["binary_gate_outcome"],
                    "quality_gate_outcome":  row["quality_gate_outcome"],
                    "ripeness_stage_outcome":row["ripeness_stage_outcome"],
                    "predicted_label":       row["predicted_label"],
                    "top1_probability":      row["top1_probability"],
                    "stages":                stages,
                })

    except Exception as exc:
        logging.warning("history endpoint error: %s", exc)
        return jsonify({"records": [], "total": 0, "page": page, "per_page": per_page, "error": str(exc)})

    return jsonify({"records": records, "total": total, "page": page, "per_page": per_page})


# ── GET /stats ─────────────────────────────────────────────────────────────
#
# Aggregates from inference_requests + pipeline_stages.
#
# Query params:
#   filter — same as /history
#
# Response:
#   total, accepted, gate_rejected, runtime_errors,
#   avg_inference_latency_ms, avg_api_latency_ms,
#   class_dist, rejection_breakdown, latency_series

@app.route("/stats", methods=["GET"])
def stats():
    filter_val = request.args.get("filter", "all").strip()
    source_tags = {"api", "cli"}
    by_source = filter_val in source_tags

    try:
        with _db() as conn:
            if filter_val == "all":
                where, params = "1=1", []
            elif by_source:
                where, params = "ir.source_tag = ?", [filter_val]
            else:
                where, params = "mr.model_path LIKE ?", [f"%{filter_val}%"]

            agg = conn.execute(
                f"""SELECT
                        COUNT(*)                                                                AS total,
                        SUM(CASE WHEN ir.request_outcome_tag='accepted'     THEN 1 ELSE 0 END) AS accepted,
                        SUM(CASE WHEN ir.request_outcome_tag='gate_rejected' THEN 1 ELSE 0 END) AS gate_rejected,
                        SUM(CASE WHEN ir.request_outcome_tag IN ('runtime_error','input_error')
                                 THEN 1 ELSE 0 END)                                            AS runtime_errors,
                        AVG(CASE WHEN ir.inference_latency_ms IS NOT NULL
                                 THEN ir.inference_latency_ms END)                             AS avg_inference_latency_ms,
                        AVG(CASE WHEN ir.api_latency_ms IS NOT NULL
                                 THEN ir.api_latency_ms END)                                   AS avg_api_latency_ms
                    FROM inference_requests ir
                    JOIN model_registry mr ON mr.model_registry_id = ir.model_registry_id
                    WHERE {where}""",
                params,
            ).fetchone()

            # class_dist — from ripeness_classification pipeline stage
            dist_rows = conn.execute(
                f"""SELECT ps.predicted_label, COUNT(*) AS cnt
                    FROM pipeline_stages ps
                    JOIN inference_requests ir ON ir.inference_id = ps.inference_id
                    JOIN model_registry mr ON mr.model_registry_id = ir.model_registry_id
                    WHERE ps.stage_tag = 'ripeness_classification'
                      AND ps.predicted_label IS NOT NULL
                      AND {where}
                    GROUP BY ps.predicted_label""",
                params,
            ).fetchall()
            class_dist = {"Ripe": 0, "Overripe": 0, "Underripe": 0}
            for r in dist_rows:
                if r["predicted_label"] in class_dist:
                    class_dist[r["predicted_label"]] = r["cnt"]

            # rejection_breakdown — error_code_tag WHERE gate_rejected
            rej_rows = conn.execute(
                f"""SELECT ir.error_code_tag, COUNT(*) AS cnt
                    FROM inference_requests ir
                    JOIN model_registry mr ON mr.model_registry_id = ir.model_registry_id
                    WHERE ir.request_outcome_tag = 'gate_rejected'
                      AND ir.error_code_tag IS NOT NULL
                      AND {where}
                    GROUP BY ir.error_code_tag
                    ORDER BY cnt DESC""",
                params,
            ).fetchall()
            rejection_breakdown = {r["error_code_tag"]: r["cnt"] for r in rej_rows}

            # latency_series — last 30 inference_latency_ms (newest-first)
            lat_rows = conn.execute(
                f"""SELECT ir.inference_latency_ms
                    FROM inference_requests ir
                    JOIN model_registry mr ON mr.model_registry_id = ir.model_registry_id
                    WHERE ir.inference_latency_ms IS NOT NULL
                      AND ir.request_outcome_tag = 'accepted'
                      AND {where}
                    ORDER BY ir.created_at_utc DESC
                    LIMIT 30""",
                params,
            ).fetchall()
            latency_series = [r["inference_latency_ms"] for r in lat_rows]

    except Exception as exc:
        logging.warning("stats endpoint error: %s", exc)
        return jsonify({
            "total": 0, "accepted": 0, "gate_rejected": 0, "runtime_errors": 0,
            "avg_inference_latency_ms": None, "avg_api_latency_ms": None,
            "class_dist": {"Ripe": 0, "Overripe": 0, "Underripe": 0},
            "rejection_breakdown": {}, "latency_series": [], "error": str(exc),
        })

    return jsonify({
        "total":                   agg["total"] or 0,
        "accepted":                agg["accepted"] or 0,
        "gate_rejected":           agg["gate_rejected"] or 0,
        "runtime_errors":          agg["runtime_errors"] or 0,
        "avg_inference_latency_ms": round(agg["avg_inference_latency_ms"], 2)
                                    if agg["avg_inference_latency_ms"] else None,
        "avg_api_latency_ms":      round(agg["avg_api_latency_ms"], 2)
                                    if agg["avg_api_latency_ms"] else None,
        "class_dist":              class_dist,
        "rejection_breakdown":     rejection_breakdown,
        "latency_series":          latency_series,
    })


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
