from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

MIGRATION_ID = "001_initial_inference_logging_schema"
MAX_RAW_JSON_LEN = 8192

SOURCE_TAGS = {"api", "cli"}
REQUEST_OUTCOME_TAGS = {"accepted", "gate_rejected", "runtime_error", "input_error"}
STAGE_TAGS = {"binary_gate", "quality_gate", "ripeness_classification"}
STAGE_ORDERS = {
    "binary_gate": 1,
    "quality_gate": 2,
    "ripeness_classification": 3,
}
STAGE_OUTCOME_TAGS = {"passed", "rejected", "not_run", "completed", "error"}
ERROR_CODE_TAGS = {
    "not_palm_fruit",
    "low_resolution",
    "bad_aspect_ratio",
    "bad_exposure",
    "low_contrast",
    "blurry_image",
    "runtime_unavailable",
    "unexpected_error",
}
QUALITY_ERROR_CODE_TAGS = {
    "low_resolution",
    "bad_aspect_ratio",
    "bad_exposure",
    "low_contrast",
    "blurry_image",
}

_init_lock = threading.Lock()
_init_db_path: Optional[str] = None
_fingerprint_lock = threading.Lock()
# cache[path] = (mtime_ns, size, sha256)
_fingerprint_cache: Dict[str, Tuple[int, int, str]] = {}


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def resolve_db_path() -> str:
    configured = os.getenv("INFERENCE_DB_PATH", "").strip()
    if not configured:
        configured = os.path.join("reports", "inference_log.db")
    if not os.path.isabs(configured):
        configured = os.path.join(_project_root(), configured)
    return os.path.abspath(configured)


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def _ddl_statements() -> List[str]:
    return [
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            migration_id TEXT PRIMARY KEY,
            applied_at_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            ddl_checksum TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS model_registry (
            model_registry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_path TEXT NOT NULL,
            labels_path TEXT NOT NULL,
            binary_model_path TEXT,
            model_fingerprint_sha256 TEXT NOT NULL,
            model_size_bytes INTEGER NOT NULL,
            fingerprint_computed_at_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            registered_at_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            UNIQUE(model_fingerprint_sha256),
            CHECK(length(model_fingerprint_sha256) = 64),
            CHECK(length(model_path) > 0),
            CHECK(length(labels_path) > 0),
            CHECK(model_size_bytes >= 0)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS inference_requests (
            inference_id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_uid TEXT NOT NULL UNIQUE,
            source_tag TEXT NOT NULL,
            model_registry_id INTEGER NOT NULL,
            request_outcome_tag TEXT NOT NULL,
            error_code_tag TEXT,
            error_message TEXT,
            hint_message TEXT,
            image_name TEXT,
            image_mime_type TEXT,
            image_size_bytes INTEGER,
            image_width_px INTEGER,
            image_height_px INTEGER,
            warmup_runs INTEGER,
            timed_runs INTEGER,
            inference_latency_ms REAL,
            api_latency_ms REAL,
            created_at_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            http_status_code INTEGER,
            raw_result_json TEXT,
            raw_error_json TEXT,
            FOREIGN KEY(model_registry_id) REFERENCES model_registry(model_registry_id),
            CHECK(source_tag IN ('api','cli')),
            CHECK(request_outcome_tag IN ('accepted','gate_rejected','runtime_error','input_error')),
            CHECK(error_code_tag IS NULL OR error_code_tag IN ('not_palm_fruit','low_resolution','bad_aspect_ratio','bad_exposure','low_contrast','blurry_image','runtime_unavailable','unexpected_error')),
            CHECK(image_size_bytes IS NULL OR image_size_bytes >= 0),
            CHECK(image_width_px IS NULL OR image_width_px > 0),
            CHECK(image_height_px IS NULL OR image_height_px > 0),
            CHECK(warmup_runs IS NULL OR warmup_runs >= 0),
            CHECK(timed_runs IS NULL OR timed_runs >= 1),
            CHECK(inference_latency_ms IS NULL OR inference_latency_ms >= 0),
            CHECK(api_latency_ms IS NULL OR api_latency_ms >= 0),
            CHECK(http_status_code IS NULL OR (http_status_code BETWEEN 100 AND 599)),
            CHECK(raw_result_json IS NULL OR length(raw_result_json) <= 8192),
            CHECK(raw_error_json IS NULL OR length(raw_error_json) <= 8192)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS pipeline_stages (
            stage_id INTEGER PRIMARY KEY AUTOINCREMENT,
            inference_id INTEGER NOT NULL,
            stage_tag TEXT NOT NULL,
            stage_order INTEGER NOT NULL,
            stage_outcome_tag TEXT NOT NULL,
            palm_score REAL,
            binary_threshold REAL,
            binary_gate_pass INTEGER,
            aspect_ratio REAL,
            brightness REAL,
            contrast_std REAL,
            sharpness REAL,
            quality_gate_pass INTEGER,
            predicted_label TEXT,
            predicted_index INTEGER,
            top1_probability REAL,
            stage_latency_ms REAL,
            stage_started_at_utc TEXT,
            stage_finished_at_utc TEXT,
            stage_error_code_tag TEXT,
            stage_error_message TEXT,
            stage_hint_message TEXT,
            stage_details_json TEXT,
            FOREIGN KEY(inference_id) REFERENCES inference_requests(inference_id),
            UNIQUE(inference_id, stage_tag),
            UNIQUE(inference_id, stage_order),
            CHECK(stage_tag IN ('binary_gate','quality_gate','ripeness_classification')),
            CHECK(stage_order IN (1,2,3)),
            CHECK(stage_outcome_tag IN ('passed','rejected','not_run','completed','error')),
            CHECK(binary_gate_pass IS NULL OR binary_gate_pass IN (0,1)),
            CHECK(quality_gate_pass IS NULL OR quality_gate_pass IN (0,1)),
            CHECK(stage_error_code_tag IS NULL OR stage_error_code_tag IN ('not_palm_fruit','low_resolution','bad_aspect_ratio','bad_exposure','low_contrast','blurry_image','runtime_unavailable','unexpected_error')),
            CHECK(top1_probability IS NULL OR (top1_probability BETWEEN 0 AND 1)),
            CHECK(stage_latency_ms IS NULL OR stage_latency_ms >= 0)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS validation_runs (
            validation_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT NOT NULL,
            dataset_path TEXT,
            threshold_tag TEXT,
            created_at_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS validation_annotations (
            annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            inference_id INTEGER NOT NULL,
            validation_run_id INTEGER NOT NULL,
            expected_source_tag TEXT NOT NULL,
            sample_key TEXT,
            FOREIGN KEY(inference_id) REFERENCES inference_requests(inference_id),
            FOREIGN KEY(validation_run_id) REFERENCES validation_runs(validation_run_id),
            UNIQUE(inference_id),
            CHECK(expected_source_tag IN ('Palm','Non-palm'))
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_requests_created_at
        ON inference_requests(created_at_utc DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_requests_source_outcome
        ON inference_requests(source_tag, request_outcome_tag, created_at_utc DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_requests_model
        ON inference_requests(model_registry_id, created_at_utc DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_stages_stage_outcome
        ON pipeline_stages(stage_tag, stage_outcome_tag)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_stages_error_code
        ON pipeline_stages(stage_error_code_tag)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_validation_expected_source
        ON validation_annotations(expected_source_tag)
        """,
        "DROP VIEW IF EXISTS v_request_pipeline_trace",
        """
        CREATE VIEW v_request_pipeline_trace AS
        SELECT
            ir.inference_id,
            ir.request_uid,
            ir.source_tag,
            ir.request_outcome_tag,
            ir.error_code_tag,
            ir.model_registry_id,
            ir.image_name,
            ir.image_mime_type,
            ir.image_size_bytes,
            ir.image_width_px,
            ir.image_height_px,
            ir.inference_latency_ms,
            ir.api_latency_ms,
            ir.http_status_code,
            ir.created_at_utc,
            MAX(CASE WHEN ps.stage_tag = 'binary_gate' THEN ps.stage_outcome_tag END) AS binary_gate_outcome,
            MAX(CASE WHEN ps.stage_tag = 'quality_gate' THEN ps.stage_outcome_tag END) AS quality_gate_outcome,
            MAX(CASE WHEN ps.stage_tag = 'ripeness_classification' THEN ps.stage_outcome_tag END) AS ripeness_stage_outcome,
            MAX(CASE WHEN ps.stage_tag = 'ripeness_classification' THEN ps.predicted_label END) AS predicted_label,
            MAX(CASE WHEN ps.stage_tag = 'ripeness_classification' THEN ps.top1_probability END) AS top1_probability
        FROM inference_requests ir
        LEFT JOIN pipeline_stages ps ON ps.inference_id = ir.inference_id
        GROUP BY
            ir.inference_id,
            ir.request_uid,
            ir.source_tag,
            ir.request_outcome_tag,
            ir.error_code_tag,
            ir.model_registry_id,
            ir.image_name,
            ir.image_mime_type,
            ir.image_size_bytes,
            ir.image_width_px,
            ir.image_height_px,
            ir.inference_latency_ms,
            ir.api_latency_ms,
            ir.http_status_code,
            ir.created_at_utc
        """,
        "DROP VIEW IF EXISTS v_validation_stage_summary",
        """
        CREATE VIEW v_validation_stage_summary AS
        SELECT
            va.expected_source_tag AS expected_source,
            CASE
                WHEN ps.stage_tag = 'binary_gate' AND ps.stage_outcome_tag = 'rejected' THEN 'binary_gate_reject'
                WHEN ps.stage_tag = 'ripeness_classification' AND ps.stage_outcome_tag = 'completed' THEN 'ripeness_classification'
            END AS stage,
            COUNT(*) AS count
        FROM validation_annotations va
        JOIN pipeline_stages ps ON ps.inference_id = va.inference_id
        WHERE
            (ps.stage_tag = 'binary_gate' AND ps.stage_outcome_tag = 'rejected')
            OR
            (ps.stage_tag = 'ripeness_classification' AND ps.stage_outcome_tag = 'completed')
        GROUP BY va.expected_source_tag, stage
        """,
    ]


def _ddl_checksum() -> str:
    data = "\n".join(_ddl_statements())
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def init_inference_db() -> None:
    global _init_db_path

    db_path = resolve_db_path()
    with _init_lock:
        if _init_db_path == db_path:
            return

        conn = _connect(db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    migration_id TEXT PRIMARY KEY,
                    applied_at_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    ddl_checksum TEXT
                )
                """
            )
            applied = conn.execute(
                "SELECT 1 FROM schema_migrations WHERE migration_id = ?",
                (MIGRATION_ID,),
            ).fetchone()

            for stmt in _ddl_statements():
                conn.execute(stmt)

            if not applied:
                conn.execute(
                    """
                    INSERT INTO schema_migrations (migration_id, ddl_checksum)
                    VALUES (?, ?)
                    """,
                    (MIGRATION_ID, _ddl_checksum()),
                )
            conn.commit()
            _init_db_path = db_path
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _normalize_path(path: str) -> str:
    return os.path.abspath(path)


def _compute_model_fingerprint_sha256(model_path: str) -> Tuple[str, int]:
    normalized = _normalize_path(model_path)
    stat = os.stat(normalized)

    with _fingerprint_lock:
        cached = _fingerprint_cache.get(normalized)
        if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return cached[2], stat.st_size

    hasher = hashlib.sha256()
    with open(normalized, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)

    digest = hasher.hexdigest()
    with _fingerprint_lock:
        _fingerprint_cache[normalized] = (stat.st_mtime_ns, stat.st_size, digest)

    return digest, stat.st_size


def _safe_debug_json(payload: Optional[Any]) -> Optional[str]:
    if payload is None:
        return None

    if isinstance(payload, str):
        raw = payload
    else:
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True, default=str)

    if len(raw) > MAX_RAW_JSON_LEN:
        return None
    return raw


def _upsert_model_registry(
    conn: sqlite3.Connection,
    model_path: str,
    labels_path: str,
    binary_model_path: Optional[str],
) -> int:
    normalized_model_path = _normalize_path(model_path)
    normalized_labels_path = _normalize_path(labels_path)
    normalized_binary_model_path = _normalize_path(binary_model_path) if binary_model_path else None

    fingerprint_sha256, model_size_bytes = _compute_model_fingerprint_sha256(normalized_model_path)

    existing = conn.execute(
        """
        SELECT model_registry_id
        FROM model_registry
        WHERE model_fingerprint_sha256 = ?
        """,
        (fingerprint_sha256,),
    ).fetchone()
    if existing:
        return int(existing[0])

    cursor = conn.execute(
        """
        INSERT INTO model_registry (
            model_path,
            labels_path,
            binary_model_path,
            model_fingerprint_sha256,
            model_size_bytes
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            normalized_model_path,
            normalized_labels_path,
            normalized_binary_model_path,
            fingerprint_sha256,
            model_size_bytes,
        ),
    )
    return int(cursor.lastrowid)


def build_stage_rows(
    *,
    gate_enabled: bool,
    request_outcome_tag: str,
    error_code_tag: Optional[str] = None,
    error_message: Optional[str] = None,
    hint_message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    prediction: Optional[Dict[str, Any]] = None,
    binary_threshold: Optional[float] = None,
    inference_latency_ms: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if request_outcome_tag not in REQUEST_OUTCOME_TAGS:
        raise ValueError(f"invalid request_outcome_tag: {request_outcome_tag}")

    details = details or {}

    def _base_row(stage_tag: str) -> Dict[str, Any]:
        return {
            "stage_tag": stage_tag,
            "stage_order": STAGE_ORDERS[stage_tag],
            "stage_outcome_tag": "not_run",
            "palm_score": None,
            "binary_threshold": None,
            "binary_gate_pass": None,
            "aspect_ratio": None,
            "brightness": None,
            "contrast_std": None,
            "sharpness": None,
            "quality_gate_pass": None,
            "predicted_label": None,
            "predicted_index": None,
            "top1_probability": None,
            "stage_latency_ms": None,
            "stage_error_code_tag": None,
            "stage_error_message": None,
            "stage_hint_message": None,
            "stage_details_json": None,
        }

    binary = _base_row("binary_gate")
    quality = _base_row("quality_gate")
    ripeness = _base_row("ripeness_classification")

    # Copy quality metrics when available so rejected quality-stage rows carry context.
    quality["aspect_ratio"] = details.get("aspect_ratio")
    quality["brightness"] = details.get("brightness")
    quality["contrast_std"] = details.get("contrast_std")
    quality["sharpness"] = details.get("sharpness")

    binary["palm_score"] = details.get("palm_score")
    binary["binary_threshold"] = binary_threshold

    if request_outcome_tag == "accepted":
        binary["stage_outcome_tag"] = "passed" if gate_enabled else "not_run"
        binary["binary_gate_pass"] = 1 if gate_enabled else None
        quality["stage_outcome_tag"] = "passed"
        quality["quality_gate_pass"] = 1
        ripeness["stage_outcome_tag"] = "completed"
        if prediction:
            ripeness["predicted_label"] = prediction.get("label")
            ripeness["predicted_index"] = prediction.get("index")
            ripeness["top1_probability"] = prediction.get("probability")
        ripeness["stage_latency_ms"] = inference_latency_ms

    elif request_outcome_tag == "gate_rejected":
        if error_code_tag == "not_palm_fruit":
            binary["stage_outcome_tag"] = "rejected"
            binary["binary_gate_pass"] = 0
            binary["stage_error_code_tag"] = error_code_tag
            binary["stage_error_message"] = error_message
            binary["stage_hint_message"] = hint_message
            binary["stage_details_json"] = _safe_debug_json(details)
        elif error_code_tag in QUALITY_ERROR_CODE_TAGS:
            binary["stage_outcome_tag"] = "passed" if gate_enabled else "not_run"
            binary["binary_gate_pass"] = 1 if gate_enabled else None
            quality["stage_outcome_tag"] = "rejected"
            quality["quality_gate_pass"] = 0
            quality["stage_error_code_tag"] = error_code_tag
            quality["stage_error_message"] = error_message
            quality["stage_hint_message"] = hint_message
            quality["stage_details_json"] = _safe_debug_json(details)
        else:
            binary["stage_outcome_tag"] = "error"
            binary["stage_error_code_tag"] = error_code_tag
            binary["stage_error_message"] = error_message
            binary["stage_hint_message"] = hint_message
            binary["stage_details_json"] = _safe_debug_json(details)

    else:
        binary["stage_outcome_tag"] = "error"
        binary["stage_error_code_tag"] = error_code_tag
        binary["stage_error_message"] = error_message
        binary["stage_hint_message"] = hint_message
        binary["stage_details_json"] = _safe_debug_json(details)

    return [binary, quality, ripeness]


def _validate_stage_rows(stage_rows: List[Dict[str, Any]]) -> None:
    if len(stage_rows) != 3:
        raise ValueError("stage_rows must contain exactly three rows")

    seen_tags = set()
    seen_orders = set()
    for row in stage_rows:
        stage_tag = row.get("stage_tag")
        stage_order = row.get("stage_order")
        stage_outcome = row.get("stage_outcome_tag")
        stage_error_code = row.get("stage_error_code_tag")

        if stage_tag not in STAGE_TAGS:
            raise ValueError(f"invalid stage_tag: {stage_tag}")
        if stage_order not in {1, 2, 3}:
            raise ValueError(f"invalid stage_order: {stage_order}")
        if stage_outcome not in STAGE_OUTCOME_TAGS:
            raise ValueError(f"invalid stage_outcome_tag: {stage_outcome}")
        if stage_error_code is not None and stage_error_code not in ERROR_CODE_TAGS:
            raise ValueError(f"invalid stage_error_code_tag: {stage_error_code}")

        seen_tags.add(stage_tag)
        seen_orders.add(stage_order)

    if seen_tags != STAGE_TAGS:
        raise ValueError("stage_rows must include binary_gate, quality_gate, and ripeness_classification")
    if seen_orders != {1, 2, 3}:
        raise ValueError("stage_rows must include stage_order values 1, 2, and 3")


def _ensure_validation_run(
    conn: sqlite3.Connection,
    run_name: str,
    dataset_path: Optional[str],
    threshold_tag: Optional[str],
) -> int:
    existing = conn.execute(
        """
        SELECT validation_run_id
        FROM validation_runs
        WHERE run_name = ? AND COALESCE(dataset_path, '') = COALESCE(?, '') AND COALESCE(threshold_tag, '') = COALESCE(?, '')
        ORDER BY validation_run_id DESC
        LIMIT 1
        """,
        (run_name, dataset_path, threshold_tag),
    ).fetchone()
    if existing:
        return int(existing[0])

    cursor = conn.execute(
        """
        INSERT INTO validation_runs (run_name, dataset_path, threshold_tag)
        VALUES (?, ?, ?)
        """,
        (run_name, dataset_path, threshold_tag),
    )
    return int(cursor.lastrowid)


def log_inference_event(
    *,
    source_tag: str,
    request_uid: str,
    model_path: str,
    labels_path: str,
    binary_model_path: Optional[str],
    request_outcome_tag: str,
    stage_rows: List[Dict[str, Any]],
    error_code_tag: Optional[str] = None,
    error_message: Optional[str] = None,
    hint_message: Optional[str] = None,
    image_name: Optional[str] = None,
    image_mime_type: Optional[str] = None,
    image_size_bytes: Optional[int] = None,
    image_width_px: Optional[int] = None,
    image_height_px: Optional[int] = None,
    warmup_runs: Optional[int] = None,
    timed_runs: Optional[int] = None,
    inference_latency_ms: Optional[float] = None,
    api_latency_ms: Optional[float] = None,
    http_status_code: Optional[int] = None,
    raw_result: Optional[Any] = None,
    raw_error: Optional[Any] = None,
    validation_annotation: Optional[Dict[str, Any]] = None,
) -> bool:
    try:
        if source_tag not in SOURCE_TAGS:
            raise ValueError(f"invalid source_tag: {source_tag}")
        if request_outcome_tag not in REQUEST_OUTCOME_TAGS:
            raise ValueError(f"invalid request_outcome_tag: {request_outcome_tag}")
        if error_code_tag is not None and error_code_tag not in ERROR_CODE_TAGS:
            raise ValueError(f"invalid error_code_tag: {error_code_tag}")

        _validate_stage_rows(stage_rows)
        init_inference_db()

        conn = _connect(resolve_db_path())
        try:
            conn.execute("BEGIN IMMEDIATE")
            model_registry_id = _upsert_model_registry(
                conn,
                model_path=model_path,
                labels_path=labels_path,
                binary_model_path=binary_model_path,
            )

            cursor = conn.execute(
                """
                INSERT INTO inference_requests (
                    request_uid,
                    source_tag,
                    model_registry_id,
                    request_outcome_tag,
                    error_code_tag,
                    error_message,
                    hint_message,
                    image_name,
                    image_mime_type,
                    image_size_bytes,
                    image_width_px,
                    image_height_px,
                    warmup_runs,
                    timed_runs,
                    inference_latency_ms,
                    api_latency_ms,
                    http_status_code,
                    raw_result_json,
                    raw_error_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_uid,
                    source_tag,
                    model_registry_id,
                    request_outcome_tag,
                    error_code_tag,
                    error_message,
                    hint_message,
                    image_name,
                    image_mime_type,
                    image_size_bytes,
                    image_width_px,
                    image_height_px,
                    warmup_runs,
                    timed_runs,
                    inference_latency_ms,
                    api_latency_ms,
                    http_status_code,
                    _safe_debug_json(raw_result),
                    _safe_debug_json(raw_error),
                ),
            )
            inference_id = int(cursor.lastrowid)

            for row in stage_rows:
                conn.execute(
                    """
                    INSERT INTO pipeline_stages (
                        inference_id,
                        stage_tag,
                        stage_order,
                        stage_outcome_tag,
                        palm_score,
                        binary_threshold,
                        binary_gate_pass,
                        aspect_ratio,
                        brightness,
                        contrast_std,
                        sharpness,
                        quality_gate_pass,
                        predicted_label,
                        predicted_index,
                        top1_probability,
                        stage_latency_ms,
                        stage_started_at_utc,
                        stage_finished_at_utc,
                        stage_error_code_tag,
                        stage_error_message,
                        stage_hint_message,
                        stage_details_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        inference_id,
                        row.get("stage_tag"),
                        row.get("stage_order"),
                        row.get("stage_outcome_tag"),
                        row.get("palm_score"),
                        row.get("binary_threshold"),
                        row.get("binary_gate_pass"),
                        row.get("aspect_ratio"),
                        row.get("brightness"),
                        row.get("contrast_std"),
                        row.get("sharpness"),
                        row.get("quality_gate_pass"),
                        row.get("predicted_label"),
                        row.get("predicted_index"),
                        row.get("top1_probability"),
                        row.get("stage_latency_ms"),
                        row.get("stage_started_at_utc"),
                        row.get("stage_finished_at_utc"),
                        row.get("stage_error_code_tag"),
                        row.get("stage_error_message"),
                        row.get("stage_hint_message"),
                        _safe_debug_json(row.get("stage_details_json")),
                    ),
                )

            if validation_annotation:
                expected_source = validation_annotation.get("expected_source_tag")
                run_name = validation_annotation.get("run_name")
                if expected_source and run_name:
                    validation_run_id = _ensure_validation_run(
                        conn,
                        run_name=run_name,
                        dataset_path=validation_annotation.get("dataset_path"),
                        threshold_tag=validation_annotation.get("threshold_tag"),
                    )
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO validation_annotations (
                            inference_id,
                            validation_run_id,
                            expected_source_tag,
                            sample_key
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            inference_id,
                            validation_run_id,
                            expected_source,
                            validation_annotation.get("sample_key"),
                        ),
                    )

            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Inference DB log write skipped: %s", exc)
        return False
