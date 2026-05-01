"""
api_additions.py — Paste into api/app.py

Three changes:
  1. /classify patch  — reads preprocess_family from request.form to support
                        per-request preprocessing (fixes web app model switching)
  2. GET /history     — queries v_request_pipeline_trace + pipeline_stages
  3. GET /stats       — aggregates from inference_requests + pipeline_stages

All field names are exact column names from inference_db.py DDL.
"""

# ── ADD TO IMPORTS at top of app.py ──────────────────────────────────────────
import json as _json
import sqlite3 as _sqlite3

# ── FIX #1: /classify preprocess_family override ─────────────────────────────
#
# Problem: pi_inference.py reads MODEL_PREPROCESS_FAMILY from env at module-load
#   time. When the web app switches model architecture (e.g. V2→ResNet18), the
#   preprocessing applied during inference is still the one set at server startup.
#
# Fix: read an optional "preprocess_family" field from the multipart form and
#   pass it to predict_bytes(). This lets the web app send the correct family per
#   architecture without restarting the server.
#
# SUPPORTED_PREPROCESS_FAMILIES (from pi_inference.py):
#   mobilenet_v2 | mobilenet_v3 | efficientnet | imagenet_timm | imagenet_torchvision | none
#
# Insert this block INSIDE the existing /classify route, just before the
# predict_bytes() call, replacing the bare:
#   result = predict_bytes(image_bytes, bundle=bundle, labels=labels, ...)
# with:

"""
PATCH for /classify route in app.py:

    # Read optional per-request preprocess family sent by web app.
    # Falls back to the module-level MODEL_PREPROCESS_FAMILY if not provided.
    from scripts.pi_inference import MODEL_PREPROCESS_FAMILY, _normalize_preprocess_family
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
        preprocess_family=effective_preprocess_family,   # <-- was hardcoded
    )
"""


# ── HELPER — DB connection ────────────────────────────────────────────────────
def _db():
    """Open a read-only WAL connection to inference_log.db."""
    from inference_db import resolve_db_path
    conn = _sqlite3.connect(resolve_db_path(), timeout=5.0)
    conn.row_factory = _sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


# ── GET /history ──────────────────────────────────────────────────────────────
#
# Queries v_request_pipeline_trace (the view defined in inference_db.py) joined
# with model_registry to get model_path, plus per-request pipeline_stages rows.
#
# Query params:
#   filter   — 'all' | model_path fragment (e.g. 'int8','float16') | source_tag ('api','cli')
#   page     — 1-based page number (default 1)
#   per_page — rows per page (default 50, max 100)
#
# Response row fields match v_request_pipeline_trace columns + model_registry.model_path:
#   request_uid, source_tag, request_outcome_tag, error_code_tag,
#   model_path (from model_registry), image_name, image_size_bytes,
#   image_width_px, image_height_px,
#   inference_latency_ms, api_latency_ms, http_status_code, created_at_utc,
#   binary_gate_outcome, quality_gate_outcome, ripeness_stage_outcome,
#   predicted_label, top1_probability,
#   stages: [{stage_tag, stage_order, stage_outcome_tag, palm_score,
#             binary_threshold, aspect_ratio, brightness, contrast_std, sharpness,
#             predicted_label, top1_probability, stage_latency_ms,
#             stage_error_code_tag, stage_error_message, stage_hint_message}]

@app.route("/history", methods=["GET"])
def history():
    filter_val = request.args.get("filter", "all").strip()
    page       = max(1, int(request.args.get("page", 1)))
    per_page   = min(100, max(1, int(request.args.get("per_page", 50))))
    offset     = (page - 1) * per_page

    # source_tag filter: 'api' | 'cli'
    source_tags = {"api", "cli"}
    by_source = filter_val in source_tags

    try:
        with _db() as conn:
            if filter_val == "all":
                where, params = "1=1", []
            elif by_source:
                # source_tag column in inference_requests
                where, params = "ir.source_tag = ?", [filter_val]
            else:
                # model_path fragment (absolute path stored in model_registry)
                where, params = "mr.model_path LIKE ?", [f"%{filter_val}%"]

            total = conn.execute(
                f"""SELECT COUNT(*)
                    FROM inference_requests ir
                    JOIN model_registry mr ON mr.model_registry_id = ir.model_registry_id
                    WHERE {where}""",
                params,
            ).fetchone()[0]

            # Use v_request_pipeline_trace for the pivoted stage outcomes
            rows = conn.execute(
                f"""SELECT
                        vt.request_uid,
                        vt.source_tag,
                        vt.request_outcome_tag,
                        vt.error_code_tag,
                        mr.model_path,
                        vt.image_name,
                        vt.image_size_bytes,
                        vt.image_width_px,
                        vt.image_height_px,
                        vt.inference_latency_ms,
                        vt.api_latency_ms,
                        vt.http_status_code,
                        vt.created_at_utc,
                        vt.binary_gate_outcome,
                        vt.quality_gate_outcome,
                        vt.ripeness_stage_outcome,
                        vt.predicted_label,
                        vt.top1_probability,
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
                # Fetch per-stage detail from pipeline_stages for this inference_id
                stage_rows = conn.execute(
                    """SELECT
                           stage_tag, stage_order, stage_outcome_tag,
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
                    # v_request_pipeline_trace columns
                    "request_uid":           row["request_uid"],
                    "source_tag":            row["source_tag"],
                    "request_outcome_tag":   row["request_outcome_tag"],
                    "error_code_tag":        row["error_code_tag"],
                    # model_registry.model_path (absolute; client shows basename)
                    "model_path":            row["model_path"],
                    "image_name":            row["image_name"],
                    "image_size_bytes":      row["image_size_bytes"],
                    "image_width_px":        row["image_width_px"],
                    "image_height_px":       row["image_height_px"],
                    # inference_requests latency fields
                    "inference_latency_ms":  row["inference_latency_ms"],
                    "api_latency_ms":        row["api_latency_ms"],
                    "http_status_code":      row["http_status_code"],
                    "created_at_utc":        row["created_at_utc"],
                    # pivoted stage outcome columns from view
                    "binary_gate_outcome":   row["binary_gate_outcome"],
                    "quality_gate_outcome":  row["quality_gate_outcome"],
                    "ripeness_stage_outcome":row["ripeness_stage_outcome"],
                    # ripeness result columns from view
                    "predicted_label":       row["predicted_label"],
                    "top1_probability":      row["top1_probability"],
                    # full stage detail array
                    "stages":                stages,
                })

    except Exception as exc:
        logging.warning("history endpoint error: %s", exc)
        return jsonify({"records": [], "total": 0, "page": page, "per_page": per_page, "error": str(exc)})

    return jsonify({"records": records, "total": total, "page": page, "per_page": per_page})


# ── GET /stats ────────────────────────────────────────────────────────────────
#
# Aggregates from inference_requests + pipeline_stages.
#
# Query params:
#   filter — same as /history
#
# Response:
#   total                  — COUNT(*) inference_requests
#   accepted               — COUNT WHERE request_outcome_tag='accepted'
#   gate_rejected          — COUNT WHERE request_outcome_tag='gate_rejected'
#   runtime_errors         — COUNT WHERE request_outcome_tag IN ('runtime_error','input_error')
#   avg_inference_latency_ms — AVG(inference_latency_ms) — model-only timing
#   avg_api_latency_ms     — AVG(api_latency_ms)
#   class_dist             — {Ripe:N, Overripe:N, Underripe:N} from pipeline_stages.predicted_label
#   rejection_breakdown    — {error_code_tag: count} from inference_requests WHERE gate_rejected
#   latency_series         — last 30 inference_latency_ms values (newest-first)

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
                        COUNT(*)                                                                    AS total,
                        SUM(CASE WHEN ir.request_outcome_tag='accepted'     THEN 1 ELSE 0 END)     AS accepted,
                        SUM(CASE WHEN ir.request_outcome_tag='gate_rejected' THEN 1 ELSE 0 END)    AS gate_rejected,
                        SUM(CASE WHEN ir.request_outcome_tag IN ('runtime_error','input_error')
                                 THEN 1 ELSE 0 END)                                                AS runtime_errors,
                        AVG(CASE WHEN ir.inference_latency_ms IS NOT NULL
                                 THEN ir.inference_latency_ms END)                                 AS avg_inference_latency_ms,
                        AVG(CASE WHEN ir.api_latency_ms IS NOT NULL
                                 THEN ir.api_latency_ms END)                                       AS avg_api_latency_ms
                    FROM inference_requests ir
                    JOIN model_registry mr ON mr.model_registry_id = ir.model_registry_id
                    WHERE {where}""",
                params,
            ).fetchone()

            # class_dist — from pipeline_stages.predicted_label WHERE stage_tag='ripeness_classification'
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

            # rejection_breakdown — error_code_tag from inference_requests WHERE gate_rejected
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

            # latency_series — last 30 inference_latency_ms (newest-first for the chart)
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
        "latency_series":          latency_series,  # newest-first; frontend reverses
    })
