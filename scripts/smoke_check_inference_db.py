import argparse
import os
import sqlite3
import sys
from typing import Iterable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference_db import MIGRATION_ID, init_inference_db, resolve_db_path


REQUIRED_TABLES = {
    "schema_migrations",
    "model_registry",
    "inference_requests",
    "pipeline_stages",
    "validation_runs",
    "validation_annotations",
}
REQUIRED_INDEXES = {
    "idx_requests_created_at",
    "idx_requests_source_outcome",
    "idx_requests_model",
    "idx_stages_stage_outcome",
    "idx_stages_error_code",
    "idx_validation_expected_source",
}
REQUIRED_VIEWS = {
    "v_request_pipeline_trace",
    "v_validation_stage_summary",
}


def _fetch_names(conn: sqlite3.Connection, kind: str) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = ?",
        (kind,),
    ).fetchall()
    return {str(row[0]) for row in rows}


def _assert_contains(found: set[str], required: set[str], label: str) -> None:
    missing = sorted(required - found)
    if missing:
        raise RuntimeError(f"Missing {label}: {', '.join(missing)}")


def _must_fail(callable_obj, failure_label: str) -> None:
    try:
        callable_obj()
    except sqlite3.IntegrityError:
        return
    raise RuntimeError(f"Constraint probe did not fail as expected: {failure_label}")


def _run_constraint_probes(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
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
                "C:/probe/model.tflite",
                "C:/probe/labels.json",
                "C:/probe/binary.tflite",
                "a" * 64,
                123,
            ),
        )
        model_registry_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

        _must_fail(
            lambda: conn.execute(
                """
                INSERT INTO inference_requests (
                    request_uid,
                    source_tag,
                    model_registry_id,
                    request_outcome_tag
                )
                VALUES (?, ?, ?, ?)
                """,
                ("probe-invalid-source", "invalid", model_registry_id, "accepted"),
            ),
            "invalid source_tag",
        )

        _must_fail(
            lambda: conn.execute(
                """
                INSERT INTO inference_requests (
                    request_uid,
                    source_tag,
                    model_registry_id,
                    request_outcome_tag,
                    raw_result_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "probe-oversized-json",
                    "api",
                    model_registry_id,
                    "accepted",
                    "x" * 9000,
                ),
            ),
            "oversized raw_result_json",
        )

        conn.execute(
            """
            INSERT INTO inference_requests (
                request_uid,
                source_tag,
                model_registry_id,
                request_outcome_tag,
                http_status_code
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            ("probe-valid", "api", model_registry_id, "accepted", 200),
        )

        _must_fail(
            lambda: conn.execute(
                """
                INSERT INTO pipeline_stages (
                    inference_id,
                    stage_tag,
                    stage_order,
                    stage_outcome_tag
                )
                VALUES (?, ?, ?, ?)
                """,
                (99999999, "binary_gate", 1, "passed"),
            ),
            "orphan pipeline_stages FK",
        )
    finally:
        conn.execute("ROLLBACK")


def _row_count(conn: sqlite3.Connection, table_name: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])


def run_checks(db_path: str) -> None:
    os.environ["INFERENCE_DB_PATH"] = db_path
    init_inference_db()

    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        tables = _fetch_names(conn, "table")
        indexes = _fetch_names(conn, "index")
        views = _fetch_names(conn, "view")

        _assert_contains(tables, REQUIRED_TABLES, "tables")
        _assert_contains(indexes, REQUIRED_INDEXES, "indexes")
        _assert_contains(views, REQUIRED_VIEWS, "views")

        migration = conn.execute(
            "SELECT migration_id FROM schema_migrations WHERE migration_id = ?",
            (MIGRATION_ID,),
        ).fetchone()
        if not migration:
            raise RuntimeError(f"Migration marker missing: {MIGRATION_ID}")

        # Verify views can be queried.
        conn.execute("SELECT * FROM v_request_pipeline_trace LIMIT 1").fetchall()
        conn.execute("SELECT * FROM v_validation_stage_summary LIMIT 1").fetchall()

        _run_constraint_probes(conn)

        print("Inference DB smoke check PASSED")
        print(f"db_path={db_path}")
        print(f"tables_checked={len(REQUIRED_TABLES)}")
        print(f"indexes_checked={len(REQUIRED_INDEXES)}")
        print(f"views_checked={len(REQUIRED_VIEWS)}")
        print(f"existing_request_rows={_row_count(conn, 'inference_requests')}")
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check SQLite inference logging schema")
    parser.add_argument(
        "--db",
        default=resolve_db_path(),
        help="Path to inference SQLite DB (defaults to INFERENCE_DB_PATH or reports/inference_log.db)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = os.path.abspath(args.db)
    try:
        run_checks(db_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Inference DB smoke check FAILED: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
