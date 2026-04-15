import argparse
import json
import mimetypes
import os
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "reports" / "inference_log.db"
DEFAULT_PALM_DIR = Path("C:/Users/jeffy/Documents/PSM/PalmDetector/Palm")
DEFAULT_NON_PALM_DIR = Path("C:/Users/jeffy/Documents/PSM/PalmDetector/Non-palm")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class StepResult:
    ok: bool
    detail: str


@dataclass
class ApiResult:
    status_code: int
    payload: Dict[str, object]


@dataclass
class CliResult:
    exit_code: int
    payload: Dict[str, object]
    stdout: str
    stderr: str


def _python_exe() -> str:
    return sys.executable


def _find_first_image(root: Path) -> Optional[Path]:
    if not root.exists() or not root.is_dir():
        return None
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    return None


def _build_multipart(file_path: Path, field_name: str = "file") -> Tuple[bytes, str]:
    boundary = f"----Boundary{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()

    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        (
            f"Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{file_path.name}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8"),
        file_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    body = b"".join(parts)
    return body, boundary


def _http_get_json(url: str, timeout: int = 10) -> Dict[str, object]:
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_for_health(base_url: str, timeout_seconds: int = 30) -> StepResult:
    deadline = time.time() + timeout_seconds
    health_url = f"{base_url}/health"
    last_error = "unknown"

    while time.time() < deadline:
        try:
            payload = _http_get_json(health_url, timeout=5)
            if isinstance(payload, dict) and payload.get("ready") is not None:
                return StepResult(True, f"health endpoint ready: {payload.get('status')}")
            last_error = f"unexpected health payload: {payload}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(1)

    return StepResult(False, f"health endpoint not ready within timeout: {last_error}")


def _post_classify(base_url: str, image_path: Path) -> ApiResult:
    body, boundary = _build_multipart(image_path)
    req = urllib.request.Request(
        url=f"{base_url}/classify",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = int(resp.getcode())
            payload = json.loads(resp.read().decode("utf-8"))
            return ApiResult(status, payload)
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8")
        payload = json.loads(body_text) if body_text else {}
        return ApiResult(int(exc.code), payload)


def _extract_json_from_stdout(stdout: str) -> Dict[str, object]:
    text = stdout.strip()
    if not text:
        return {}

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:  # noqa: BLE001
        return {}


def _run_cli(image_path: Path, db_path: Path) -> CliResult:
    env = os.environ.copy()
    env["INFERENCE_DB_PATH"] = str(db_path)
    cmd = [
        _python_exe(),
        str(PROJECT_ROOT / "scripts" / "pi_inference.py"),
        "--image",
        str(image_path),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = _extract_json_from_stdout(completed.stdout)
    return CliResult(completed.returncode, payload, completed.stdout, completed.stderr)


def _run_smoke_check(db_path: Path) -> StepResult:
    env = os.environ.copy()
    env["INFERENCE_DB_PATH"] = str(db_path)
    cmd = [
        _python_exe(),
        str(PROJECT_ROOT / "scripts" / "smoke_check_inference_db.py"),
        "--db",
        str(db_path),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return StepResult(True, completed.stdout.strip())
    return StepResult(False, (completed.stdout + "\n" + completed.stderr).strip())


def _db_connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _db_baseline(db_path: Path) -> Tuple[int, int, int]:
    if not db_path.exists():
        return 0, 0, 0
    conn = _db_connect(db_path)
    try:
        max_inference_id = int(conn.execute("SELECT COALESCE(MAX(inference_id), 0) FROM inference_requests").fetchone()[0])
        req_count = int(conn.execute("SELECT COUNT(*) FROM inference_requests").fetchone()[0])
        stage_count = int(conn.execute("SELECT COUNT(*) FROM pipeline_stages").fetchone()[0])
        return max_inference_id, req_count, stage_count
    finally:
        conn.close()


def _db_postcheck(db_path: Path, baseline_max_id: int) -> Dict[str, object]:
    conn = _db_connect(db_path)
    try:
        new_req = int(
            conn.execute(
                "SELECT COUNT(*) FROM inference_requests WHERE inference_id > ?",
                (baseline_max_id,),
            ).fetchone()[0]
        )
        new_stage = int(
            conn.execute(
                "SELECT COUNT(*) FROM pipeline_stages WHERE inference_id > ?",
                (baseline_max_id,),
            ).fetchone()[0]
        )
        source_counts = conn.execute(
            """
            SELECT source_tag, COUNT(*)
            FROM inference_requests
            WHERE inference_id > ?
            GROUP BY source_tag
            ORDER BY source_tag
            """,
            (baseline_max_id,),
        ).fetchall()
        latest = conn.execute(
            """
            SELECT request_uid, source_tag, request_outcome_tag, error_code_tag, image_name, http_status_code
            FROM inference_requests
            WHERE inference_id > ?
            ORDER BY inference_id DESC
            LIMIT 8
            """,
            (baseline_max_id,),
        ).fetchall()

        return {
            "new_requests": new_req,
            "new_stages": new_stage,
            "source_counts": source_counts,
            "latest_rows": latest,
        }
    finally:
        conn.close()


def _start_api(db_path: Path, port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["INFERENCE_DB_PATH"] = str(db_path)
    env["PORT"] = str(port)
    cmd = [_python_exe(), str(PROJECT_ROOT / "api" / "app.py")]
    return subprocess.Popen(  # noqa: S603
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _stop_api(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full pre-deploy dry-run for API/CLI inference logging")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite DB path")
    parser.add_argument("--port", type=int, default=5003, help="Local API port for dry-run")
    parser.add_argument("--palm-dir", default=str(DEFAULT_PALM_DIR), help="Palm image directory")
    parser.add_argument("--non-palm-dir", default=str(DEFAULT_NON_PALM_DIR), help="Non-palm image directory")
    parser.add_argument("--palm-image", default="", help="Explicit palm image path override")
    parser.add_argument("--non-palm-image", default="", help="Explicit non-palm image path override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).resolve()

    palm_image = Path(args.palm_image).resolve() if args.palm_image else _find_first_image(Path(args.palm_dir))
    non_palm_image = (
        Path(args.non_palm_image).resolve() if args.non_palm_image else _find_first_image(Path(args.non_palm_dir))
    )

    failures = []

    if not palm_image or not palm_image.exists():
        failures.append("Palm image not found")
    if not non_palm_image or not non_palm_image.exists():
        failures.append("Non-palm image not found")

    if failures:
        print("PREDEPLOY_DRY_RUN: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    baseline_max_id, baseline_req_count, baseline_stage_count = _db_baseline(db_path)
    print(f"Baseline DB: requests={baseline_req_count}, stages={baseline_stage_count}, max_inference_id={baseline_max_id}")
    print(f"Palm image: {palm_image}")
    print(f"Non-palm image: {non_palm_image}")

    proc = _start_api(db_path=db_path, port=args.port)
    api_stdout_tail = ""
    try:
        health = _wait_for_health(base_url=f"http://127.0.0.1:{args.port}")
        if not health.ok:
            failures.append(health.detail)
        else:
            print(f"API health: {health.detail}")

        if not failures:
            api_palm = _post_classify(base_url=f"http://127.0.0.1:{args.port}", image_path=palm_image)
            api_non = _post_classify(base_url=f"http://127.0.0.1:{args.port}", image_path=non_palm_image)
            print(f"API palm status={api_palm.status_code} payload={api_palm.payload}")
            print(f"API non_palm status={api_non.status_code} payload={api_non.payload}")

            if api_palm.status_code not in {200, 422, 400, 503}:
                failures.append(f"Unexpected API palm status: {api_palm.status_code}")
            if api_non.status_code not in {200, 422, 400, 503}:
                failures.append(f"Unexpected API non-palm status: {api_non.status_code}")

    finally:
        _stop_api(proc)
        if proc.stdout is not None:
            try:
                remaining = proc.stdout.read() or ""
                api_stdout_tail = remaining[-4000:]
            except Exception:  # noqa: BLE001
                api_stdout_tail = ""

    cli_palm = _run_cli(palm_image, db_path)
    cli_non = _run_cli(non_palm_image, db_path)
    print(f"CLI palm exit={cli_palm.exit_code} payload={cli_palm.payload}")
    print(f"CLI non_palm exit={cli_non.exit_code} payload={cli_non.payload}")

    if cli_palm.exit_code not in {0, 1, 2}:
        failures.append(f"Unexpected CLI palm exit code: {cli_palm.exit_code}")
    if cli_non.exit_code not in {0, 1, 2}:
        failures.append(f"Unexpected CLI non-palm exit code: {cli_non.exit_code}")

    smoke = _run_smoke_check(db_path)
    print(f"Smoke check: {'PASS' if smoke.ok else 'FAIL'}")
    print(smoke.detail)
    if not smoke.ok:
        failures.append("Smoke check script failed")

    post = _db_postcheck(db_path, baseline_max_id)
    print(f"DB delta requests={post['new_requests']} stages={post['new_stages']}")
    print(f"DB source counts (new rows)={post['source_counts']}")
    print("DB latest rows (new)")
    for row in post["latest_rows"]:
        print(f"- {row}")

    source_counts = dict(post["source_counts"])
    if post["new_requests"] < 4:
        failures.append("Expected at least 4 new request rows (2 API + 2 CLI)")
    if post["new_stages"] != post["new_requests"] * 3:
        failures.append("Stage rows are not exactly 3x request rows")
    if source_counts.get("api", 0) < 2:
        failures.append("Expected at least 2 API rows in this dry run")
    if source_counts.get("cli", 0) < 2:
        failures.append("Expected at least 2 CLI rows in this dry run")

    if failures:
        print("PREDEPLOY_DRY_RUN: FAIL")
        for failure in failures:
            print(f"- {failure}")
        if api_stdout_tail:
            print("API output tail:")
            print(api_stdout_tail)
        raise SystemExit(1)

    print("PREDEPLOY_DRY_RUN: PASS")


if __name__ == "__main__":
    main()
