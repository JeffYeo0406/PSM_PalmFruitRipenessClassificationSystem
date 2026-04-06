import io
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import requests
from PIL import Image, ImageFilter

API_URL = os.getenv("API_URL", "http://localhost:5000/classify")
IMAGE_PATH = os.getenv("IMAGE_PATH", "sample.jpg")
NON_PALM_DIR = os.getenv("NON_PALM_DIR", "")
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))
BLUR_RADIUS = float(os.getenv("BLUR_RADIUS", "10"))
EXPECTED_BLUR_STATUS = int(os.getenv("EXPECTED_BLUR_STATUS", "422"))
EXPECTED_NON_PALM_STATUS = int(os.getenv("EXPECTED_NON_PALM_STATUS", "422"))

QUALITY_GATE_CODES = {
    "low_resolution",
    "bad_aspect_ratio",
    "bad_exposure",
    "low_contrast",
    "blurry_image",
}


def post_image_bytes(api_url: str, filename: str, image_bytes: bytes) -> Tuple[int, str]:
    files = {"file": (filename, io.BytesIO(image_bytes), "image/jpeg")}
    try:
        response = requests.post(api_url, files=files, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not connect to API endpoint {api_url}. "
            "Start the API first: .venv\\Scripts\\python.exe api\\app.py"
        ) from exc
    return response.status_code, response.text


def derive_health_url(api_url: str) -> str:
    stripped = api_url.rstrip("/")
    if stripped.endswith("/classify"):
        return stripped.rsplit("/", 1)[0] + "/health"
    return stripped + "/health"


def ensure_api_reachable(api_url: str) -> None:
    health_url = derive_health_url(api_url)
    try:
        response = requests.get(health_url, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"API not reachable at {api_url}. "
            "Start server with: .venv\\Scripts\\python.exe api\\app.py"
        ) from exc

    if response.status_code >= 500:
        raise RuntimeError(
            f"API health endpoint returned {response.status_code}. "
            "Check server logs before running the gate test."
        )


def make_blurry_jpeg(image_path: str, radius: float) -> bytes:
    img = Image.open(image_path).convert("RGB")

    # Combine heavy blur + down/up sampling to reliably trigger the blur gate.
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    tiny_w = max(16, img.width // 8)
    tiny_h = max(16, img.height // 8)
    blurred = blurred.resize((tiny_w, tiny_h), Image.Resampling.BILINEAR)
    blurred = blurred.resize((img.width, img.height), Image.Resampling.BILINEAR)

    buffer = io.BytesIO()
    blurred.save(buffer, format="JPEG", quality=65)
    return buffer.getvalue()


def pretty_body(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw), indent=2)
    except Exception:
        return raw


def first_image_from_dir(root_dir: str) -> Optional[str]:
    if not root_dir or not os.path.isdir(root_dir):
        return None

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for current_root, _, files in os.walk(root_dir):
        for name in sorted(files):
            _, ext = os.path.splitext(name.lower())
            if ext in exts:
                return os.path.join(current_root, name)
    return None


def extract_error_code(raw: str) -> Optional[str]:
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    code = payload.get("error_code")
    return code if isinstance(code, str) else None


def gate_origin_from_response(status_code: int, raw: str) -> str:
    if status_code < 400:
        return "accepted"

    code = extract_error_code(raw)
    if code == "not_palm_fruit":
        return "binary_non_palm_gate"
    if code in QUALITY_GATE_CODES:
        return "quality_gate"
    if code:
        return f"other_gate:{code}"
    return "unknown"


def run_case(case_name: str, filename: str, image_bytes: bytes) -> Tuple[int, str, str]:
    status, body = post_image_bytes(API_URL, filename, image_bytes)
    origin = gate_origin_from_response(status, body)
    print(f"\n{case_name}")
    print("Status:", status)
    print("Gate origin:", origin)
    print("Body:\n", pretty_body(body))
    return status, body, origin


if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: IMAGE_PATH not found: {IMAGE_PATH}")
        sys.exit(1)

    print(f"API_URL={API_URL}")
    print(f"IMAGE_PATH={IMAGE_PATH}")
    print(f"NON_PALM_DIR={NON_PALM_DIR or '(not set)'}")

    try:
        ensure_api_reachable(API_URL)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    with open(IMAGE_PATH, "rb") as f:
        valid_bytes = f.read()

    try:
        valid_status, valid_body, valid_origin = run_case(
            "[1/3] Valid image request",
            os.path.basename(IMAGE_PATH),
            valid_bytes,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    blurry_bytes = make_blurry_jpeg(IMAGE_PATH, BLUR_RADIUS)
    try:
        blurry_status, blurry_body, blurry_origin = run_case(
            "[2/3] Intentionally blurry image request",
            "intentionally_blurry.jpg",
            blurry_bytes,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    non_palm_status = None
    non_palm_origin = "not_run"
    non_palm_path = first_image_from_dir(NON_PALM_DIR)
    if NON_PALM_DIR and not non_palm_path:
        print("\n[3/3] Non-palm image request")
        print("WARN: NON_PALM_DIR was provided but no image file was found.")
    elif non_palm_path:
        with open(non_palm_path, "rb") as f:
            non_palm_bytes = f.read()
        try:
            non_palm_status, non_palm_body, non_palm_origin = run_case(
                "[3/3] Non-palm image request",
                os.path.basename(non_palm_path),
                non_palm_bytes,
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        print("Non-palm source image:", non_palm_path)
        if non_palm_origin == "quality_gate":
            print("Interpretation: rejected by quality gate before semantic palm check.")
        elif non_palm_origin == "binary_non_palm_gate":
            print("Interpretation: rejected by binary palm/non-palm semantic gate.")
        elif non_palm_origin == "accepted":
            print("Interpretation: request accepted (likely binary semantic gate is disabled or threshold not met).")
        else:
            print("Interpretation: request rejected by another/unknown gate.")
    else:
        print("\n[3/3] Non-palm image request")
        print("SKIP: Set NON_PALM_DIR to run an explicit non-palm test.")

    print("\nQuick check summary")
    print("Valid status:", valid_status)
    print("Valid gate origin:", valid_origin)
    print("Blurry status:", blurry_status)
    print("Blurry gate origin:", blurry_origin)
    if blurry_status == EXPECTED_BLUR_STATUS:
        print(f"PASS: blurry image returned expected status {EXPECTED_BLUR_STATUS}.")
    else:
        print(
            f"WARN: blurry image returned {blurry_status}, expected {EXPECTED_BLUR_STATUS}. "
            "Tune BLUR_RADIUS or quality thresholds if needed."
        )

    if non_palm_status is not None:
        print("Non-palm status:", non_palm_status)
        print("Non-palm gate origin:", non_palm_origin)
        if non_palm_status == EXPECTED_NON_PALM_STATUS:
            print(f"PASS: non-palm image returned expected status {EXPECTED_NON_PALM_STATUS}.")
        else:
            print(
                f"WARN: non-palm image returned {non_palm_status}, expected {EXPECTED_NON_PALM_STATUS}. "
                "If quality gate triggered first, improve sample quality to test semantic gate."
            )
