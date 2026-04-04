import os
import json
import requests

API_URL = os.getenv("API_URL", "http://localhost:5000/classify")
IMAGE_PATH = os.getenv("IMAGE_PATH", "sample.jpg")

if __name__ == "__main__":
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": (os.path.basename(IMAGE_PATH), f, "image/jpeg")}
        resp = requests.post(API_URL, files=files, timeout=30)
        print("Status:", resp.status_code)
        print("Body:", resp.text)

    if resp.ok:
        payload = resp.json()
        req_id = payload.get("request_id")
        if req_id:
            result_url = API_URL.rstrip("/").rsplit("/", 1)[0] + f"/result/{req_id}"
            result_resp = requests.get(result_url, timeout=30)
            print("Result Status:", result_resp.status_code)
            try:
                print("Result Body:", json.dumps(result_resp.json(), indent=2))
            except Exception:
                print("Result Body:", result_resp.text)
