"""Smoke-test the running prediction server against a local image.

Usage:
    python test.py                                    # TestData/test3.jpg
    python test.py TestData/test1.jpg
    python test.py TestData/test1.jpg http://192.168.1.20:4000
"""
import sys
import os
from pathlib import Path

import requests

path_prefix = Path(__file__).parent.absolute()

image_path = sys.argv[1] if len(sys.argv) > 1 else str(path_prefix / "TestData" / "test3.jpg")
base_url = sys.argv[2] if len(sys.argv) > 2 else os.environ.get(
    "BASE_URL", "http://127.0.0.1:4000"
)

if not os.path.exists(image_path):
    sys.exit(f"No such image: {image_path}")

with open(image_path, "rb") as fh:
    response = requests.post(f"{base_url}/predict", files={"file": fh})

# The server answers with an annotated JPEG on success, or JSON explaining why
# it could not predict.
content_type = response.headers.get("Content-Type", "")

if response.ok and content_type.startswith("image/"):
    output_dir = path_prefix / "TestResult"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / Path(image_path).name
    output_path.write_bytes(response.content)
    print(f"OK {response.status_code} - annotated image saved to {output_path}")
else:
    try:
        payload = response.json()
        print(f"FAILED {response.status_code} [{payload.get('code')}] {payload.get('error')}")
    except ValueError:
        print(f"FAILED {response.status_code} - unexpected body: {response.text[:200]!r}")
    sys.exit(1)
