"""Download the stored result image for a logged prediction.

    ./myenv/bin/python scripts/get_prediction_image.py 42
    ./myenv/bin/python scripts/get_prediction_image.py 42 --out /tmp/check.jpg

The stored image is the annotated result -- the prediction is already drawn on
it -- so this only fetches it from wherever it lives (object storage in
production, local disk without a bucket configured).
"""
import argparse
import os
import sys

os.environ.setdefault("LOAD_MODELS", "false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Before anything imports `config`, which reads the environment once at
# class-definition time -- import it after this and it freezes the SQLite
# default no matter what .env says.
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import storage  # noqa: E402
from app import create_app  # noqa: E402
from extensions import db  # noqa: E402
from models import PredictionLog  # noqa: E402


def fetch(log_id, out_path):
    app = create_app()
    with app.app_context():
        log = db.session.get(PredictionLog, log_id)
        if log is None:
            sys.exit(f"No prediction with id {log_id}")
        if not log.image_key:
            sys.exit(
                f"Prediction {log_id} has no stored image "
                "(it failed, or uploads were turned off)"
            )

        raw = storage.load_upload(log.image_key, app.config_object)
        if raw is None:
            sys.exit(f"{log.image_key} could not be read")

        with open(out_path, "wb") as fh:
            fh.write(raw)

        print(
            f"id={log.id} map={log.map_type} zone={log.zone_number} "
            f"rating={log.rating or '-'} -> {out_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_id", type=int)
    parser.add_argument("--out", default=None, help="defaults to ./prediction-<id>.jpg")
    args = parser.parse_args()
    fetch(args.log_id, args.out or f"prediction-{args.log_id}.jpg")
