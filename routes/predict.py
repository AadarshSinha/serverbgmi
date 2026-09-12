"""The prediction endpoint.

Auth is optional: signed-out visitors get the same result, their request is
just logged without a user id. Every call -- success or failure -- writes one
PredictionLog row; successful ones also persist the annotated result image, so
a stored image always shows what the user was shown.
"""
import io
import time
from datetime import timedelta, timezone

import cv2
import numpy as np
from flask import Blueprint, request, jsonify, send_file, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity

import predictor
import storage
from extensions import db
from models import PredictionLog, utcnow
from errors import ApiError, PredictionError

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
@jwt_required(optional=True)
def predict():
    started = time.perf_counter()

    identity = get_jwt_identity()
    user_id = int(identity) if identity else None

    raw = _read_upload_bytes()

    # The stored image is the annotated result, so it cannot be written until
    # the prediction has actually run. A request that fails stores no image.
    log = PredictionLog(
        user_id=user_id,
        ip_hash=PredictionLog.hash_ip(_client_ip(), current_app.config["JWT_SECRET_KEY"]),
        user_agent=(request.headers.get("User-Agent") or "")[:256] or None,
        status_code=500,
        succeeded=False,
    )

    try:
        image = _decode(raw)
        result = predictor.predict_zone(image)

        log.map_type = result["map_type"]
        log.zone_number = result["zone_number"]
        _record_geometry(log, result)

        annotated = predictor.annotate(image, result)
        success, buffer = cv2.imencode(".jpg", annotated)
        if not success:
            raise ApiError("ENCODE_FAILED", "Failed to encode the result image.", 500)

        # Keep the image the user was shown, prediction drawn on and all.
        annotated_bytes = buffer.tobytes()
        log.image_key = storage.save_upload(
            annotated_bytes, "result.jpg", current_app.config_object
        )
        log.image_bytes = len(annotated_bytes)

        log.succeeded = True
        log.status_code = 200
        _finalise(log, started)

        io_buf = io.BytesIO(buffer)
        io_buf.seek(0)
        response = send_file(io_buf, mimetype="image/jpeg")
        # The body is a JPEG, so the row id travels in a header. CORS has to
        # expose it explicitly or the browser hides it from JavaScript.
        if log.id is not None:
            response.headers["X-Prediction-Id"] = str(log.id)
        return response

    except ApiError as err:
        log.error_code = err.code
        log.status_code = err.status
        _finalise(log, started)
        print(f"Prediction failed [{err.code}]: {err.message}")
        return jsonify(err.to_dict()), err.status

    except Exception:
        import traceback

        log.error_code = "INTERNAL_ERROR"
        log.status_code = 500
        _finalise(log, started)
        print(traceback.format_exc())
        return (
            jsonify(
                {
                    "code": "INTERNAL_ERROR",
                    "error": "Something went wrong while processing that image.",
                }
            ),
            500,
        )


# How long a prediction stays ratable. Long enough to come back to the tab,
# short enough that an id cannot be rated months later by someone guessing.
FEEDBACK_WINDOW = timedelta(hours=24)


@predict_bp.route("/predict/<int:log_id>/feedback", methods=["POST"])
@jwt_required(optional=True)
def feedback(log_id):
    """Record how good the user thought a prediction was.

    Deliberately forgiving about who can answer -- signed-out visitors make
    most of the predictions, so requiring an account would throw away most of
    the signal. A prediction made while signed in can only be rated by that
    same user; everything else is protected by the row being ratable once,
    within a day, and only if it succeeded.
    """
    payload = request.get_json(silent=True) or {}
    rating = (payload.get("rating") or "").strip()

    if rating not in PredictionLog.RATINGS:
        raise ApiError(
            "INVALID_RATING",
            "Rating must be one of: " + ", ".join(PredictionLog.RATINGS) + ".",
            400,
        )

    log = db.session.get(PredictionLog, log_id)
    if log is None or not log.succeeded:
        raise ApiError("UNKNOWN_PREDICTION", "That prediction could not be found.", 404)

    identity = get_jwt_identity()
    user_id = int(identity) if identity else None
    if log.user_id is not None and log.user_id != user_id:
        raise ApiError("NOT_YOUR_PREDICTION", "That prediction belongs to someone else.", 403)

    if log.rating is not None:
        raise ApiError("ALREADY_RATED", "That prediction has already been rated.", 409)

    if log.created_at and (utcnow() - _as_utc(log.created_at)) > FEEDBACK_WINDOW:
        raise ApiError(
            "FEEDBACK_WINDOW_CLOSED",
            "That prediction is too old to rate.",
            422,
        )

    log.rating = rating
    log.rated_at = utcnow()
    db.session.commit()

    return jsonify({"ok": True, "rating": log.rating}), 200


def _record_geometry(log, result):
    """Store the numbers the annotated image is drawn from.

    Keeping these means the result image is reproducible from the original
    screenshot, so only one image per prediction has to be stored -- and the
    clean one, which is the one worth retraining on.
    """
    detected = result.get("frame_center")
    predicted = result.get("predicted_center")
    if detected is not None:
        log.detected_center_x = int(detected[0])
        log.detected_center_y = int(detected[1])
    if predicted is not None:
        log.predicted_center_x = int(predicted[0])
        log.predicted_center_y = int(predicted[1])
    if result.get("predicted_radius") is not None:
        log.predicted_radius = int(result["predicted_radius"])


def _as_utc(value):
    """Postgres hands back an aware datetime; SQLite hands back a naive one.

    Both store UTC, so the naive case just needs the tzinfo attached -- without
    this, subtracting them raises TypeError on SQLite and the whole feedback
    window check dies.
    """
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _read_upload_bytes():
    if "file" not in request.files:
        raise PredictionError("NO_FILE", "No image was uploaded.", status=400)
    raw = request.files["file"].read()
    if not raw:
        raise PredictionError("EMPTY_FILE", "The uploaded file was empty.", status=400)
    return raw


def _decode(raw):
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise PredictionError(
            "INVALID_IMAGE",
            "That file could not be read as an image. Please upload a PNG or JPG "
            "screenshot.",
            status=400,
        )
    return image


def _client_ip():
    # X-Forwarded-For is a client-controlled header, but behind the ALB/nginx
    # the left-most entry is the closest thing to a real client address. It is
    # only ever hashed, never trusted for authorisation.
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr


def _finalise(log, started):
    """Write the usage row. Analytics must never break a prediction."""
    log.duration_ms = int((time.perf_counter() - started) * 1000)
    try:
        db.session.add(log)
        db.session.commit()
    except Exception as exc:  # noqa: BLE001
        db.session.rollback()
        print(f"WARNING: could not write prediction log ({type(exc).__name__}: {exc})")
