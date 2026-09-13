"""Product feedback: one question, one free-text answer.

Separate from `/predict/<id>/feedback`, which rates a single prediction. This
is the site-wide prompt asking what someone actually wants from the tool.
"""
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity

from extensions import db
from errors import ApiError
from models import Feedback, PredictionLog

feedback_bp = Blueprint("feedback", __name__)

MAX_MESSAGE_LENGTH = 2000


@feedback_bp.route("/feedback", methods=["POST"])
@jwt_required(optional=True)
def submit_feedback():
    payload = request.get_json(silent=True) or {}

    message = (payload.get("message") or "").strip()
    game = (payload.get("game") or "").strip() or None

    if game is not None and game not in Feedback.GAMES:
        raise ApiError(
            "VALIDATION_ERROR",
            "Game must be one of: " + ", ".join(Feedback.GAMES) + ".",
            400,
        )

    # A tap on its own is a complete answer -- requiring text alongside it is
    # what kills the response rate this prompt exists to protect.
    if not message and game is None:
        raise ApiError("VALIDATION_ERROR", "Please pick an option or write something.", 400)

    if len(message) > MAX_MESSAGE_LENGTH:
        raise ApiError(
            "VALIDATION_ERROR",
            f"Please keep it under {MAX_MESSAGE_LENGTH} characters.",
            400,
        )

    identity = get_jwt_identity()

    entry = Feedback(
        game=game,
        message=message or None,
        user_id=int(identity) if identity else None,
        # Sent by the browser; both are plain strings like "Asia/Kolkata" and
        # "en-US", and neither requires the user to grant anything.
        timezone=(payload.get("timezone") or "")[:64] or None,
        locale=(payload.get("locale") or "")[:32] or None,
        ip_hash=PredictionLog.hash_ip(_client_ip(), current_app.config["JWT_SECRET_KEY"]),
        user_agent=(request.headers.get("User-Agent") or "")[:256] or None,
    )

    db.session.add(entry)
    db.session.commit()

    return jsonify({"ok": True}), 201


def _client_ip():
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr
