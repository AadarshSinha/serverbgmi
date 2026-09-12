"""Email + password authentication.

Accounts are optional today: predictions work signed out. Auth exists so that
usage can be attributed to a person and so the paid tier has something to hang
off once Cashfree is switched on.
"""
import re
from datetime import timedelta

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from sqlalchemy.exc import IntegrityError

from extensions import db
from models import User, utcnow
from errors import ApiError

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8


def _issue_token(user):
    expires = timedelta(hours=current_app.config["JWT_ACCESS_TOKEN_EXPIRES_HOURS"])
    # flask-jwt-extended requires the subject to be a string.
    return create_access_token(identity=str(user.id), expires_delta=expires)


def _clean_credentials(payload):
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""

    if not EMAIL_RE.match(email):
        raise ApiError("VALIDATION_ERROR", "Please enter a valid email address.", 400)
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ApiError(
            "VALIDATION_ERROR",
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters.",
            400,
        )
    return email, password


@auth_bp.route("/signup", methods=["POST"])
def signup():
    payload = request.get_json(silent=True) or {}
    email, password = _clean_credentials(payload)
    name = (payload.get("name") or "").strip() or None

    user = User(email=email, name=name)
    user.set_password(password)

    db.session.add(user)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        raise ApiError(
            "EMAIL_TAKEN", "An account with that email already exists.", 409
        )

    return jsonify({"token": _issue_token(user), "user": user.to_dict()}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""

    user = User.query.filter_by(email=email).first()
    # Same message either way so the endpoint cannot be used to enumerate emails.
    if user is None or not user.check_password(password):
        raise ApiError("INVALID_CREDENTIALS", "Incorrect email or password.", 401)
    if not user.is_active:
        raise ApiError("ACCOUNT_DISABLED", "This account has been disabled.", 403)

    user.last_login_at = utcnow()
    db.session.commit()

    return jsonify({"token": _issue_token(user), "user": user.to_dict()})


@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    user = db.session.get(User, int(get_jwt_identity()))
    if user is None or not user.is_active:
        raise ApiError("ACCOUNT_DISABLED", "This account is no longer active.", 403)
    return jsonify({"user": user.to_dict()})
