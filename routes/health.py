"""Liveness endpoints.

The frontend polls /health to decide whether to show the "server offline"
banner, so this must stay cheap and must never depend on the ML models being
warm.
"""
from flask import Blueprint, jsonify, current_app

import predictor
from config import SUPPORTED_MAPS

health_bp = Blueprint("health", __name__)

API_VERSION = "1.1.0"


def _payload():
    return {
        "status": "ok",
        "version": API_VERSION,
        "modelsLoaded": predictor.is_ready(),
        "supportedMaps": list(SUPPORTED_MAPS.values()),
        "billingEnabled": bool(current_app.config.get("BILLING_ENABLED")),
    }


@health_bp.route("/", methods=["GET"])
def home():
    # Kept for backwards compatibility with the original health check.
    payload = _payload()
    payload["message"] = "Server is live!"
    return jsonify(payload)


@health_bp.route("/health", methods=["GET"])
def health():
    return jsonify(_payload())
