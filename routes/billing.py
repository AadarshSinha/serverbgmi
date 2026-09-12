"""Cashfree billing scaffold.

Nothing here charges anyone yet. The endpoints, database rows and webhook
signature check are in place so that switching billing on is a configuration
change (BILLING_ENABLED + credentials) plus filling in the one marked TODO,
rather than a new feature built under time pressure.
"""
import base64
import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity

from extensions import db
from models import User, Payment
from errors import ApiError

billing_bp = Blueprint("billing", __name__, url_prefix="/billing")

# Prices are illustrative until the first real plan is published. The "free"
# plan id is persisted on User.plan, so it stays as-is; only the display name
# is neutral, since the entry tier is not promised to stay at zero forever.
PLANS = {
    "free": {
        "id": "free",
        "name": "Starter",
        "amount": 0,
        "currency": "INR",
        "features": [
            "Next-zone predictions",
            "Erangel and Miramar",
            "Predicted circle and direction drawn on your screenshot",
        ],
    },
    "pro": {
        "id": "pro",
        "name": "Pro",
        "amount": 199,
        "currency": "INR",
        "interval": "month",
        "features": [
            "Everything in Starter",
            "Priority prediction queue",
            "Multi-zone lookahead",
            "Match history and accuracy tracking",
        ],
    },
}


@billing_bp.route("/plans", methods=["GET"])
def plans():
    return jsonify(
        {
            "plans": list(PLANS.values()),
            "billingEnabled": bool(current_app.config.get("BILLING_ENABLED")),
        }
    )


@billing_bp.route("/checkout", methods=["POST"])
@jwt_required()
def checkout():
    """Create a Cashfree order and return its payment session id.

    Returns 503 until billing is switched on, so the frontend can render the
    upgrade flow today without being able to take money by accident.
    """
    if not current_app.config.get("BILLING_ENABLED"):
        raise ApiError(
            "BILLING_DISABLED",
            "Payments are not enabled yet. Pro plans are coming soon.",
            503,
        )

    payload = request.get_json(silent=True) or {}
    plan_id = (payload.get("plan") or "").strip().lower()
    plan = PLANS.get(plan_id)
    if plan is None or plan_id == "free":
        raise ApiError("INVALID_PLAN", "Choose a valid paid plan.", 400)

    user = db.session.get(User, int(get_jwt_identity()))
    if user is None:
        raise ApiError("ACCOUNT_DISABLED", "This account is no longer active.", 403)

    order = Payment(
        user_id=user.id,
        order_id=f"zp_{uuid.uuid4().hex[:24]}",
        plan=plan_id,
        amount=plan["amount"],
        currency=plan["currency"],
        status="created",
    )
    db.session.add(order)
    db.session.commit()

    # TODO(billing): POST to https://api.cashfree.com/pg/orders with
    # x-client-id / x-client-secret headers, then store the returned
    # payment_session_id on `order` and hand it to the frontend SDK.
    raise ApiError(
        "BILLING_NOT_IMPLEMENTED",
        "Checkout is not wired up yet. Pro plans are coming soon.",
        503,
    )


@billing_bp.route("/webhook", methods=["POST"])
def webhook():
    """Receive payment status updates from Cashfree.

    Signature verification runs first: an unverified webhook must never be
    allowed to upgrade an account.
    """
    raw_body = request.get_data(as_text=True)
    timestamp = request.headers.get("x-webhook-timestamp", "")
    signature = request.headers.get("x-webhook-signature", "")

    if not _verify_signature(raw_body, timestamp, signature):
        raise ApiError("INVALID_SIGNATURE", "Webhook signature verification failed.", 401)

    event = json.loads(raw_body or "{}")
    order_id = (event.get("data", {}).get("order", {}) or {}).get("order_id")
    if not order_id:
        raise ApiError("INVALID_PAYLOAD", "Webhook payload had no order id.", 400)

    order = Payment.query.filter_by(order_id=order_id).first()
    if order is None:
        # 200 so Cashfree stops retrying a webhook we can never match.
        return jsonify({"status": "ignored", "reason": "unknown order"}), 200

    event_type = (event.get("type") or "").upper()
    order.raw_payload = raw_body

    if "SUCCESS" in event_type:
        order.status = "paid"
        user = db.session.get(User, order.user_id) if order.user_id else None
        if user is not None:
            user.plan = order.plan
            user.plan_expires_at = _next_renewal()
    elif "FAILED" in event_type:
        order.status = "failed"
    elif "DROPPED" in event_type or "CANCEL" in event_type:
        order.status = "cancelled"

    db.session.commit()
    return jsonify({"status": "ok"})


def _verify_signature(raw_body, timestamp, signature):
    secret = current_app.config.get("CASHFREE_WEBHOOK_SECRET")
    if not secret or not signature or not timestamp:
        return False
    digest = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}{raw_body}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)


def _next_renewal():
    now = datetime.now(timezone.utc)
    month = now.month + 1
    year = now.year + (month > 12)
    month = month - 12 if month > 12 else month
    day = min(now.day, 28)  # avoid month-length edge cases
    return now.replace(year=year, month=month, day=day)
