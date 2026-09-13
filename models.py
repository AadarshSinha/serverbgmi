"""Database models.

Kept deliberately small: a users table, an append-only log of every prediction
request, and a payments table that exists so billing can be switched on later
without a migration scramble.
"""
import hashlib
from datetime import datetime, timezone

from werkzeug.security import generate_password_hash, check_password_hash

from extensions import db


def utcnow():
    return datetime.now(timezone.utc)


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    name = db.Column(db.String(120), nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)

    # Monetisation hooks. Predictions are unlimited for everyone today; `plan`
    # is what gates features once Cashfree is switched on.
    plan = db.Column(db.String(32), nullable=False, default="free")
    plan_expires_at = db.Column(db.DateTime(timezone=True), nullable=True)
    cashfree_customer_id = db.Column(db.String(128), nullable=True)

    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    last_login_at = db.Column(db.DateTime(timezone=True), nullable=True)

    predictions = db.relationship("PredictionLog", back_populates="user", lazy="dynamic")

    def set_password(self, raw_password):
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password):
        return check_password_hash(self.password_hash, raw_password)

    @property
    def is_pro(self):
        if self.plan != "pro":
            return False
        if self.plan_expires_at is None:
            return True
        expires = self.plan_expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return expires > utcnow()

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "plan": self.plan,
            "isPro": self.is_pro,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }


class PredictionLog(db.Model):
    """One row per /predict call, successful or not.

    Doubles as usage analytics and as an index over the uploaded screenshots,
    which are the raw material for growing the training set.
    """

    __tablename__ = "prediction_logs"

    # Three options, not five stars: one click, and every answer means
    # something. A 5-point scale mostly collects 3s.
    RATINGS = ("spot_on", "close", "way_off")

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(
        db.DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )

    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    user = db.relationship("User", back_populates="predictions")

    succeeded = db.Column(db.Boolean, nullable=False, default=False, index=True)
    status_code = db.Column(db.Integer, nullable=False)
    error_code = db.Column(db.String(64), nullable=True, index=True)

    map_type = db.Column(db.String(8), nullable=True, index=True)
    zone_number = db.Column(db.Integer, nullable=True)
    duration_ms = db.Column(db.Integer, nullable=True)

    # The prediction itself, in the uploaded image's own pixel coordinates.
    # Together with the stored screenshot these are everything `annotate()`
    # needs, so the result image can be redrawn exactly whenever it is wanted --
    # and unlike an image, they can be queried, averaged and compared between
    # model versions.
    detected_center_x = db.Column(db.Integer, nullable=True)
    detected_center_y = db.Column(db.Integer, nullable=True)
    predicted_center_x = db.Column(db.Integer, nullable=True)
    predicted_center_y = db.Column(db.Integer, nullable=True)
    predicted_radius = db.Column(db.Integer, nullable=True)

    # Where the annotated result image was persisted: an object key in
    # production, a relative path under LOCAL_UPLOAD_DIR in development. Null on
    # a failed prediction (there is no result to store), or if storage was
    # disabled or the write failed -- which never blocks the prediction itself.
    image_key = db.Column(db.String(512), nullable=True)
    image_bytes = db.Column(db.Integer, nullable=True)

    # Hashed, not raw: enough to count unique visitors without storing an
    # identifier we do not need.
    ip_hash = db.Column(db.String(64), nullable=True, index=True)
    user_agent = db.Column(db.String(256), nullable=True)

    # How good the user said the prediction was. One of PredictionLog.RATINGS,
    # or null for the great majority of rows nobody rates. Indexed because the
    # quality queries all filter on it.
    rating = db.Column(db.String(16), nullable=True, index=True)
    rated_at = db.Column(db.DateTime(timezone=True), nullable=True)

    @staticmethod
    def hash_ip(ip, salt):
        if not ip:
            return None
        return hashlib.sha256(f"{salt}:{ip}".encode("utf-8")).hexdigest()

    def to_dict(self):
        return {
            "id": self.id,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "succeeded": self.succeeded,
            "statusCode": self.status_code,
            "errorCode": self.error_code,
            "mapType": self.map_type,
            "zoneNumber": self.zone_number,
            "durationMs": self.duration_ms,
            "rating": self.rating,
            "predictedCenter": (
                [self.predicted_center_x, self.predicted_center_y]
                if self.predicted_center_x is not None
                else None
            ),
            "predictedRadius": self.predicted_radius,
        }


class Feedback(db.Model):
    """Free-text answers to "what are you actually looking for?".

    Deliberately one field. The prompt asks a single question and takes a
    single answer -- anything more and the response rate collapses.
    """

    __tablename__ = "feedback"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(
        db.DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )

    message = db.Column(db.Text, nullable=False)

    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Where they are, as far as we can tell without asking permission. The
    # browser's own timezone and locale place someone to a region -- no
    # geolocation prompt, no third-party lookup, no raw address stored.
    timezone = db.Column(db.String(64), nullable=True, index=True)
    locale = db.Column(db.String(32), nullable=True)

    ip_hash = db.Column(db.String(64), nullable=True, index=True)
    user_agent = db.Column(db.String(256), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "message": self.message,
            "timezone": self.timezone,
            "locale": self.locale,
        }


class Payment(db.Model):
    """Cashfree order records. Unused until BILLING_ENABLED is turned on."""

    __tablename__ = "payments"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = db.Column(
        db.DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Cashfree's identifiers.
    order_id = db.Column(db.String(128), unique=True, nullable=False, index=True)
    cf_payment_id = db.Column(db.String(128), nullable=True, index=True)
    payment_session_id = db.Column(db.String(256), nullable=True)

    plan = db.Column(db.String(32), nullable=False)
    amount = db.Column(db.Numeric(10, 2), nullable=False)
    currency = db.Column(db.String(8), nullable=False, default="INR")
    # created | paid | failed | refunded | cancelled
    status = db.Column(db.String(32), nullable=False, default="created", index=True)

    raw_payload = db.Column(db.Text, nullable=True)

    def to_dict(self):
        return {
            "orderId": self.order_id,
            "plan": self.plan,
            "amount": float(self.amount) if self.amount is not None else None,
            "currency": self.currency,
            "status": self.status,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }
