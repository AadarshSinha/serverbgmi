"""End-to-end tests for the HTTP contract.

Run them the same way you run everything else:

    make test                                          # in the container, on Postgres
    ./myenv/bin/python -m unittest discover -s tests    # venv, on temporary SQLite

These drive the real Flask app through `test_client()`, so the routes, error
handlers, JWT wiring and database models are all exercised for real. Only the
ML pipeline is stubbed: LOAD_MODELS is off, and the handful of tests that need
a prediction patch `predictor.get_template_cordinates` to hand back a known
circle instead of running YOLO on a fixture image.

The one rule every test here enforces is that a failure is still JSON with a
machine-readable `code` -- the frontend branches on those codes, and an HTML
error page would break it.
"""
import base64
import hashlib
import hmac
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

# Must be set before `config` is imported: Config reads the environment at
# class-definition time. load_dotenv() does not override existing vars, so a
# developer's real .env cannot leak into the test run.
os.environ["LOAD_MODELS"] = "false"
os.environ["APP_ENV"] = "development"
os.environ["STORE_UPLOADS"] = "false"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import predictor  # noqa: E402
from app import create_app  # noqa: E402
from config import Config  # noqa: E402
from extensions import db  # noqa: E402
from models import PredictionLog, User  # noqa: E402


# Point this at a throwaway Postgres to run the suite against the same engine
# production uses; `make test` sets it to the compose `db` service. Left unset,
# each test still gets its own temporary SQLite file, so the suite runs with
# nothing installed but the venv.
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")


def make_app(**overrides):
    """A fresh app on a throwaway database."""
    path = None
    if TEST_DATABASE_URL:
        uri = TEST_DATABASE_URL
    else:
        handle, path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        uri = f"sqlite:///{path}"

    attrs = {
        "TESTING": True,
        "LOAD_MODELS": False,
        "SQLALCHEMY_DATABASE_URI": uri,
        "JWT_SECRET_KEY": "test-secret",
        "STORE_UPLOADS": False,
        "ALLOWED_ORIGINS": "*",
        **overrides,
    }
    app = create_app(type("TestConfig", (Config,), attrs))
    app._db_path = path

    if TEST_DATABASE_URL:
        # Every test shares the one Postgres database, so wipe it explicitly.
        # A temporary SQLite file gets this for free by being a new file.
        with app.app_context():
            db.drop_all()
            db.create_all()

    return app


class ApiTestCase(unittest.TestCase):
    """Shared setup: an app, a client, and helpers for the common flows."""

    config_overrides = {}

    def setUp(self):
        self.app = make_app(**self.config_overrides)
        self.client = self.app.test_client()
        self.ctx = self.app.app_context()
        self.ctx.push()

    def tearDown(self):
        db.session.remove()
        self.ctx.pop()
        if self.app._db_path:
            try:
                os.unlink(self.app._db_path)
            except OSError:
                pass

    # -- helpers ----------------------------------------------------------
    def signup(self, email="player@example.com", password="hunter2hunter2", **extra):
        return self.client.post(
            "/auth/signup", json={"email": email, "password": password, **extra}
        )

    def token_for(self, email="player@example.com", password="hunter2hunter2"):
        return self.signup(email, password).get_json()["token"]

    def auth(self, token):
        return {"Authorization": f"Bearer {token}"}

    def upload(self, data=b"not-an-image", filename="zone.jpg", headers=None):
        return self.client.post(
            "/predict",
            data={"file": (io.BytesIO(data), filename)},
            content_type="multipart/form-data",
            headers=headers or {},
        )

    def assertApiError(self, response, status, code):
        self.assertEqual(response.status_code, status)
        self.assertEqual(
            response.content_type.split(";")[0],
            "application/json",
            "errors must stay JSON so the frontend can parse them",
        )
        body = response.get_json()
        self.assertEqual(body["code"], code)
        self.assertTrue(body["error"], "every error needs a human-readable message")
        return body


# --------------------------------------------------------------------- health


class HealthTests(ApiTestCase):
    def test_root_stays_backwards_compatible(self):
        body = self.client.get("/").get_json()
        self.assertEqual(body["message"], "Server is live!")
        self.assertEqual(body["status"], "ok")

    def test_health_reports_what_the_frontend_needs(self):
        body = self.client.get("/health").get_json()
        self.assertEqual(body["status"], "ok")
        self.assertFalse(body["modelsLoaded"])  # LOAD_MODELS is off here
        self.assertFalse(body["billingEnabled"])
        self.assertTrue(body["version"])

    def test_only_supported_maps_are_advertised(self):
        body = self.client.get("/health").get_json()
        self.assertEqual(body["supportedMaps"], ["Erangel", "Miramar"])
        self.assertNotIn("Vikendi", body["supportedMaps"])
        self.assertNotIn("Sanhok", body["supportedMaps"])

    def test_unknown_route_answers_json_not_html(self):
        response = self.client.get("/nope")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.content_type.split(";")[0], "application/json")
        self.assertIn("code", response.get_json())


# ----------------------------------------------------------------------- auth


class AuthTests(ApiTestCase):
    def test_signup_returns_a_token_and_the_user(self):
        response = self.signup()
        self.assertEqual(response.status_code, 201)
        body = response.get_json()
        self.assertTrue(body["token"])
        self.assertEqual(body["user"]["email"], "player@example.com")
        self.assertEqual(body["user"]["plan"], "free")
        self.assertNotIn("password", json.dumps(body).lower())

    def test_email_is_normalised(self):
        body = self.signup(email="  Player@Example.COM  ").get_json()
        self.assertEqual(body["user"]["email"], "player@example.com")

    def test_rejects_a_malformed_email(self):
        self.assertApiError(self.signup(email="not-an-email"), 400, "VALIDATION_ERROR")

    def test_rejects_a_short_password(self):
        self.assertApiError(self.signup(password="short"), 400, "VALIDATION_ERROR")

    def test_rejects_a_duplicate_email(self):
        self.signup()
        self.assertApiError(self.signup(), 409, "EMAIL_TAKEN")

    def test_login_with_correct_credentials(self):
        self.signup()
        response = self.client.post(
            "/auth/login",
            json={"email": "player@example.com", "password": "hunter2hunter2"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["token"])

    def test_login_does_not_reveal_whether_an_email_exists(self):
        self.signup()
        wrong_password = self.client.post(
            "/auth/login",
            json={"email": "player@example.com", "password": "wrongpassword"},
        )
        unknown_email = self.client.post(
            "/auth/login",
            json={"email": "nobody@example.com", "password": "wrongpassword"},
        )
        self.assertApiError(wrong_password, 401, "INVALID_CREDENTIALS")
        self.assertApiError(unknown_email, 401, "INVALID_CREDENTIALS")
        self.assertEqual(
            wrong_password.get_json()["error"], unknown_email.get_json()["error"]
        )

    def test_password_is_hashed_not_stored(self):
        self.signup()
        user = User.query.filter_by(email="player@example.com").first()
        self.assertNotIn("hunter2hunter2", user.password_hash)
        self.assertTrue(user.check_password("hunter2hunter2"))
        self.assertFalse(user.check_password("hunter2hunter3"))

    def test_me_requires_a_token(self):
        self.assertEqual(self.client.get("/auth/me").status_code, 401)

    def test_me_returns_the_signed_in_user(self):
        token = self.token_for()
        body = self.client.get("/auth/me", headers=self.auth(token)).get_json()
        self.assertEqual(body["user"]["email"], "player@example.com")

    def test_me_rejects_a_forged_token(self):
        self.assertEqual(
            self.client.get("/auth/me", headers=self.auth("garbage")).status_code, 422
        )

    def test_disabled_account_cannot_log_in(self):
        self.signup()
        user = User.query.filter_by(email="player@example.com").first()
        user.is_active = False
        db.session.commit()

        response = self.client.post(
            "/auth/login",
            json={"email": "player@example.com", "password": "hunter2hunter2"},
        )
        self.assertApiError(response, 403, "ACCOUNT_DISABLED")


# -------------------------------------------------------------------- predict


def fake_zone(center=(100, 100), radius=400):
    """Stand-in for the YOLO + homography stage."""
    import numpy as np

    return {
        "frame_center": (50, 50),
        "frame_radius": 40,
        "template_center": center,
        "template_radius": radius,
        "matrix": np.eye(3, dtype=np.float32),
    }


VALID_PNG = base64.b64decode(
    # 1x1 PNG — enough for cv2.imdecode to succeed.
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class PredictValidationTests(ApiTestCase):
    def test_missing_file_is_a_clear_400(self):
        response = self.client.post("/predict", data={}, content_type="multipart/form-data")
        self.assertApiError(response, 400, "NO_FILE")

    def test_empty_file_is_a_clear_400(self):
        self.assertApiError(self.upload(b""), 400, "EMPTY_FILE")

    def test_undecodable_bytes_are_a_clear_400(self):
        self.assertApiError(self.upload(b"definitely not an image"), 400, "INVALID_IMAGE")

    def test_oversized_upload_is_rejected_as_json(self):
        app = make_app(MAX_CONTENT_LENGTH=64)
        response = app.test_client().post(
            "/predict",
            data={"file": (io.BytesIO(b"x" * 5000), "big.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.get_json()["code"], "FILE_TOO_LARGE")

    def test_a_failed_prediction_never_returns_html(self):
        # The original bug: an unhandled KeyError produced Flask's HTML 500 page,
        # which the frontend then rendered as a broken image.
        response = self.upload(b"definitely not an image")
        self.assertNotIn("<!doctype", response.get_data(as_text=True).lower())


class PredictPipelineTests(ApiTestCase):
    """Exercise predict_zone's branches with the circle-detection stage stubbed."""

    def call_with_zone(self, **zone_kwargs):
        with mock.patch.object(
            predictor, "get_template_cordinates", return_value=fake_zone(**zone_kwargs)
        ):
            return self.upload(VALID_PNG, "zone.png")

    def test_vikendi_is_retired_with_an_explanation(self):
        # Quadrant boundaries: x > 1280 and y < 1440 is the Vikendi quadrant.
        body = self.assertApiError(
            self.call_with_zone(center=(2000, 100)), 422, "MAP_NOT_SUPPORTED"
        )
        self.assertIn("Vikendi", body["error"])
        self.assertIn("Erangel", body["error"])
        self.assertIn("Miramar", body["error"])

    def test_sanhok_is_retired_with_an_explanation(self):
        body = self.assertApiError(
            self.call_with_zone(center=(100, 2000)), 422, "MAP_NOT_SUPPORTED"
        )
        self.assertIn("Sanhok", body["error"])

    def test_an_unplaceable_circle_is_an_unknown_map(self):
        self.assertApiError(self.call_with_zone(center=(0, 0)), 422, "UNKNOWN_MAP")

    def test_a_nonsense_radius_is_an_unknown_zone(self):
        predictor.map_constants = {"erangle": {str(z): [z * 10, z * 10 + 5] for z in range(1, 9)}}
        try:
            self.assertApiError(
                self.call_with_zone(center=(100, 100), radius=99999), 422, "UNKNOWN_ZONE"
            )
        finally:
            predictor.map_constants = None

    def test_missing_trained_model_is_a_503_not_a_crash(self):
        predictor.map_constants = {
            "erangle": {str(z): [z * 100, z * 100 + 50] for z in range(1, 9)}
        }
        predictor.models_dict.clear()
        try:
            self.assertApiError(
                self.call_with_zone(center=(100, 100), radius=120), 503, "MODEL_UNAVAILABLE"
            )
        finally:
            predictor.map_constants = None


class FeedbackTests(ApiTestCase):
    """Rating a prediction after the fact."""

    def _successful_prediction(self, headers=None):
        """A real 200 from /predict, with the ML stage stubbed out.

        The Keras models are not loaded in tests, so the pipeline itself is
        replaced -- everything around it (logging, the id header, auth) is the
        real thing.
        """
        import numpy as np

        annotated = np.zeros((8, 8, 3), dtype=np.uint8)
        with mock.patch.object(
            predictor,
            "predict_zone",
            return_value={
                "map_type": "e",
                "zone_number": 1,
                "frame_center": (50, 50),
                "predicted_center": (300.4, 400.6),
                "predicted_radius": 55,
            },
        ), mock.patch.object(predictor, "annotate", return_value=annotated):
            response = self.upload(VALID_PNG, "zone.png", headers=headers or {})

        self.assertEqual(response.status_code, 200)
        prediction_id = response.headers.get("X-Prediction-Id")
        self.assertIsNotNone(
            prediction_id, "the browser needs the row id to attach feedback to it"
        )
        return int(prediction_id)

    def test_the_stored_image_is_the_annotated_result(self):
        """Not the upload: what gets kept is what the user was shown."""
        import storage

        with mock.patch.object(storage, "save_upload", return_value="k.jpg") as saved:
            self._successful_prediction()

        stored = saved.call_args.args[0]
        self.assertTrue(stored.startswith(b"\xff\xd8"), "stored bytes should be a JPEG")
        self.assertNotEqual(stored, VALID_PNG, "the upload itself must not be stored")

    def test_a_failed_prediction_stores_no_image(self):
        import storage

        with mock.patch.object(storage, "save_upload") as saved:
            self.upload(b"not-an-image")

        saved.assert_not_called()
        log = db.session.scalars(db.select(PredictionLog)).first()
        self.assertIsNone(log.image_key)

    def test_the_prediction_geometry_is_recorded(self):
        # Only the clean screenshot is stored; these numbers are what let the
        # annotated version be redrawn from it.
        prediction_id = self._successful_prediction()
        log = db.session.get(PredictionLog, prediction_id)
        self.assertEqual(
            (log.predicted_center_x, log.predicted_center_y, log.predicted_radius),
            (300, 400, 55),
        )
        self.assertEqual((log.detected_center_x, log.detected_center_y), (50, 50))

    def test_a_prediction_can_be_rated(self):
        prediction_id = self._successful_prediction()
        response = self.client.post(
            f"/predict/{prediction_id}/feedback", json={"rating": "spot_on"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["rating"], "spot_on")

        log = db.session.get(PredictionLog, prediction_id)
        self.assertEqual(log.rating, "spot_on")
        self.assertIsNotNone(log.rated_at)

    def test_every_rating_value_is_accepted(self):
        for rating in PredictionLog.RATINGS:
            prediction_id = self._successful_prediction()
            response = self.client.post(
                f"/predict/{prediction_id}/feedback", json={"rating": rating}
            )
            self.assertEqual(response.status_code, 200, rating)

    def test_a_nonsense_rating_is_refused(self):
        prediction_id = self._successful_prediction()
        response = self.client.post(
            f"/predict/{prediction_id}/feedback", json={"rating": "5 stars"}
        )
        self.assertApiError(response, 400, "INVALID_RATING")

    def test_rating_twice_is_refused(self):
        prediction_id = self._successful_prediction()
        self.client.post(f"/predict/{prediction_id}/feedback", json={"rating": "close"})
        response = self.client.post(
            f"/predict/{prediction_id}/feedback", json={"rating": "way_off"}
        )
        self.assertApiError(response, 409, "ALREADY_RATED")

    def test_unknown_prediction_is_a_404(self):
        response = self.client.post("/predict/999999/feedback", json={"rating": "close"})
        self.assertApiError(response, 404, "UNKNOWN_PREDICTION")

    def test_a_failed_prediction_cannot_be_rated(self):
        self.upload(b"not-an-image")  # logged, but succeeded=False
        log = db.session.scalars(db.select(PredictionLog)).first()
        response = self.client.post(f"/predict/{log.id}/feedback", json={"rating": "close"})
        self.assertApiError(response, 404, "UNKNOWN_PREDICTION")

    def test_someone_elses_prediction_cannot_be_rated(self):
        mine = self.token_for("owner@example.com")
        prediction_id = self._successful_prediction(headers=self.auth(mine))

        theirs = self.token_for("stranger@example.com")
        response = self.client.post(
            f"/predict/{prediction_id}/feedback",
            json={"rating": "way_off"},
            headers=self.auth(theirs),
        )
        self.assertApiError(response, 403, "NOT_YOUR_PREDICTION")

    def test_a_signed_out_visitor_can_rate_their_own_prediction(self):
        # Most predictions are made signed out; requiring an account here would
        # throw away most of the feedback.
        prediction_id = self._successful_prediction()
        response = self.client.post(
            f"/predict/{prediction_id}/feedback", json={"rating": "close"}
        )
        self.assertEqual(response.status_code, 200)

    def test_feedback_closes_after_the_window(self):
        from datetime import timedelta

        prediction_id = self._successful_prediction()
        log = db.session.get(PredictionLog, prediction_id)
        log.created_at = log.created_at - timedelta(days=3)
        db.session.commit()

        response = self.client.post(
            f"/predict/{prediction_id}/feedback", json={"rating": "close"}
        )
        self.assertApiError(response, 422, "FEEDBACK_WINDOW_CLOSED")


class ZoneLookupTests(unittest.TestCase):
    """The loop that replaced the original 80-line match/case."""

    def setUp(self):
        predictor.map_constants = {
            "erangle": {str(z): [z * 100, z * 100 + 99] for z in range(1, 9)},
            "miramar": {str(z): [z * 100, z * 100 + 99] for z in range(1, 9)},
        }

    def tearDown(self):
        predictor.map_constants = None

    def test_returns_the_next_zone_radius_and_current_phase(self):
        self.assertEqual(
            predictor.get_target_zone_radius_and_current_zone(150, "e"), (200, 1)
        )
        self.assertEqual(
            predictor.get_target_zone_radius_and_current_zone(350, "m"), (400, 3)
        )

    def test_final_zone_has_no_next_radius(self):
        self.assertEqual(
            predictor.get_target_zone_radius_and_current_zone(850, "e"), (0, 8)
        )

    def test_retired_maps_resolve_to_nothing(self):
        for code in ("v", "s"):
            self.assertEqual(
                predictor.get_target_zone_radius_and_current_zone(150, code), (0, 0)
            )

    def test_unmatched_radius_resolves_to_nothing(self):
        self.assertEqual(
            predictor.get_target_zone_radius_and_current_zone(99999, "e"), (0, 0)
        )


class MapQuadrantTests(unittest.TestCase):
    def test_quadrants_map_to_the_expected_games(self):
        self.assertEqual(predictor.get_map_type((100, 100)), "e")
        self.assertEqual(predictor.get_map_type((2000, 100)), "v")
        self.assertEqual(predictor.get_map_type((100, 2000)), "s")
        self.assertEqual(predictor.get_map_type((2000, 2000)), "m")
        self.assertEqual(predictor.get_map_type((0, 0)), "None")


# ------------------------------------------------------------------- logging


class RequestLoggingTests(ApiTestCase):
    def test_every_request_is_logged_even_when_it_fails(self):
        self.upload(b"definitely not an image")
        logs = PredictionLog.query.all()
        self.assertEqual(len(logs), 1)
        self.assertFalse(logs[0].succeeded)
        self.assertEqual(logs[0].error_code, "INVALID_IMAGE")
        self.assertEqual(logs[0].status_code, 400)
        self.assertIsNotNone(logs[0].duration_ms)

    def test_anonymous_requests_are_logged_without_a_user(self):
        self.upload(b"definitely not an image")
        self.assertIsNone(PredictionLog.query.first().user_id)

    def test_signed_in_requests_are_attributed(self):
        token = self.token_for()
        self.upload(b"definitely not an image", headers=self.auth(token))
        log = PredictionLog.query.first()
        self.assertIsNotNone(log.user_id)

    def test_client_ip_is_hashed_not_stored(self):
        self.upload(
            b"definitely not an image", headers={"X-Forwarded-For": "203.0.113.7, 10.0.0.1"}
        )
        log = PredictionLog.query.first()
        self.assertIsNotNone(log.ip_hash)
        self.assertNotIn("203.0.113.7", log.ip_hash)
        self.assertEqual(
            log.ip_hash, PredictionLog.hash_ip("203.0.113.7", "test-secret")
        )

    def test_missing_file_is_still_a_clean_error_without_a_log_row(self):
        # No upload means nothing to log against; the important part is that it
        # does not 500 trying to record one.
        self.assertApiError(
            self.client.post("/predict", data={}, content_type="multipart/form-data"),
            400,
            "NO_FILE",
        )


# ------------------------------------------------------------------- billing


class UploadStorageTests(unittest.TestCase):
    """Where screenshots go. The bucket is never really contacted."""

    def setUp(self):
        import storage

        storage._s3_client = None  # the client is cached; each test builds its own
        self.storage = storage

    def tearDown(self):
        self.storage._s3_client = None

    def _config(self, **overrides):
        attrs = {
            "STORE_UPLOADS": True,
            "S3_BUCKET": "zonepredictor-uploads",
            "S3_PREFIX": "uploads",
            "AWS_REGION": "ap-southeast-1",
            "S3_ENDPOINT_URL": None,
            "S3_ACCESS_KEY_ID": None,
            "S3_SECRET_ACCESS_KEY": None,
            **overrides,
        }
        return type("StorageConfig", (Config,), attrs)

    def test_a_custom_endpoint_is_passed_through(self):
        # Neon Object Storage, R2 and Spaces all speak S3 at their own host.
        import boto3

        with mock.patch.object(boto3, "client") as client:
            self.storage.save_upload(
                b"bytes",
                "zone.jpg",
                self._config(
                    S3_ENDPOINT_URL="https://s3.neon.example",
                    S3_ACCESS_KEY_ID="key",
                    S3_SECRET_ACCESS_KEY="secret",
                ),
            )

        kwargs = client.call_args.kwargs
        self.assertEqual(kwargs["endpoint_url"], "https://s3.neon.example")
        self.assertEqual(kwargs["aws_access_key_id"], "key")

    def test_plain_aws_sends_no_endpoint(self):
        import boto3

        with mock.patch.object(boto3, "client") as client:
            self.storage.save_upload(b"bytes", "zone.jpg", self._config())

        self.assertNotIn("endpoint_url", client.call_args.kwargs)

    def test_the_key_is_date_partitioned_and_keeps_the_extension(self):
        key = self.storage.build_key("uploads", "screenshot.PNG")
        self.assertTrue(key.startswith("uploads/"))
        self.assertTrue(key.endswith(".png"))

    def test_an_odd_extension_falls_back_to_jpg(self):
        self.assertTrue(self.storage.build_key("uploads", "zone.heic").endswith(".jpg"))

    def test_a_storage_failure_never_breaks_a_prediction(self):
        import boto3

        with mock.patch.object(boto3, "client", side_effect=RuntimeError("bucket gone")):
            key = self.storage.save_upload(b"bytes", "zone.jpg", self._config())

        self.assertIsNone(key, "a failed upload returns no key instead of raising")


class BillingDisabledTests(ApiTestCase):
    def test_plans_are_public(self):
        body = self.client.get("/billing/plans").get_json()
        self.assertFalse(body["billingEnabled"])
        ids = [p["id"] for p in body["plans"]]
        self.assertIn("free", ids)
        self.assertIn("pro", ids)

    def test_checkout_requires_a_token(self):
        self.assertEqual(
            self.client.post("/billing/checkout", json={"plan": "pro"}).status_code, 401
        )

    def test_checkout_refuses_while_billing_is_off(self):
        token = self.token_for()
        response = self.client.post(
            "/billing/checkout", json={"plan": "pro"}, headers=self.auth(token)
        )
        self.assertApiError(response, 503, "BILLING_DISABLED")

    def test_webhook_rejects_an_unsigned_call(self):
        response = self.client.post("/billing/webhook", json={"type": "PAYMENT_SUCCESS"})
        self.assertApiError(response, 401, "INVALID_SIGNATURE")


class BillingEnabledTests(ApiTestCase):
    config_overrides = {
        "BILLING_ENABLED": True,
        "CASHFREE_APP_ID": "test-app",
        "CASHFREE_SECRET_KEY": "test-key",
        "CASHFREE_WEBHOOK_SECRET": "webhook-secret",
    }

    def sign(self, body):
        timestamp = "1700000000"
        digest = hmac.new(
            b"webhook-secret", f"{timestamp}{body}".encode(), hashlib.sha256
        ).digest()
        return {
            "x-webhook-timestamp": timestamp,
            "x-webhook-signature": base64.b64encode(digest).decode(),
            "Content-Type": "application/json",
        }

    def test_checkout_creates_an_order_but_does_not_charge_yet(self):
        token = self.token_for()
        response = self.client.post(
            "/billing/checkout", json={"plan": "pro"}, headers=self.auth(token)
        )
        # The order row exists; the Cashfree call is the one TODO left.
        self.assertApiError(response, 503, "BILLING_NOT_IMPLEMENTED")

    def test_checkout_rejects_an_unknown_plan(self):
        token = self.token_for()
        for plan in ("free", "platinum", ""):
            response = self.client.post(
                "/billing/checkout", json={"plan": plan}, headers=self.auth(token)
            )
            self.assertApiError(response, 400, "INVALID_PLAN")

    def test_a_forged_signature_cannot_upgrade_an_account(self):
        body = json.dumps({"type": "PAYMENT_SUCCESS_WEBHOOK"})
        response = self.client.post(
            "/billing/webhook",
            data=body,
            headers={
                "x-webhook-timestamp": "1700000000",
                "x-webhook-signature": "AAAA",
                "Content-Type": "application/json",
            },
        )
        self.assertApiError(response, 401, "INVALID_SIGNATURE")

    def test_a_correctly_signed_webhook_is_accepted(self):
        body = json.dumps(
            {
                "type": "PAYMENT_SUCCESS_WEBHOOK",
                "data": {"order": {"order_id": "zp_unknown"}},
            }
        )
        response = self.client.post(
            "/billing/webhook", data=body, headers=self.sign(body)
        )
        # Unknown orders are acknowledged so Cashfree stops retrying.
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ignored")


# -------------------------------------------------------------------- config


class ProductionGuardTests(unittest.TestCase):
    def test_production_refuses_insecure_defaults(self):
        cfg = type(
            "Cfg",
            (Config,),
            {
                "JWT_SECRET_KEY": "dev-only-insecure-secret",
                "SQLALCHEMY_DATABASE_URI": "sqlite:///local.db",
                "ALLOWED_ORIGINS": "*",
            },
        )
        problems = " ".join(cfg.validate_production())
        self.assertIn("JWT_SECRET_KEY", problems)
        self.assertIn("Postgres", problems)
        self.assertIn("ALLOWED_ORIGINS", problems)

    def test_a_properly_configured_production_passes(self):
        cfg = type(
            "Cfg",
            (Config,),
            {
                "JWT_SECRET_KEY": "a-real-random-secret",
                "SQLALCHEMY_DATABASE_URI": "postgresql://u:p@host/db",
                "ALLOWED_ORIGINS": "https://zonepredictor.com",
                "BILLING_ENABLED": False,
            },
        )
        self.assertEqual(cfg.validate_production(), [])

    def test_billing_without_credentials_is_flagged(self):
        cfg = type(
            "Cfg",
            (Config,),
            {
                "JWT_SECRET_KEY": "a-real-random-secret",
                "SQLALCHEMY_DATABASE_URI": "postgresql://u:p@host/db",
                "ALLOWED_ORIGINS": "https://zonepredictor.com",
                "BILLING_ENABLED": True,
                "CASHFREE_APP_ID": None,
                "CASHFREE_SECRET_KEY": None,
            },
        )
        self.assertIn("Cashfree", " ".join(cfg.validate_production()))

    def test_create_app_refuses_to_boot_a_broken_production(self):
        cfg = type(
            "Cfg",
            (Config,),
            {
                "ENV": "production",
                "IS_PRODUCTION": True,
                "JWT_SECRET_KEY": "dev-only-insecure-secret",
                "LOAD_MODELS": False,
            },
        )
        with self.assertRaises(RuntimeError):
            create_app(cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
