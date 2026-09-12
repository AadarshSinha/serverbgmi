"""Environment-driven configuration.

Every setting has a development-friendly default so `python app.py` works with
no .env at all. Production hardening is opt-in via environment variables, and
`Config.validate_production()` refuses to start if a required secret is missing.
"""
import os
from pathlib import Path

PATH_PREFIX = Path(__file__).parent.absolute()


def _bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _normalise_db_url(url):
    # SQLAlchemy 2.x dropped the legacy "postgres://" scheme that several hosts
    # still hand out in DATABASE_URL.
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql://", 1)
    return url


class Config:
    ENV = os.environ.get("APP_ENV", "development")
    IS_PRODUCTION = ENV == "production"

    PORT = int(os.environ.get("PORT", 4000))

    # --- database -----------------------------------------------------------
    # Defaults to a local SQLite file. On EC2 point this at a managed Postgres
    # (RDS/Neon/Supabase) so the data outlives the instance.
    SQLALCHEMY_DATABASE_URI = _normalise_db_url(
        os.environ.get("DATABASE_URL", f"sqlite:///{PATH_PREFIX / 'zonepredictor.db'}")
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}

    # --- auth ---------------------------------------------------------------
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-only-insecure-secret")
    JWT_ACCESS_TOKEN_EXPIRES_HOURS = int(
        os.environ.get("JWT_ACCESS_TOKEN_EXPIRES_HOURS", 24 * 7)
    )

    # --- uploads ------------------------------------------------------------
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))

    # Persist every uploaded screenshot. These double as future training data,
    # so the frontend discloses it to the user.
    STORE_UPLOADS = _bool("STORE_UPLOADS", True)
    S3_BUCKET = os.environ.get("S3_BUCKET")  # unset -> store on local disk
    S3_PREFIX = os.environ.get("S3_PREFIX", "uploads")
    AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
    LOCAL_UPLOAD_DIR = os.environ.get("LOCAL_UPLOAD_DIR", str(PATH_PREFIX / "uploads"))

    # Any S3-compatible store: Neon Object Storage, Cloudflare R2, DigitalOcean
    # Spaces, MinIO. Unset means real AWS S3, where boto3 works out the endpoint
    # from the region on its own.
    #
    # AWS_ENDPOINT_URL_S3 is the name the AWS SDKs use, and the name Neon hands
    # you in its .env snippet -- accepting both means that snippet can be pasted
    # in unedited.
    S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL") or os.environ.get(
        "AWS_ENDPOINT_URL_S3"
    )
    # Left unset, boto3 falls back to the usual AWS credential chain
    # (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, instance role, ~/.aws), which
    # is how Neon's snippet authenticates without these being set at all.
    S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
    S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")

    # --- cors ---------------------------------------------------------------
    ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")

    # --- billing (Cashfree) -------------------------------------------------
    # Scaffolding only. While BILLING_ENABLED is false the endpoints exist and
    # are documented but refuse to start a real transaction.
    BILLING_ENABLED = _bool("BILLING_ENABLED", False)
    CASHFREE_APP_ID = os.environ.get("CASHFREE_APP_ID")
    CASHFREE_SECRET_KEY = os.environ.get("CASHFREE_SECRET_KEY")
    CASHFREE_ENV = os.environ.get("CASHFREE_ENV", "sandbox")  # sandbox | production
    CASHFREE_WEBHOOK_SECRET = os.environ.get("CASHFREE_WEBHOOK_SECRET")
    CASHFREE_RETURN_URL = os.environ.get("CASHFREE_RETURN_URL")

    # --- ml -----------------------------------------------------------------
    # Tests flip this off so they can import the app without TensorFlow.
    LOAD_MODELS = _bool("LOAD_MODELS", True)

    @classmethod
    def validate_production(cls):
        """Fail fast on misconfiguration rather than serving insecurely."""
        problems = []
        if cls.JWT_SECRET_KEY == "dev-only-insecure-secret":
            problems.append("JWT_SECRET_KEY must be set to a random secret")
        if cls.SQLALCHEMY_DATABASE_URI.startswith("sqlite:"):
            problems.append(
                "DATABASE_URL should point at a managed Postgres in production; "
                "a SQLite file on the instance is lost when the instance dies"
            )
        if cls.ALLOWED_ORIGINS == "*":
            problems.append("ALLOWED_ORIGINS should list your real site origins")
        if cls.BILLING_ENABLED and not (cls.CASHFREE_APP_ID and cls.CASHFREE_SECRET_KEY):
            problems.append("BILLING_ENABLED is on but Cashfree credentials are missing")
        return problems


def parse_origins(value):
    if value == "*":
        return "*"
    return [o.strip() for o in value.split(",") if o.strip()]


# Maps the prediction pipeline currently supports. Vikendi and Sanhok were
# retired: their models were trained on too few matches to be trustworthy.
SUPPORTED_MAPS = {
    "e": "Erangel",
    "m": "Miramar",
}

RETIRED_MAPS = {
    "v": "Vikendi",
    "s": "Sanhok",
}
