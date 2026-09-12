"""ZonePredictor API.

    python app.py            # development server on PORT (default 4000)
    gunicorn "app:create_app()" --bind 0.0.0.0:4000 --workers 2

Each gunicorn worker loads its own copy of the ML models (~11MB of Keras
weights plus the YOLO detector and the stitched map), so raise --workers with
an eye on memory.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # read .env before Config is imported

from flask import Flask, jsonify  # noqa: E402
from werkzeug.exceptions import HTTPException  # noqa: E402

import predictor  # noqa: E402
from config import Config, parse_origins  # noqa: E402
from errors import ApiError  # noqa: E402
from extensions import db, migrate, jwt, cors  # noqa: E402
from routes import health_bp, auth_bp, predict_bp, billing_bp  # noqa: E402


def create_app(config_object=Config):
    app = Flask(__name__)
    app.config.from_object(config_object)
    # Handy handle for code that needs the class itself (e.g. storage helpers).
    app.config_object = config_object

    if config_object.IS_PRODUCTION:
        problems = config_object.validate_production()
        if problems:
            raise RuntimeError(
                "Refusing to start in production with: " + "; ".join(problems)
            )

    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    cors.init_app(
        app,
        resources={r"/*": {"origins": parse_origins(config_object.ALLOWED_ORIGINS)}},
        # /predict answers with a JPEG, so it returns the row id in a header.
        # Without this the browser receives it but refuses to let JS read it.
        expose_headers=["X-Prediction-Id"],
    )

    app.register_blueprint(health_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(billing_bp)

    _register_error_handlers(app)

    with app.app_context():
        # Fine for a single-instance deployment. Switch to `flask db upgrade`
        # once more than one process can start at the same time.
        db.create_all()

    if config_object.LOAD_MODELS:
        predictor.initialize_heavy_objects()

    print(f"env={config_object.ENV} database={describe_database(app)}")

    return app


def describe_database(app):
    """Which database am I actually talking to?

    Printed on every boot. Connection strings differ between environments by a
    hostname buried in the middle of a URL that also contains a password, so
    this prints the identifying half and never the credentials.
    """
    url = app.config["SQLALCHEMY_DATABASE_URI"]
    if url.startswith("sqlite"):
        return f"sqlite {url.rsplit('/', 1)[-1]}"
    try:
        from sqlalchemy.engine import make_url

        parsed = make_url(url)
        return f"{parsed.drivername} {parsed.host}/{parsed.database}"
    except Exception:  # never let a log line stop the app booting
        return "unknown"


def _register_error_handlers(app):
    @app.errorhandler(ApiError)
    def handle_api_error(error):
        return jsonify(error.to_dict()), error.status

    @app.errorhandler(413)
    def handle_too_large(error):
        limit_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return (
            jsonify(
                {
                    "code": "FILE_TOO_LARGE",
                    "error": f"That image is too large. Please upload a file under {limit_mb}MB.",
                }
            ),
            413,
        )

    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        # Keeps 404/405/401 answering JSON rather than Flask's HTML pages, so
        # the frontend can always parse an error body.
        return (
            jsonify({"code": error.name.upper().replace(" ", "_"), "error": error.description}),
            error.code,
        )

    @app.errorhandler(Exception)
    def handle_unexpected(error):
        import traceback

        print(traceback.format_exc())
        return (
            jsonify({"code": "INTERNAL_ERROR", "error": "Something went wrong."}),
            500,
        )


if __name__ == "__main__":
    application = create_app()
    port = Config.PORT
    print(f"Starting server on port {port} (env={Config.ENV})...")
    application.run(host="0.0.0.0", port=port)
