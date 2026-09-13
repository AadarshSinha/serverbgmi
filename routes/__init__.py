from routes.health import health_bp
from routes.auth import auth_bp
from routes.predict import predict_bp
from routes.billing import billing_bp
from routes.feedback import feedback_bp

__all__ = ["health_bp", "auth_bp", "predict_bp", "billing_bp", "feedback_bp"]
