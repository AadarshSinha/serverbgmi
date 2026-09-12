"""Error types shared across blueprints."""


class ApiError(Exception):
    """An error we understand and can explain to the caller.

    Carries a machine-readable code so the frontend can react specifically,
    plus the HTTP status to respond with.
    """

    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def to_dict(self):
        return {"code": self.code, "error": self.message}


class PredictionError(ApiError):
    """Prediction could not be produced.

    Defaults to 422: the image was readable, but we could not get a usable
    prediction out of it.
    """

    def __init__(self, code, message, status=422):
        super().__init__(code, message, status)
