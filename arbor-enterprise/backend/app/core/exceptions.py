"""Custom exceptions for A.R.B.O.R."""


class ArborException(Exception):
    """Base exception for A.R.B.O.R."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class EntityNotFoundError(ArborException):
    def __init__(self, entity_id: str):
        super().__init__(f"Entity not found: {entity_id}", status_code=404)


class AuthenticationError(ArborException):
    def __init__(self, detail: str = "Authentication required"):
        super().__init__(detail, status_code=401)


class PermissionDeniedError(ArborException):
    def __init__(self, detail: str = "Permission denied"):
        super().__init__(detail, status_code=403)


class RateLimitExceededError(ArborException):
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(detail, status_code=429)
