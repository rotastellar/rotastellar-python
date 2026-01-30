"""
RotaStellar SDK - Custom Exceptions

All custom exceptions raised by the RotaStellar SDK.
"""

from typing import Optional, Dict, Any


class RotaStellarError(Exception):
    """Base exception for all RotaStellar SDK errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - {self.details}"
        return self.message


class AuthenticationError(RotaStellarError):
    """Raised when authentication fails."""

    pass


class MissingAPIKeyError(AuthenticationError):
    """Raised when API key is missing or empty."""

    def __init__(self):
        super().__init__(
            "API key is required. Get your key at https://rotastellar.com/dashboard"
        )


class InvalidAPIKeyError(AuthenticationError):
    """Raised when API key format is invalid."""

    def __init__(self, api_key: str):
        masked = api_key[:10] + "..." if len(api_key) > 10 else api_key
        super().__init__(
            f"Invalid API key format: {masked}. "
            "Keys should start with 'rs_live_' or 'rs_test_'"
        )


class APIError(RotaStellarError):
    """Raised when the API returns an error response."""

    def __init__(
        self,
        message: str,
        status_code: int,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.status_code = status_code
        self.request_id = request_id

    def __str__(self) -> str:
        parts = [f"[{self.status_code}] {self.message}"]
        if self.request_id:
            parts.append(f"(request_id: {self.request_id})")
        return " ".join(parts)


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        retry_after: Optional[int] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            "Rate limit exceeded",
            status_code=429,
            request_id=request_id,
            details={"retry_after": retry_after},
        )
        self.retry_after = retry_after


class NotFoundError(APIError):
    """Raised when a requested resource is not found."""

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            f"{resource_type} not found: {resource_id}",
            status_code=404,
            request_id=request_id,
            details={"resource_type": resource_type, "resource_id": resource_id},
        )


class ValidationError(RotaStellarError):
    """Raised when input validation fails."""

    def __init__(self, field: str, message: str):
        super().__init__(f"Validation error on '{field}': {message}")
        self.field = field


class NetworkError(RotaStellarError):
    """Raised when a network error occurs."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class TimeoutError(NetworkError):
    """Raised when a request times out."""

    def __init__(self, timeout_seconds: float):
        super().__init__(f"Request timed out after {timeout_seconds} seconds")
        self.timeout_seconds = timeout_seconds
