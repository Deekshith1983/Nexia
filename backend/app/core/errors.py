"""
Nexia Core Error Hierarchy.
Structured, safe domain exceptions categorized for production error handling and non-sensitive API responses.
"""
from typing import Any, Dict, Optional


class NexiaError(Exception):
    """Base class for all structured Nexia domain errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class ValidationError(NexiaError):
    """Raised when an input or validation rule check fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="VALIDATION_ERROR", status_code=400, details=details)


class ModelUnavailableError(NexiaError):
    """Raised when the language model provider is unreachable or offline."""

    def __init__(self, message: str = "Language model service is unavailable.", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="MODEL_UNAVAILABLE", status_code=503, details=details)


class ModelTimeoutError(NexiaError):
    """Raised when a language model generation times out."""

    def __init__(self, message: str = "Language model request timed out.", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="MODEL_TIMEOUT", status_code=504, details=details)


class KnowledgeProviderError(NexiaError):
    """Raised when an external knowledge source fails or times out."""

    def __init__(self, message: str = "External knowledge provider failed.", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="KNOWLEDGE_PROVIDER_ERROR", status_code=502, details=details)


class PersistenceError(NexiaError):
    """Raised when a database or filesystem persistence operation fails."""

    def __init__(self, message: str = "Database operation failed.", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="PERSISTENCE_ERROR", status_code=500, details=details)


class SessionError(NexiaError):
    """Raised when research session operations or lifecycle transitions fail."""

    def __init__(self, message: str = "Research session error.", status_code: int = 404, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="SESSION_ERROR", status_code=status_code, details=details)


class CancellationError(NexiaError):
    """Raised when an operation or stream is cancelled by user/client disconnect."""

    def __init__(self, message: str = "Operation was cancelled.", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="CANCELLED", status_code=499, details=details)


class ConfigurationError(NexiaError):
    """Raised when configuration parameters or runtime settings are invalid."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="CONFIGURATION_ERROR", status_code=500, details=details)


class SecurityViolationError(NexiaError):
    """Raised when a security boundary, injection attempt, or unauthorized access is detected."""

    def __init__(self, message: str = "Security boundary check failed.", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="SECURITY_VIOLATION", status_code=403, details=details)
