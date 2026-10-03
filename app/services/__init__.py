"""Services package for FitBuddy."""

from app.services.ai import (
    GeminiAIService,
    get_ai_service,
    AIServiceError,
    GeminiConfigurationError,
    GeminiAuthenticationError,
    GeminiModelNotFoundError,
    GeminiQuotaExhaustedError,
    GeminiSafetyBlockedError,
    GeminiEmptyResponseError,
    GeminiOutputValidationError,
    GeminiTimeoutError,
    GeminiNetworkError,
    AIResult,
)

__all__ = [
    "GeminiAIService",
    "get_ai_service",
    "AIServiceError",
    "GeminiConfigurationError",
    "GeminiAuthenticationError",
    "GeminiModelNotFoundError",
    "GeminiQuotaExhaustedError",
    "GeminiSafetyBlockedError",
    "GeminiEmptyResponseError",
    "GeminiOutputValidationError",
    "GeminiTimeoutError",
    "GeminiNetworkError",
    "AIResult",
]
