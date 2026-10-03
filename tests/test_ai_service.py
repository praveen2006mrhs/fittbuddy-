"""Comprehensive unit and integration tests for FitBuddy Gemini integration service."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from pydantic import ValidationError

from google.genai import errors, types

from app.config import Settings
from app.schemas import (
    AnonymousFitnessProfile,
    FitnessProfileSchema,
    WorkoutPlanResponseSchema,
    FitnessTipResponseSchema,
)
from app.services.ai import (
    GeminiAIService,
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


@pytest.fixture
def sample_profile_schema():
    """Valid fitness profile schema input."""
    return FitnessProfileSchema(
        display_name="Jordan Fit",
        age=28,
        weight_kg=75.5,
        fitness_goal="muscle_gain",
        workout_intensity="high",
        experience_level="intermediate",
        equipment="dumbbells",
        available_minutes=45,
        exercise_limitations="Mild lower back stiffness in morning.",
    )


# ==============================================================================
# 1. Data Minimization & Privacy Tests
# ==============================================================================

def test_data_minimization_strips_pii(sample_profile_schema):
    """Verify that AnonymousFitnessProfile strictly extracts physical parameters and excludes PII."""
    anon = AnonymousFitnessProfile.from_profile(sample_profile_schema)

    # Assert only non-identifying metrics are preserved
    assert anon.age == 28
    assert anon.weight_kg == 75.5
    assert anon.fitness_goal == "muscle_gain"
    assert anon.workout_intensity == "high"
    assert anon.experience_level == "intermediate"
    assert anon.equipment == "dumbbells"
    assert anon.available_minutes == 45
    assert anon.exercise_limitations == "Mild lower back stiffness in morning."

    # Assert no PII fields exist on schema
    assert not hasattr(anon, "display_name")
    assert not hasattr(anon, "email")
    assert not hasattr(anon, "user_id")
    assert not hasattr(anon, "password")
    assert not hasattr(anon, "id")

    # Assert prompt context string contains NO user identifiers
    prompt_context = anon.to_prompt_context()
    assert "Jordan" not in prompt_context
    assert "display_name" not in prompt_context
    assert "75.5 kg" in prompt_context
    assert "Muscle Gain" in prompt_context
    assert "Dumbbells" in prompt_context


# ==============================================================================
# 2. Configuration & Secret Masking Tests
# ==============================================================================

def test_key_masking_never_exposes_full_key():
    """Verify that masked_gemini_key hides the middle of credentials and never leaks secrets."""
    s1 = Settings(gemini_api_key="")
    assert s1.masked_gemini_key == "Not configured"
    assert not s1.is_gemini_configured

    s2 = Settings(gemini_api_key="your_gemini_api_key_here")
    assert not s2.is_gemini_configured

    real_looking_key = "AIzaSyABC1234567890XYZabcdefghijklmno"
    s3 = Settings(gemini_api_key=real_looking_key)
    assert s3.is_gemini_configured
    masked = s3.masked_gemini_key
    assert masked.startswith("AIza")
    assert masked.endswith("lmno")
    assert "..." in masked
    assert real_looking_key not in masked


def test_missing_credentials_raises_configuration_error(sample_profile_schema):
    """Verify that missing API key raises GeminiConfigurationError with helpful instructions."""
    settings = Settings(gemini_api_key="", gemini_mock_enabled=False)
    service = GeminiAIService(settings=settings)
    anon = AnonymousFitnessProfile.from_profile(sample_profile_schema)

    assert not service.is_configured

    with pytest.raises(GeminiConfigurationError) as exc_info:
        asyncio.run(service.generate_workout_plan(anon))

    err = exc_info.value
    assert "GEMINI_API_KEY" in err.user_message
    assert err.code == "MISSING_CREDENTIALS"
    assert err.status_code == 503


# ==============================================================================
# 3. Deterministic Mock Mode Tests
# ==============================================================================

def test_mock_mode_returns_structured_mock_plan(sample_profile_schema):
    """Verify that explicit mock mode generates complete 7-day plan with is_mock=True."""
    settings = Settings(gemini_api_key="", gemini_mock_enabled=True)
    service = GeminiAIService(settings=settings)
    anon = AnonymousFitnessProfile.from_profile(sample_profile_schema)

    result = asyncio.run(service.generate_workout_plan(anon))

    assert isinstance(result, AIResult)
    assert result.is_mock is True
    assert result.model_identifier == "mock-fitbuddy-v1"
    assert isinstance(result.data, WorkoutPlanResponseSchema)
    assert len(result.data.schedule) == 7

    # Check each day
    for i, day in enumerate(result.data.schedule, 1):
        assert day.day_number == i
        assert day.day_name
        assert day.focus
        assert len(day.exercises) > 0
        for ex in day.exercises:
            assert ex.name
            assert ex.sets >= 1
            assert len(ex.form_cues) >= 1


def test_mock_mode_returns_structured_tips(sample_profile_schema):
    """Verify that explicit mock mode generates structured nutrition and recovery tips."""
    settings = Settings(gemini_api_key="", gemini_mock_enabled=True)
    service = GeminiAIService(settings=settings)
    anon = AnonymousFitnessProfile.from_profile(sample_profile_schema)

    nutr_res = asyncio.run(service.generate_fitness_tip(anon, category="nutrition"))
    assert nutr_res.is_mock is True
    assert isinstance(nutr_res.data, FitnessTipResponseSchema)
    assert nutr_res.data.category == "nutrition"
    assert len(nutr_res.data.actionable_steps) >= 2

    rec_res = asyncio.run(service.generate_fitness_tip(anon, category="recovery"))
    assert rec_res.is_mock is True
    assert isinstance(rec_res.data, FitnessTipResponseSchema)
    assert rec_res.data.category == "recovery"
    assert len(rec_res.data.actionable_steps) >= 2


def test_mock_mode_minimal_test():
    """Verify minimal test request works in mock mode."""
    settings = Settings(gemini_mock_enabled=True)
    service = GeminiAIService(settings=settings)
    res = asyncio.run(service.generate_minimal_test())
    assert res.is_mock is True
    assert "mock active" in res.data


def test_live_failure_never_falls_back_to_mock(sample_profile_schema):
    """Verify that live API failure raises exception honestly and never replaces with mock."""
    settings = Settings(
        gemini_api_key="AIzaFakeKeyForTest",
        gemini_mock_enabled=False,  # Live mode!
        gemini_max_retries=0,
    )
    service = GeminiAIService(settings=settings)
    anon = AnonymousFitnessProfile.from_profile(sample_profile_schema)

    # Mock client generate_content to simulate auth rejection (401)
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=errors.ClientError(code=401, response_json={"error": "API_KEY_INVALID"})
    )

    with patch.object(service, "_get_client", return_value=mock_client):
        with pytest.raises(GeminiAuthenticationError):
            asyncio.run(service.generate_workout_plan(anon))


# ==============================================================================
# 4. Error Handling & Retry Logic Tests
# ==============================================================================

def test_authentication_error_not_retried(sample_profile_schema):
    """Verify 401/403 ClientError raises GeminiAuthenticationError immediately without retrying."""
    settings = Settings(
        gemini_api_key="AIzaFakeKeyForTest",
        gemini_mock_enabled=False,
        gemini_max_retries=3,
    )
    service = GeminiAIService(settings=settings)
    anon = AnonymousFitnessProfile.from_profile(sample_profile_schema)

    mock_client = MagicMock()
    mock_call = AsyncMock(
        side_effect=errors.ClientError(code=403, response_json={"error": "PERMISSION_DENIED"})
    )
    mock_client.aio.models.generate_content = mock_call

    with patch.object(service, "_get_client", return_value=mock_client):
        with pytest.raises(GeminiAuthenticationError) as exc_info:
            asyncio.run(service.generate_workout_plan(anon))

        assert exc_info.value.code == "INVALID_CREDENTIALS"
        # Must have been called exactly once (no retries)
        assert mock_call.call_count == 1


def test_model_not_found_not_retried(sample_profile_schema):
    """Verify 404 ClientError raises GeminiModelNotFoundError immediately."""
    settings = Settings(
        gemini_api_key="AIzaFakeKeyForTest",
        gemini_workout_model="nonexistent-gemini-model",
        gemini_mock_enabled=False,
        gemini_max_retries=3,
    )
    service = GeminiAIService(settings=settings)
    anon = AnonymousFitnessProfile.from_profile(sample_profile_schema)

    mock_client = MagicMock()
    mock_call = AsyncMock(
        side_effect=errors.ClientError(code=404, response_json={"error": "MODEL_NOT_FOUND"})
    )
    mock_client.aio.models.generate_content = mock_call

    with patch.object(service, "_get_client", return_value=mock_client):
        with pytest.raises(GeminiModelNotFoundError) as exc_info:
            asyncio.run(service.generate_workout_plan(anon))

        assert exc_info.value.code == "MODEL_NOT_FOUND"
        assert mock_call.call_count == 1


def test_rate_limit_bounded_retries_and_exhaustion(sample_profile_schema):
    """Verify 429 errors are retried up to max_retries before raising GeminiQuotaExhaustedError."""
    settings = Settings(
        gemini_api_key="AIzaFakeKeyForTest",
        gemini_mock_enabled=False,
        gemini_max_retries=2,
        gemini_retry_delay_seconds=0.01,  # Fast for testing
    )
    service = GeminiAIService(settings=settings)
    anon = AnonymousFitnessProfile.from_profile(sample_profile_schema)

    mock_client = MagicMock()
    mock_call = AsyncMock(
        side_effect=errors.ClientError(code=429, response_json={"error": "RESOURCE_EXHAUSTED"})
    )
    mock_client.aio.models.generate_content = mock_call

    with patch.object(service, "_get_client", return_value=mock_client):
        with pytest.raises(GeminiQuotaExhaustedError) as exc_info:
            asyncio.run(service.generate_workout_plan(anon))

        assert exc_info.value.code == "QUOTA_EXHAUSTED"
        # Called initial attempt (1) + 2 retries = 3 calls total
        assert mock_call.call_count == 3


def test_transient_server_error_recovers_on_retry(sample_profile_schema):
    """Verify transient 503 ServerError recovers on subsequent retry."""
    settings = Settings(
        gemini_api_key="AIzaFakeKeyForTest",
        gemini_mock_enabled=False,
        gemini_max_retries=2,
        gemini_retry_delay_seconds=0.01,
    )
    service = GeminiAIService(settings=settings)
    anon = AnonymousFitnessProfile.from_profile(sample_profile_schema)

    # Valid response to return on second attempt
    mock_plan = service._build_deterministic_mock_plan(anon)
    success_response = MagicMock()
    success_response.parsed = mock_plan
    success_response.text = mock_plan.model_dump_json()
    success_response.candidates = [MagicMock(finish_reason="STOP")]
    success_response.prompt_feedback = None
    success_response.usage_metadata = None

    mock_client = MagicMock()
    mock_call = AsyncMock(
        side_effect=[
            errors.ServerError(code=503, response_json={"error": "UNAVAILABLE"}),
            success_response,
        ]
    )
    mock_client.aio.models.generate_content = mock_call

    with patch.object(service, "_get_client", return_value=mock_client):
        result = asyncio.run(service.generate_workout_plan(anon))
        assert result.is_mock is False
        assert mock_call.call_count == 2


def test_safety_blocked_response_raises_safety_error(sample_profile_schema):
    """Verify safety finish_reason triggers GeminiSafetyBlockedError."""
    settings = Settings(
        gemini_api_key="AIzaFakeKeyForTest",
        gemini_mock_enabled=False,
        gemini_max_retries=1,
    )
    service = GeminiAIService(settings=settings)
    anon = AnonymousFitnessProfile.from_profile(sample_profile_schema)

    blocked_response = MagicMock()
    blocked_response.prompt_feedback = None
    blocked_candidate = MagicMock()
    blocked_candidate.finish_reason = "SAFETY"
    blocked_response.candidates = [blocked_candidate]

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=blocked_response)

    with patch.object(service, "_get_client", return_value=mock_client):
        with pytest.raises(GeminiSafetyBlockedError) as exc_info:
            asyncio.run(service.generate_workout_plan(anon))

        assert exc_info.value.code == "SAFETY_BLOCKED"


def test_malformed_json_raises_output_validation_error(sample_profile_schema):
    """Verify unparseable response raises GeminiOutputValidationError."""
    settings = Settings(
        gemini_api_key="AIzaFakeKeyForTest",
        gemini_mock_enabled=False,
        gemini_max_retries=0,
    )
    service = GeminiAIService(settings=settings)
    anon = AnonymousFitnessProfile.from_profile(sample_profile_schema)

    malformed_response = MagicMock()
    malformed_response.parsed = None
    malformed_response.text = "This is not valid JSON content"
    malformed_response.candidates = [MagicMock(finish_reason="STOP")]
    malformed_response.prompt_feedback = None

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=malformed_response)

    with patch.object(service, "_get_client", return_value=mock_client):
        with pytest.raises(GeminiOutputValidationError) as exc_info:
            asyncio.run(service.generate_workout_plan(anon))

        assert exc_info.value.code == "OUTPUT_VALIDATION_ERROR"


def test_empty_candidates_raises_empty_response_error(sample_profile_schema):
    """Verify empty candidate list raises GeminiEmptyResponseError."""
    settings = Settings(
        gemini_api_key="AIzaFakeKeyForTest",
        gemini_mock_enabled=False,
        gemini_max_retries=0,
    )
    service = GeminiAIService(settings=settings)
    anon = AnonymousFitnessProfile.from_profile(sample_profile_schema)

    empty_response = MagicMock()
    empty_response.candidates = []
    empty_response.prompt_feedback = None

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=empty_response)

    with patch.object(service, "_get_client", return_value=mock_client):
        with pytest.raises(GeminiEmptyResponseError) as exc_info:
            asyncio.run(service.generate_workout_plan(anon))

        assert exc_info.value.code == "EMPTY_RESPONSE"
