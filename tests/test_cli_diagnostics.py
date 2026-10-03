"""Tests for FitBuddy CLI Gemini diagnostics command."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.cli import run_gemini_diagnostics
from app.config import Settings
from app.services.ai import GeminiAIService, AIResult


def test_cli_diagnostics_when_unconfigured(capsys):
    """Verify gemini-diagnostics reports missing config and instructions when key is missing."""
    test_settings = Settings(gemini_api_key="", gemini_mock_enabled=False)
    success = run_gemini_diagnostics(test_request=False, settings=test_settings)

    captured = capsys.readouterr()
    assert success is False
    assert "[CONFIG CHECK] Status: NOT CONFIGURED" in captured.out
    assert "GEMINI_API_KEY=your_actual_key_here" in captured.out
    assert "Live integration remains unverified." in captured.out
    assert "https://aistudio.google.com/" in captured.out


def test_cli_diagnostics_lists_models(capsys):
    """Verify gemini-diagnostics lists models when key is configured."""
    test_settings = Settings(
        gemini_api_key="AIzaSyDummyKeyForTestingOnly12345",
        gemini_mock_enabled=False,
    )

    mock_models = [
        {
            "name": "models/gemini-2.5-flash",
            "display_name": "Gemini 2.5 Flash",
            "description": "Fast multimodal model",
            "input_token_limit": 1048576,
            "output_token_limit": 8192,
            "supported_actions": ["generateContent"],
        },
        {
            "name": "models/gemini-2.5-pro",
            "display_name": "Gemini 2.5 Pro",
            "description": "Reasoning model",
            "input_token_limit": 2097152,
            "output_token_limit": 8192,
            "supported_actions": ["generateContent"],
        },
    ]

    with patch.object(GeminiAIService, "list_compatible_models", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = mock_models
        success = run_gemini_diagnostics(test_request=False, settings=test_settings)

    captured = capsys.readouterr()
    assert success is True
    assert "[CONFIG CHECK] Status: CONFIGURED" in captured.out
    assert "models/gemini-2.5-flash" in captured.out
    assert "AIzaSyDummyKeyForTestingOnly12345" not in captured.out  # NEVER PRINT KEY!
    assert test_settings.masked_gemini_key in captured.out


def test_cli_diagnostics_runs_minimal_test(capsys):
    """Verify gemini-diagnostics --test executes minimal request."""
    test_settings = Settings(
        gemini_api_key="AIzaSyDummyKeyForTestingOnly12345",
        gemini_mock_enabled=False,
    )

    mock_models = [
        {
            "name": "models/gemini-2.5-flash",
            "display_name": "Gemini 2.5 Flash",
            "supported_actions": ["generateContent"],
        }
    ]

    mock_result = AIResult(
        data="FitBuddy AI diagnostic check successful.",
        raw_text="FitBuddy AI diagnostic check successful.",
        model_identifier="gemini-2.5-flash",
        is_mock=False,
        finish_reason="STOP",
        usage_metadata={"prompt_tokens": 10, "candidates_tokens": 6, "total_tokens": 16},
    )

    with patch.object(GeminiAIService, "list_compatible_models", new_callable=AsyncMock, return_value=mock_models):
        with patch.object(GeminiAIService, "generate_minimal_test", new_callable=AsyncMock, return_value=mock_result):
            success = run_gemini_diagnostics(test_request=True, settings=test_settings)

    captured = capsys.readouterr()
    assert success is True
    assert "[MINIMAL TEST]" in captured.out
    assert "FitBuddy AI diagnostic check successful." in captured.out
    assert "Live Gemini integration verified successfully!" in captured.out
