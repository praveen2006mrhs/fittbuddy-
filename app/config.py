"""Configuration management for FitBuddy using Pydantic Settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # Core Application Settings
    app_name: str = "FitBuddy"
    app_env: str = "development"
    debug: bool = True
    host: str = "127.0.0.1"
    port: int = 8000

    # Security & Session
    session_secret: str = "fitbuddy-super-secret-session-key-change-me"

    # Database
    database_url: str = "sqlite:///./fitbuddy.db"

    # Google Gemini GenAI Settings
    gemini_api_key: str = ""
    gemini_workout_model: str = "gemini-2.5-flash"
    gemini_tip_model: str = "gemini-2.5-flash"
    gemini_timeout_seconds: float = 30.0
    gemini_max_retries: int = 3
    gemini_retry_delay_seconds: float = 1.0
    gemini_mock_enabled: bool = False

    @property
    def is_gemini_configured(self) -> bool:
        """Check if a non-empty, non-placeholder Gemini API key is configured."""
        key = (self.gemini_api_key or "").strip()
        return bool(key and "your_gemini_api_key" not in key.lower())

    @property
    def masked_gemini_key(self) -> str:
        """Return a safe masked representation of the API key, never exposing the full key."""
        key = (self.gemini_api_key or "").strip()
        if not key:
            return "Not configured"
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()
