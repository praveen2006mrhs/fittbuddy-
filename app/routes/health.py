"""Health check endpoint for FitBuddy."""

from fastapi import APIRouter, Depends
from app.config import Settings, get_settings

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(settings: Settings = Depends(get_settings)):
    """Return health status of the application and configuration summary."""
    gemini_configured = settings.is_gemini_configured
    
    return {
        "status": "healthy",
        "app": settings.app_name,
        "environment": settings.app_env,
        "database": {
            "type": "sqlite",
            "url": settings.database_url,
        },
        "gemini": {
            "configured": gemini_configured,
            "mock_enabled": settings.gemini_mock_enabled,
            "mode": "live" if gemini_configured else ("mock" if settings.gemini_mock_enabled else "unconfigured"),
            "workout_model": settings.gemini_workout_model,
            "tip_model": settings.gemini_tip_model,
            "timeout_seconds": settings.gemini_timeout_seconds,
            "max_retries": settings.gemini_max_retries,
        },
    }
