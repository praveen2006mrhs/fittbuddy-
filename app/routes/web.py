"""Web page routes rendering Jinja2 templates for FitBuddy."""

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import Settings, get_settings
from app.models import User
from app.services.security import get_current_user_optional, generate_csrf_token

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter(tags=["Web"])


@router.get("/", response_class=HTMLResponse)
async def home_page(
    request: Request,
    settings: Settings = Depends(get_settings),
    user: Optional[User] = Depends(get_current_user_optional),
):
    """Render FitBuddy home landing page."""
    gemini_active = settings.is_gemini_configured
    mock_enabled = settings.gemini_mock_enabled
    csrf_token = generate_csrf_token(subject=str(user.id) if user else "anonymous")
    
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": settings.app_name,
            "app_env": settings.app_env,
            "gemini_active": gemini_active,
            "mock_enabled": mock_enabled,
            "workout_model": settings.gemini_workout_model,
            "tip_model": settings.gemini_tip_model,
            "user": user,
            "csrf_token": csrf_token,
        },
    )
