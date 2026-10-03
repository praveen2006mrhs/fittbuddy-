"""Profile routes for reading, updating, and deleting the authenticated user's profile."""

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, Request, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import (
    User,
    FitnessProfile,
    FitnessGoal,
    WorkoutIntensity,
    ExperienceLevel,
    Equipment,
)
from app.schemas import FitnessProfileSchema
from app.services.security import (
    get_current_user,
    generate_csrf_token,
    verify_csrf_token,
    verify_password,
    SESSION_COOKIE_NAME,
)

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter(prefix="/profile", tags=["Profile"])
settings = get_settings()


@router.get("", response_class=HTMLResponse)
async def view_profile(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """View and edit the current user's fitness profile."""
    profile = db.query(FitnessProfile).filter(FitnessProfile.user_id == user.id).first()
    csrf_token = generate_csrf_token(subject=str(user.id))

    return templates.TemplateResponse(
        request=request,
        name="profile/index.html",
        context={
            "app_name": settings.app_name,
            "user": user,
            "profile": profile,
            "csrf_token": csrf_token,
            "goals": [g.value for g in FitnessGoal],
            "intensities": [i.value for i in WorkoutIntensity],
            "levels": [e.value for e in ExperienceLevel],
            "equipment_options": [eq.value for eq in Equipment],
            "errors": [],
            "success_message": None,
            "form_data": {},
        },
    )


@router.post("", response_class=HTMLResponse)
async def update_profile(
    request: Request,
    display_name: str = Form(...),
    age: str = Form(...),
    weight_kg: str = Form(...),
    fitness_goal: str = Form(...),
    workout_intensity: str = Form(...),
    experience_level: str = Form(...),
    equipment: str = Form(...),
    available_minutes: str = Form(...),
    exercise_limitations: Optional[str] = Form(None),
    csrf_token: str = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Validate and persist the authenticated user's fitness profile."""
    # Retain entered values for repopulating form on validation error
    raw_form = {
        "display_name": display_name,
        "age": age,
        "weight_kg": weight_kg,
        "fitness_goal": fitness_goal,
        "workout_intensity": workout_intensity,
        "experience_level": experience_level,
        "equipment": equipment,
        "available_minutes": available_minutes,
        "exercise_limitations": exercise_limitations or "",
    }

    # Verify CSRF token
    if not verify_csrf_token(csrf_token, subject=str(user.id)):
        new_csrf = generate_csrf_token(subject=str(user.id))
        return templates.TemplateResponse(
            request=request,
            name="profile/index.html",
            context={
                "app_name": settings.app_name,
                "user": user,
                "profile": user.profile,
                "csrf_token": new_csrf,
                "goals": [g.value for g in FitnessGoal],
                "intensities": [i.value for i in WorkoutIntensity],
                "levels": [e.value for e in ExperienceLevel],
                "equipment_options": [eq.value for eq in Equipment],
                "errors": ["Security check failed (invalid or expired CSRF token). Please try again."],
                "success_message": None,
                "form_data": raw_form,
            },
            status_code=status.HTTP_403_FORBIDDEN,
        )

    # Validate inputs with Pydantic
    errors = []
    try:
        validated = FitnessProfileSchema(
            display_name=display_name,
            age=age,
            weight_kg=weight_kg,
            fitness_goal=fitness_goal,
            workout_intensity=workout_intensity,
            experience_level=experience_level,
            equipment=equipment,
            available_minutes=available_minutes,
            exercise_limitations=exercise_limitations,
        )
    except ValidationError as exc:
        for err in exc.errors():
            errors.append(err["msg"])
        new_csrf = generate_csrf_token(subject=str(user.id))
        return templates.TemplateResponse(
            request=request,
            name="profile/index.html",
            context={
                "app_name": settings.app_name,
                "user": user,
                "profile": user.profile,
                "csrf_token": new_csrf,
                "goals": [g.value for g in FitnessGoal],
                "intensities": [i.value for i in WorkoutIntensity],
                "levels": [e.value for e in ExperienceLevel],
                "equipment_options": [eq.value for eq in Equipment],
                "errors": errors,
                "success_message": None,
                "form_data": raw_form,
            },
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    # Persist or update profile
    profile = db.query(FitnessProfile).filter(FitnessProfile.user_id == user.id).first()
    if profile:
        profile.display_name = validated.display_name
        profile.age = validated.age
        profile.weight_kg = validated.weight_kg
        profile.fitness_goal = validated.fitness_goal.value
        profile.workout_intensity = validated.workout_intensity.value
        profile.experience_level = validated.experience_level.value
        profile.equipment = validated.equipment.value
        profile.available_minutes = validated.available_minutes
        profile.exercise_limitations = validated.exercise_limitations
    else:
        profile = FitnessProfile(
            user_id=user.id,
            display_name=validated.display_name,
            age=validated.age,
            weight_kg=validated.weight_kg,
            fitness_goal=validated.fitness_goal.value,
            workout_intensity=validated.workout_intensity.value,
            experience_level=validated.experience_level.value,
            equipment=validated.equipment.value,
            available_minutes=validated.available_minutes,
            exercise_limitations=validated.exercise_limitations,
        )
        db.add(profile)

    db.commit()
    db.refresh(profile)

    new_csrf = generate_csrf_token(subject=str(user.id))
    return templates.TemplateResponse(
        request=request,
        name="profile/index.html",
        context={
            "app_name": settings.app_name,
            "user": user,
            "profile": profile,
            "csrf_token": new_csrf,
            "goals": [g.value for g in FitnessGoal],
            "intensities": [i.value for i in WorkoutIntensity],
            "levels": [e.value for e in ExperienceLevel],
            "equipment_options": [eq.value for eq in Equipment],
            "errors": [],
            "success_message": "Fitness profile saved successfully!",
            "form_data": {},
        },
    )


@router.get("/delete", response_class=HTMLResponse)
async def delete_account_page(
    request: Request,
    user: User = Depends(get_current_user),
):
    """Render account deletion confirmation page."""
    csrf_token = generate_csrf_token(subject=str(user.id))
    return templates.TemplateResponse(
        request=request,
        name="profile/delete.html",
        context={
            "app_name": settings.app_name,
            "user": user,
            "csrf_token": csrf_token,
            "errors": [],
        },
    )


@router.post("/delete")
async def handle_delete_account(
    request: Request,
    password: str = Form(...),
    confirmation_text: str = Form(...),
    csrf_token: str = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Permanently delete user account and all cascading dependent data."""
    if not verify_csrf_token(csrf_token, subject=str(user.id)):
        new_csrf = generate_csrf_token(subject=str(user.id))
        return templates.TemplateResponse(
            request=request,
            name="profile/delete.html",
            context={
                "app_name": settings.app_name,
                "user": user,
                "csrf_token": new_csrf,
                "errors": ["Security check failed (invalid or expired CSRF token). Please try again."],
            },
            status_code=status.HTTP_403_FORBIDDEN,
        )

    # Require explicit confirmation phrase
    if confirmation_text.strip() != "DELETE":
        new_csrf = generate_csrf_token(subject=str(user.id))
        return templates.TemplateResponse(
            request=request,
            name="profile/delete.html",
            context={
                "app_name": settings.app_name,
                "user": user,
                "csrf_token": new_csrf,
                "errors": ["You must type DELETE exactly into the confirmation field."],
            },
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    # Verify password before deletion
    if not verify_password(password, user.password_hash):
        new_csrf = generate_csrf_token(subject=str(user.id))
        return templates.TemplateResponse(
            request=request,
            name="profile/delete.html",
            context={
                "app_name": settings.app_name,
                "user": user,
                "csrf_token": new_csrf,
                "errors": ["Incorrect password. Account deletion was not completed."],
            },
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    # Delete user (foreign-key cascade removes profile, sessions, plans, feedbacks, tips)
    db.delete(user)
    db.commit()

    response = RedirectResponse(
        url="/register?deleted=1",
        status_code=status.HTTP_303_SEE_OTHER,
    )
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
    )
    return response
