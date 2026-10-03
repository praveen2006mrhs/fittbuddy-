"""Routes for generating, listing, and retrieving personalized 7-day workout plans."""

import json
import logging
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from fastapi import (
    APIRouter,
    Depends,
    Request,
    Form,
    HTTPException,
    status,
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import SessionLocal, get_db
from app.models import User, FitnessProfile, WorkoutPlan, PlanFeedback
from app.schemas import (
    AnonymousFitnessProfile,
    WorkoutPlanResponseSchema,
    PlanRevisionInputSchema,
)
from app.services.ai import get_ai_service, AIServiceError, GeminiOutputValidationError
from app.services.security import (
    get_current_user,
    generate_csrf_token,
    verify_csrf_token,
)

logger = logging.getLogger("fitbuddy.plans")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter(tags=["Workout Plans"])
settings = get_settings()

# In-memory tracking of active generations to prevent accidental concurrent double submissions
_active_generations: set[int] = set()
_generation_lock = asyncio.Lock()


def _is_json_request(request: Request) -> bool:
    """Determine if client expects a JSON response."""
    accept = request.headers.get("accept", "")
    content_type = request.headers.get("content-type", "")
    return (
        "application/json" in accept
        or "application/json" in content_type
        or request.url.path.startswith("/api/")
    )


# ==============================================================================
# 1. Generate & Save Workout Plan
# ==============================================================================

@router.post("/plans/generate")
@router.post("/api/plans/generate")
async def generate_and_save_plan(
    request: Request,
    csrf_token: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
):
    """Generate and persist a personalized 7-day workout plan for the authenticated user.

    Safety & Concurrency Guarantees:
    - Protects against accidental duplicate submissions via per-user lock and debounce.
    - Database transaction is NOT held open during the AI network call.
    - Stores full profile snapshot used for generation and creation metadata.
    - Rejects malformed AI output without creating a broken plan record in the database.
    """
    is_json = _is_json_request(request)

    # 1. CSRF Verification for HTML form requests
    if not is_json:
        if not csrf_token or not verify_csrf_token(csrf_token, subject=str(user.id)):
            new_csrf = generate_csrf_token(subject=str(user.id))
            return templates.TemplateResponse(
                request=request,
                name="plans/index.html",
                context={
                    "app_name": settings.app_name,
                    "user": user,
                    "csrf_token": new_csrf,
                    "plans": [],
                    "errors": ["Security check failed (invalid or expired CSRF token). Please try again."],
                    "has_profile": True,
                },
                status_code=status.HTTP_403_FORBIDDEN,
            )

    # 2. Duplicate submission protection (active generation lock)
    async with _generation_lock:
        if user.id in _active_generations:
            err_msg = "A workout plan is already currently being generated for your account. Please wait a moment."
            if is_json:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=err_msg)
            new_csrf = generate_csrf_token(subject=str(user.id))
            return templates.TemplateResponse(
                request=request,
                name="plans/index.html",
                context={
                    "app_name": settings.app_name,
                    "user": user,
                    "csrf_token": new_csrf,
                    "plans": [],
                    "errors": [err_msg],
                    "has_profile": True,
                },
                status_code=status.HTTP_409_CONFLICT,
            )
        _active_generations.add(user.id)

    try:
        # 3. Retrieve user profile and snapshot in a dedicated short DB transaction
        profile_snapshot_json: str
        anonymous_profile: AnonymousFitnessProfile
        profile_snapshot_data: dict

        with SessionLocal() as db:
            profile = db.query(FitnessProfile).filter(FitnessProfile.user_id == user.id).first()
            if not profile:
                err_msg = "Please complete your fitness profile before generating a workout plan."
                if is_json:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)
                new_csrf = generate_csrf_token(subject=str(user.id))
                return templates.TemplateResponse(
                    request=request,
                    name="plans/index.html",
                    context={
                        "app_name": settings.app_name,
                        "user": user,
                        "csrf_token": new_csrf,
                        "plans": [],
                        "errors": [err_msg],
                        "has_profile": False,
                    },
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

            # Check debounce: avoid rapid spamming within 3 seconds
            latest_recent = (
                db.query(WorkoutPlan)
                .filter(WorkoutPlan.owner_id == user.id)
                .order_by(WorkoutPlan.created_at.desc())
                .first()
            )
            if latest_recent and latest_recent.created_at:
                now_utc = datetime.now(timezone.utc)
                plan_time = latest_recent.created_at
                if plan_time.tzinfo is None:
                    plan_time = plan_time.replace(tzinfo=timezone.utc)
                diff = (now_utc - plan_time).total_seconds()
                if diff < 3.0:
                    err_msg = "Please wait a few moments before generating another workout plan."
                    if is_json:
                        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=err_msg)
                    new_csrf = generate_csrf_token(subject=str(user.id))
                    return templates.TemplateResponse(
                        request=request,
                        name="plans/index.html",
                        context={
                            "app_name": settings.app_name,
                            "user": user,
                            "csrf_token": new_csrf,
                            "plans": [],
                            "errors": [err_msg],
                            "has_profile": True,
                        },
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    )

            profile_snapshot_data = {
                "age": profile.age,
                "weight_kg": profile.weight_kg,
                "fitness_goal": profile.fitness_goal,
                "workout_intensity": profile.workout_intensity,
                "experience_level": profile.experience_level,
                "equipment": profile.equipment,
                "available_minutes": profile.available_minutes,
                "exercise_limitations": profile.exercise_limitations,
                "snapshot_at": datetime.now(timezone.utc).isoformat(),
            }
            profile_snapshot_json = json.dumps(profile_snapshot_data)
            anonymous_profile = AnonymousFitnessProfile.from_profile(profile)

        # Database session is CLOSED here. ZERO DB transactions held open during AI call!

        # 4. Invoke AI Generation Service
        ai_service = get_ai_service()
        try:
            ai_result = await ai_service.generate_workout_plan(anonymous_profile=anonymous_profile)
        except AIServiceError as exc:
            logger.error("AI service error generating workout plan: %s", exc.message)
            if is_json:
                raise HTTPException(
                    status_code=exc.status_code if exc.status_code != 500 else status.HTTP_502_BAD_GATEWAY,
                    detail=exc.user_message,
                )
            new_csrf = generate_csrf_token(subject=str(user.id))
            return templates.TemplateResponse(
                request=request,
                name="plans/index.html",
                context={
                    "app_name": settings.app_name,
                    "user": user,
                    "csrf_token": new_csrf,
                    "plans": [],
                    "errors": [exc.user_message],
                    "has_profile": True,
                },
                status_code=exc.status_code if exc.status_code != 500 else status.HTTP_502_BAD_GATEWAY,
            )

        # 5. Persist the validated plan in a new short DB transaction
        created_plan_id: int
        next_version: int
        with SessionLocal() as db:
            latest = (
                db.query(WorkoutPlan)
                .filter(WorkoutPlan.owner_id == user.id)
                .order_by(WorkoutPlan.version.desc())
                .first()
            )
            next_version = (latest.version + 1) if latest else 1

            new_plan = WorkoutPlan(
                owner_id=user.id,
                version=next_version,
                parent_plan_id=latest.id if latest else None,
                profile_snapshot=profile_snapshot_json,
                validated_plan_json=ai_result.data.model_dump_json(),
                model_identifier=ai_result.model_identifier,
                created_at=datetime.now(timezone.utc),
            )
            db.add(new_plan)
            db.commit()
            db.refresh(new_plan)
            created_plan_id = new_plan.id

        logger.info(
            "Successfully created and persisted workout plan #%d (v%d) for user #%d",
            created_plan_id,
            next_version,
            user.id,
        )

        if is_json:
            return JSONResponse(
                content={
                    "status": "success",
                    "plan_id": created_plan_id,
                    "version": next_version,
                    "model_identifier": ai_result.model_identifier,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "plan": ai_result.data.model_dump(),
                    "profile_snapshot": profile_snapshot_data,
                },
                status_code=status.HTTP_201_CREATED,
            )

        return RedirectResponse(
            url=f"/plans/{created_plan_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    finally:
        async with _generation_lock:
            _active_generations.discard(user.id)


# ==============================================================================
# 2. List Current User's Plans
# ==============================================================================

@router.get("/plans", response_class=HTMLResponse)
@router.get("/api/plans")
async def list_user_plans(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all workout plans owned by the authenticated user."""
    plans = (
        db.query(WorkoutPlan)
        .filter(WorkoutPlan.owner_id == user.id)
        .order_by(WorkoutPlan.created_at.desc())
        .all()
    )

    profile = db.query(FitnessProfile).filter(FitnessProfile.user_id == user.id).first()
    has_profile = profile is not None

    plan_items = []
    for p in plans:
        try:
            plan_obj = json.loads(p.validated_plan_json)
        except Exception:
            plan_obj = {}
        try:
            snapshot_obj = json.loads(p.profile_snapshot)
        except Exception:
            snapshot_obj = {}

        plan_items.append({
            "id": p.id,
            "version": p.version,
            "parent_plan_id": p.parent_plan_id,
            "created_at": p.created_at,
            "model_identifier": p.model_identifier,
            "plan_title": plan_obj.get("plan_title", f"Workout Plan v{p.version}"),
            "goal": plan_obj.get("goal", snapshot_obj.get("fitness_goal", "General Fitness")),
            "summary": plan_obj.get("summary", ""),
            "days_count": len(plan_obj.get("days", plan_obj.get("schedule", []))),
            "profile_snapshot": snapshot_obj,
            "plan": plan_obj,
        })

    if _is_json_request(request):
        # Convert created_at to ISO string for JSON serialization
        json_items = []
        for item in plan_items:
            json_copy = dict(item)
            if hasattr(json_copy["created_at"], "isoformat"):
                json_copy["created_at"] = json_copy["created_at"].isoformat()
            json_items.append(json_copy)
        return JSONResponse(content=json_items)

    csrf_token = generate_csrf_token(subject=str(user.id))
    return templates.TemplateResponse(
        request=request,
        name="plans/index.html",
        context={
            "app_name": settings.app_name,
            "user": user,
            "profile": profile,
            "has_profile": has_profile,
            "plans": plan_items,
            "csrf_token": csrf_token,
            "errors": [],
        },
    )


# ==============================================================================
# 3. Retrieve Single Owned Plan
# ==============================================================================

def _get_plan_lineage(db: Session, current_plan: WorkoutPlan, user_id: int) -> list[dict]:
    """Retrieve the full version lineage for a workout plan family.

    Traverses ancestors to locate the root plan, discovers all descendants,
    and returns versions ordered chronologically by version number.
    """
    # 1. Walk up parent_plan_id to find root plan
    curr = current_plan
    ancestor_ids = {curr.id}
    while curr.parent_plan_id:
        parent = (
            db.query(WorkoutPlan)
            .filter(WorkoutPlan.id == curr.parent_plan_id, WorkoutPlan.owner_id == user_id)
            .first()
        )
        if not parent or parent.id in ancestor_ids:
            break
        ancestor_ids.add(parent.id)
        curr = parent
    root_id = curr.id

    # 2. Get all plans for this user in this lineage
    all_user_plans = (
        db.query(WorkoutPlan)
        .filter(WorkoutPlan.owner_id == user_id)
        .all()
    )

    family_plan_ids = {root_id}
    changed = True
    while changed:
        changed = False
        for p in all_user_plans:
            if p.id not in family_plan_ids and p.parent_plan_id in family_plan_ids:
                family_plan_ids.add(p.id)
                changed = True

    family_plan_ids.add(current_plan.id)
    family_plans = [p for p in all_user_plans if p.id in family_plan_ids]
    family_plans.sort(key=lambda p: (p.version, p.id))

    # 3. Gather feedback and change summaries for each version
    lineage = []
    for p in family_plans:
        fb_record = (
            db.query(PlanFeedback)
            .filter(PlanFeedback.resulting_plan_id == p.id)
            .order_by(PlanFeedback.created_at.desc())
            .first()
        )
        feedback_text = fb_record.feedback_text if fb_record else None

        change_summary = None
        try:
            p_data = json.loads(p.validated_plan_json)
            change_summary = p_data.get("change_summary") or p_data.get("explanation_of_changes")
        except Exception:
            pass

        if not change_summary:
            change_summary = (
                "Initial 7-day baseline routine."
                if p.version == 1
                else "Plan revision updated based on user feedback."
            )

        lineage.append({
            "plan_id": p.id,
            "version": p.version,
            "is_current": (p.id == current_plan.id),
            "created_at": p.created_at,
            "feedback": feedback_text,
            "change_summary": change_summary,
            "model_identifier": p.model_identifier,
        })

    return lineage


@router.get("/plans/{plan_id}", response_class=HTMLResponse)
@router.get("/api/plans/{plan_id}")
async def get_workout_plan(
    plan_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve one specific workout plan owned by the authenticated user.

    Ownership Isolation:
    - If plan is owned by another user and current user is not admin, returns 403 Forbidden.
    """
    plan = db.query(WorkoutPlan).filter(WorkoutPlan.id == plan_id).first()
    if not plan:
        if _is_json_request(request):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workout plan not found.")
        return templates.TemplateResponse(
            request=request,
            name="plans/index.html",
            context={
                "app_name": settings.app_name,
                "user": user,
                "plans": [],
                "csrf_token": generate_csrf_token(subject=str(user.id)),
                "errors": [f"Workout plan #{plan_id} was not found."],
                "has_profile": True,
            },
            status_code=status.HTTP_404_NOT_FOUND,
        )

    # Ownership check
    if plan.owner_id != user.id and user.role != "admin":
        if _is_json_request(request):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to view this workout plan.",
            )
        return templates.TemplateResponse(
            request=request,
            name="plans/index.html",
            context={
                "app_name": settings.app_name,
                "user": user,
                "plans": [],
                "csrf_token": generate_csrf_token(subject=str(user.id)),
                "errors": ["You do not have permission to view this workout plan."],
                "has_profile": True,
            },
            status_code=status.HTTP_403_FORBIDDEN,
        )

    try:
        plan_data = json.loads(plan.validated_plan_json)
    except Exception as exc:
        logger.error("Failed to parse validated_plan_json for plan #%d: %s", plan.id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored workout plan data is corrupted or invalid.",
        )

    try:
        snapshot_data = json.loads(plan.profile_snapshot)
    except Exception:
        snapshot_data = {}

    days = plan_data.get("days") or plan_data.get("schedule") or []
    lineage = _get_plan_lineage(db, plan, user.id)

    if _is_json_request(request):
        json_lineage = []
        for item in lineage:
            item_copy = dict(item)
            if hasattr(item_copy["created_at"], "isoformat"):
                item_copy["created_at"] = item_copy["created_at"].isoformat()
            json_lineage.append(item_copy)

        return JSONResponse(
            content={
                "id": plan.id,
                "version": plan.version,
                "owner_id": plan.owner_id,
                "parent_plan_id": plan.parent_plan_id,
                "model_identifier": plan.model_identifier,
                "created_at": plan.created_at.isoformat() if hasattr(plan.created_at, "isoformat") else str(plan.created_at),
                "plan": plan_data,
                "profile_snapshot": snapshot_data,
                "version_history": json_lineage,
            }
        )

    csrf_token = generate_csrf_token(subject=str(user.id))
    return templates.TemplateResponse(
        request=request,
        name="plans/detail.html",
        context={
            "app_name": settings.app_name,
            "user": user,
            "plan_id": plan.id,
            "version": plan.version,
            "created_at": plan.created_at,
            "model_identifier": plan.model_identifier,
            "plan": plan_data,
            "days": days,
            "profile_snapshot": snapshot_data,
            "version_history": lineage,
            "csrf_token": csrf_token,
            "errors": [],
            "feedback_text": "",
        },
    )


# ==============================================================================
# 4. Revise Workout Plan (Feedback-based updates)
# ==============================================================================

@router.post("/plans/{plan_id}/revise")
@router.post("/api/plans/{plan_id}/revise")
async def revise_workout_plan(
    plan_id: int,
    request: Request,
    csrf_token: Optional[str] = Form(None),
    feedback_text: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
):
    """Revise an existing workout plan based on user feedback to generate a new version.

    Requirements & Safety Guarantees:
    - Ownership Check: Verified BEFORE calling Gemini (403 Forbidden for non-owners).
    - Feedback Validation: 5 to 1000 characters; rejected if empty or whitespace (422).
    - Concurrency & Debounce: Protected with active generation lock per user (409 Conflict).
    - Security Isolation: Feedback wrapped as untrusted data in prompt tags.
    - Non-destructive: Original plan is untouched; failure keeps current plan and returns feedback for retry.
    - Profile Invariance: User's saved fitness profile is NOT modified; feedback affects only this plan.
    - Transaction Safety: Database connection is closed during AI network call.
    - Version Lineage: Increments version, sets parent_plan_id, and records PlanFeedback.
    """
    is_json = _is_json_request(request)

    # 1. Parse feedback text from JSON body or Form
    raw_feedback = ""
    if is_json:
        try:
            body = await request.json()
            raw_feedback = body.get("feedback_text") or body.get("feedback") or ""
        except Exception:
            raw_feedback = ""
    else:
        raw_feedback = feedback_text or ""

    clean_feedback = raw_feedback.strip() if raw_feedback else ""

    # Helper function to render detail page on error
    def _render_error_page(source_plan_obj, error_msg: str, status_code: int = 400):
        try:
            p_data = json.loads(source_plan_obj.validated_plan_json)
        except Exception:
            p_data = {}
        try:
            s_data = json.loads(source_plan_obj.profile_snapshot)
        except Exception:
            s_data = {}
        with SessionLocal() as db_inner:
            lineage = _get_plan_lineage(db_inner, source_plan_obj, user.id)
        new_csrf = generate_csrf_token(subject=str(user.id))
        return templates.TemplateResponse(
            request=request,
            name="plans/detail.html",
            context={
                "app_name": settings.app_name,
                "user": user,
                "plan_id": source_plan_obj.id,
                "version": source_plan_obj.version,
                "created_at": source_plan_obj.created_at,
                "model_identifier": source_plan_obj.model_identifier,
                "plan": p_data,
                "days": p_data.get("days") or p_data.get("schedule") or [],
                "profile_snapshot": s_data,
                "version_history": lineage,
                "csrf_token": new_csrf,
                "errors": [error_msg],
                "feedback_text": clean_feedback,  # Preserved for retry!
            },
            status_code=status_code,
        )

    # 2. Verify CSRF for HTML form submissions
    if not is_json:
        if not csrf_token or not verify_csrf_token(csrf_token, subject=str(user.id)):
            with SessionLocal() as db:
                target_p = db.query(WorkoutPlan).filter(WorkoutPlan.id == plan_id).first()
            if target_p and (target_p.owner_id == user.id or user.role == "admin"):
                return _render_error_page(
                    target_p,
                    "Security check failed (invalid or expired CSRF token). Please try again.",
                    status.HTTP_403_FORBIDDEN,
                )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF verification failed.")

    # 3. Ownership Check: confirm source-plan ownership BEFORE calling Gemini
    source_plan_snapshot_json: str
    source_plan_data_json: str
    source_plan_id: int
    source_plan_version: int
    anonymous_profile: AnonymousFitnessProfile
    source_plan_copy: WorkoutPlan

    with SessionLocal() as db:
        source_plan = db.query(WorkoutPlan).filter(WorkoutPlan.id == plan_id).first()
        if not source_plan:
            if is_json:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source workout plan not found.")
            return templates.TemplateResponse(
                request=request,
                name="plans/index.html",
                context={
                    "app_name": settings.app_name,
                    "user": user,
                    "plans": [],
                    "csrf_token": generate_csrf_token(subject=str(user.id)),
                    "errors": [f"Workout plan #{plan_id} was not found."],
                    "has_profile": True,
                },
                status_code=status.HTTP_404_NOT_FOUND,
            )

        # Check ownership BEFORE invoking Gemini or processing further!
        if source_plan.owner_id != user.id and user.role != "admin":
            if is_json:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to revise this workout plan.",
                )
            return templates.TemplateResponse(
                request=request,
                name="plans/index.html",
                context={
                    "app_name": settings.app_name,
                    "user": user,
                    "plans": [],
                    "csrf_token": generate_csrf_token(subject=str(user.id)),
                    "errors": ["You do not have permission to revise this workout plan."],
                    "has_profile": True,
                },
                status_code=status.HTTP_403_FORBIDDEN,
            )

        # 4. Validate feedback length and reject empty submissions
        if not clean_feedback or len(clean_feedback) < 5:
            err_msg = "Feedback is required and must be at least 5 characters (e.g. 'Add more rest days')."
            if is_json:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=err_msg)
            return _render_error_page(source_plan, err_msg, status.HTTP_422_UNPROCESSABLE_ENTITY)

        if len(clean_feedback) > 1000:
            err_msg = "Feedback must not exceed 1000 characters."
            if is_json:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=err_msg)
            return _render_error_page(source_plan, err_msg, status.HTTP_422_UNPROCESSABLE_ENTITY)

        source_plan_snapshot_json = source_plan.profile_snapshot
        source_plan_data_json = source_plan.validated_plan_json
        source_plan_id = source_plan.id
        source_plan_version = source_plan.version
        source_plan_copy = source_plan

        try:
            snapshot_dict = json.loads(source_plan_snapshot_json)
            anonymous_profile = AnonymousFitnessProfile(
                age=int(snapshot_dict.get("age", 25)),
                weight_kg=float(snapshot_dict.get("weight_kg", 70.0)),
                fitness_goal=str(snapshot_dict.get("fitness_goal", "general_fitness")),
                workout_intensity=str(snapshot_dict.get("workout_intensity", "moderate")),
                experience_level=str(snapshot_dict.get("experience_level", "beginner")),
                equipment=str(snapshot_dict.get("equipment", "bodyweight")),
                available_minutes=int(snapshot_dict.get("available_minutes", 45)),
                exercise_limitations=snapshot_dict.get("exercise_limitations"),
            )
        except Exception as exc:
            logger.warning("Could not parse profile snapshot from plan, falling back to current profile: %s", exc)
            user_profile = db.query(FitnessProfile).filter(FitnessProfile.user_id == user.id).first()
            if not user_profile:
                raise HTTPException(status_code=400, detail="Fitness profile missing.")
            anonymous_profile = AnonymousFitnessProfile.from_profile(user_profile)
            source_plan_snapshot_json = json.dumps({
                "age": user_profile.age,
                "weight_kg": user_profile.weight_kg,
                "fitness_goal": user_profile.fitness_goal,
                "workout_intensity": user_profile.workout_intensity,
                "experience_level": user_profile.experience_level,
                "equipment": user_profile.equipment,
                "available_minutes": user_profile.available_minutes,
                "exercise_limitations": user_profile.exercise_limitations,
                "snapshot_at": datetime.now(timezone.utc).isoformat(),
            })

    # DB session is closed here. No DB lock or open transaction during Gemini network call!

    # 5. Prevent duplicate concurrent submissions
    async with _generation_lock:
        if user.id in _active_generations:
            err_msg = "A workout plan revision is already being processed for your account. Please wait a moment."
            if is_json:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=err_msg)
            return _render_error_page(source_plan_copy, err_msg, status.HTTP_409_CONFLICT)
        _active_generations.add(user.id)

    try:
        # 6. Call AI Service with feedback
        ai_service = get_ai_service()
        try:
            ai_result = await ai_service.generate_workout_plan(
                anonymous_profile=anonymous_profile,
                feedback_history=[clean_feedback],
                previous_plan_context=source_plan_data_json,
            )
        except AIServiceError as exc:
            logger.error("AI service error revising workout plan: %s", exc.message)
            if is_json:
                raise HTTPException(
                    status_code=exc.status_code if exc.status_code != 500 else status.HTTP_502_BAD_GATEWAY,
                    detail=exc.user_message,
                )
            return _render_error_page(
                source_plan_copy,
                exc.user_message,
                exc.status_code if exc.status_code != 500 else status.HTTP_502_BAD_GATEWAY,
            )

        # 7. Persist validated new version in DB and record PlanFeedback
        created_plan_id: int
        next_version: int

        with SessionLocal() as db:
            # Safe concurrency version assignment
            max_v = (
                db.query(func.max(WorkoutPlan.version))
                .filter(WorkoutPlan.owner_id == user.id)
                .scalar()
            ) or 0
            next_version = max(max_v, source_plan_version) + 1

            new_plan = WorkoutPlan(
                owner_id=user.id,
                version=next_version,
                parent_plan_id=source_plan_id,
                profile_snapshot=source_plan_snapshot_json,
                validated_plan_json=ai_result.data.model_dump_json(),
                model_identifier=ai_result.model_identifier,
                created_at=datetime.now(timezone.utc),
            )
            db.add(new_plan)
            db.flush()  # assign new_plan.id

            feedback_record = PlanFeedback(
                source_plan_id=source_plan_id,
                resulting_plan_id=new_plan.id,
                feedback_text=clean_feedback,
                created_at=datetime.now(timezone.utc),
            )
            db.add(feedback_record)
            db.commit()
            db.refresh(new_plan)
            created_plan_id = new_plan.id

        logger.info(
            "Successfully created revised workout plan #%d (v%d) from source #%d for user #%d",
            created_plan_id,
            next_version,
            source_plan_id,
            user.id,
        )

        if is_json:
            return JSONResponse(
                content={
                    "status": "success",
                    "plan_id": created_plan_id,
                    "version": next_version,
                    "parent_plan_id": source_plan_id,
                    "feedback": clean_feedback,
                    "change_summary": ai_result.data.change_summary,
                    "model_identifier": ai_result.model_identifier,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "plan": ai_result.data.model_dump(),
                },
                status_code=status.HTTP_201_CREATED,
            )

        return RedirectResponse(
            url=f"/plans/{created_plan_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    finally:
        async with _generation_lock:
            _active_generations.discard(user.id)

