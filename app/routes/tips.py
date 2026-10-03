"""Routes for requesting, listing, and retrieving daily nutrition and recovery tips."""

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
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import SessionLocal, get_db
from app.models import User, FitnessProfile, FitnessTip
from app.schemas import (
    FitnessTipResponseSchema,
    FitnessTipRequestSchema,
)
from app.services.ai import (
    get_ai_service,
    AIServiceError,
    GeminiQuotaExhaustedError,
)
from app.services.security import (
    get_current_user,
    generate_csrf_token,
    verify_csrf_token,
)

logger = logging.getLogger("fitbuddy.tips")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter(tags=["Nutrition & Recovery Tips"])
settings = get_settings()

# In-memory tracking for rate limiting & duplicate click prevention (local MVP single-process)
_active_tip_users: set[int] = set()
_tip_lock = asyncio.Lock()
_last_tip_timestamps: dict[int, datetime] = {}


def _is_json_request(request: Request) -> bool:
    """Determine if client expects a JSON response."""
    accept = request.headers.get("accept", "")
    content_type = request.headers.get("content-type", "")
    return (
        "application/json" in accept
        or "application/json" in content_type
        or request.url.path.startswith("/api/")
    )


def _format_tip_item(tip: FitnessTip) -> dict:
    """Safely deserialize tip_text JSON into structured presentation dictionary."""
    try:
        data = json.loads(tip.tip_text)
    except Exception:
        data = {}

    short_title = data.get("short_title") or data.get("title") or f"{tip.category.title()} Tip"
    practical_tip = data.get("practical_tip") or data.get("summary") or ""
    brief_explanation = data.get("brief_explanation") or data.get("scientific_rationale") or ""
    cautions = data.get("cautions") or []
    actionable_steps = data.get("actionable_steps") or []

    is_mock = "mock" in tip.model_identifier.lower()

    return {
        "id": tip.id,
        "owner_id": tip.owner_id,
        "goal": tip.goal,
        "category": tip.category,
        "model_identifier": tip.model_identifier,
        "is_mock": is_mock,
        "created_at": tip.created_at,
        "short_title": short_title,
        "practical_tip": practical_tip,
        "brief_explanation": brief_explanation,
        "cautions": cautions,
        "actionable_steps": actionable_steps,
    }


# ==============================================================================
# 1. View Tips Page (Latest Tip + History)
# ==============================================================================

@router.get("/tips", response_class=HTMLResponse)
async def tips_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Render daily tips dashboard with the latest saved tip and previous tips history."""
    profile = db.query(FitnessProfile).filter(FitnessProfile.user_id == user.id).first()
    has_profile = profile is not None
    user_goal = profile.fitness_goal if profile else "general_wellness"

    raw_tips = (
        db.query(FitnessTip)
        .filter(FitnessTip.owner_id == user.id)
        .order_by(FitnessTip.created_at.desc())
        .limit(15)
        .all()
    )

    formatted_tips = [_format_tip_item(t) for t in raw_tips]
    latest_tip = formatted_tips[0] if formatted_tips else None
    history_tips = formatted_tips[1:] if len(formatted_tips) > 1 else []

    csrf_token = generate_csrf_token(subject=str(user.id))
    return templates.TemplateResponse(
        request=request,
        name="tips/index.html",
        context={
            "app_name": settings.app_name,
            "user": user,
            "has_profile": has_profile,
            "profile": profile,
            "user_goal": user_goal,
            "latest_tip": latest_tip,
            "history_tips": history_tips,
            "csrf_token": csrf_token,
            "errors": [],
            "selected_category": "nutrition",
        },
    )


# ==============================================================================
# 2. Generate Daily Tip (Nutrition or Recovery)
# ==============================================================================

@router.post("/tips/generate")
@router.post("/api/tips/generate")
async def generate_tip(
    request: Request,
    csrf_token: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
):
    """Generate and save a concise nutrition or recovery tip based on the user's saved goal.

    Safety, Scoping & Isolation Guarantees:
    - Sends ONLY the context needed for the tip (goal and category; no PII).
    - Rate-limited and debounce-protected for local MVP single process.
    - Duplicate click lock prevents concurrent duplicate calls.
    - Zero database transactions held open during the AI network call.
    - Persists successful tips with owner, goal, category, model identifier, and timestamp.
    - Never replaces a failed real request with an unlabeled static or mock result.
    """
    is_json = _is_json_request(request)

    # 1. Parse category from JSON or form
    req_category: str = ""
    if is_json:
        try:
            body = await request.json()
            req_category = body.get("category", "")
        except Exception:
            req_category = ""
    else:
        req_category = category or ""

    clean_category = str(req_category).strip().lower()
    if clean_category not in ("nutrition", "recovery"):
        err_msg = "Category must be either 'nutrition' or 'recovery'."
        if is_json:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=err_msg)
        return _render_tips_error(request, user, [err_msg], status.HTTP_422_UNPROCESSABLE_ENTITY)

    # 2. Verify CSRF for HTML requests
    if not is_json:
        if not csrf_token or not verify_csrf_token(csrf_token, subject=str(user.id)):
            return _render_tips_error(
                request,
                user,
                ["Security check failed (invalid or expired CSRF token). Please try again."],
                status.HTTP_403_FORBIDDEN,
            )

    # 3. Retrieve user profile and goal in a short DB transaction
    saved_goal: str
    with SessionLocal() as db:
        profile = db.query(FitnessProfile).filter(FitnessProfile.user_id == user.id).first()
        if not profile:
            err_msg = "Please set up your fitness profile before generating daily tips."
            if is_json:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)
            return _render_tips_error(request, user, [err_msg], status.HTTP_400_BAD_REQUEST)
        saved_goal = profile.fitness_goal

        # 4. Rate Limiting & Cooldown Protection (3-second debounce on recent tip)
        latest_recent = (
            db.query(FitnessTip)
            .filter(FitnessTip.owner_id == user.id)
            .order_by(FitnessTip.created_at.desc())
            .first()
        )
        if latest_recent and latest_recent.created_at:
            now_utc = datetime.now(timezone.utc)
            tip_time = latest_recent.created_at
            if tip_time.tzinfo is None:
                tip_time = tip_time.replace(tzinfo=timezone.utc)
            diff = (now_utc - tip_time).total_seconds()
            if diff < 3.0:
                err_msg = "Please wait a moment before requesting another tip."
                if is_json:
                    raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=err_msg)
                return _render_tips_error(request, user, [err_msg], status.HTTP_429_TOO_MANY_REQUESTS)

    # 5. Duplicate Click / In-Flight Concurrency Protection
    async with _tip_lock:
        if user.id in _active_tip_users:
            err_msg = "A tip request is already currently in progress. Please wait a moment."
            if is_json:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=err_msg)
            return _render_tips_error(request, user, [err_msg], status.HTTP_409_CONFLICT)
        _active_tip_users.add(user.id)

    try:
        # 6. Invoke AI Generation Service
        # (Zero DB sessions open during the AI network call!)
        ai_service = get_ai_service()
        try:
            ai_result = await ai_service.generate_fitness_tip(
                goal=saved_goal,
                category=clean_category,
            )
        except GeminiQuotaExhaustedError as exc:
            logger.warning("Gemini quota exhausted while generating tip: %s", exc)
            if is_json:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="AI service quota reached or rate-limited. Please wait a short while and retry.",
                )
            return _render_tips_error(
                request,
                user,
                ["AI service quota reached or rate-limited. Please wait a short while and retry."],
                status.HTTP_429_TOO_MANY_REQUESTS,
            )
        except AIServiceError as exc:
            logger.error("AI service error generating tip: %s", exc.message)
            status_code = exc.status_code if exc.status_code != 500 else status.HTTP_502_BAD_GATEWAY
            if is_json:
                raise HTTPException(status_code=status_code, detail=exc.user_message)
            return _render_tips_error(request, user, [exc.user_message], status_code)

        # 7. Persist Validated Tip in Short DB Transaction
        created_tip_id: int
        with SessionLocal() as db:
            new_tip = FitnessTip(
                owner_id=user.id,
                goal=saved_goal,
                category=clean_category,
                tip_text=ai_result.data.model_dump_json(),
                model_identifier=ai_result.model_identifier,
                created_at=datetime.now(timezone.utc),
            )
            db.add(new_tip)
            db.commit()
            db.refresh(new_tip)
            created_tip_id = new_tip.id

        _last_tip_timestamps[user.id] = datetime.now(timezone.utc)

        logger.info(
            "Successfully generated and saved %s tip #%d for user #%d (goal=%s)",
            clean_category,
            created_tip_id,
            user.id,
            saved_goal,
        )

        if is_json:
            return JSONResponse(
                content={
                    "status": "success",
                    "tip_id": created_tip_id,
                    "category": clean_category,
                    "goal": saved_goal,
                    "model_identifier": ai_result.model_identifier,
                    "is_mock": ai_result.is_mock,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "tip": ai_result.data.model_dump(),
                },
                status_code=status.HTTP_201_CREATED,
            )

        return RedirectResponse(url="/tips", status_code=status.HTTP_303_SEE_OTHER)

    finally:
        async with _tip_lock:
            _active_tip_users.discard(user.id)


def _render_tips_error(request: Request, user: User, errors: list[str], status_code: int):
    """Helper to render tips template with error feedback."""
    with SessionLocal() as db:
        profile = db.query(FitnessProfile).filter(FitnessProfile.user_id == user.id).first()
        raw_tips = (
            db.query(FitnessTip)
            .filter(FitnessTip.owner_id == user.id)
            .order_by(FitnessTip.created_at.desc())
            .limit(15)
            .all()
        )
    formatted = [_format_tip_item(t) for t in raw_tips]
    return templates.TemplateResponse(
        request=request,
        name="tips/index.html",
        context={
            "app_name": settings.app_name,
            "user": user,
            "has_profile": profile is not None,
            "profile": profile,
            "user_goal": profile.fitness_goal if profile else "general_wellness",
            "latest_tip": formatted[0] if formatted else None,
            "history_tips": formatted[1:] if len(formatted) > 1 else [],
            "csrf_token": generate_csrf_token(subject=str(user.id)),
            "errors": errors,
            "selected_category": "nutrition",
        },
        status_code=status_code,
    )


# ==============================================================================
# 3. JSON List & Retrieval Endpoints with Strict Ownership Isolation
# ==============================================================================

@router.get("/api/tips")
async def list_user_tips(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all tips owned by the authenticated user in JSON format."""
    tips = (
        db.query(FitnessTip)
        .filter(FitnessTip.owner_id == user.id)
        .order_by(FitnessTip.created_at.desc())
        .all()
    )

    result = []
    for t in tips:
        item = _format_tip_item(t)
        if hasattr(item["created_at"], "isoformat"):
            item["created_at"] = item["created_at"].isoformat()
        result.append(item)

    return JSONResponse(content=result)


@router.get("/api/tips/{tip_id}")
async def get_single_tip(
    tip_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve one specific tip owned by the authenticated user.

    Strict Ownership Isolation:
    - If tip belongs to another user and current user is not admin, returns 403 Forbidden.
    """
    tip = db.query(FitnessTip).filter(FitnessTip.id == tip_id).first()
    if not tip:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tip not found.")

    if tip.owner_id != user.id and user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this tip.",
        )

    item = _format_tip_item(tip)
    if hasattr(item["created_at"], "isoformat"):
        item["created_at"] = item["created_at"].isoformat()

    return JSONResponse(content=item)
