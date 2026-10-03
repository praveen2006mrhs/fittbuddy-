"""Protected administrative dashboard routes for FitBuddy.

Read-only inspection of platform metrics, users, workout plans, revisions,
and Google Gemini / mock generation metadata.
"""

import json
import math
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import (
    APIRouter,
    Depends,
    Request,
    Query,
    HTTPException,
    status,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import (
    User,
    FitnessProfile,
    WorkoutPlan,
    PlanFeedback,
    FitnessTip,
    UserRole,
)
from app.services.security import (
    require_admin_user,
    generate_csrf_token,
)

logger = logging.getLogger("fitbuddy.admin")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter(tags=["Admin"])
settings = get_settings()


def _is_json_request(request: Request) -> bool:
    """Determine if client expects a JSON response."""
    accept = request.headers.get("accept", "")
    content_type = request.headers.get("content-type", "")
    return (
        "application/json" in accept
        or "application/json" in content_type
        or request.url.path.startswith("/api/")
    )


def _is_mock_mode(model_identifier: Optional[str]) -> bool:
    """Determine whether the output was generated live or via mock simulation."""
    if not model_identifier:
        return True
    m = model_identifier.lower()
    return "mock" in m or "demo" in m or "simulation" in m


def _safe_json_loads(data: Optional[str]) -> Dict[str, Any]:
    """Safely parse JSON string into a dictionary."""
    if not data:
        return {}
    try:
        return json.loads(data)
    except Exception:
        return {}


# ==============================================================================
# 1. Admin Dashboard Overview & Statistics
# ==============================================================================

@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    q: Optional[str] = Query(None),
    admin: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    """Render the protected Admin Dashboard with database metrics and user listing."""
    # 1. Actual counts directly from database
    total_users = db.query(func.count(User.id)).scalar() or 0
    total_plans = db.query(func.count(WorkoutPlan.id)).scalar() or 0
    total_revisions = db.query(func.count(PlanFeedback.id)).scalar() or 0
    total_tips = db.query(func.count(FitnessTip.id)).scalar() or 0

    stats = {
        "users": total_users,
        "plans": total_plans,
        "revisions": total_revisions,
        "tips": total_tips,
    }

    # 2. Query users with search and pagination
    user_query = db.query(User).outerjoin(FitnessProfile)
    q_clean = q.strip() if q else ""
    if q_clean:
        user_query = user_query.filter(
            or_(
                User.email.ilike(f"%{q_clean}%"),
                FitnessProfile.display_name.ilike(f"%{q_clean}%"),
            )
        )

    total_matching = user_query.count()
    total_pages = max(1, math.ceil(total_matching / per_page))
    if page > total_pages and total_pages > 0:
        page = total_pages

    users_db = (
        user_query.order_by(User.created_at.desc(), User.id.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    # Format users for display (Strictly omitting password_hash and tokens!)
    users_list = []
    for u in users_db:
        profile = u.profile
        users_list.append({
            "id": u.id,
            "email": u.email,
            "role": u.role,
            "display_name": profile.display_name if profile else "—",
            "fitness_goal": (
                profile.fitness_goal.replace("_", " ").title()
                if profile and profile.fitness_goal
                else "—"
            ),
            "plans_count": len(u.workout_plans),
            "created_at": (
                u.created_at.strftime("%Y-%m-%d %H:%M UTC")
                if u.created_at
                else "—"
            ),
        })

    # 3. Recent plans across all users
    recent_plans_db = (
        db.query(WorkoutPlan)
        .order_by(WorkoutPlan.created_at.desc())
        .limit(5)
        .all()
    )
    recent_plans = []
    for p in recent_plans_db:
        p_json = _safe_json_loads(p.validated_plan_json)
        is_mock = _is_mock_mode(p.model_identifier)
        recent_plans.append({
            "id": p.id,
            "owner_email": p.owner.email if p.owner else "—",
            "title": p_json.get("plan_title", f"Plan #{p.id}"),
            "version": p.version,
            "model_identifier": p.model_identifier,
            "is_mock": is_mock,
            "mode_label": "Mock Simulation" if is_mock else f"Live AI ({p.model_identifier})",
            "created_at": (
                p.created_at.strftime("%Y-%m-%d %H:%M UTC")
                if p.created_at
                else "—"
            ),
        })

    if _is_json_request(request):
        return JSONResponse({
            "stats": stats,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_matching": total_matching,
                "total_pages": total_pages,
                "q": q_clean,
            },
            "users": users_list,
            "recent_plans": recent_plans,
        })

    csrf_token = generate_csrf_token(subject=str(admin.id))
    return templates.TemplateResponse(
        request=request,
        name="admin/index.html",
        context={
            "app_name": settings.app_name,
            "user": admin,
            "csrf_token": csrf_token,
            "stats": stats,
            "users": users_list,
            "recent_plans": recent_plans,
            "page": page,
            "per_page": per_page,
            "total_matching": total_matching,
            "total_pages": total_pages,
            "q": q_clean,
            "has_prev": page > 1,
            "has_next": page < total_pages,
            "prev_page": page - 1,
            "next_page": page + 1,
        },
    )


@router.get("/api/admin/overview", response_class=JSONResponse)
async def api_admin_overview(
    admin: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    """JSON API returning platform statistics."""
    total_users = db.query(func.count(User.id)).scalar() or 0
    total_plans = db.query(func.count(WorkoutPlan.id)).scalar() or 0
    total_revisions = db.query(func.count(PlanFeedback.id)).scalar() or 0
    total_tips = db.query(func.count(FitnessTip.id)).scalar() or 0

    return {
        "counts": {
            "users": total_users,
            "plans": total_plans,
            "revisions": total_revisions,
            "tips": total_tips,
        }
    }


@router.get("/api/admin/users", response_class=JSONResponse)
async def api_admin_users(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    q: Optional[str] = Query(None),
    admin: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    """JSON API returning paginated users list without sensitive fields."""
    user_query = db.query(User).outerjoin(FitnessProfile)
    q_clean = q.strip() if q else ""
    if q_clean:
        user_query = user_query.filter(
            or_(
                User.email.ilike(f"%{q_clean}%"),
                FitnessProfile.display_name.ilike(f"%{q_clean}%"),
            )
        )

    total_matching = user_query.count()
    total_pages = max(1, math.ceil(total_matching / per_page))

    users_db = (
        user_query.order_by(User.created_at.desc(), User.id.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    users_data = []
    for u in users_db:
        profile = u.profile
        users_data.append({
            "id": u.id,
            "email": u.email,
            "role": u.role,
            "display_name": profile.display_name if profile else None,
            "fitness_goal": profile.fitness_goal if profile else None,
            "plans_count": len(u.workout_plans),
            "created_at": u.created_at.isoformat() if u.created_at else None,
        })

    return {
        "page": page,
        "per_page": per_page,
        "total_count": total_matching,
        "total_pages": total_pages,
        "users": users_data,
    }


# ==============================================================================
# 2. User Detail & History Inspection
# ==============================================================================

@router.get("/admin/users/{user_id}", response_class=HTMLResponse)
@router.get("/api/admin/users/{user_id}", response_class=JSONResponse)
async def admin_inspect_user(
    request: Request,
    user_id: int,
    admin: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    """Inspect a user's fitness profile snapshot, saved plans, and tips history."""
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} was not found.",
        )

    # Profile info
    profile = target_user.profile
    profile_data = None
    if profile:
        profile_data = {
            "display_name": profile.display_name,
            "age": profile.age,
            "weight_kg": profile.weight_kg,
            "fitness_goal": profile.fitness_goal,
            "workout_intensity": profile.workout_intensity,
            "experience_level": profile.experience_level,
            "equipment": profile.equipment,
            "available_minutes": profile.available_minutes,
            "exercise_limitations": profile.exercise_limitations or "None declared",
            "updated_at": (
                profile.updated_at.strftime("%Y-%m-%d %H:%M UTC")
                if profile.updated_at
                else "—"
            ),
        }

    # Plans history
    plans_db = (
        db.query(WorkoutPlan)
        .filter(WorkoutPlan.owner_id == target_user.id)
        .order_by(WorkoutPlan.created_at.desc(), WorkoutPlan.version.desc())
        .all()
    )

    plans_data = []
    for p in plans_db:
        p_json = _safe_json_loads(p.validated_plan_json)
        is_mock = _is_mock_mode(p.model_identifier)
        revisions_count = (
            db.query(func.count(WorkoutPlan.id))
            .filter(WorkoutPlan.parent_plan_id == p.id)
            .scalar()
            or 0
        )
        plans_data.append({
            "id": p.id,
            "version": p.version,
            "parent_plan_id": p.parent_plan_id,
            "title": p_json.get("plan_title", f"Workout Plan #{p.id}"),
            "goal": p_json.get("goal", "—"),
            "model_identifier": p.model_identifier,
            "is_mock": is_mock,
            "mode_label": "Mock Simulation" if is_mock else f"Live AI ({p.model_identifier})",
            "revisions_count": revisions_count,
            "created_at": (
                p.created_at.strftime("%Y-%m-%d %H:%M UTC")
                if p.created_at
                else "—"
            ),
            "created_at_iso": p.created_at.isoformat() if p.created_at else None,
        })

    # Tips history
    tips_db = (
        db.query(FitnessTip)
        .filter(FitnessTip.owner_id == target_user.id)
        .order_by(FitnessTip.created_at.desc())
        .all()
    )

    tips_data = []
    for t in tips_db:
        t_json = _safe_json_loads(t.tip_text)
        is_mock = _is_mock_mode(t.model_identifier)
        tips_data.append({
            "id": t.id,
            "goal": t.goal,
            "category": t.category,
            "title": t_json.get("short_title", f"Tip #{t.id}"),
            "practical_tip": t_json.get("practical_tip", ""),
            "brief_explanation": t_json.get("brief_explanation", ""),
            "model_identifier": t.model_identifier,
            "is_mock": is_mock,
            "mode_label": "Mock Simulation" if is_mock else f"Live AI ({t.model_identifier})",
            "created_at": (
                t.created_at.strftime("%Y-%m-%d %H:%M UTC")
                if t.created_at
                else "—"
            ),
            "created_at_iso": t.created_at.isoformat() if t.created_at else None,
        })

    user_info = {
        "id": target_user.id,
        "email": target_user.email,
        "role": target_user.role,
        "created_at": (
            target_user.created_at.strftime("%Y-%m-%d %H:%M UTC")
            if target_user.created_at
            else "—"
        ),
        "created_at_iso": target_user.created_at.isoformat() if target_user.created_at else None,
    }

    if _is_json_request(request):
        return JSONResponse({
            "user": user_info,
            "profile": profile_data,
            "plans": plans_data,
            "tips": tips_data,
        })

    csrf_token = generate_csrf_token(subject=str(admin.id))
    return templates.TemplateResponse(
        request=request,
        name="admin/user_detail.html",
        context={
            "app_name": settings.app_name,
            "user": admin,
            "csrf_token": csrf_token,
            "target_user": user_info,
            "profile": profile_data,
            "plans": plans_data,
            "tips": tips_data,
        },
    )


# ==============================================================================
# 3. Plan Detail, Version Lineage & Feedback Inspection
# ==============================================================================

@router.get("/admin/plans/{plan_id}", response_class=HTMLResponse)
@router.get("/api/admin/plans/{plan_id}", response_class=JSONResponse)
async def admin_inspect_plan(
    request: Request,
    plan_id: int,
    admin: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    """Inspect full structured 7-day schedule, parent/version lineage, feedback, and model mode."""
    plan = db.query(WorkoutPlan).filter(WorkoutPlan.id == plan_id).first()
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workout plan with ID {plan_id} was not found.",
        )

    # Parse stored JSON payloads
    plan_data = _safe_json_loads(plan.validated_plan_json)
    profile_snapshot = _safe_json_loads(plan.profile_snapshot)
    is_mock = _is_mock_mode(plan.model_identifier)

    # 1. Feedback received on this plan (feedbacks submitted where source_plan_id == plan.id)
    feedbacks_received_db = (
        db.query(PlanFeedback)
        .filter(PlanFeedback.source_plan_id == plan.id)
        .order_by(PlanFeedback.created_at.desc())
        .all()
    )
    feedbacks_received = []
    for f in feedbacks_received_db:
        feedbacks_received.append({
            "id": f.id,
            "resulting_plan_id": f.resulting_plan_id,
            "feedback_text": f.feedback_text,
            "created_at": (
                f.created_at.strftime("%Y-%m-%d %H:%M UTC")
                if f.created_at
                else "—"
            ),
            "created_at_iso": f.created_at.isoformat() if f.created_at else None,
        })

    # 2. Triggering feedback (if this plan is a revision resulting from a parent plan)
    triggering_feedback = None
    if plan.parent_plan_id:
        tf_record = (
            db.query(PlanFeedback)
            .filter(PlanFeedback.resulting_plan_id == plan.id)
            .first()
        )
        if tf_record:
            triggering_feedback = {
                "id": tf_record.id,
                "source_plan_id": tf_record.source_plan_id,
                "feedback_text": tf_record.feedback_text,
                "created_at": (
                    tf_record.created_at.strftime("%Y-%m-%d %H:%M UTC")
                    if tf_record.created_at
                    else "—"
                ),
                "created_at_iso": tf_record.created_at.isoformat() if tf_record.created_at else None,
            }

    # 3. Revisions derived from this plan
    revisions_db = (
        db.query(WorkoutPlan)
        .filter(WorkoutPlan.parent_plan_id == plan.id)
        .order_by(WorkoutPlan.version.asc())
        .all()
    )
    revisions_derived = []
    for r in revisions_db:
        r_json = _safe_json_loads(r.validated_plan_json)
        revisions_derived.append({
            "id": r.id,
            "version": r.version,
            "title": r_json.get("plan_title", f"Revision #{r.id}"),
            "created_at": (
                r.created_at.strftime("%Y-%m-%d %H:%M UTC")
                if r.created_at
                else "—"
            ),
        })

    owner_info = {
        "id": plan.owner.id if plan.owner else None,
        "email": plan.owner.email if plan.owner else "—",
        "display_name": (
            plan.owner.profile.display_name
            if plan.owner and plan.owner.profile
            else "—"
        ),
    }

    plan_metadata = {
        "id": plan.id,
        "version": plan.version,
        "parent_plan_id": plan.parent_plan_id,
        "model_identifier": plan.model_identifier,
        "is_mock": is_mock,
        "mode_label": "Mock Simulation" if is_mock else f"Live AI ({plan.model_identifier})",
        "created_at": (
            plan.created_at.strftime("%Y-%m-%d %H:%M UTC")
            if plan.created_at
            else "—"
        ),
        "created_at_iso": plan.created_at.isoformat() if plan.created_at else None,
    }

    if _is_json_request(request):
        return JSONResponse({
            "plan": plan_metadata,
            "owner": owner_info,
            "plan_schedule": plan_data,
            "profile_snapshot": profile_snapshot,
            "triggering_feedback": triggering_feedback,
            "feedbacks_received": feedbacks_received,
            "revisions_derived": revisions_derived,
        })

    csrf_token = generate_csrf_token(subject=str(admin.id))
    return templates.TemplateResponse(
        request=request,
        name="admin/plan_detail.html",
        context={
            "app_name": settings.app_name,
            "user": admin,
            "csrf_token": csrf_token,
            "plan": plan_metadata,
            "owner": owner_info,
            "plan_schedule": plan_data,
            "profile_snapshot": profile_snapshot,
            "triggering_feedback": triggering_feedback,
            "feedbacks_received": feedbacks_received,
            "revisions_derived": revisions_derived,
        },
    )
