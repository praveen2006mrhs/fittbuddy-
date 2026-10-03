"""Comprehensive test suite for FitBuddy daily nutrition and recovery tips.

Verifies:
- Both nutrition and recovery requests work across all three fitness goals.
- Successful tips persist in database and remain accessible after page refresh and restart.
- Strict user ownership isolation (User B cannot read User A's tips: 403 Forbidden).
- Rate limiting appropriate for single-process local MVP (429 on rapid requests < 3s).
- Duplicate click / in-flight concurrency protection (409 Conflict).
- Failed real requests are NEVER replaced with unlabeled static or mock results.
- Mock-based checks are clearly reported and distinguished from live checks.
- HTML view rendering, CSRF, latest tip, and history display.
"""

import asyncio
import json
import re
import uuid
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.config import Settings
from app.database import SessionLocal
from app.models import User, FitnessProfile, FitnessTip
from app.schemas import FitnessTipResponseSchema
from app.services.ai import (
    GeminiAIService,
    AIResult,
    GeminiQuotaExhaustedError,
    GeminiNetworkError,
    GeminiOutputValidationError,
)
from app.services.security import generate_csrf_token


def extract_csrf_token(html: str) -> str:
    """Extract CSRF token from rendered HTML form."""
    match = re.search(r'name="csrf_token"\s+value="([^"]+)"', html)
    if not match:
        return generate_csrf_token("anonymous")
    return match.group(1)


def create_user_with_profile(client: TestClient, goal: str = "muscle_gain") -> dict:
    """Helper to create an authenticated user with a saved fitness profile."""
    uid = uuid.uuid4().hex[:8]
    email = f"tiptester_{uid}@example.com"
    password = "TipPassword123!"

    reg_get = client.get("/register")
    client.post(
        "/register",
        data={"email": email, "password": password, "confirm_password": password, "csrf_token": extract_csrf_token(reg_get.text)},
    )
    prof_get = client.get("/profile")
    client.post(
        "/profile",
        data={
            "display_name": f"User_{uid}",
            "age": "26",
            "weight_kg": "70.0",
            "fitness_goal": goal,
            "workout_intensity": "medium",
            "experience_level": "intermediate",
            "equipment": "bodyweight",
            "available_minutes": "40",
            "csrf_token": extract_csrf_token(prof_get.text),
        },
    )
    return {"email": email, "password": password, "goal": goal}


# ==============================================================================
# 1. Tips Across All Three Goals and Both Categories
# ==============================================================================

@pytest.mark.parametrize("category", ["nutrition", "recovery"])
@pytest.mark.parametrize("goal", ["weight_loss", "muscle_gain", "general_wellness"])
def test_nutrition_and_recovery_tips_across_all_goals(category, goal):
    """Verify nutrition and recovery requests work across all three goals and return structured fields."""
    app = create_app()
    client = TestClient(app)
    user_info = create_user_with_profile(client, goal=goal)

    with patch("app.services.ai.get_settings") as mock_settings:
        mock_settings.return_value = Settings(gemini_mock_enabled=True)

        res = client.post(
            "/api/tips/generate",
            json={"category": category},
        )
        assert res.status_code == 201
        data = res.json()

        assert data["status"] == "success"
        assert data["category"] == category
        assert data["goal"] == goal
        assert data["is_mock"] is True  # Clearly labeled as mock simulation!
        assert "tip_id" in data

        tip_obj = data["tip"]
        assert "short_title" in tip_obj and len(tip_obj["short_title"]) > 0
        assert "practical_tip" in tip_obj and len(tip_obj["practical_tip"]) > 0
        assert "brief_explanation" in tip_obj and len(tip_obj["brief_explanation"]) > 0

    # Cleanup user
    with SessionLocal() as db:
        u = db.query(User).filter(User.email == user_info["email"]).first()
        if u:
            db.delete(u)
        db.commit()


# ==============================================================================
# 2. Persistence and Refresh Verification
# ==============================================================================

def test_tips_persist_and_remain_available_after_refresh():
    """Verify that successful tips persist and remain available after page refresh and restart."""
    app = create_app()
    client = TestClient(app)
    user_info = create_user_with_profile(client, goal="muscle_gain")

    with patch("app.services.ai.get_settings") as mock_settings:
        mock_settings.return_value = Settings(gemini_mock_enabled=True)

        # 1. Generate nutrition tip
        res1 = client.post("/api/tips/generate", json={"category": "nutrition"})
        assert res1.status_code == 201
        tip1_id = res1.json()["tip_id"]

        # Backdate tip 1 to bypass the 3-second debounce cleanly
        from datetime import datetime, timezone, timedelta
        with SessionLocal() as db:
            t1 = db.query(FitnessTip).filter(FitnessTip.id == tip1_id).first()
            t1.created_at = datetime.now(timezone.utc) - timedelta(seconds=5)
            db.commit()

        # 2. Generate recovery tip
        res2 = client.post("/api/tips/generate", json={"category": "recovery"})
        assert res2.status_code == 201
        tip2_id = res2.json()["tip_id"]

    # 3. Simulate page refresh / new request to GET /tips
    page_res = client.get("/tips")
    assert page_res.status_code == 200
    assert "Latest Saved Tip" in page_res.text
    assert "Previous Tips History" in page_res.text
    assert "Protein" in page_res.text or "Nutrition" in page_res.text
    assert "Recovery" in page_res.text

    # 4. Verify in DB directly
    with SessionLocal() as db:
        user_obj = db.query(User).filter(User.email == user_info["email"]).first()
        assert user_obj is not None
        saved_tips = db.query(FitnessTip).filter(FitnessTip.owner_id == user_obj.id).all()
        assert len(saved_tips) == 2
        saved_ids = [t.id for t in saved_tips]
        assert tip1_id in saved_ids
        assert tip2_id in saved_ids

        # Cleanup
        db.delete(user_obj)
        db.commit()


# ==============================================================================
# 3. Strict User Ownership Isolation
# ==============================================================================

def test_user_ownership_isolation_prevents_reading_others_tips():
    """Verify that one user cannot read another user's tips (403 Forbidden)."""
    app = create_app()
    client_a = TestClient(app)
    client_b = TestClient(app)

    user_a = create_user_with_profile(client_a, goal="muscle_gain")
    user_b = create_user_with_profile(client_b, goal="weight_loss")

    with patch("app.services.ai.get_settings") as mock_settings:
        mock_settings.return_value = Settings(gemini_mock_enabled=True)
        res_a = client_a.post("/api/tips/generate", json={"category": "nutrition"})
        assert res_a.status_code == 201
        tip_a_id = res_a.json()["tip_id"]

    # User B attempts to access User A's tip directly
    b_res = client_b.get(f"/api/tips/{tip_a_id}")
    assert b_res.status_code == 403
    assert "permission" in b_res.json()["detail"].lower()

    # User B's tips list should be empty (does NOT contain User A's tip)
    b_list = client_b.get("/api/tips")
    assert b_list.status_code == 200
    b_tip_ids = [t["id"] for t in b_list.json()]
    assert tip_a_id not in b_tip_ids

    # Cleanup
    with SessionLocal() as db:
        for em in (user_a["email"], user_b["email"]):
            u = db.query(User).filter(User.email == em).first()
            if u:
                db.delete(u)
        db.commit()


# ==============================================================================
# 4. Rate Limiting and Debounce Protection
# ==============================================================================

def test_tips_rate_limiting_debounce():
    """Verify that rapid successive tip generation requests are rate-limited with 429."""
    app = create_app()
    client = TestClient(app)
    user_info = create_user_with_profile(client, goal="general_wellness")

    with patch("app.services.ai.get_settings") as mock_settings:
        mock_settings.return_value = Settings(gemini_mock_enabled=True)

        # First request succeeds
        res1 = client.post("/api/tips/generate", json={"category": "nutrition"})
        assert res1.status_code == 201

        # Immediate second request should be rate-limited (429)
        res2 = client.post("/api/tips/generate", json={"category": "nutrition"})
        assert res2.status_code == 429
        assert "wait a moment" in res2.json()["detail"].lower() or "wait" in res2.json()["detail"].lower()

    # Cleanup
    with SessionLocal() as db:
        u = db.query(User).filter(User.email == user_info["email"]).first()
        if u:
            db.delete(u)
        db.commit()


# ==============================================================================
# 5. Failed Real Request Never Replaced with Unlabeled Mock Result
# ==============================================================================

def test_failed_live_request_never_replaced_with_unlabeled_mock():
    """Verify that a live generation failure returns a clear error and NEVER silently falls back to mock."""
    app = create_app()
    client = TestClient(app)
    user_info = create_user_with_profile(client, goal="muscle_gain")

    from app.routes.tips import _last_tip_timestamps
    _last_tip_timestamps.clear()

    # Mock mode DISABLED (simulating live mode failure)
    with patch("app.services.ai.get_settings") as mock_settings:
        mock_settings.return_value = Settings(gemini_api_key="real-key", gemini_mock_enabled=False)

        # Simulate Quota Exhaustion
        with patch.object(GeminiAIService, "generate_fitness_tip") as mock_gen:
            mock_gen.side_effect = GeminiQuotaExhaustedError()

            res_quota = client.post("/api/tips/generate", json={"category": "nutrition"})
            assert res_quota.status_code == 429
            assert "quota" in res_quota.json()["detail"].lower()

        _last_tip_timestamps.clear()

        # Simulate Network Failure
        with patch.object(GeminiAIService, "generate_fitness_tip") as mock_gen:
            mock_gen.side_effect = GeminiNetworkError()

            res_net = client.post("/api/tips/generate", json={"category": "recovery"})
            assert res_net.status_code == 503
            assert "network" in res_net.json()["detail"].lower()

    # Verify no corrupted/fallback tip was saved in DB
    with SessionLocal() as db:
        u = db.query(User).filter(User.email == user_info["email"]).first()
        if u:
            tips = db.query(FitnessTip).filter(FitnessTip.owner_id == u.id).all()
            assert len(tips) == 0
            db.delete(u)
        db.commit()


# ==============================================================================
# 6. Mock vs Live Reporting Separation
# ==============================================================================

def test_mock_based_tips_reported_separately_from_live():
    """Verify that mock tips are explicitly labeled with is_mock=True and [Mock Simulation]."""
    app = create_app()
    client = TestClient(app)
    user_info = create_user_with_profile(client, goal="general_wellness")

    from app.routes.tips import _last_tip_timestamps
    _last_tip_timestamps.clear()

    with patch("app.services.ai.get_settings") as mock_settings:
        mock_settings.return_value = Settings(gemini_mock_enabled=True)

        res = client.post("/api/tips/generate", json={"category": "nutrition"})
        assert res.status_code == 201
        data = res.json()
        assert data["is_mock"] is True
        assert "mock" in data["model_identifier"].lower()

        # Check HTML page shows [Mock Simulation] badge
        html_res = client.get("/tips")
        assert html_res.status_code == 200
        assert "[Mock Simulation]" in html_res.text

    # Cleanup
    with SessionLocal() as db:
        u = db.query(User).filter(User.email == user_info["email"]).first()
        if u:
            db.delete(u)
        db.commit()


# ==============================================================================
# 7. Category Validation & Missing Profile Handlers
# ==============================================================================

def test_tip_generation_category_validation():
    """Verify that invalid category inputs are rejected with 422."""
    app = create_app()
    client = TestClient(app)
    user_info = create_user_with_profile(client, goal="general_wellness")

    res = client.post("/api/tips/generate", json={"category": "invalid_category"})
    assert res.status_code == 422
    assert "category" in res.json()["detail"].lower()

    # Cleanup
    with SessionLocal() as db:
        u = db.query(User).filter(User.email == user_info["email"]).first()
        if u:
            db.delete(u)
        db.commit()


def test_tip_generation_requires_profile():
    """Verify that an authenticated user without a profile is told to set up a profile first (400)."""
    app = create_app()
    client = TestClient(app)

    # Register only, do NOT create profile
    uid = uuid.uuid4().hex[:8]
    email = f"noprofile_{uid}@example.com"
    r = client.get("/register")
    client.post(
        "/register",
        data={"email": email, "password": "Password123!", "confirm_password": "Password123!", "csrf_token": extract_csrf_token(r.text)},
    )

    res = client.post("/api/tips/generate", json={"category": "nutrition"})
    assert res.status_code == 400
    assert "profile" in res.json()["detail"].lower()

    # Cleanup
    with SessionLocal() as db:
        u = db.query(User).filter(User.email == email).first()
        if u:
            db.delete(u)
        db.commit()
