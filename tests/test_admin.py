"""Comprehensive test suite for the protected FitBuddy Admin Dashboard.

Verifies:
1. Strict Role-Based Access Control:
   - Direct URL access to /admin and subroutes by an unauthenticated visitor is denied (303 to /login or 401).
   - Direct URL access to /admin and subroutes by an ordinary authenticated user is denied (403 Forbidden).
   - "Check: A normal user cannot access /admin by typing the URL."
2. Administrative Command:
   - Admin status can only be granted via CLI (manage.py create-admin / cli.create_admin).
   - Public registration or profile updates cannot elevate an account to admin.
3. Statistics Telemetry:
   - User, workout plan, revision, and tip counts reflect actual dynamic database records.
4. User Directory & Inspection:
   - Paginated user list with email/name search filtering.
   - User detail inspection shows profile snapshot, saved plans history, and generated tips.
5. Plan Inspection & Feedback Audit:
   - Full 7-day schedule inspection with day-by-day exercises, warm-ups, and cooldowns.
   - Version lineage tracking (parent plan link and derived child revisions).
   - Triggering and received revision feedback notes.
6. Model Telemetry & Mock vs Live Badge:
   - Clearly distinguishes between Mock Simulation and Live AI generation.
7. Security & Scope Boundaries:
   - Strict read-only MVP (no user impersonation or bulk destructive actions).
   - Never exposes password hashes, session secrets, API keys, or raw tokens.
"""

import json
import re
import uuid
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database import SessionLocal
from app.models import (
    User,
    FitnessProfile,
    WorkoutPlan,
    PlanFeedback,
    FitnessTip,
    UserRole,
)
from app.config import Settings
from app.cli import create_admin
from app.services.security import generate_csrf_token, verify_password


def extract_csrf_token(html: str) -> str:
    """Extract CSRF token from HTML form response."""
    match = re.search(r'name="csrf_token"\s+value="([^"]+)"', html)
    if not match:
        return generate_csrf_token("anonymous")
    return match.group(1)


def register_normal_user(client: TestClient, email: str = None, password: str = "NormalUser123!") -> dict:
    """Helper to register and log in an ordinary (non-admin) user with a profile."""
    uid = uuid.uuid4().hex[:8]
    if not email:
        email = f"user_{uid}@fitbuddy.test"

    reg_get = client.get("/register")
    client.post(
        "/register",
        data={
            "email": email,
            "password": password,
            "confirm_password": password,
            "csrf_token": extract_csrf_token(reg_get.text),
        },
        follow_redirects=False,
    )

    prof_get = client.get("/profile")
    client.post(
        "/profile",
        data={
            "display_name": f"Regular User {uid}",
            "age": "29",
            "weight_kg": "72.5",
            "fitness_goal": "muscle_gain",
            "workout_intensity": "medium",
            "experience_level": "intermediate",
            "equipment": "dumbbells",
            "available_minutes": "45",
            "exercise_limitations": "Minor knee soreness",
            "csrf_token": extract_csrf_token(prof_get.text),
        },
        follow_redirects=False,
    )
    return {"email": email, "password": password, "display_name": f"Regular User {uid}"}


def create_and_login_admin(client: TestClient) -> dict:
    """Helper to provision an admin via CLI and log them into the TestClient session."""
    uid = uuid.uuid4().hex[:8]
    email = f"admin_{uid}@fitbuddy.test"
    password = "SuperAdminPassword123!"

    # Provision exclusively through the documented local administrative command
    success = create_admin(email=email, password=password)
    assert success is True

    # Log in through web interface
    login_get = client.get("/login")
    login_res = client.post(
        "/login",
        data={
            "email": email,
            "password": password,
            "csrf_token": extract_csrf_token(login_get.text),
        },
        follow_redirects=False,
    )
    assert login_res.status_code == 303
    return {"email": email, "password": password}


# ==============================================================================
# 1. Access Control: Normal Users & Visitors are Denied Access
# ==============================================================================

def test_unauthenticated_visitor_cannot_access_admin_dashboard_or_endpoints():
    """Unauthenticated visitor cannot access admin routes (redirects or 401)."""
    client = TestClient(app)

    # 1. Browser GET to /admin should redirect to /login
    resp = client.get("/admin", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login"

    # 2. Browser GET with redirect followed should land on /login
    resp_followed = client.get("/admin", follow_redirects=True)
    assert resp_followed.status_code == 200
    assert "Log In" in resp_followed.text
    assert "Platform Administration" not in resp_followed.text

    # 3. Direct access to user inspection endpoint
    resp_user = client.get("/admin/users/1", follow_redirects=False)
    assert resp_user.status_code == 303

    # 4. Direct access to plan inspection endpoint
    resp_plan = client.get("/admin/plans/1", follow_redirects=False)
    assert resp_plan.status_code == 303

    # 5. API overview endpoint returns 401
    resp_api = client.get("/api/admin/overview")
    assert resp_api.status_code == 401
    assert "detail" in resp_api.json()


def test_normal_user_cannot_access_admin_by_typing_url():
    """Check: A normal user cannot access /admin by typing the URL (returns 403 Forbidden)."""
    client = TestClient(app)
    user_creds = register_normal_user(client)

    # Normal user types "/admin" in the address bar
    admin_get = client.get("/admin")
    assert admin_get.status_code == 403
    assert "admin authorization required" in admin_get.text.lower() or "denied" in admin_get.text.lower()

    # Normal user attempts to access user inspection page by URL
    user_inspect = client.get("/admin/users/1")
    assert user_inspect.status_code == 403

    # Normal user attempts to access plan inspection page by URL
    plan_inspect = client.get("/admin/plans/1")
    assert plan_inspect.status_code == 403

    # Normal user calls API overview
    api_get = client.get("/api/admin/overview")
    assert api_get.status_code == 403
    assert "admin authorization required" in api_get.json()["detail"].lower()

    # Normal user calls API users list
    api_users = client.get("/api/admin/users")
    assert api_users.status_code == 403

    # Cleanup DB
    with SessionLocal() as db:
        u = db.query(User).filter(User.email == user_creds["email"]).first()
        if u:
            db.delete(u)
            db.commit()


# ==============================================================================
# 2. Local Administrative Command & Role Elevation Immutability
# ==============================================================================

def test_admin_status_assigned_only_via_cli_not_public_registration_or_profile():
    """Verify admin status cannot be assigned through public registration or profile edit."""
    client = TestClient(app)
    uid = uuid.uuid4().hex[:8]
    email = f"sneaky_{uid}@example.com"
    password = "SneakyPassword123!"

    # 1. Attempt to pass role="admin" in registration form
    reg_get = client.get("/register")
    client.post(
        "/register",
        data={
            "email": email,
            "password": password,
            "confirm_password": password,
            "role": "admin",  # Sneaky parameter attempt
            "csrf_token": extract_csrf_token(reg_get.text),
        },
        follow_redirects=False,
    )

    with SessionLocal() as db:
        user_db = db.query(User).filter(User.email == email).first()
        assert user_db is not None
        assert user_db.role == UserRole.USER.value  # Role must strictly remain 'user'

    # 2. Attempt to pass role="admin" in profile edit
    prof_get = client.get("/profile")
    client.post(
        "/profile",
        data={
            "display_name": "Hacker",
            "age": "30",
            "weight_kg": "75.0",
            "fitness_goal": "weight_loss",
            "workout_intensity": "low",
            "experience_level": "beginner",
            "equipment": "bodyweight",
            "available_minutes": "30",
            "role": "admin",  # Sneaky parameter attempt
            "csrf_token": extract_csrf_token(prof_get.text),
        },
        follow_redirects=False,
    )

    with SessionLocal() as db:
        user_db = db.query(User).filter(User.email == email).first()
        assert user_db.role == UserRole.USER.value  # Still strictly 'user'

    # 3. Legitimate elevation via documented CLI command
    elevated = create_admin(email=email, password=password)
    assert elevated is True

    with SessionLocal() as db:
        user_db = db.query(User).filter(User.email == email).first()
        assert user_db.role == UserRole.ADMIN.value  # Now 'admin' via CLI

        # Cleanup
        db.delete(user_db)
        db.commit()


# ==============================================================================
# 3. Database Statistics Telemetry
# ==============================================================================

def test_actual_counts_of_users_plans_and_revisions_from_database():
    """Verify statistics displayed in the admin dashboard reflect live database counts."""
    admin_client = TestClient(app)
    admin_creds = create_and_login_admin(admin_client)

    # 1. Fetch live DB counts
    with SessionLocal() as db:
        expected_users = db.query(User).count()
        expected_plans = db.query(WorkoutPlan).count()
        expected_revisions = db.query(PlanFeedback).count()
        expected_tips = db.query(FitnessTip).count()

    # 2. Query admin overview HTML page
    dash_res = admin_client.get("/admin")
    assert dash_res.status_code == 200
    assert "Platform Administration" in dash_res.text
    assert f'id="count-users">\n        {expected_users}' in dash_res.text
    assert f'id="count-plans">\n        {expected_plans}' in dash_res.text
    assert f'id="count-revisions">\n        {expected_revisions}' in dash_res.text

    # 3. Query admin overview JSON API
    api_res = admin_client.get("/api/admin/overview")
    assert api_res.status_code == 200
    counts = api_res.json()["counts"]
    assert counts["users"] == expected_users
    assert counts["plans"] == expected_plans
    assert counts["revisions"] == expected_revisions
    assert counts["tips"] == expected_tips

    # 4. Now create a user, plan, and revision, and verify telemetry increments dynamically
    user_client = TestClient(app)
    user_creds = register_normal_user(user_client)

    with patch("app.services.ai.get_settings") as mock_settings:
        mock_settings.return_value = Settings(gemini_mock_enabled=True)
        gen_res = user_client.post("/api/plans/generate")
        assert gen_res.status_code == 201
        source_plan_id = gen_res.json()["plan_id"]

        # Submit revision
        rev_res = user_client.post(
            f"/api/plans/{source_plan_id}/revise",
            json={"feedback_text": "Please add more recovery days for my joints"},
        )
        assert rev_res.status_code == 201

        # Generate a tip
        tip_res = user_client.post("/api/tips/generate", json={"category": "nutrition"})
        assert tip_res.status_code == 201

    # Verify counts incremented in admin view
    api_res_after = admin_client.get("/api/admin/overview")
    new_counts = api_res_after.json()["counts"]
    assert new_counts["users"] == expected_users + 1
    assert new_counts["plans"] == expected_plans + 2  # initial + revision
    assert new_counts["revisions"] == expected_revisions + 1
    assert new_counts["tips"] == expected_tips + 1

    # Cleanup DB
    with SessionLocal() as db:
        for em in (admin_creds["email"], user_creds["email"]):
            u = db.query(User).filter(User.email == em).first()
            if u:
                db.delete(u)
        db.commit()


# ==============================================================================
# 4. Paginated User List and Search
# ==============================================================================

def test_admin_paginated_user_list_and_search():
    """Verify paginated user list and search filtering by email or display name."""
    admin_client = TestClient(app)
    admin_creds = create_and_login_admin(admin_client)

    # Create two users with distinct emails and display names
    c1 = TestClient(app)
    u1 = register_normal_user(c1, email=f"alpha_{uuid.uuid4().hex[:6]}@example.com")
    c2 = TestClient(app)
    u2 = register_normal_user(c2, email=f"beta_{uuid.uuid4().hex[:6]}@example.com")

    # 1. Search for alpha
    search_alpha = admin_client.get("/admin?q=alpha")
    assert search_alpha.status_code == 200
    assert u1["email"] in search_alpha.text
    assert u2["email"] not in search_alpha.text

    # 2. Search via API
    api_search = admin_client.get("/api/admin/users?q=beta")
    assert api_search.status_code == 200
    results = api_search.json()
    emails = [u["email"] for u in results["users"]]
    assert u2["email"] in emails
    assert u1["email"] not in emails

    # 3. Pagination bounds
    page_res = admin_client.get("/api/admin/users?page=1&per_page=1")
    assert page_res.status_code == 200
    page_data = page_res.json()
    assert page_data["page"] == 1
    assert page_data["per_page"] == 1
    assert len(page_data["users"]) == 1

    # Cleanup
    with SessionLocal() as db:
        for em in (admin_creds["email"], u1["email"], u2["email"]):
            u = db.query(User).filter(User.email == em).first()
            if u:
                db.delete(u)
        db.commit()


# ==============================================================================
# 5. Open User's Profile and Saved Plan History
# ==============================================================================

def test_admin_open_user_profile_and_saved_plan_history():
    """Admin can open a user's profile and saved plan history at /admin/users/{user_id}."""
    admin_client = TestClient(app)
    admin_creds = create_and_login_admin(admin_client)

    user_client = TestClient(app)
    user_creds = register_normal_user(user_client)

    # User generates a workout plan and a tip
    with patch("app.services.ai.get_settings") as mock_settings:
        mock_settings.return_value = Settings(gemini_mock_enabled=True)
        gen_res = user_client.post("/api/plans/generate")
        plan_id = gen_res.json()["plan_id"]
        tip_res = user_client.post("/api/tips/generate", json={"category": "nutrition"})
        tip_id = tip_res.json()["tip_id"]

    # Get target user's ID
    with SessionLocal() as db:
        target_user = db.query(User).filter(User.email == user_creds["email"]).first()
        target_id = target_user.id

    # Admin visits /admin/users/{user_id}
    inspect_res = admin_client.get(f"/admin/users/{target_id}")
    assert inspect_res.status_code == 200
    assert user_creds["email"] in inspect_res.text
    assert "Fitness Profile Parameters" in inspect_res.text
    assert "Minor knee soreness" in inspect_res.text  # Profile limitation
    assert f"#{plan_id}" in inspect_res.text  # Saved plan listed
    assert f"#{tip_id}" in inspect_res.text  # Generated tip listed

    # Admin visits non-existent user -> 404
    missing_res = admin_client.get("/admin/users/999999")
    assert missing_res.status_code == 404

    # Cleanup
    with SessionLocal() as db:
        for em in (admin_creds["email"], user_creds["email"]):
            u = db.query(User).filter(User.email == em).first()
            if u:
                db.delete(u)
        db.commit()


# ==============================================================================
# 6. Inspect Plan Versions and Related Feedback
# ==============================================================================

def test_admin_inspect_plan_versions_and_related_feedback():
    """Admin can inspect complete 7-day schedule, version lineage, and user feedback."""
    admin_client = TestClient(app)
    admin_creds = create_and_login_admin(admin_client)

    user_client = TestClient(app)
    user_creds = register_normal_user(user_client)

    # 1. User generates plan v1
    feedback_note = "Focus more on bodyweight cardio, reduce rest times to 30s"
    with patch("app.services.ai.get_settings") as mock_settings:
        mock_settings.return_value = Settings(gemini_mock_enabled=True)
        p1_res = user_client.post("/api/plans/generate")
        p1_id = p1_res.json()["plan_id"]

        # 2. User revises plan to create v2
        p2_res = user_client.post(
            f"/api/plans/{p1_id}/revise",
            json={"feedback_text": feedback_note},
        )
        p2_id = p2_res.json()["plan_id"]

    # Admin inspects plan v1
    v1_inspect = admin_client.get(f"/admin/plans/{p1_id}")
    assert v1_inspect.status_code == 200
    assert "Version Lineage &amp; Adaptation Feedback" in v1_inspect.text
    assert f"#{p2_id}" in v1_inspect.text  # Derived revision reference
    assert feedback_note in v1_inspect.text  # Feedback note received on v1

    # Admin inspects plan v2
    v2_inspect = admin_client.get(f"/admin/plans/{p2_id}")
    assert v2_inspect.status_code == 200
    assert "Version 2" in v2_inspect.text
    assert f"Plan #{p1_id}" in v2_inspect.text  # Parent plan reference
    assert feedback_note in v2_inspect.text  # Triggering feedback displayed
    # All 7 days are structured in the schedule
    for day_num in range(1, 8):
        assert f"Day {day_num}" in v2_inspect.text

    # JSON inspection endpoint
    api_plan = admin_client.get(f"/api/admin/plans/{p2_id}")
    assert api_plan.status_code == 200
    plan_json = api_plan.json()
    assert plan_json["plan"]["version"] == 2
    assert plan_json["plan"]["parent_plan_id"] == p1_id
    assert plan_json["triggering_feedback"]["feedback_text"] == feedback_note
    assert len(plan_json["plan_schedule"]["days"]) == 7

    # Non-existent plan -> 404
    missing_plan = admin_client.get("/admin/plans/999999")
    assert missing_plan.status_code == 404

    # Cleanup
    with SessionLocal() as db:
        for em in (admin_creds["email"], user_creds["email"]):
            u = db.query(User).filter(User.email == em).first()
            if u:
                db.delete(u)
        db.commit()


# ==============================================================================
# 7. Mock vs Live Generation Mode Indicator
# ==============================================================================

def test_admin_shows_output_generated_live_or_mock_mode():
    """Verify whether output was generated live or in mock/demo mode is transparently badged."""
    admin_client = TestClient(app)
    admin_creds = create_and_login_admin(admin_client)

    user_client = TestClient(app)
    user_creds = register_normal_user(user_client)

    # 1. Generate plan in mock mode
    with patch("app.services.ai.get_settings") as mock_settings:
        mock_settings.return_value = Settings(gemini_mock_enabled=True)
        mock_gen = user_client.post("/api/plans/generate")
        mock_plan_id = mock_gen.json()["plan_id"]

    mock_inspect = admin_client.get(f"/admin/plans/{mock_plan_id}")
    assert mock_inspect.status_code == 200
    assert "Mock Simulation Mode" in mock_inspect.text

    # 2. Insert a simulated live plan with live model identifier
    with SessionLocal() as db:
        u = db.query(User).filter(User.email == user_creds["email"]).first()
        mock_p = db.query(WorkoutPlan).filter(WorkoutPlan.id == mock_plan_id).first()

        live_plan = WorkoutPlan(
            owner_id=u.id,
            version=10,
            parent_plan_id=None,
            profile_snapshot=mock_p.profile_snapshot,
            validated_plan_json=mock_p.validated_plan_json,
            model_identifier="gemini-2.5-flash",
        )
        db.add(live_plan)
        db.commit()
        db.refresh(live_plan)
        live_plan_id = live_plan.id

    live_inspect = admin_client.get(f"/admin/plans/{live_plan_id}")
    assert live_inspect.status_code == 200
    assert "Live AI Generation (gemini-2.5-flash)" in live_inspect.text

    # Cleanup
    with SessionLocal() as db:
        for em in (admin_creds["email"], user_creds["email"]):
            user_obj = db.query(User).filter(User.email == em).first()
            if user_obj:
                db.delete(user_obj)
        db.commit()


# ==============================================================================
# 8. Security & Read-Only Constraints
# ==============================================================================

def test_sensitive_credentials_never_exposed_in_admin_views():
    """Verify password hashes, session secrets, API keys, or raw tokens are never exposed."""
    admin_client = TestClient(app)
    admin_creds = create_and_login_admin(admin_client)

    user_client = TestClient(app)
    user_creds = register_normal_user(user_client)

    # Query all admin HTML and JSON endpoints
    endpoints = [
        "/admin",
        "/api/admin/overview",
        "/api/admin/users",
        f"/admin/users/1",
        f"/api/admin/users/1",
    ]

    for ep in endpoints:
        res = admin_client.get(ep)
        body = res.text
        # Password hashes (Argon2id identifiers)
        assert "$argon2id$" not in body
        assert "password_hash" not in body
        # Session secrets or tokens
        assert "session_token" not in body
        assert "SESSION_SECRET" not in body
        # Raw API key pattern
        assert "AIza" not in body

    # Cleanup
    with SessionLocal() as db:
        for em in (admin_creds["email"], user_creds["email"]):
            u = db.query(User).filter(User.email == em).first()
            if u:
                db.delete(u)
        db.commit()


def test_admin_dashboard_strictly_read_only():
    """Verify admin endpoints are read-only: POST, PUT, DELETE return 405 Method Not Allowed."""
    admin_client = TestClient(app)
    admin_creds = create_and_login_admin(admin_client)

    # Disallow destructive modifications through admin routes
    assert admin_client.post("/admin").status_code == 405
    assert admin_client.delete("/admin").status_code == 405
    assert admin_client.post("/admin/users/1").status_code == 405
    assert admin_client.delete("/admin/users/1").status_code == 405
    assert admin_client.post("/admin/plans/1").status_code == 405
    assert admin_client.delete("/admin/plans/1").status_code == 405

    # Cleanup
    with SessionLocal() as db:
        u = db.query(User).filter(User.email == admin_creds["email"]).first()
        if u:
            db.delete(u)
            db.commit()
