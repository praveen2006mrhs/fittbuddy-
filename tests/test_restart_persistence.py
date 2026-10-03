"""Explicit verification of database persistence across application restart and multi-user profile isolation."""

import pytest
import re
import uuid
from fastapi.testclient import TestClient

from app.main import create_app
from app.database import SessionLocal
from app.models import User
from app.services.security import generate_csrf_token


def extract_csrf_token(html: str) -> str:
    match = re.search(r'name="csrf_token"\s+value="([^"]+)"', html)
    if not match:
        return generate_csrf_token("anonymous")
    return match.group(1)


def test_app_restart_persistence_and_multiuser_isolation():
    """
    Simulates:
    1. User A registers and saves profile on App Instance 1.
    2. App shuts down and restarts (Instance 2).
    3. User A logs in on Instance 2 and verifies their profile is restored.
    4. User B registers on Instance 2 and verifies they cannot see User A's data.
    """
    uid = uuid.uuid4().hex[:8]
    email_a = f"dan_{uid}@fitbuddy.test"
    email_b = f"elena_{uid}@fitbuddy.test"

    # -------------------------------------------------------------
    # Session 1: App Instance 1
    # -------------------------------------------------------------
    app1 = create_app()
    client1 = TestClient(app1)

    reg_get = client1.get("/register")
    csrf1 = extract_csrf_token(reg_get.text)

    # Register User A (Dan)
    reg_res = client1.post(
        "/register",
        data={
            "email": email_a,
            "password": "DanPassword123",
            "confirm_password": "DanPassword123",
            "csrf_token": csrf1,
        },
        follow_redirects=False,
    )
    assert reg_res.status_code == 303

    p_get1 = client1.get("/profile")
    csrf_profile1 = extract_csrf_token(p_get1.text)

    # Save Dan's profile
    save_res = client1.post(
        "/profile",
        data={
            "display_name": "Dan Power",
            "age": "32",
            "weight_kg": "85.5",
            "fitness_goal": "muscle_gain",
            "workout_intensity": "high",
            "experience_level": "advanced",
            "equipment": "full_gym",
            "available_minutes": "75",
            "exercise_limitations": "Rotator cuff precaution",
            "csrf_token": csrf_profile1,
        },
    )
    assert save_res.status_code == 200
    assert "Fitness profile saved successfully!" in save_res.text

    # Log out Dan
    logout_csrf = extract_csrf_token(save_res.text)
    client1.post("/logout", data={"csrf_token": logout_csrf})

    # Simulate App Shutdown
    del client1
    del app1

    # -------------------------------------------------------------
    # Session 2: App Restart (New App Instance 2)
    # -------------------------------------------------------------
    app2 = create_app()
    client2_dan = TestClient(app2)

    login_get = client2_dan.get("/login")
    login_csrf = extract_csrf_token(login_get.text)

    # Dan logs back in after restart
    login_res = client2_dan.post(
        "/login",
        data={
            "email": email_a,
            "password": "DanPassword123",
            "csrf_token": login_csrf,
        },
        follow_redirects=False,
    )
    assert login_res.status_code == 303

    # Dan reads profile on restarted app: data is persisted!
    dan_restarted_profile = client2_dan.get("/profile")
    assert dan_restarted_profile.status_code == 200
    assert "Dan Power" in dan_restarted_profile.text
    assert 'value="85.5"' in dan_restarted_profile.text
    assert "Rotator cuff precaution" in dan_restarted_profile.text

    # -------------------------------------------------------------
    # Session 3: User B (Elena) on App Instance 2
    # -------------------------------------------------------------
    client2_elena = TestClient(app2)
    reg_elena = client2_elena.get("/register")
    csrf_elena = extract_csrf_token(reg_elena.text)

    client2_elena.post(
        "/register",
        data={
            "email": email_b,
            "password": "ElenaPassword456",
            "confirm_password": "ElenaPassword456",
            "csrf_token": csrf_elena,
        },
        follow_redirects=False,
    )

    # Elena accesses profile: should NOT see Dan's details
    elena_profile = client2_elena.get("/profile")
    assert elena_profile.status_code == 200
    assert "Dan Power" not in elena_profile.text
    assert "Rotator cuff precaution" not in elena_profile.text
    assert 'value="85.5"' not in elena_profile.text

    # Clean up test users
    db = SessionLocal()
    u_a = db.query(User).filter(User.email == email_a).first()
    u_b = db.query(User).filter(User.email == email_b).first()
    if u_a:
        db.delete(u_a)
    if u_b:
        db.delete(u_b)
    db.commit()
    db.close()
